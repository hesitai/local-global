"""Grouped-gate regression tests; no training or real cache writes.

Run with PYTHONDONTWRITEBYTECODE=1 and the joint-cu118-test Python.
The user backup is only read, including when importing its source modules.
"""
import argparse
import contextlib
import importlib.util
import io
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

STUDENT = Path(__file__).resolve().parents[1]
ROOT = STUDENT.parent
BACKUP = ROOT.parent / 'joint2D copy'
sys.path.insert(0, str(STUDENT))
from kernel_response import ResponseConditionedLocal, response_loss
from model import Uformer
from options import Options
from dataset.dataset_motiondeblur import Online2DPatchTrainDataset
import online_2d

torch.set_num_threads(2)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_kwargs():
    return dict(img_size=256, embed_dim=8, depths=[1]*9,
                num_heads=[1, 1, 2, 4, 8, 8, 4, 2, 1], use_ddem=True,
                ddem_dim=8, ddem_blocks=1, use_local_branch=True,
                use_channel_aware=True, double_process=True,
                use_kernel_response=True, drop_path_rate=0)


class GroupedControllerTests(unittest.TestCase):
    def test_initialization_rng_and_legacy_weights(self):
        reference = import_file('_reference_response', BACKUP / 'only student  3000/kernel_response.py')
        torch.manual_seed(1234)
        old = reference.ResponseConditionedLocal(8)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(1234)
        one = ResponseConditionedLocal(8, groups=1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.manual_seed(1234)
        grouped = ResponseConditionedLocal(8, groups=4)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, one.state_dict()[key]), key)
            expected = value.repeat(4, 1, 1, 1) if key == 'gate.weight' else value.repeat(4) if key == 'gate.bias' else value
            self.assertTrue(torch.equal(expected, grouped.state_dict()[key]), key)
        x, r = torch.rand(2, 16, 24, 8), torch.rand(2, 64, 2, 3)
        torch.testing.assert_close(old(x, r), one(x, r), rtol=0, atol=0)
        torch.testing.assert_close(one(x, r), grouped(x, r), rtol=1e-6, atol=1e-6)
        self.assertEqual(sum(p.numel() for p in grouped.parameters()) - sum(p.numel() for p in one.parameters()), 396)
        with self.assertRaises(RuntimeError):
            grouped.load_state_dict(old.state_dict(), strict=True)
        for groups in [0, -1, 3]:
            with self.assertRaises(ValueError):
                ResponseConditionedLocal(8, groups=groups)

    def test_group_expert_axis_and_channel_mapping(self):
        model = ResponseConditionedLocal(8, groups=4)
        probabilities = torch.tensor([[.7, .1, .1, .1], [.1, .7, .1, .1],
                                      [.1, .1, .7, .1], [.1, .1, .1, .7]])
        with torch.no_grad():
            model.gate.weight.zero_()
            model.gate.bias.copy_(probabilities.log().reshape(-1))
            for i, expert in enumerate(model.experts):
                expert.weight.zero_()
                expert.weight[:, 0, 1, 1] = i + 1
        x = torch.rand(2, 16, 24, 8)
        response = torch.rand(2, 64, 2, 3)
        output, stats = model(x, response, shift_size=4, return_stats=True)
        gains = (probabilities * torch.arange(1., 5.)).sum(1).repeat_interleave(2)
        torch.testing.assert_close(output, x * gains, atol=1e-6, rtol=1e-6)
        self.assertEqual(stats.shape, (2, 4, 7))
        self.assertFalse(stats.requires_grad)
        torch.testing.assert_close(stats[:, :, :4].sum(-1), torch.ones(2, 4))
        torch.testing.assert_close(stats[0, :, :4], probabilities)
        self.assertGreater(float(stats[:, :, -1].min()), 0)
        self.assertTrue(torch.isfinite(stats).all())

    def test_stats_do_not_change_outputs_rng_or_gradients(self):
        model = ResponseConditionedLocal(8, groups=4)
        x = torch.rand(2, 16, 24, 8, requires_grad=True)
        r = torch.rand(2, 64, 2, 3, requires_grad=True)
        rng = torch.get_rng_state().clone()
        y = model(x, r)
        grads = torch.autograd.grad(y.square().mean(), [x, r, model.gate.weight])
        z, stats = model(x, r, return_stats=True)
        grads_stats = torch.autograd.grad(z.square().mean(), [x, r, model.gate.weight])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(y, z, rtol=0, atol=0)
        for a, b in zip(grads, grads_stats):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        group_grads = grads[-1].reshape(4, 4, 32)
        self.assertTrue((group_grads.abs().sum((1, 2)) > 0).all())
        self.assertFalse(torch.equal(group_grads[0], group_grads[1]))


class GroupedModelTests(unittest.TestCase):
    def test_full_graph_parity_and_gradient(self):
        # Build the actual backed-up model with its own response implementation,
        # not an old model silently importing the newly edited helper.
        old_response = import_file('_old_response_for_model', BACKUP / 'only student  3000/kernel_response.py')
        old_model = import_file('_old_group_baseline_model', BACKUP / 'only student  3000/model.py')
        torch.manual_seed(1234)
        with patch.dict(sys.modules, {'kernel_response': old_response}):
            baseline = old_model.Uformer(**model_kwargs()).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(1234)
        grouped = Uformer(**model_kwargs(), kernel_response_groups=4).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.manual_seed(1234)
        one = Uformer(**model_kwargs(), kernel_response_groups=1).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, one.state_dict()[key]), key)
            expected = value
            if '.response_local_second.gate.weight' in key:
                expected = value.repeat(4, 1, 1, 1)
            elif '.response_local_second.gate.bias' in key:
                expected = value.repeat(4)
            self.assertTrue(torch.equal(expected, grouped.state_dict()[key]), key)
        self.assertEqual(sum(p.numel() for p in grouped.parameters()) - sum(p.numel() for p in one.parameters()), 3168)
        controllers = [(n, m) for n, m in grouped.named_modules() if isinstance(m, ResponseConditionedLocal)]
        self.assertEqual(len(controllers), 8)
        self.assertTrue(all(m.groups == 4 and not n.startswith('conv.') for n, m in controllers))
        x = torch.rand(2, 3, 128, 256)
        with torch.no_grad():
            old_result = baseline(x)
            torch.testing.assert_close(one(x), old_result, rtol=0, atol=0)
            grouped_result, prior = grouped(x, return_ddem_features=True, collect_response_stats=True)
            torch.testing.assert_close(grouped_result, old_result, rtol=2e-5, atol=2e-6)
            self.assertEqual(len(prior['response_gate_stats']), 8)
            for stats in prior['response_gate_stats'].values():
                self.assertEqual(stats.shape, (2, 4, 7))
        del baseline, one
        grouped.train()
        x = torch.rand(1, 3, 256, 256)
        restored, prior = grouped(x, return_ddem_features=True, collect_response_stats=True)
        self.assertEqual(prior['kernel_response'].shape, (1, 64, 16, 16))
        auxiliary = response_loss(prior['kernel_response'], torch.zeros_like(prior['kernel_response']))[0]
        ((restored - x).square().mean() + .01 * auxiliary).backward()
        for name, module in controllers:
            grad = module.gate.weight.grad.reshape(4, -1)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertTrue((grad.abs().sum(1) > 0).all(), name)
        self.assertGreater(float(grouped.ddem.response_head[-1].weight.grad.abs().sum()), 0)
        torch.nn.utils.clip_grad_norm_(grouped.parameters(), 1.)
        torch.optim.AdamW(grouped.parameters(), lr=3e-4, weight_decay=.02).step()
        self.assertTrue(all(torch.isfinite(p).all() for p in grouped.parameters()))


class ProtocolTests(unittest.TestCase):
    def test_generator_label_loader_sources_unchanged(self):
        protected = [
            'motion-flow-syn-python-flat/generate_online_2d_epoch.py',
            'motion-flow-syn-python-flat/script_uav_global_blur_multisubframe_no_depth.py',
            'motion-flow-syn-python-flat/kernel_response_labels.py',
            'only student  3000/online_2d.py',
            'only student  3000/dataset/dataset_motiondeblur.py',
        ]
        for path in protected:
            self.assertEqual((ROOT / path).read_bytes(), (BACKUP / path).read_bytes(), path)

    def test_all_sources_all_500_epochs_select_identical_variants(self):
        reference = import_file('_previous_group_selection', BACKUP / 'only student  3000/dataset/dataset_motiondeblur.py')
        gt = STUDENT / 'dataset/UAV_single_image_pose_split/fixed_protocol_first_half_60_78_visdrone_all/train_full_7520'
        paths = sorted(p.resolve() for p in gt.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})
        self.assertEqual(len(paths), 7520)
        # Metadata-only enumeration of every epoch/source; no 90240-image
        # generation or repeated filesystem existence checks are required.
        old = SimpleNamespace(sample_seed=1234)
        new = SimpleNamespace(sample_seed=1234)
        rows = [dict(source_path=str(p), variant_index=str(v),
                     input_path='old-cache/{:06d}_v{:02d}.png'.format(i, v))
                for i, p in enumerate(paths) for v in range(12)]
        new_rows = [dict(r, input_path=r['input_path'].replace('old-cache', 'g4-cache')) for r in rows]
        for epoch in range(1, 501):
            old.training_epoch = new.training_epoch = epoch
            a = reference.Online2DPatchTrainDataset._sample_epoch_rows(old, rows)
            b = Online2DPatchTrainDataset._sample_epoch_rows(new, new_rows)
            self.assertEqual([r['variant_index'] for r in a], [r['variant_index'] for r in b], epoch)
        print('PARITY PASS: 7520 actual source identities x 500 epochs; cache path change does not affect selection', flush=True)

    def test_launcher_and_generator_command_parity(self):
        env = os.environ.copy()
        env.update(KERNEL_RESPONSE='1', DATASET_SCOPE='full', EPOCHS='500', BATCH_SIZE='8',
                   GEN_WORKERS='24', BLUR_BACKEND='cpu', POOL_VARIANTS='12', POOL_REFRESH_EPOCHS='50',
                   DRY_RUN='1', RESUME='0', KEEP_CACHE='0', RUN_TAG='', PYTHONDONTWRITEBYTECODE='1')
        def command(script, groups):
            result = subprocess.run([sys.executable, str(script)], env=dict(env, RESPONSE_GROUPS=str(groups)),
                                    capture_output=True, text=True, check=True)
            parts = shlex.split(result.stdout.strip().splitlines()[-1])
            return Options().init(argparse.ArgumentParser()).parse_args(parts[2:])
        current = command(STUDENT / 'train_online_2d.py', 4)
        previous = command(STUDENT / 'train_online_2d.py', 1)
        # Evaluate backed-up launcher source at the ORIGINAL project identity.
        # No backup file is edited; its __file__-based constants now resolve to
        # the paths actually used by the previous formal experiment.
        original_path = STUDENT / 'train_online_2d.py'
        backup_path = BACKUP / 'only student  3000/train_online_2d.py'
        namespace = {'__file__': str(original_path), '__name__': '_read_only_backup_launcher'}
        exec(compile(backup_path.read_text(), str(backup_path), 'exec'), namespace)
        printed = io.StringIO()
        # Select legacy groups explicitly: the backup may now default to v2.
        with patch.dict(os.environ, dict(env, RESPONSE_GROUPS='1')), contextlib.redirect_stdout(printed):
            self.assertEqual(namespace['main'](), 0)
        old_parts = shlex.split(printed.getvalue().strip().splitlines()[-1])
        old_options = Options().init(argparse.ArgumentParser()).parse_args(old_parts[2:])
        for key, value in vars(old_options).items():
            if key not in ('kernel_response_groups', 'response_gate_log_interval',
                           'env', 'online_2d_cache_root'):
                self.assertEqual(value, vars(current)[key], key)
        self.assertEqual(current.kernel_response_groups, 4)
        self.assertEqual(current.nepoch, 500)
        self.assertEqual(current.online_2d_workers, 24)
        self.assertEqual(current.batch_size, 8)
        self.assertEqual(current.response_gate_log_interval, 1)
        self.assertIn('serial_g4_v2', current.env)
        self.assertIn('serial_v1_', previous.env)
        self.assertNotEqual(current.online_2d_cache_root, previous.online_2d_cache_root)
        self.assertEqual(previous.env, old_options.env)
        self.assertEqual(previous.online_2d_cache_root, old_options.online_2d_cache_root)
        for epoch in [1, 51, 101, 451]:
            a, b = online_2d.generation_command(old_options, epoch), online_2d.generation_command(current, epoch)
            a[a.index('--cache-root') + 1] = 'same-cache'
            b[b.index('--cache-root') + 1] = 'same-cache'
            self.assertEqual(a, b)
        # Ensure parser/model-level defaults retain compatibility with old tests.
        defaults = Options().init(argparse.ArgumentParser()).parse_args([])
        self.assertEqual(defaults.kernel_response_groups, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

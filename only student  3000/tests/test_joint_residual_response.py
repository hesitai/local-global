"""Regression checks for joint content/response residual control.

The default suite runs entirely on CPU and creates no dataset caches, training
logs, or real checkpoints. Set RUN_CUDA_TESTS=1 to include the small AMP check.
"""

import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernel_response import (
    JointResidualResponseConditionedLocal,
    ResponseConditionedLocal,
)
from model import Uformer

torch.set_num_threads(2)


def model_kwargs():
    """Exercise all eight injection stages with a small actual Uformer."""
    return dict(
        img_size=256, embed_dim=8, depths=[1] * 9,
        num_heads=[1, 1, 2, 4, 8, 8, 4, 2, 1],
        use_ddem=True, ddem_dim=8, ddem_blocks=1,
        use_local_branch=True, use_channel_aware=True,
        double_process=True, use_kernel_response=True,
        kernel_response_groups=4, drop_path_rate=0,
    )


def make_model(variant='joint_residual'):
    return Uformer(kernel_response_variant=variant, **model_kwargs())


def import_reference(name, path):
    """Load the user's pre-change source without modifying the backup."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class JointResidualControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_initialization_and_expert_geometry(self):
        module = JointResidualResponseConditionedLocal(8, groups=4)
        x = torch.randn(2, 16, 24, 8)
        response = torch.randn(2, 64, 2, 3)
        output, stats = module(x, response, return_stats=True)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(stats.shape, (2, 4, 10))
        self.assertFalse(stats.requires_grad)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(stats).all())
        torch.testing.assert_close(stats[:, :, :4].sum(-1), torch.ones(2, 4))
        torch.testing.assert_close(
            stats[:, :, 7], torch.full((2, 4), .1), atol=.01, rtol=0)
        self.assertTrue((stats[:, :, 8] >= 0).all())
        self.assertTrue((stats[:, :, 9] > 0).all())
        for index, expert in enumerate(module.experts, 1):
            self.assertEqual(expert.kernel_size, (3, 3))
            self.assertEqual(expert.dilation, (index, index))
            self.assertEqual(expert.groups, 8)
            self.assertGreater(float(expert.weight.abs().sum()), 0)
            # Residual experts must not receive the legacy +identity center.
            self.assertLess(float(expert.weight[:, 0, 1, 1].abs().mean()), .5)

    def test_restoration_gradient_reaches_every_component(self):
        module = JointResidualResponseConditionedLocal(8, groups=4)
        x = torch.randn(2, 16, 24, 8, requires_grad=True)
        response = torch.randn(2, 64, 2, 3, requires_grad=True)
        target = torch.randn_like(x)
        # No response auxiliary loss: the restoration path itself must use R.
        (module(x, response) - target).square().mean().backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        self.assertGreater(float(response.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(response.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        torch.optim.AdamW(module.parameters(), lr=3e-4).step()
        self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))

    def test_group_mapping_strength_and_identity(self):
        module = JointResidualResponseConditionedLocal(8, groups=4)
        probabilities = torch.tensor([
            [.7, .1, .1, .1], [.1, .7, .1, .1],
            [.1, .1, .7, .1], [.1, .1, .1, .7],
        ])
        strengths = torch.tensor([.1, .2, .3, .4])
        with torch.no_grad():
            module.gate.weight.zero_()
            module.gate.bias.copy_(probabilities.log().reshape(-1))
            module.strength.weight.zero_()
            module.strength.bias.copy_(torch.logit(strengths))
            for index, expert in enumerate(module.experts, 1):
                expert.weight.zero_()
                expert.weight[:, 0, 1, 1] = index
        x = torch.randn(2, 16, 24, 8)
        response = torch.randn(2, 64, 2, 3)
        output, stats = module(x, response, return_stats=True)
        delta = (probabilities * torch.arange(1., 5.)).sum(1)
        gains = (1 + strengths * delta).repeat_interleave(2)
        torch.testing.assert_close(output, x * gains, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(stats[0, :, :4], probabilities)
        torch.testing.assert_close(stats[0, :, 7], strengths)
        self.assertTrue((stats[:, :, 8] < 1e-6).all())
        with torch.no_grad():
            for expert in module.experts:
                expert.weight.zero_()
        torch.testing.assert_close(module(x, response), x, rtol=0, atol=0)

    def test_joint_input_alignment_under_window_shift(self):
        module = JointResidualResponseConditionedLocal(8, groups=4).eval()
        captured = []
        handle = module.fusion.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone()))
        x = torch.randn(2, 16, 24, 8)
        response = torch.randn(2, 64, 2, 3)
        try:
            with torch.no_grad():
                module(x, response, shift_size=0)
                module(torch.roll(x, (-4, -4), (1, 2)), response, shift_size=4)
        finally:
            handle.remove()
        self.assertEqual(captured[0].shape[1], 48)
        torch.testing.assert_close(
            captured[1], torch.roll(captured[0], (-4, -4), (-2, -1)),
            rtol=0, atol=0)
        # Product channels must describe the aligned C/Q interaction.
        torch.testing.assert_close(
            captured[0][:, 32:], captured[0][:, :16] * captured[0][:, 16:32],
            rtol=0, atol=0)

    def test_stats_preserve_outputs_gradients_and_rng(self):
        module = JointResidualResponseConditionedLocal(8, groups=4)
        x = torch.randn(2, 16, 24, 8, requires_grad=True)
        response = torch.randn(2, 64, 2, 3, requires_grad=True)
        state = torch.get_rng_state().clone()
        parameters = [x, response, *module.parameters()]
        output = module(x, response)
        grads = torch.autograd.grad(output.square().mean(), parameters)
        with_stats, stats = module(x, response, return_stats=True)
        grads_with_stats = torch.autograd.grad(with_stats.square().mean(), parameters)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(stats.requires_grad)
        torch.testing.assert_close(output, with_stats, rtol=0, atol=0)
        for first, second in zip(grads, grads_with_stats):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_response_changes_output_and_checkpoint_versions_are_strict(self):
        module = JointResidualResponseConditionedLocal(8, groups=4).eval()
        x = torch.randn(2, 16, 24, 8)
        response = torch.randn(2, 64, 2, 3)
        with torch.no_grad():
            correct = module(x, response)
            changed = module(x, -response)
        self.assertGreater(float((correct - changed).abs().max()), 0)
        legacy = ResponseConditionedLocal(8, groups=4)
        with self.assertRaises(RuntimeError):
            module.load_state_dict(legacy.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            legacy.load_state_dict(module.state_dict(), strict=True)

    def test_invalid_configuration(self):
        for kwargs in (
            {'groups': 0}, {'groups': 3}, {'initial_strength': 0},
            {'initial_strength': 1}, {'initial_strength': float('nan')},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                JointResidualResponseConditionedLocal(8, **kwargs)


class LegacyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.backup = PROJECT_ROOT.parent / 'only student  3000 copy'
        if not (self.backup / 'kernel_response.py').is_file():
            self.skipTest('Read-only pre-change user backup is unavailable')
        self.reference_response = import_reference(
            '_joint_residual_reference_response', self.backup / 'kernel_response.py')

    def test_serial_controller_matches_user_backup_exactly(self):
        torch.manual_seed(1234)
        reference = self.reference_response.ResponseConditionedLocal(8, groups=4)
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(1234)
        current = ResponseConditionedLocal(8, groups=4)
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        self.assertEqual(set(reference.state_dict()), set(current.state_dict()))
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
        x = torch.randn(2, 16, 24, 8)
        response = torch.randn(2, 64, 2, 3)
        with torch.no_grad():
            expected, old_stats = reference(x, response, shift_size=4, return_stats=True)
            actual, new_stats = current(x, response, shift_size=4, return_stats=True)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(old_stats, new_stats, rtol=0, atol=0)

    def test_model_common_initialization_and_global_rng_are_preserved(self):
        model_path = self.backup / 'model.py'
        if not model_path.is_file():
            self.skipTest('Pre-change model source is unavailable')
        reference_model = import_reference('_joint_residual_reference_model', model_path)
        torch.manual_seed(1234)
        # The backed-up model must instantiate its own original response class.
        with patch.dict(sys.modules, {'kernel_response': self.reference_response}):
            reference = reference_model.Uformer(**model_kwargs()).eval()
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(1234)
        serial = make_model('serial').eval()
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        self.assertEqual(set(reference.state_dict()), set(serial.state_dict()))
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, serial.state_dict()[name], rtol=0, atol=0)
        torch.manual_seed(1234)
        joint = make_model().eval()
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        for name, value in reference.state_dict().items():
            if '.response_local_second.' not in name:
                torch.testing.assert_close(value, joint.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(any('ddem.response_head.' in key for key in reference.state_dict()))
        x = torch.randn(1, 3, 128, 256)
        with torch.no_grad():
            torch.testing.assert_close(reference(x), serial(x), rtol=0, atol=0)


class JointResidualModelTests(unittest.TestCase):
    def test_full_model_backward_optimizer_and_state_roundtrip(self):
        torch.manual_seed(1234)
        model = make_model()
        controllers = [module for module in model.modules()
                       if isinstance(module, JointResidualResponseConditionedLocal)]
        self.assertEqual(len(controllers), 8)
        self.assertTrue(all(module.groups == 4 for module in controllers))
        x = torch.rand(1, 3, 256, 256)
        restored, prior = model(
            x, return_ddem_features=True, collect_response_stats=True)
        self.assertEqual(restored.shape, x.shape)
        self.assertEqual(prior['kernel_response'].shape, (1, 64, 16, 16))
        self.assertEqual(len(prior['response_gate_stats']), 8)
        self.assertTrue(all(stats.shape == (1, 4, 10)
                            for stats in prior['response_gate_stats'].values()))
        (restored - torch.rand_like(restored)).square().mean().backward()
        for index, controller in enumerate(controllers):
            for name, parameter in controller.named_parameters():
                self.assertIsNotNone(parameter.grad, (index, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (index, name))
                self.assertGreater(float(parameter.grad.abs().sum()), 0, (index, name))
        self.assertGreater(float(model.ddem.response_head[-1].weight.grad.abs().sum()), 0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.02)
        optimizer.step()
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        model.eval()
        rectangular = torch.rand(1, 3, 128, 256)
        with torch.no_grad():
            expected = model(rectangular)
        stream = io.BytesIO()
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu')
        restored_model = make_model().eval()
        restored_model.load_state_dict(saved['state_dict'], strict=True)
        with torch.no_grad():
            actual = restored_model(rectangular)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            make_model('serial').load_state_dict(saved['state_dict'], strict=True)

    @unittest.skipUnless(
        os.environ.get('RUN_CUDA_TESTS') == '1' and torch.cuda.is_available(),
        'CUDA unavailable or optional RUN_CUDA_TESTS not enabled')
    def test_optional_cuda_amp(self):
        module = JointResidualResponseConditionedLocal(8, groups=4).cuda()
        x = torch.randn(2, 16, 24, 8, device='cuda', requires_grad=True)
        response = torch.randn(2, 64, 2, 3, device='cuda', requires_grad=True)
        with torch.cuda.amp.autocast():
            output, stats = module(x, response, return_stats=True)
            loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(stats).all())
        self.assertTrue(torch.isfinite(response.grad).all())
        self.assertGreater(float(response.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()

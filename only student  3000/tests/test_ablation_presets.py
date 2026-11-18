"""Check independent ablation flags, budgets, cache isolation, and clipping."""
import argparse
import ast
import contextlib
import io
import os
import shlex
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_online_2d as launcher
from options import Options

EXPECTED = {
    'full': (True, True, True, True, True),
    'uformer': (False, False, False, False, False),
    'local': (True, False, False, False, False),
    'local_cam': (True, True, False, False, False),
    'local_cam_z0z1': (True, True, True, True, False),
    'local_cam_double': (True, True, True, False, False),
}
FEATURES = ('use_local_branch', 'use_channel_aware', 'double_process',
            'use_ddem', 'use_kernel_response')


class AblationPresetTests(unittest.TestCase):
    def test_architectures_protocol_and_isolated_paths(self):
        image = SimpleNamespace(suffix='.png', is_file=lambda: True)
        experiments, caches = set(), set()
        for name, flags in EXPECTED.items():
            with self.subTest(name=name), patch.dict(os.environ, {
                    'DRY_RUN': '1', 'RESUME': '0', 'MAX_ITERS': '200000',
                    'GPUS': '0', 'BATCH_SIZE': '4', 'DATASET_SCOPE': 'full', 'RUN_TAG': '',
                    'WARMUP_ITERS': '2820', 'RESPONSE_VARIANT': 'joint_residual'}), \
                    patch.object(Path, 'exists', return_value=True), \
                    patch.object(Path, 'iterdir', return_value=[image] * 7520):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(launcher.main(ablation=name), 0)
                command = shlex.split(output.getvalue().splitlines()[-1])
                options = Options().init(argparse.ArgumentParser()).parse_args(command[2:])
                self.assertEqual(tuple(getattr(options, f) for f in FEATURES), flags)
                self.assertEqual(options.ddem_scratch_student, options.use_ddem)
                self.assertEqual((options.max_iters, options.nepoch), (200000, 107))
                self.assertEqual((options.gpu, options.batch_size), ('0', 4))
                self.assertIn('_b4', options.env)
                self.assertEqual((options.lr_initial, options.warmup_iters, options.grad_clip),
                                 (.0003, 2820, 1.))
                self.assertEqual(options.ddem_kd_weight, 0)
                self.assertFalse(options.resume)
                if not options.use_kernel_response:
                    self.assertEqual(options.kernel_response_variant, 'serial')
                    self.assertEqual(options.kernel_response_groups, 1)
                experiments.add(options.env)
                caches.add(options.online_2d_cache_root)
        self.assertEqual(len(experiments), len(EXPECTED))
        self.assertEqual(len(caches), len(EXPECTED))

    def test_gradient_clipping_does_not_depend_on_ddem(self):
        tree = ast.parse((ROOT / 'train/train_motiondeblur.py').read_text())
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'grad_clip' for t in node.targets)]
        self.assertEqual(len(assignments), 2)
        for node in assignments:
            expression = compile(ast.Expression(node.value), '<grad_clip>', 'eval')
            for enabled in [False, True]:
                for threshold in [0., 1.]:
                    options = SimpleNamespace(use_ddem=enabled, grad_clip=threshold)
                    self.assertEqual(eval(expression, {'opt': options}), threshold or None)


if __name__ == '__main__':
    unittest.main()

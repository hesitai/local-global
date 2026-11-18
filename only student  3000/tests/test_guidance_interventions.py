"""Safety and metric checks for the standalone guidance diagnostics."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from skimage.metrics import peak_signal_noise_ratio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    'guidance_diagnostics', ROOT / 'tools/check_guidance_contributions.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)
from model import Uformer, Z0StageAdapter, Z1StageAdapter
from kernel_response import JointResidualResponseConditionedLocal

torch.set_num_threads(2)


class GuidanceInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        cls.model = Uformer(
            img_size=256, embed_dim=8, depths=[1] * 9,
            num_heads=[1, 1, 2, 4, 8, 8, 4, 2, 1],
            use_ddem=True, ddem_dim=8, ddem_blocks=1,
            use_local_branch=True, use_channel_aware=True, double_process=True,
            use_kernel_response=True, kernel_response_groups=4,
            kernel_response_variant='joint_residual', drop_path_rate=0,
        ).eval()

    def test_modes_restore_flags_and_response_bypass_is_identity(self):
        adapters = [m for m in self.model.modules()
                    if isinstance(m, (Z0StageAdapter, Z1StageAdapter))]
        controllers = [m for m in self.model.modules()
                       if isinstance(m, JointResidualResponseConditionedLocal)]
        self.assertEqual(len(controllers), 8)
        before = [m.guidance_enabled for m in adapters]
        forward = controllers[0].forward
        for mode, (z0, z1, response) in diagnostics.MODES.items():
            with diagnostics.guidance_intervention(self.model, mode, {}):
                for module in adapters:
                    expected = z0 if isinstance(module, Z0StageAdapter) else z1
                    self.assertEqual(module.guidance_enabled, expected)
                if not response:
                    local = torch.randn(1, 16, 16, controllers[0].channels)
                    r = torch.randn(1, 64, 1, 1)
                    torch.testing.assert_close(
                        controllers[0](local, r), local, rtol=0, atol=0)
            self.assertEqual([m.guidance_enabled for m in adapters], before)
            self.assertEqual(controllers[0].forward, forward)

    def test_exception_restores_flags_forward_and_hooks(self):
        modules = [m for m in self.model.modules()
                   if isinstance(m, (Z0StageAdapter, Z1StageAdapter))]
        flags = [m.guidance_enabled for m in modules]
        hooks = len(self.model.ddem._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with diagnostics.guidance_intervention(self.model, 'none'):
                raise RuntimeError('probe')
        self.assertEqual([m.guidance_enabled for m in modules], flags)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with diagnostics.guidance_intervention(self.model, 'mean_R', {}):
                raise RuntimeError('probe')
        self.assertEqual(len(self.model.ddem._forward_hooks), hooks)

    def test_replacement_changes_only_response_and_is_restored(self):
        image = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            original = self.model.ddem(image)
            replacement = original['kernel_response'].flip(0).clone() + .25
            for mode in ['shuffled_R', 'mean_R']:
                with diagnostics.guidance_intervention(
                        self.model, mode, {'response': replacement}):
                    changed = self.model.ddem(image)
                for key in ['z0', 'z1', 'z_tilde']:
                    torch.testing.assert_close(changed[key], original[key], rtol=0, atol=0)
                torch.testing.assert_close(changed['kernel_response'], replacement)
            restored = self.model.ddem(image)
        torch.testing.assert_close(
            restored['kernel_response'], original['kernel_response'], rtol=0, atol=0)

    def test_full_mode_matches_model_and_preserves_weights(self):
        image = torch.rand(1, 3, 128, 128)
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            original = self.model(image)
            with diagnostics.guidance_intervention(self.model, 'full'):
                actual = self.model(image)
        torch.testing.assert_close(original, actual, rtol=0, atol=0)
        for name, parameter in self.model.state_dict().items():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_psnr_matches_skimage_per_image(self):
        target = torch.rand(3, 3, 16, 16)
        predicted = torch.rand_like(target) * 1.2 - .1
        actual = diagnostics.paired_psnr(predicted, target)
        for i, value in enumerate(actual):
            expected = peak_signal_noise_ratio(
                target[i].numpy(), predicted[i].clamp(0, 1).numpy(), data_range=1.)
            self.assertAlmostEqual(value, expected, places=6)
        summary = diagnostics.describe_delta([20., 22.], [19., 23.])
        self.assertEqual(summary['mean_full_minus_intervention_db'], 0.)
        self.assertEqual(summary['fraction_full_better'], .5)


if __name__ == '__main__':
    unittest.main()

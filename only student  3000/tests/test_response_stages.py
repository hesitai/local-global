"""Check isolated R bypass/scaling and restoration of hooks and weights."""
import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('response_stages', ROOT/'tools/check_response_stages.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
from kernel_response import JointResidualResponseConditionedLocal

torch.set_num_threads(2)


class StageSettingsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.model = nn.Module()
        self.model.control = JointResidualResponseConditionedLocal(8, groups=4).eval()
        self.x = torch.randn(2, 16, 24, 8)
        self.r = torch.randn(2, 64, 2, 3)

    def test_residual_formula_and_exact_reference(self):
        module = self.model.control
        with torch.no_grad():
            reference = module(self.x, self.r)
            for scale in [0., .5, .75, 1., 1.25, 1.5]:
                with tool.response_setting(self.model, scale=scale):
                    actual = module(self.x, self.r)
                expected = self.x + scale * (reference-self.x)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=5e-7)
                if scale == 1.:
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_bypass_and_exception_restore(self):
        module = self.model.control
        forward = module.forward
        with tool.response_setting(self.model, bypass_stage='control'):
            torch.testing.assert_close(module(self.x, self.r), self.x, rtol=0, atol=0)
        self.assertEqual(module.forward, forward)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with tool.response_setting(self.model, scale=1.5):
                raise RuntimeError('probe')
        self.assertTrue(all(not expert._forward_hooks for expert in module.experts))
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with tool.response_setting(self.model, bypass_stage='control'):
                raise RuntimeError('probe')
        self.assertEqual(module.forward, forward)

    def test_weights_and_outputs_preserved(self):
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        with torch.no_grad():
            reference = self.model.control(self.x, self.r)
            with tool.response_setting(self.model, scale=.5):
                self.model.control(self.x, self.r)
            actual = self.model.control(self.x, self.r)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for key,value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_invalid_settings(self):
        for settings in [dict(scale=-1),dict(scale=float('nan')),dict(bypass_stage='missing'),
                         dict(bypass_stage='control',scale=.5)]:
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                with tool.response_setting(self.model, **settings):
                    pass


if __name__ == '__main__':
    unittest.main()

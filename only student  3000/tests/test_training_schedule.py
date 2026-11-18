"""CPU-only checks for iteration budgets and learning-rate boundaries."""

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training_schedule import cosine_learning_rate_factor


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        'online_training_launcher', PROJECT_ROOT / 'train_online_2d.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingScheduleTests(unittest.TestCase):
    def test_warmup_and_cosine_boundaries(self):
        def factor(step):
            return cosine_learning_rate_factor(
                step, total_iters=600_000, warmup_iters=2_820,
                min_factor=1e-6 / 3e-4)

        self.assertAlmostEqual(factor(0), 1 / 2_820)
        self.assertAlmostEqual(factor(2_819), 1.0)
        self.assertAlmostEqual(factor(2_820), 1.0)
        self.assertGreater(factor(300_000), factor(470_000))
        self.assertGreater(factor(470_000), factor(599_999))
        self.assertAlmostEqual(factor(600_000) * 3e-4, 1e-6)
        self.assertEqual(factor(600_001), factor(600_000))

    def test_zero_warmup(self):
        self.assertEqual(
            cosine_learning_rate_factor(0, total_iters=100), 1.0)

    def test_invalid_schedule(self):
        for kwargs in (
            {'total_iters': 0},
            {'total_iters': 100, 'warmup_iters': -1},
            {'total_iters': 100, 'warmup_iters': 100},
            {'total_iters': 100, 'min_factor': 2},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cosine_learning_rate_factor(0, **kwargs)

    def test_launcher_accepts_disabled_iteration_mode(self):
        from unittest.mock import patch
        launcher = load_launcher()
        with patch.dict('os.environ', {'MAX_ITERS': '0', 'WARMUP_ITERS': '0'}):
            self.assertEqual(launcher.env_int('MAX_ITERS', 600_000, minimum=0), 0)
            self.assertEqual(launcher.env_int('WARMUP_ITERS', 2_820, minimum=0), 0)
        with patch.dict('os.environ', {'BATCH_SIZE': '0'}):
            with self.assertRaises(ValueError):
                launcher.env_int('BATCH_SIZE', 8)


if __name__ == '__main__':
    unittest.main()

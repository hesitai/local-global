"""Seed reconstruction and metadata mismatch checks for response recovery."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'response_recovery', ROOT / 'tools/restore_validation_response_labels.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class ResponseRecoveryTests(unittest.TestCase):
    def test_seed_reconstruction_keeps_original_case_sampling(self):
        row = dict(sample_id='000022_source_v00', input_path='/tmp/000022_source_v00.png',
                   target_path='/tmp/000022_source_v00.png', source_path='/tmp/source.png',
                   epoch='0', variant_index='0', seed=str(20260810 + 22 * 10007),
                   patch_size='256')
        summary = dict(patch_size=256, strength_range=[1., 1.8], adaptive_subframes=True)
        task = recovery.infer_task(row, Path('/tmp/recovery-test'), summary)
        self.assertEqual(task.base_seed, 20260810)
        self.assertEqual(task.source_index, 22)
        self.assertEqual(task.cases, tuple(recovery.generator.motion.blur_cases))
        self.assertTrue(task.kernel_response)
        self.assertEqual(task.response_backend, 'cpu')
        with self.assertRaises(ValueError):
            recovery.infer_task(dict(row, seed=str(int(row['seed']) + 1)),
                                Path('/tmp/recovery-test'), summary)

    def test_verification_detects_metadata_error_without_changing_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yy, xx = np.mgrid[:768, :1024]
            source = np.stack((xx % 251, yy % 251, (xx + yy) % 251), axis=-1).astype(np.uint8)
            source_path = root / 'source.png'
            cv2.imwrite(str(source_path), source)
            original, restored = root / 'original', root / 'restored'
            for folder in [original, restored]:
                (folder / 'input').mkdir(parents=True)
                (folder / 'target').mkdir()
            name = '000000_source_v00.png'
            summary = dict(patch_size=256, strength_range=[1., 1.8], adaptive_subframes=True)
            row = dict(sample_id=Path(name).stem,
                       input_path=str(original / 'input' / name),
                       target_path=str(original / 'target' / name),
                       source_path=str(source_path), epoch='0', variant_index='0',
                       seed='20260810', patch_size='256')
            result = recovery.generator.render_task(
                recovery.infer_task(row, original, summary))
            original_row = dict(result, input_path=row['input_path'],
                                target_path=row['target_path'])
            original_row.pop('output_name')
            hashes = {key: recovery.sha256(row[key])
                      for key in ['input_path', 'target_path', 'source_path']}
            check = recovery.verify_one(original_row, str(restored), summary)
            self.assertEqual(check['differences'], [])
            self.assertTrue(check['image_parity']['input_path']['pixel_identical'])
            self.assertEqual(check['label_shape'], [64, 16, 16])
            bad = recovery.verify_one(dict(original_row, strength='0.00000000'),
                                      str(restored), summary)
            self.assertIn('strength', [item['field'] for item in bad['differences']])
            for key, before in hashes.items():
                self.assertEqual(recovery.sha256(row[key]), before)


if __name__ == '__main__':
    unittest.main()

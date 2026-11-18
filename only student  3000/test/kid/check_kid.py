"""Offline KID checks. No pretrained weights, network, or GPU required."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from evaluate_kid import ImageDataset, image_manifest, kid_subsets, run


class FakeExtractor(torch.nn.Module):
    def forward(self, image, *args):
        feature = image.mean(dim=(1, 2, 3)).reshape(-1, 1).repeat(1, 2048)
        return [feature[:, :, None, None]]


class KIDChecks(unittest.TestCase):
    def test_estimator_matches_explicit_unbiased_formula_and_keeps_negative(self):
        x = np.random.RandomState(7).randn(8, 5)
        y = np.random.RandomState(8).randn(9, 5)
        scores, ri, gi = kid_subsets(x, y, 6, 4, 123)
        for score, a, b in zip(scores, ri, gi):
            xx, yy = x[a], y[b]
            kernel = lambda u, v: (u @ v / 5 + 1) ** 3
            expected = sum(kernel(u, v) for i, u in enumerate(xx) for j, v in enumerate(xx) if i != j) / 30
            expected += sum(kernel(u, v) for i, u in enumerate(yy) for j, v in enumerate(yy) if i != j) / 30
            expected -= 2 * sum(kernel(u, v) for u in xx for v in yy) / 36
            self.assertAlmostEqual(score, expected, places=12)
        negative, _, _ = kid_subsets(x, x, 8, 3, 1)
        self.assertTrue(np.all(negative < 0))

    def test_sampling_reproducibility_shared_reference_and_validation(self):
        x = np.arange(40).reshape(10, 4)
        first = kid_subsets(x, x + 2, 6, 3, 10)
        second = kid_subsets(x, x + 7, 6, 3, 10)
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertTrue(all(len(set(row)) == 6 for row in first[1]))
        for subset in (1, 11):
            with self.assertRaises(ValueError):
                kid_subsets(x, x, subset, 3, 1)

    def test_full_pipeline_cache_and_input_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real, generated = root / 'real', root / 'generated'
            real.mkdir(); generated.mkdir()
            for i in range(4):
                Image.new('RGB', (64, 80), (30 * i, 50, 70)).save(real / ('图%d.png' % i))
                Image.new('RGB', (64, 80), (30 * i + 3, 50, 70)).save(generated / ('图%d.png' % i))
            config = {'real_dir': str(real), 'generated_dir': str(generated),
                      'method_name': 'test', 'output_dir': str(root / 'first'),
                      'cache_dir': str(root / 'cache'), 'device': 'cpu', 'seed': 123,
                      'batch_size': 2, 'num_workers': 0, 'subset_size': 3, 'num_subsets': 5,
                      'require_equal_counts': True, 'center_crop_size': 0}
            with patch('evaluate_kid.build_extractor', return_value=FakeExtractor()), redirect_stdout(io.StringIO()):
                first = run(config)
                config['output_dir'] = str(root / 'second')
                second = run(config)
                Image.new('RGB', (64, 80), (170, 50, 70)).save(generated / '图0.png')
                config['output_dir'] = str(root / 'third')
                third = run(config)
            self.assertEqual(first['summary'], second['summary'])
            self.assertEqual(first['real_features_cache'], third['real_features_cache'])
            self.assertNotEqual(first['generated_features_cache'], third['generated_features_cache'])
            self.assertEqual(first['summary']['real_count'], 4)
            stored = json.loads((root / 'first' / 'summary.json').read_text())
            self.assertIn('not a CI', stored['std_definition'])
            self.assertTrue((root / 'first' / 'sampling_indices.npz').is_file())
            with self.assertRaises(FileExistsError):
                run(config)

    def test_clean_preprocessing_matches_pyiqa_and_crop_preflight(self):
        from pyiqa.archs.fid_arch import ResizeDataset
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb = np.random.RandomState(123).randint(0, 256, (65, 81, 3), dtype=np.uint8)
            path = root / 'image.png'
            Image.fromarray(rgb).save(path)
            actual = ImageDataset([path], 0)[0]
            expected = ResizeDataset([str(path)], 'clean')[0]
            self.assertTrue(torch.equal(actual, expected))
            with self.assertRaises(ValueError):
                ImageDataset([path], 128)[0]
            Image.new('RGB', (64, 64)).save(root / 'second.png')
            (root / 'broken.jpg').write_bytes(b'broken')
            with self.assertRaises(Exception):
                image_manifest(root, 0)


if __name__ == '__main__':
    unittest.main()

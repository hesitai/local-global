"""CPU-only entrypoint regression tests; real datasets/results are never written."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    'joint2d_evaluation_entrypoint', ROOT / 'test/test_gopro_hide.py')
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class ToyModel(torch.nn.Module):
    def forward(self, image):
        return image * 1.7 - .3 + image.mean((-2, -1), keepdim=True) * .02


class EvaluationEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.old_args = evaluation.args
        self.old_model = evaluation.model_restoration

    def tearDown(self):
        evaluation.args = self.old_args
        evaluation.model_restoration = self.old_model

    def test_import_is_safe_and_does_not_parse_cli(self):
        module = importlib.util.module_from_spec(SPEC)
        with patch.object(sys, 'argv', ['test', '--not-a-test-option']), \
                patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA queried')), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            SPEC.loader.exec_module(module)
        self.assertEqual(out.getvalue(), '')
        self.assertIsNone(module.args)
        self.assertIsNone(module.model_restoration)

    def test_no_arguments_match_previous_complete_model_command(self):
        defaults = evaluation.parse_args([])
        explicit = evaluation.parse_args([
            '--input_dir', str(evaluation.DEFAULT_INPUT_DIR),
            '--weights', str(evaluation.DEFAULT_WEIGHTS),
            '--result_dir', str(evaluation.DEFAULT_RESULT_DIR),
            '--gpus', '0', '--arch', 'Uformer_B', '--train_ps', '256',
            '--use_local_branch', '--double_process', '--use_channel_aware',
            '--use_ddem', '--ddem_stage', '2', '--ddem_dim', '32', '--ddem_blocks', '2',
            '--ddem_backbone', 'local_global', '--ddem_heads', '1', '--ddem_win_size', '8',
            '--use_kernel_response', '--kernel_response_groups', '4',
            '--tile_mode', 'native', '--tile_size', '256', '--tile_overlap', '64',
            '--tile_blend', 'hann', '--save_images', '--save_per_image_metrics'])
        self.assertEqual(vars(defaults), vars(explicit))
        self.assertTrue(defaults.weights.endswith('model_epoch_639.pth'))
        self.assertTrue(defaults.result_dir.endswith('joint2d_i600000_final_fixed_full_Lwf4ZZ'))

    def test_relative_paths_are_repository_relative_not_cwd(self):
        before = os.getcwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                options = evaluation.parse_args([
                    '--input_dir', 'dataset/example', '--weights', 'checkpoints/model.pth',
                    '--result_dir', 'results/example'])
            finally:
                os.chdir(before)
        self.assertEqual(options.input_dir, str(ROOT / 'dataset/example'))
        self.assertEqual(options.weights, str(ROOT / 'checkpoints/model.pth'))
        self.assertEqual(options.result_dir, str(ROOT / 'results/example'))

    def test_explicit_negative_switches_and_invalid_geometry(self):
        options = evaluation.parse_args([
            '--no-save-images', '--no_save_per_image_metrics',
            '--no_use_kernel_response', '--kernel_response_groups', '1',
            '--no_use_ddem', '--no_use_local_branch', '--no_double_process',
            '--no_use_channel_aware'])
        for name in ('save_images', 'save_per_image_metrics', 'use_kernel_response',
                     'use_ddem', 'use_local_branch', 'double_process', 'use_channel_aware'):
            self.assertFalse(getattr(options, name))
        for cli in (['--tile_size', '-1'], ['--tile_size', '255'],
                    ['--tile_overlap', '256'], ['--batch_size', '2'],
                    ['--no_use_kernel_response'], ['--gpus', ''], ['--max_images', '-1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(cli)

    def test_backup_preserves_all_existing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'result'
            output.mkdir()
            (output / 'old.png').write_bytes(b'old image')
            (output / 'psnr_ssim.txt').write_text('old scores')
            with contextlib.redirect_stdout(io.StringIO()):
                backup = Path(evaluation.prepare_result_directory(output, 'backup'))
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(backup.parent, output.parent)
            self.assertEqual((backup / 'old.png').read_bytes(), b'old image')
            self.assertEqual((backup / 'psnr_ssim.txt').read_text(), 'old scores')
            (output / 'new.txt').write_text('new')
            with self.assertRaises(FileExistsError):
                evaluation.prepare_result_directory(output, 'error')
            self.assertEqual((output / 'new.txt').read_text(), 'new')

    def fixture(self, root):
        data = root / 'data'
        for kind in ('input', 'target'):
            (data / kind).mkdir(parents=True)
            for name, size in [('b.png', (477, 283)), ('a.png', (77, 45))]:
                color = (40, 90, 130) if kind == 'input' else (45, 85, 138)
                Image.new('RGB', size, color).save(data / kind / name)
        weight = root / 'weights.pth'
        weight.write_bytes(b'fixture; loading stubbed')
        options = evaluation.parse_args([
            '--input_dir', str(data), '--weights', str(weight),
            '--result_dir', str(root / 'result'), '--num_workers', '0'])
        return options

    def test_native_outputs_metrics_csv_mean_and_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            options = self.fixture(Path(temp))
            dataset = evaluation.validate_inputs(options)
            before = {p: p.read_bytes() for p in Path(options.input_dir).rglob('*.png')}
            evaluation.args = options
            evaluation.model_restoration = ToyModel().eval()
            evaluation.prepare_result_directory(options.result_dir, 'error')
            with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), \
                    patch('torch.cuda.amp.autocast', lambda *a, **kw: contextlib.nullcontext()), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                rows, mean_psnr, mean_ssim = evaluation.evaluate_dataset(
                    DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
                for index, row in enumerate(rows):
                    target, image, name = dataset[index]
                    restored = evaluation.restore_tiled_native(
                        image[None], None, 256, 64).clamp(0, 1)
                    rgb = restored[0].permute(1, 2, 0).numpy()
                    gt = target.permute(1, 2, 0).numpy()
                    self.assertEqual(row['filename'], name)
                    self.assertEqual(row['psnr'], float(peak_signal_noise_ratio(rgb, gt)))
                    self.assertEqual(row['ssim'], float(structural_similarity(
                        rgb, gt, channel_axis=-1, data_range=1.0)))
                    saved = np.array(Image.open(Path(options.result_dir) / (name + '.PNG')))
                    np.testing.assert_array_equal(saved, img_as_ubyte(rgb))
                    self.assertEqual(saved.shape, gt.shape)
            self.assertAlmostEqual(mean_psnr, sum(r['psnr'] for r in rows) / 2, places=12)
            self.assertAlmostEqual(mean_ssim, sum(r['ssim'] for r in rows) / 2, places=12)
            with (Path(options.result_dir) / 'psnr_ssim_per_image.csv').open(newline='') as stream:
                saved_rows = list(csv.DictReader(stream))
            self.assertEqual([r['filename'] for r in saved_rows], ['a', 'b'])
            self.assertTrue((Path(options.result_dir) / 'psnr_ssim.txt').is_file())
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_check_only_never_touches_existing_outputs_or_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            options = self.fixture(Path(temp))
            result = Path(options.result_dir)
            result.mkdir()
            (result / 'old.txt').write_text('keep')
            cli = ['--input_dir', options.input_dir, '--weights', options.weights,
                   '--result_dir', options.result_dir, '--check_only']
            with patch.object(evaluation, 'load_evaluation_model', return_value=(ToyModel(), {})), \
                    patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA queried')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluation.main(cli), 0)
            self.assertEqual((result / 'old.txt').read_text(), 'keep')
            self.assertEqual(list(result.parent.glob('result_backup_*')), [])

    def test_failed_run_has_log_partial_csv_but_no_final_average(self):
        class FailsSecond(ToyModel):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def cuda(self):
                return self

            def forward(self, image):
                self.calls += 1
                return super().forward(image) if self.calls == 1 else image * float('nan')

        with tempfile.TemporaryDirectory() as temp:
            options = self.fixture(Path(temp))
            cli = ['--input_dir', options.input_dir, '--weights', options.weights,
                   '--result_dir', options.result_dir, '--num_workers', '0']
            with patch.object(evaluation, 'load_evaluation_model', return_value=(FailsSecond(), {})), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), \
                    patch('torch.cuda.amp.autocast', lambda *a, **kw: contextlib.nullcontext()), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, 'Non-finite'):
                evaluation.main(cli)
            output = Path(options.result_dir)
            metadata = json.loads((output / 'inference_config.json').read_text())
            self.assertEqual(metadata['status'], 'failed')
            self.assertEqual(metadata['processed_images'], 1)
            self.assertFalse((output / 'psnr_ssim.txt').exists())
            self.assertIn('Non-finite', (output / 'test.log').read_text())

    def test_successful_main_saves_log_metadata_and_complete_results(self):
        class CpuFixture(ToyModel):
            def cuda(self):
                return self

        with tempfile.TemporaryDirectory() as temp:
            options = self.fixture(Path(temp))
            cli = ['--input_dir', options.input_dir, '--weights', options.weights,
                   '--result_dir', options.result_dir, '--num_workers', '0']
            with patch.object(evaluation, 'load_evaluation_model',
                              return_value=(CpuFixture(), {'checkpoint_key': 'state_dict'})), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), \
                    patch('torch.cuda.amp.autocast', lambda *a, **kw: contextlib.nullcontext()), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(evaluation.main(cli), 0)
            output = Path(options.result_dir)
            metadata = json.loads((output / 'inference_config.json').read_text())
            self.assertEqual((metadata['status'], metadata['processed_images']), ('complete', 2))
            self.assertEqual(metadata['checkpoint']['checkpoint_key'], 'state_dict')
            self.assertEqual(metadata['metrics']['aggregation'],
                             'arithmetic mean of per-image values; before PNG quantization')
            for name in ('test.log', 'psnr_ssim.txt', 'psnr_ssim_per_image.csv', 'a.PNG', 'b.PNG'):
                self.assertTrue((output / name).is_file(), name)
            self.assertIn('PSNR:', (output / 'test.log').read_text())


if __name__ == '__main__':
    unittest.main()

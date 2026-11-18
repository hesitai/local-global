"""CPU checks for verified-label prediction and true-R inference diagnostics."""

import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location(
    'response_prediction_diagnostics', ROOT / 'tools/check_response_prediction.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)
torch.set_num_threads(2)


class ToyDDEM(nn.Module):
    def forward(self, image):
        descriptor = image.mean(dim=(-2, -1))
        return dict(
            kernel_response=image * 2,
            z0=descriptor,
            z1=descriptor.unsqueeze(-1) * descriptor.unsqueeze(-2),
            z_tilde=descriptor + 1,
        )


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ddem = ToyDDEM()


class ResponseMetricTests(unittest.TestCase):
    def test_identical_prediction_has_zero_error(self):
        response = torch.rand(2, 64, 3, 5)
        errors = diagnostics.response_error_metrics(response, response)
        self.assertEqual(set(errors), set(diagnostics.METRIC_NAMES))
        for name, values in errors.items():
            self.assertEqual(values.shape, (2, 4), name)
            torch.testing.assert_close(values, torch.zeros(2, 4), rtol=0, atol=0)

    def test_radius_and_real_imag_layout_with_known_errors(self):
        target = torch.zeros(2, 4, 8, 2, 3, 5)
        target[:, :, :, 0] = 1.
        prediction = target.clone()
        changes = torch.tensor([.1, .2, .3, .4])
        for radius, change in enumerate(changes):
            prediction[:, radius, :, 0] += change
        errors = diagnostics.response_error_metrics(
            prediction.reshape(2, 64, 3, 5), target.reshape(2, 64, 3, 5))
        expected = {
            'complex_mae': changes / 2,
            'complex_rmse': changes / math.sqrt(2),
            'complex_smooth_l1': changes.square() / 4,
            'magnitude_mae': changes,
            'magnitude_rmse': changes,
        }
        for name, values in errors.items():
            torch.testing.assert_close(values, expected[name].expand(2, -1),
                                       rtol=1e-5, atol=1e-7)
        actual = diagnostics.aggregate_metric(errors, 'complex_rmse')
        expected_rmse = (changes.square().mean() / 2).sqrt().expand(2)
        torch.testing.assert_close(actual, expected_rmse, rtol=1e-5, atol=1e-7)

    def test_phase_change_is_not_magnitude_error(self):
        target = torch.zeros(1, 64, 2, 2)
        target[:, ::2] = 1.
        prediction = torch.zeros_like(target)
        prediction[:, 1::2] = 1.
        errors = diagnostics.response_error_metrics(prediction, target)
        torch.testing.assert_close(errors['magnitude_mae'], torch.zeros(1, 4))
        torch.testing.assert_close(errors['complex_mae'], torch.ones(1, 4))
        torch.testing.assert_close(errors['complex_smooth_l1'], torch.full((1, 4), .5))

    def test_invalid_response_tensors_are_rejected(self):
        for prediction, target in [
            (torch.zeros(1, 32, 2, 2), torch.zeros(1, 32, 2, 2)),
            (torch.zeros(1, 64, 2, 2), torch.zeros(2, 64, 2, 2)),
            (torch.full((1, 64, 2, 2), float('nan')), torch.zeros(1, 64, 2, 2)),
        ]:
            with self.assertRaises(ValueError):
                diagnostics.response_error_metrics(prediction, target)


class OracleHookTests(unittest.TestCase):
    def test_interpolation_changes_only_response_and_restores_hook(self):
        model = ToyModel()
        image = torch.rand(2, 64, 2, 2)
        original = model.ddem(image)
        target = torch.rand_like(original['kernel_response'])
        hooks_before = len(model.ddem._forward_hooks)
        for fraction in [0., .25, .5, .75, 1.]:
            state = dict(target=target, fraction=fraction)
            with diagnostics.oracle_response_intervention(model, state):
                changed = model.ddem(image)
            expected = ((1 - fraction) * original['kernel_response'] + fraction * target)
            torch.testing.assert_close(changed['kernel_response'], expected)
            if fraction == 0:
                self.assertIs(changed['kernel_response'].dtype,
                              original['kernel_response'].dtype)
                torch.testing.assert_close(changed['kernel_response'],
                                           original['kernel_response'], rtol=0, atol=0)
            for name in ['z0', 'z1', 'z_tilde']:
                torch.testing.assert_close(changed[name], original[name], rtol=0, atol=0)
            self.assertEqual(len(model.ddem._forward_hooks), hooks_before)
        restored = model.ddem(image)
        torch.testing.assert_close(restored['kernel_response'],
                                   original['kernel_response'], rtol=0, atol=0)

    def test_error_and_invalid_state_remove_hook(self):
        model = ToyModel()
        image = torch.rand(1, 64, 2, 2)
        hooks_before = len(model.ddem._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with diagnostics.oracle_response_intervention(model, {}):
                raise RuntimeError('probe')
        self.assertEqual(len(model.ddem._forward_hooks), hooks_before)
        for state in [
            dict(target=image, fraction=-.1),
            dict(target=image, fraction=1.1),
            dict(target=image, fraction=float('nan')),
            dict(target=torch.rand(2, 64, 2, 2), fraction=1.),
            dict(target=torch.full_like(image, float('nan')), fraction=1.),
            dict(fraction=1.),
        ]:
            with self.subTest(state_keys=list(state)), self.assertRaises(ValueError):
                with diagnostics.oracle_response_intervention(model, state):
                    model.ddem(image)
            self.assertEqual(len(model.ddem._forward_hooks), hooks_before)


class ParityReportTests(unittest.TestCase):
    def make_fixture(self, directory):
        rows = []
        for sample in ['a', 'b']:
            paths = {}
            for name in ['input_path', 'target_path', 'kernel_response_path']:
                path = directory / (sample + '_' + name)
                path.write_bytes(name.encode())
                paths[name] = str(path)
            rows.append(dict(sample_id=sample, **paths))
        manifest = directory / 'pairs.tsv'
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        report = dict(
            status='verified', expected_count=2, verified_count=2,
            mismatches=[], failures=[], label_manifest=str(manifest),
            label_manifest_sha256=diagnostics.file_sha256(manifest),
            verified_samples=[dict(
                sample_id=row['sample_id'], differences=[],
                kernel_response_path=row['kernel_response_path'],
                kernel_response_sha256=diagnostics.file_sha256(row['kernel_response_path']),
                image_parity={name: dict(pixel_identical=True,
                                        original_sha256=diagnostics.file_sha256(row[name]))
                              for name in ['input_path', 'target_path']},
            ) for row in rows],
        )
        return manifest, report, rows

    def test_verified_report_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, report, expected_rows = self.make_fixture(Path(temporary))
            manifest.with_name('parity_report.json').write_text(json.dumps(report))
            actual_report, rows = diagnostics.validate_parity_report(manifest)
            self.assertEqual(actual_report['status'], 'verified')
            self.assertEqual(rows, expected_rows)

    def test_mismatch_and_incomplete_reports_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, report, _rows = self.make_fixture(Path(temporary))
            for change in [
                dict(status='running'), dict(status='failed'),
                dict(expected_count=3), dict(verified_count=1),
                dict(mismatches=[dict(sample_id='a')]),
                dict(failures=[dict(sample_id='a')]),
                dict(label_manifest=str(manifest.with_name('other.tsv'))),
            ]:
                candidate = dict(report, **change)
                manifest.with_name('parity_report.json').write_text(json.dumps(candidate))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    diagnostics.validate_parity_report(manifest)

    def test_modified_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, report, _rows = self.make_fixture(Path(temporary))
            manifest.with_name('parity_report.json').write_text(json.dumps(report))
            manifest.write_text(manifest.read_text() + '\n')
            with self.assertRaises(ValueError):
                diagnostics.validate_parity_report(manifest)


if __name__ == '__main__':
    unittest.main()

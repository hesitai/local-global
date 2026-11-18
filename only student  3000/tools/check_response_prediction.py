#!/usr/bin/env python3
"""Evaluate R prediction quality and non-deployable true-R interventions.

The input manifest must come from pixel-verified regeneration of the existing
validation images. This script never trains, changes model weights, or edits
validation data. Oracle/mixed R diagnostics can have distribution shift and are
not a strict performance upper bound or deployable benchmark results.
"""

import argparse
import contextlib
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_guidance_contributions import build_model, paired_psnr, file_sha256
from kernel_response import label_spec

MODE_FRACTIONS = {'pred': 0.0, 'mix25': 0.25, 'mix50': 0.5,
                  'mix75': 0.75, 'oracle': 1.0}
METRIC_NAMES = ('complex_mae', 'complex_rmse', 'complex_smooth_l1',
                'magnitude_mae', 'magnitude_rmse')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def validate_parity_report(manifest_path, report_path=None):
    """Require complete, successful pixel/metadata parity before using labels."""
    manifest_path = Path(manifest_path).resolve()
    report_path = Path(report_path) if report_path else manifest_path.with_name('parity_report.json')
    report = json.loads(report_path.read_text())
    if report.get('status') != 'verified':
        raise ValueError('Label parity report is not verified')
    expected, verified = report.get('expected_count'), report.get('verified_count')
    if not isinstance(expected, int) or expected < 1 or expected != verified:
        raise ValueError('Label parity report is incomplete or has invalid counts')
    if report.get('mismatches') != [] or report.get('failures') != []:
        raise ValueError('Label regeneration contains mismatches or failures')
    with manifest_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if len(rows) != expected:
        raise ValueError('Label manifest count differs from verified count')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs in label manifest')
    recorded = report.get('label_manifest')
    if recorded and Path(recorded).resolve() != manifest_path:
        raise ValueError('Parity report refers to a different label manifest')
    manifest_hash = report.get('label_manifest_sha256')
    if manifest_hash and manifest_hash != file_sha256(manifest_path):
        raise ValueError('Label manifest changed after pixel-parity verification')
    verified_samples = report.get('verified_samples')
    samples_by_id = None
    if verified_samples is not None:
        samples_by_id = {sample['sample_id']: sample for sample in verified_samples}
        if (len(samples_by_id) != len(verified_samples)
                or set(samples_by_id) != {row['sample_id'] for row in rows}):
            raise ValueError('Verified sample records do not cover the manifest exactly')
    for row in rows:
        for key in ('input_path', 'target_path', 'kernel_response_path'):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        if samples_by_id is not None:
            sample = samples_by_id[row['sample_id']]
            if sample.get('differences') != []:
                raise ValueError('Verified sample contains metadata differences')
            for key in ('input_path', 'target_path'):
                parity = sample.get('image_parity', {}).get(key, {})
                if parity.get('pixel_identical') is not True:
                    raise ValueError('Verified sample lacks exact input/target pixel parity')
                if parity.get('original_sha256') != file_sha256(row[key]):
                    raise ValueError('Original validation image changed after verification')
            if (Path(sample['kernel_response_path']).resolve()
                    != Path(row['kernel_response_path']).resolve()):
                raise ValueError('Verified sample refers to a different response label')
            if sample.get('kernel_response_sha256') != file_sha256(row['kernel_response_path']):
                raise ValueError('Response label changed after verification')
    return report, rows


def response_error_metrics(prediction, target):
    """Return five [batch, 4 radii] error tensors, with no relative-error blowup.

    ``complex_*`` errors average real and imaginary components, exactly as the
    auxiliary training loss does. Magnitude errors first combine re/im pairs.
    Channel order is radius(4), direction(8), interleaved real/imag(2).
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 64:
        raise ValueError('Expected equal [batch, 64, height, width] response tensors')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite response predictions or targets')
    p = prediction.float().reshape(prediction.shape[0], 4, 8, 2, *prediction.shape[-2:])
    t = target.float().reshape_as(p)
    difference = p - t
    absolute = difference.abs()
    component_axes = (2, 3, 4, 5)
    pm, tm = p.square().sum(3).sqrt(), t.square().sum(3).sqrt()
    magnitude_difference = pm - tm
    return {
        'complex_mae': absolute.mean(component_axes),
        'complex_rmse': difference.square().mean(component_axes).sqrt(),
        'complex_smooth_l1': torch.where(absolute < 1., .5 * difference.square(),
                                         absolute - .5).mean(component_axes),
        'magnitude_mae': magnitude_difference.abs().mean((2, 3, 4)),
        'magnitude_rmse': magnitude_difference.square().mean((2, 3, 4)).sqrt(),
    }


@contextlib.contextmanager
def oracle_response_intervention(model, state):
    """Mix the current predicted R with true R, leaving z0/z1 unchanged.

    Before each forward call, set state['target'] to the true [B,64,H,W] R and
    state['fraction'] to a number in [0,1]. The forward hook is always removed.
    """
    def replace(_module, _inputs, outputs):
        fraction = float(state.get('fraction', 0.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError('Oracle mixing fraction must be between 0 and 1')
        target = state.get('target')
        predicted = outputs['kernel_response']
        if target is None or target.shape != predicted.shape:
            raise ValueError('True R is missing or does not match predicted response shape')
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite true response')
        updated = dict(outputs)
        # Preserve the exact normal path at fraction=0, including dtype.
        if fraction == 0.0:
            mixed = predicted
        elif fraction == 1.0:
            mixed = target.to(device=predicted.device, dtype=predicted.dtype)
        else:
            mixed = predicted + fraction * (target.to(predicted) - predicted)
        updated['kernel_response'] = mixed
        return updated

    handle = model.ddem.register_forward_hook(replace)
    try:
        yield
    finally:
        handle.remove()


class VerifiedResponseDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        specification = label_spec()
        self.responses = []
        for row in rows:
            expected = {key: row[key] for key in
                        ('sample_id', 'seed', 'variant_index', 'crop_top', 'crop_left', 'source_path')}
            value = specification.load_label(
                row['kernel_response_path'], int(row['patch_size']), 'cpu', expected)
            self.responses.append(torch.from_numpy(value))
        self.responses = torch.stack(self.responses)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = []
        for key in ('input_path', 'target_path'):
            with Image.open(row[key]) as image:
                array = np.array(image.convert('RGB'), dtype=np.float32) / 255.
            if array.shape != (256, 256, 3):
                raise ValueError('Expected unchanged fixed 256x256 validation patches')
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        return images[0], images[1], self.responses[index], row['sample_id']


def aggregate_metric(per_radius, name):
    value = per_radius[name]
    # Root mean square must pool squared radius errors, not mean the roots.
    return value.square().mean(1).sqrt() if name.endswith('rmse') else value.mean(1)


def quality_summary(prediction, target):
    errors = response_error_metrics(prediction, target)
    mean_target = target.mean(dim=(0, 2, 3), keepdim=True).expand_as(target)
    mean_errors = response_error_metrics(mean_target, target)
    result = {}
    for label, values in [('prediction', errors), ('validation_mean_baseline', mean_errors)]:
        result[label] = {}
        for name in METRIC_NAMES:
            # Dataset RMSE pools squared per-image/radius errors.
            result[label][name] = (float(values[name].square().mean().sqrt())
                                    if name.endswith('rmse') else float(values[name].mean()))
    p, t = prediction.double(), target.double()
    p_centered = p - p.mean(0, keepdim=True)
    t_centered = t - t.mean(0, keepdim=True)
    pred_rms = float(p_centered.square().mean().sqrt())
    target_rms = float(t_centered.square().mean().sqrt())
    result['cross_image_centered_rms'] = dict(
        prediction=pred_rms, target=target_rms,
        prediction_to_target_ratio=pred_rms / max(target_rms, 1e-12))
    result['prediction_bounds'] = dict(
        component_min=float(prediction.min()), component_max=float(prediction.max()),
        fraction_components_abs_gt_1=float((prediction.abs() > 1).float().mean()),
        fraction_complex_magnitude_gt_1=float((prediction.reshape(
            prediction.shape[0], 4, 8, 2, *prediction.shape[-2:]).square().sum(3).sqrt() > 1).float().mean()))
    result['mean_baseline_note'] = ('Mean R is computed from the entire validation label set for '
                                    'descriptive error comparison only; it is not a learned, '
                                    'held-out, or deployable baseline.')
    result['complex_metric_definition'] = 'Real/imaginary component errors, not complex-modulus errors'
    return result, errors, mean_errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--label_manifest', type=Path, required=True)
    parser.add_argument('--parity_report', type=Path)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--modes', nargs='+', choices=list(MODE_FRACTIONS), default=list(MODE_FRACTIONS))
    parser.add_argument('--expected_full_psnr', type=float, default=27.057698)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Require batch_size >= 1 and num_workers >= 0')
    for name in ('weights', 'label_manifest', 'output_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir.exists():
        parser.error('Choose a new output directory; previous results are never overwritten')
    if not args.weights.is_file():
        parser.error('Checkpoint not found')
    args.modes = ['pred'] + [mode for mode in dict.fromkeys(args.modes) if mode != 'pred']
    return args


def main():
    args = parse_args()
    torch.set_num_threads(2)
    report, rows = validate_parity_report(args.label_manifest, args.parity_report)
    dataset = VerifiedResponseDataset(rows)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use an available GPU or --device cpu')
    model, checkpoint = build_model(args.weights)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(status='running', checkpoint=checkpoint,
                  options={key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
                  images=len(dataset), verified_label_manifest_sha256=file_sha256(args.label_manifest),
                  parity_report=report,
                  diagnostic_source_sha256=file_sha256(Path(__file__)),
                  metric='Mean per-image float RGB PSNR, clipped [0,1], no PNG quantization',
                  interpretation='True/mixed R are non-deployable diagnostics, not a strict performance upper bound; '
                                 'z0/z1 stay the current image predictions. Label/input domain can differ '
                                 '(linear-RGB operator descriptor versus sRGB restoration).')
    write_json(args.output_dir/'config.json', config)
    predictions = []
    with torch.no_grad():
        for inputs, _target, _true_response, _names in loader:
            predicted = model.ddem(inputs.to(device))['kernel_response'].float().cpu()
            if not torch.isfinite(predicted).all():
                raise FloatingPointError('Nonfinite predicted R')
            predictions.append(predicted)
        predictions = torch.cat(predictions)
        truth = dataset.responses
        quality, errors, mean_errors = quality_summary(predictions, truth)
        np.savez_compressed(args.output_dir/'predictions.npz',
                            predicted=predictions.numpy(), target=truth.numpy(),
                            filenames=np.asarray([row['sample_id'] for row in rows]))
        write_json(args.output_dir/'response_quality.json', quality)
        print('Response prediction quality:', json.dumps(quality, ensure_ascii=False), flush=True)
        with (args.output_dir/'per_radius.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['predictor', 'radius_index', 'radius_cycles_per_pixel', *METRIC_NAMES])
            writer.writeheader()
            for label, values in [('prediction', errors), ('validation_mean_baseline', mean_errors)]:
                for radius in range(4):
                    record = {name: float(values[name][:, radius].square().mean().sqrt())
                              if name.endswith('rmse') else float(values[name][:, radius].mean())
                              for name in METRIC_NAMES}
                    writer.writerow(dict(predictor=label, radius_index=radius,
                                         radius_cycles_per_pixel=[1/64, 1/32, 1/16, 1/8][radius], **record))
        scores = {}
        summary = dict(status='running', checkpoint=checkpoint, images=len(dataset),
                       response_quality=quality, modes={})
        fields = ['mode', 'sample_id', 'case_name', 'strength', 'psnr', 'mode_minus_pred_db',
                  *METRIC_NAMES]
        with (args.output_dir/'per_image.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for mode in args.modes:
                start = time.monotonic()
                values, names, state = [], [], {}
                state['fraction'] = MODE_FRACTIONS[mode]
                with oracle_response_intervention(model, state):
                    for batch_index, (inputs, target, true_response, batch_names) in enumerate(loader):
                        state['target'] = true_response.to(device)
                        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                            restored = model(inputs.to(device))
                        values.extend(paired_psnr(restored, target.to(device)))
                        names.extend(batch_names)
                        if (batch_index + 1) % 25 == 0:
                            print(f'{mode}: {len(values)}/{len(dataset)}', flush=True)
                if names != [row['sample_id'] for row in rows]:
                    raise RuntimeError('Sample order mismatch in oracle evaluation')
                scores[mode] = values
                mean = statistics.mean(values)
                if mode == 'pred' and args.expected_full_psnr is not None:
                    if abs(mean - args.expected_full_psnr) > .001:
                        raise RuntimeError(f'Prediction parity failed: {mean:.6f} vs {args.expected_full_psnr:.6f}')
                differences = [value - baseline for value, baseline in zip(values, scores['pred'])]
                summary['modes'][mode] = dict(true_R_fraction=MODE_FRACTIONS[mode], mean_psnr_db=mean,
                    mean_mode_minus_pred_db=statistics.mean(differences),
                    median_mode_minus_pred_db=statistics.median(differences),
                    fraction_mode_better=sum(value > 0 for value in differences)/len(differences))
                for index, (row, value, difference) in enumerate(zip(rows, values, differences)):
                    record = {name: float(aggregate_metric(errors, name)[index]) for name in METRIC_NAMES}
                    writer.writerow(dict(mode=mode, sample_id=row['sample_id'],
                        case_name=row.get('case_name', ''), strength=row.get('strength', ''),
                        psnr=f'{value:.10f}', mode_minus_pred_db=f'{difference:.10f}', **record))
                stream.flush()
                write_json(args.output_dir/'summary.json', summary)
                print(f'{mode}: PSNR={mean:.6f}, change={statistics.mean(differences):+.6f}, '
                      f'seconds={time.monotonic()-start:.1f}', flush=True)
    with (args.output_dir/'per_motion_case.csv').open('x', newline='') as stream:
        fields = ['mode', 'case_name', 'images', 'mean_psnr_db', 'mean_mode_minus_pred_db', *METRIC_NAMES]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode, values in scores.items():
            for case in sorted({row.get('case_name', '') for row in rows}):
                indices = [index for index, row in enumerate(rows) if row.get('case_name', '') == case]
                differences = [values[index]-scores['pred'][index] for index in indices]
                record = {
                    name: float(aggregate_metric(errors, name)[indices].square().mean().sqrt())
                    if name.endswith('rmse') else float(aggregate_metric(errors, name)[indices].mean())
                    for name in METRIC_NAMES
                }
                writer.writerow(dict(mode=mode, case_name=case, images=len(indices),
                    mean_psnr_db=statistics.mean(values[index] for index in indices),
                    mean_mode_minus_pred_db=statistics.mean(differences), **record))
    summary['status'] = config['status'] = 'complete'
    write_json(args.output_dir/'summary.json', summary)
    write_json(args.output_dir/'config.json', config)
    print('Completed response prediction diagnostics:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()

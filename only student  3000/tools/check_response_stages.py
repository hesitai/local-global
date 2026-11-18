#!/usr/bin/env python3
"""Frozen-checkpoint validation of R injection sites and residual strength.

These inference interventions are not retrained ablations. No production
model sources, weights, datasets, or existing result directories are changed.
"""

import argparse
import contextlib
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_guidance_contributions import (
    DEFAULT_INPUT, build_model, bypass_response, file_sha256, paired_psnr,
)
from dataset.dataset_motiondeblur import get_validation_deblur_data
from kernel_response import JointResidualResponseConditionedLocal


def find_controllers(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, JointResidualResponseConditionedLocal)}


@contextlib.contextmanager
def response_setting(model, bypass_stage=None, scale=1.0):
    """Temporarily bypass one controller or scale all R residual corrections.

    Scale each expert's output, before FP32 mixture/strength multiplication.
    This implements L + scale * alpha * sum(w * E(L)) without changing gates,
    weights, the identity path, or clipping the learned strength. Scale 1 has
    no hooks and follows the exact original computation.
    """
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('scale must be finite and nonnegative')
    controllers = find_controllers(model)
    if not controllers:
        raise ValueError('Expected joint_residual response controllers')
    if bypass_stage is not None and bypass_stage not in controllers:
        raise ValueError('Unknown response controller: ' + str(bypass_stage))
    if bypass_stage is not None and scale != 1.0:
        raise ValueError('Vary location or global scale independently')
    with contextlib.ExitStack() as stack:
        if bypass_stage is not None:
            stack.enter_context(patch.object(
                controllers[bypass_stage], 'forward', bypass_response))
        if scale != 1.0:
            def rescale(_module, _inputs, output):
                return output.float() * scale
            for controller in controllers.values():
                for expert in controller.experts:
                    handle = expert.register_forward_hook(rescale)
                    stack.callback(handle.remove)
        yield


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input_dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--scales', type=float, nargs='+', default=[.5, .75, 1., 1.25, 1.5])
    parser.add_argument('--expected_full_psnr', type=float, default=27.057698)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Require batch_size >= 1 and num_workers >= 0')
    if any(not math.isfinite(value) or value < 0 for value in args.scales):
        parser.error('Scales must be finite and nonnegative')
    for key in ['weights', 'input_dir', 'output_dir']:
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.output_dir.exists():
        parser.error('Choose a new output directory; existing files are never overwritten')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    model, metadata = build_model(args.weights)
    if metadata['kernel_response_variant'] != 'joint_residual':
        raise ValueError('Residual scaling requires the joint_residual architecture')
    controllers = find_controllers(model)
    if len(controllers) != 8:
        raise ValueError('Expected exactly eight R injection sites')
    model.to(device).eval()
    dataset = get_validation_deblur_data(str(args.input_dir), img_options=None)
    if not len(dataset):
        raise ValueError('Validation set is empty')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    manifest_path = args.input_dir / 'pairs.tsv'
    with manifest_path.open(newline='') as stream:
        manifest = {row['sample_id']: row for row in csv.DictReader(stream, delimiter='\t')}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = [('full', None, 1.)]
    settings += [('bypass_' + name.split('.')[0], name, 1.) for name in controllers]
    settings += [('scale_' + format(scale, 'g'), None, scale)
                 for scale in dict.fromkeys(args.scales) if scale != 1.]
    config = dict(status='running', checkpoint=metadata, images=len(dataset),
                  options={key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
                  stages=list(controllers), source_sha256=file_sha256(Path(__file__)),
                  manifest_sha256=file_sha256(manifest_path),
                  interpretation='Same-weight inference interventions, not retrained ablations. '
                  'Configuration selection uses validation only; not a fresh generalization test.')
    write_json(args.output_dir / 'config.json', config)
    results, names_reference = {}, None
    fields = ['mode', 'filename', 'case_name', 'strength', 'psnr', 'mode_minus_full_db']
    with (args.output_dir / 'per_image.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for mode, stage, scale in settings:
                started = time.monotonic()
                values, names = [], []
                with response_setting(model, stage, scale):
                    for target, inputs, batch_names in loader:
                        if inputs.shape[-2:] != (256, 256):
                            raise ValueError('Use the unchanged 256x256 validation patches')
                        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                            output = model(inputs.to(device))
                        values.extend(paired_psnr(output, target.to(device)))
                        names.extend(batch_names)
                if names_reference is None:
                    names_reference = names
                elif names_reference != names:
                    raise RuntimeError('Sample ordering changed')
                if mode == 'full' and abs(statistics.mean(values) - args.expected_full_psnr) > .001:
                    raise RuntimeError('Full-model validation parity failed')
                results[mode] = values
                changes = [value - reference for value, reference in zip(values, results['full'])]
                for name, value, change in zip(names, values, changes):
                    row = manifest[name]
                    writer.writerow(dict(mode=mode, filename=name, case_name=row['case_name'],
                                         strength=row['strength'], psnr=value,
                                         mode_minus_full_db=change))
                stream.flush()
                summary = {key: dict(mean_psnr=statistics.mean(scores),
                           mean_change=statistics.mean(a-b for a,b in zip(scores, results['full'])),
                           median_change=statistics.median(a-b for a,b in zip(scores, results['full'])),
                           fraction_better=sum(a>b for a,b in zip(scores, results['full']))/len(scores))
                           for key, scores in results.items()}
                write_json(args.output_dir / 'summary.json', dict(status='running', images=len(dataset), modes=summary))
                print(f'{mode}: PSNR={statistics.mean(values):.6f}, '
                      f'change={statistics.mean(changes):+.6f}, seconds={time.monotonic()-started:.1f}', flush=True)
        # Hooks and patched forwards must leave the original computation intact.
        target, inputs, _ = next(iter(loader))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            restored = model(inputs.to(device))
        recovery = paired_psnr(restored, target.to(device))
        if max(abs(a-b) for a,b in zip(recovery, results['full'])) > 1e-5:
            raise RuntimeError('Full-model output did not recover after interventions')
    write_json(args.output_dir / 'summary.json', dict(status='complete', images=len(dataset), modes=summary,
               full_output_restored=True))
    config['status'] = 'complete'
    write_json(args.output_dir / 'config.json', config)


if __name__ == '__main__':
    main()

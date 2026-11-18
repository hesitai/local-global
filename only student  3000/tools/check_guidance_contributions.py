#!/usr/bin/env python3
"""Measure same-checkpoint z0/z1/R interventions on a fixed validation set.

These are inference interventions, not retrained architecture ablations. They
measure the checkpoint's dependence on its guides, including interactions and
possible out-of-distribution effects after removing a guide. No model weights,
training configuration, image data, or previous results are modified.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_motiondeblur import get_validation_deblur_data
from kernel_response import ResponseConditionedLocal, JointResidualResponseConditionedLocal
from model import Z0StageAdapter, Z1StageAdapter
from utils.model_utils import get_arch

DEFAULT_INPUT = PROJECT_ROOT / (
    'dataset/UAV_single_image_pose_split/fixed_protocol_first_half_60_78_visdrone_all/'
    'validation_fixed_online_2d/epoch_000000')
MODES = {
    'full': (True, True, True),
    'no_z0': (False, True, True),
    'no_z1': (True, False, True),
    'no_R': (True, True, False),
    'only_z0': (True, False, False),
    'only_z1': (False, True, False),
    'only_R': (False, False, True),
    'none': (False, False, False),
    'shuffled_R': (True, True, True),
    'mean_R': (True, True, True),
}
DEFAULT_MODES = tuple(MODES)
MODES.update({
    'shuffled_z0': (True, True, True),
    'mean_z0': (True, True, True),
    'shuffled_z1': (True, True, True),
    'mean_z1': (True, True, True),
})
REPLACED_GUIDES = {
    'shuffled_R': ('kernel_response', 'shuffle'),
    'mean_R': ('kernel_response', 'mean'),
    'shuffled_z0': ('z0', 'shuffle'),
    'mean_z0': ('z0', 'mean'),
    'shuffled_z1': ('z1', 'shuffle'),
    'mean_z1': ('z1', 'mean'),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input_dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_images', type=int, default=0,
                        help='0: all images; otherwise evenly spaced diagnostic subset')
    parser.add_argument('--modes', nargs='+', choices=list(MODES), default=list(DEFAULT_MODES))
    parser.add_argument('--expected_full_psnr', type=float, default=None,
                        help='Optional validation parity guard, tolerance 0.001 dB')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_images < 0:
        parser.error('Require batch_size >= 1, num_workers >= 0, max_images >= 0')
    if 'full' not in args.modes:
        parser.error('--modes must include full as the reference')
    # Always establish the correct-guide reference first.
    args.modes = ['full'] + [mode for mode in dict.fromkeys(args.modes) if mode != 'full']
    for key in ('weights', 'input_dir', 'output_dir'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.weights.is_file():
        parser.error('Checkpoint does not exist: ' + str(args.weights))
    if not args.input_dir.is_dir():
        parser.error('Validation directory does not exist: ' + str(args.input_dir))
    if args.output_dir.exists():
        parser.error('Choose a new output directory; existing results are never overwritten')
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def build_model(weights):
    checkpoint = torch.load(weights, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected a checkpoint with state_dict')
    config_path = weights.parent.parent / 'kernel_response_config.json'
    saved = json.loads(config_path.read_text())['options'] if config_path.is_file() else {}
    variant = checkpoint.get('kernel_response_variant', saved.get('kernel_response_variant', 'serial'))
    config = dict(arch='Uformer_B', train_ps=256, embed_dim=32, dd_in=3,
                  use_local_branch=True, use_channel_aware=True, double_process=True,
                  use_ddem=True, ddem_stage=2, ddem_dim=32, ddem_blocks=2,
                  ddem_backbone='local_global', ddem_heads=1, ddem_win_size=8,
                  local_expansion=2, use_kernel_response=True,
                  kernel_response_groups=4, kernel_response_variant=variant)
    for key in config:
        if key in saved and key != 'kernel_response_variant':
            config[key] = saved[key]
    if not (config['use_ddem'] and config['use_kernel_response'] and config['ddem_stage'] == 2):
        raise ValueError('Diagnostic requires a blur-only checkpoint with z0/z1/R enabled')
    model = get_arch(SimpleNamespace(**config))
    state = {key[7:] if key.startswith('module.') else key: value
             for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = dict(epoch=checkpoint.get('epoch'),
                    global_iteration=checkpoint.get('global_iteration'),
                    kernel_response_variant=variant, architecture=config,
                    weights_sha256=file_sha256(weights))
    return model, metadata


def bypass_response(local, response, shift_size=0, return_stats=False):
    """Identity bypass of the full R controller, not replacement of R by zero."""
    if return_stats:
        raise ValueError('R-bypass diagnostic does not collect controller gate stats')
    return local


@contextlib.contextmanager
def guidance_intervention(model, mode, replacement_state=None):
    """Temporarily disable guides or replace one guide; restore state on exit.

    Before each batch, replacement_state contains the replacement under 'z0',
    'z1', or the legacy-compatible 'response' key for R. The other two guides
    remain the checkpoint's predictions for the actual current input.
    """
    if mode not in MODES:
        raise ValueError('Unknown intervention mode: ' + str(mode))
    z0_enabled, z1_enabled, r_enabled = MODES[mode]
    old_flags = []
    with contextlib.ExitStack() as stack:
        try:
            for module in model.modules():
                if isinstance(module, (Z0StageAdapter, Z1StageAdapter)):
                    old_flags.append((module, module.guidance_enabled))
                    module.guidance_enabled = z0_enabled if isinstance(module, Z0StageAdapter) else z1_enabled
                if not r_enabled and isinstance(
                        module, (ResponseConditionedLocal, JointResidualResponseConditionedLocal)):
                    stack.enter_context(patch.object(module, 'forward', bypass_response))
            if mode in REPLACED_GUIDES:
                if replacement_state is None:
                    raise ValueError('Guide replacement requires a per-batch replacement_state')
                guide_key, _replacement_kind = REPLACED_GUIDES[mode]
                state_key = 'response' if guide_key == 'kernel_response' else guide_key

                def replace_response(_module, _inputs, outputs):
                    replacement = replacement_state.get(state_key)
                    if replacement is None or replacement.shape != outputs[guide_key].shape:
                        raise ValueError('Replacement ' + guide_key + ' is missing or has an unexpected shape')
                    updated = dict(outputs)
                    updated[guide_key] = replacement
                    return updated

                handle = model.ddem.register_forward_hook(replace_response)
                stack.callback(handle.remove)
            yield
        finally:
            for module, enabled in old_flags:
                module.guidance_enabled = enabled


def paired_psnr(restored, target):
    """Float RGB PSNR before PNG quantization; mean of per-image dB values."""
    error = restored.float().clamp(0, 1).double() - target.float().clamp(0, 1).double()
    mse = error.square().mean(dim=(1, 2, 3))
    if torch.any(mse <= 0):
        raise ValueError('Nonpositive MSE yields infinite PSNR; inspect this diagnostic sample')
    score = -10 * torch.log10(mse)
    if not torch.isfinite(score).all():
        raise FloatingPointError('Nonfinite PSNR')
    return score.cpu().tolist()


def describe_delta(reference, intervention):
    difference = [a - b for a, b in zip(reference, intervention)]
    return dict(mean_full_minus_intervention_db=statistics.mean(difference),
                median_full_minus_intervention_db=statistics.median(difference),
                fraction_full_better=sum(value > 0 for value in difference) / len(difference),
                minimum_full_minus_intervention_db=min(difference),
                maximum_full_minus_intervention_db=max(difference))


def main():
    args = parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select an available device or use --device cpu')
    model, checkpoint_metadata = build_model(args.weights)
    model.to(device)
    dataset = get_validation_deblur_data(str(args.input_dir), img_options=None)
    total = len(dataset)
    if not total:
        raise ValueError('Validation set is empty')
    indices = list(range(total))
    if args.max_images and args.max_images < total:
        indices = torch.linspace(0, total - 1, args.max_images).round().long().tolist()
    selected = Subset(dataset, indices)
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    if any(mode.startswith('shuffled_') for mode in args.modes) and len(indices) < 2:
        raise ValueError('Global guide derangement requires at least two validation images')
    manifest_path = args.input_dir / 'pairs.tsv'
    manifest = {}
    if manifest_path.is_file():
        with manifest_path.open(newline='') as stream:
            manifest = {Path(row['input_path']).stem: row
                        for row in csv.DictReader(stream, delimiter='\t')}
    first_target, first_input, _first_name = selected[0]
    if (tuple(first_input.shape[-2:]) != (256, 256)
            or first_target.shape != first_input.shape):
        raise ValueError('Use matched original fixed 256x256 validation patches')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(status='running', checkpoint=checkpoint_metadata,
                  options={key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
                  dataset_images=total, selected_indices=indices,
                  intervention_flags=MODES,
                  diagnostic_source_sha256=file_sha256(Path(__file__)),
                  metric='mean per-image float RGB PSNR, range [0,1], no crop, no PNG quantization',
                  inference='fixed validation patches; no full-image tiled inference',
                  interpretation='Same-checkpoint interventions measure reliance and interactions; '
                                 'they are not retrained ablations and can introduce distribution shift.')
    write_json(args.output_dir / 'config.json', config)
    # Cache all predicted guides once so global replacement remains effective
    # at batch_size=1 and never changes the two guides not being intervened on.
    cached_guides = {key: [] for key in ('z0', 'z1', 'kernel_response')}
    with torch.no_grad():
        if any(mode in REPLACED_GUIDES for mode in args.modes):
            for _target, inputs, _names in loader:
                if tuple(inputs.shape[-2:]) != (256, 256):
                    raise ValueError('Use the original fixed 256x256 validation patches')
                prediction = model.ddem(inputs.to(device))
                for key in cached_guides:
                    cached_guides[key].append(prediction[key].cpu())
            cached_guides = {key: torch.cat(values) for key, values in cached_guides.items()}
            mean_guides = {key: values.mean(dim=0, keepdim=True)
                           for key, values in cached_guides.items()}
            # R becomes spatially constant; z0/z1 retain descriptor dimensions.
            mean_guides['kernel_response'] = cached_guides['kernel_response'].mean(
                dim=(0, 2, 3), keepdim=True)
            derangement_shift = max(1, len(indices) // 2)
            print(f'Cached {len(indices)} z0/z1/R predictions; '
                  f'derangement offset={derangement_shift}', flush=True)
        scores, names_reference, input_psnr = {}, None, []
        per_image_path = args.output_dir / 'per_image.csv'
        fields = ['mode', 'filename', 'psnr', 'full_minus_mode_db', 'case_name', 'strength']
        with per_image_path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for mode in args.modes:
                started = time.monotonic()
                values, names = [], []
                replacement = {}
                with guidance_intervention(model, mode, replacement):
                    for batch_index, (target, inputs, batch_names) in enumerate(loader):
                        if tuple(inputs.shape[-2:]) != (256, 256):
                            raise ValueError('Use the original fixed 256x256 validation patches')
                        offset, size = len(values), len(batch_names)
                        if mode in REPLACED_GUIDES:
                            key, kind = REPLACED_GUIDES[mode]
                            state_key = 'response' if key == 'kernel_response' else key
                            values_cache = cached_guides[key]
                            if kind == 'shuffle':
                                shuffled = [(i + derangement_shift) % len(indices)
                                            for i in range(offset, offset + size)]
                                replacement[state_key] = values_cache[shuffled].to(device)
                            else:
                                replacement[state_key] = mean_guides[key].expand(
                                    size, *values_cache.shape[1:]).to(device)
                        inputs, target = inputs.to(device), target.to(device)
                        if mode == 'full':
                            input_psnr.extend(paired_psnr(inputs, target))
                        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                            restored = model(inputs)
                        values.extend(paired_psnr(restored, target))
                        names.extend(batch_names)
                        if (batch_index + 1) % 25 == 0:
                            print(f'{mode}: {len(values)}/{len(indices)}', flush=True)
                if names_reference is None:
                    names_reference = names
                elif names != names_reference:
                    raise RuntimeError('Validation image order changed between intervention modes')
                scores[mode] = values
                mean = statistics.mean(values)
                if mode == 'full' and args.expected_full_psnr is not None:
                    if abs(mean - args.expected_full_psnr) > .001:
                        raise RuntimeError(f'Full-model validation parity failed: {mean:.6f} '
                                           f'vs expected {args.expected_full_psnr:.6f}')
                for index, (name, value) in enumerate(zip(names, values)):
                    row = manifest.get(name, {})
                    writer.writerow(dict(mode=mode, filename=name, psnr=f'{value:.10f}',
                                         full_minus_mode_db=f'{scores["full"][index] - value:.10f}',
                                         case_name=row.get('case_name', ''), strength=row.get('strength', '')))
                stream.flush()
                print(f'{mode}: mean PSNR={mean:.6f} dB, '
                      f'full-minus-mode={statistics.mean(scores["full"]) - mean:+.6f}, '
                      f'seconds={time.monotonic() - started:.1f}', flush=True)
                summary = dict(status='running', checkpoint=checkpoint_metadata,
                               images=len(indices),
                               input_mean_psnr_db=statistics.mean(input_psnr), modes={
                                   key: dict(mean_psnr_db=statistics.mean(value),
                                             **describe_delta(scores['full'], value))
                                   for key, value in scores.items()})
                write_json(args.output_dir / 'summary.json', summary)
    summary['status'] = 'complete'
    write_json(args.output_dir / 'summary.json', summary)
    config['status'] = 'complete'
    write_json(args.output_dir / 'config.json', config)
    print('Completed diagnostics:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()

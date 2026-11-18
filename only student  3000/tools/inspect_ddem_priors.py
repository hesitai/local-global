#!/usr/bin/env python3
"""Inspect DDEM dynamics and stage-last guided-attention changes."""

import argparse
import math
import os
import sys
from collections import defaultdict, OrderedDict

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'dataset'))

from dataset.dataset_motiondeblur import get_validation_deblur_data
from model import (
    Uformer, WindowAttention, Z0StageAdapter, Z1StageAdapter)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure DDEM priors and guided Q/local/K/V changes')
    parser.add_argument('--weights', required=True)
    parser.add_argument(
        '--input_dir',
        default='./dataset/datasets/deblurring/GoPro/test')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ddem_stage', type=int, choices=[1, 2], default=2)
    parser.add_argument('--ddem_dim', type=int, default=32)
    parser.add_argument('--ddem_blocks', type=int, default=2)
    parser.add_argument(
        '--ddem_backbone', choices=['local_global', 'transformer', 'conv'],
        default='local_global')
    parser.add_argument('--ddem_heads', type=int, default=1)
    parser.add_argument('--ddem_win_size', type=int, default=8)
    return parser.parse_args()


def build_model(args):
    return Uformer(
        img_size=256, embed_dim=32, win_size=8,
        token_projection='linear', token_mlp='leff',
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        modulator=True, dd_in=3,
        use_local_branch=True, use_channel_aware=True,
        double_process=True, use_ddem=True,
        ddem_stage=args.ddem_stage, ddem_dim=args.ddem_dim,
        ddem_blocks=args.ddem_blocks,
        ddem_backbone=args.ddem_backbone,
        ddem_heads=args.ddem_heads,
        ddem_win_size=args.ddem_win_size)


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    state = OrderedDict(
        (key[7:] if key.startswith('module.') else key, value)
        for key, value in checkpoint['state_dict'].items())
    model.load_state_dict(state, strict=True)
    return checkpoint.get('epoch', 'unknown')


def pad_pair(input_image, target, factor=128):
    _, _, height, width = input_image.shape
    padded_h = int(math.ceil(height / float(factor)) * factor)
    padded_w = int(math.ceil(width / float(factor)) * factor)
    pad_h, pad_w = padded_h - height, padded_w - width
    if pad_h or pad_w:
        input_image = F.pad(
            input_image, (0, pad_w, 0, pad_h), mode='reflect')
        target = F.pad(target, (0, pad_w, 0, pad_h), mode='reflect')
    mask = input_image.new_zeros(1, 1, padded_h, padded_w)
    mask[:, :, :height, :width] = 1
    return input_image, target, mask


def tensor_summary(tensor):
    tensor = tensor.float()
    return (
        float(tensor.mean()), float(tensor.std(unbiased=False)),
        float(tensor.min()), float(tensor.max()))


def cross_image_summary(name, values):
    values = torch.stack(values).float().flatten(1)
    feature_std = values.std(dim=0, unbiased=False).mean()
    centered = values - values.mean(dim=0, keepdim=True)
    rms_variation = centered.square().mean().sqrt()
    mean_abs = values.abs().mean().clamp_min(1e-8)
    if values.shape[0] > 1:
        normalized = F.normalize(values, dim=1)
        similarity = normalized @ normalized.transpose(0, 1)
        off_diagonal = similarity[~torch.eye(
            values.shape[0], dtype=torch.bool)]
        cosine = float(off_diagonal.mean())
    else:
        cosine = float('nan')
    print(
        f'{name}: cross-image feature-std={float(feature_std):.6e}, '
        f'RMS variation={float(rms_variation):.6e}, '
        f'relative variation={float(rms_variation / mean_abs):.6e}, '
        f'mean cosine={cosine:.6f}')


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = build_model(args)
    epoch = load_checkpoint(model, args.weights)
    model.to(device).eval()

    dataset = get_validation_deblur_data(args.input_dir, img_options=None)
    sample_count = min(args.num_samples, len(dataset))
    sample_indices = torch.linspace(
        0, len(dataset) - 1, steps=sample_count).round().long().tolist()

    z0_values, z1_values = [], []
    layer_values = defaultdict(lambda: defaultdict(list))
    adapter_values = defaultdict(lambda: defaultdict(list))
    print(f'checkpoint epoch: {epoch}; samples: {args.num_samples}')
    print('Per-image prior summary: mean / std / min / max')

    with torch.no_grad():
        for dataset_index in sample_indices:
            target, input_image, filename = dataset[dataset_index]
            input_image = input_image.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            input_image, target, valid_mask = pad_pair(input_image, target)
            kwargs = dict(
                valid_mask=valid_mask, return_ddem_features=True,
                collect_prior_stats=True)
            if args.ddem_stage == 1:
                kwargs['gt'] = target
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                _, prior = model(input_image, **kwargs)

            z0 = prior['z0'].detach().cpu().squeeze(0)
            z1 = prior['z1'].detach().cpu().squeeze(0)
            z0_values.append(z0)
            z1_values.append(z1)
            print(
                f'{str(filename):30s} Z0={tensor_summary(z0)} '
                f'Z1={tensor_summary(z1)}')

            modules = dict(model.named_modules())
            for name, module in modules.items():
                if not isinstance(module, Z0StageAdapter):
                    continue
                gamma, beta = module.modulation(prior['z0'])
                adapter_values[name]['gamma'].append(
                    gamma.detach().cpu().squeeze(0))
                adapter_values[name]['beta'].append(
                    beta.detach().cpu().squeeze(0))
                z1_name = name.replace(
                    'z0_adapter_second', 'z1_adapter_second')
                z1_adapter = modules[z1_name]
                matrix, strength = z1_adapter(prior['z1'])
                adapter_values[name]['matrix'].append(
                    matrix.detach().cpu().squeeze(0))
                adapter_values[name]['strength'].append(float(strength))

            for name, module in model.named_modules():
                if not isinstance(module, WindowAttention):
                    continue
                stats = module.last_prior_stats
                if stats is None:
                    continue
                for key, value in stats.items():
                    layer_values[name][key].append(value)

    if not z0_values:
        raise RuntimeError('No images were analyzed')

    print('\nCross-image dynamics (near-zero variation means collapse):')
    cross_image_summary('Z0', z0_values)
    cross_image_summary('Z1', z1_values)

    print('\nStage-adapter cross-image dynamics:')
    for name in sorted(adapter_values):
        cross_image_summary(name + '/gamma', adapter_values[name]['gamma'])
        cross_image_summary(name + '/beta', adapter_values[name]['beta'])
        cross_image_summary(name + '/M', adapter_values[name]['matrix'])

    print('\nPer-layer effect averaged across images:')
    print(
        f'{"Attention layer":58s} {"dQ/Q":>11s} {"dK/K":>11s} '
        f'{"dV/V":>11s} {"dL/L":>11s} {"alpha":>9s} '
        f'{"M entropy":>11s} {"M max":>9s}')
    for name in sorted(layer_values):
        q_change = sum(layer_values[name]['q_relative_change']) / len(
            layer_values[name]['q_relative_change'])
        k_change = sum(layer_values[name]['k_relative_change']) / len(
            layer_values[name]['k_relative_change'])
        v_change = sum(layer_values[name]['v_relative_change']) / len(
            layer_values[name]['v_relative_change'])
        local_change = sum(
            layer_values[name]['local_relative_change']) / len(
                layer_values[name]['local_relative_change'])
        strength = sum(layer_values[name]['z1_strength']) / len(
            layer_values[name]['z1_strength'])
        entropy = sum(layer_values[name]['local_matrix_entropy']) / len(
            layer_values[name]['local_matrix_entropy'])
        matrix_max = sum(layer_values[name]['local_matrix_max']) / len(
            layer_values[name]['local_matrix_max'])
        print(
            f'{name:58s} {q_change:11.6e} {k_change:11.6e} '
            f'{v_change:11.6e} {local_change:11.6e} '
            f'{strength:9.6f} {entropy:11.6f} {matrix_max:9.6f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Reproduce the complete model's paired, native-resolution test.

From the repository root, run: python test/test_gopro_hide.py
All paths and model options have project-relative defaults and remain overridable.
Importing this module never parses CLI options, creates results, or runs inference.
"""
import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / (
    'dataset/UAV_single_image_pose_split/'
    'fixed_protocol_first_half_60_78_visdrone_all/test_fixed_full')
DEFAULT_WEIGHTS = PROJECT_ROOT / (
    'logs/motiondeblur/Online2D/'
    'Uformer_B_single_image_online_2d_patch256_full_k12_r50_i600000_'
    'kr16_c64_all8_serial_g4_v2_complex_w0p01_18eeafbbe4/'
    'models/model_epoch_639.pth')
DEFAULT_RESULT_DIR = PROJECT_ROOT / (
    'results/deblurring/Online2D/joint2d_i600000_final_fixed_full_Lwf4ZZ')

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Retain the public four-argument tile helpers used by existing protocol audits.
# These two runtime bindings are populated only by main(), never at import.
args = None
model_restoration = None


def add_boolean_option(parser, name, default, help_text):
    """Python 3.8-compatible positive/negative switches, retaining old CLI names."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--' + name, dest=name, action='store_true', help=help_text)
    group.add_argument('--no_' + name, '--no-' + name.replace('_', '-'),
                       dest=name, action='store_false', help='Disable: ' + help_text)
    parser.set_defaults(**{name: default})


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate the complete 600k-iteration UAV deblurring model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', default=str(DEFAULT_INPUT_DIR),
        type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default=str(DEFAULT_RESULT_DIR),
        type=str, help='Directory for results')
    parser.add_argument('--weights', default=str(DEFAULT_WEIGHTS),
        type=str, help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
    parser.add_argument('--batch_size', default=1, type=int, help='One full-resolution image per loader batch; only 1 is supported')
    add_boolean_option(parser, 'save_images', True, 'Save full-resolution restored PNG images')
    add_boolean_option(parser, 'save_per_image_metrics', True,
                       'Save filename, PSNR, SSIM and image size as CSV')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension (architecture dependent)')
    parser.add_argument('--win_size', type=int, default=8, help='Window size (architecture dependent)')
    parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
    add_boolean_option(parser, 'use_local_branch', True, 'Enable use local branch')
    add_boolean_option(parser, 'double_process', True, 'Enable double process')
    parser.add_argument('--local_expansion', type=int, default=2)
    add_boolean_option(parser, 'use_channel_aware', True, 'Enable use channel aware')
    add_boolean_option(parser, 'use_ddem', True, 'Enable use ddem')
    add_boolean_option(parser, 'use_kernel_response', True,
                       'Build the response head and eight serial grouped controllers')
    parser.add_argument('--kernel_response_groups', type=int, choices=[1, 4], default=4,
                        help='must match training: 1 for previous model, 4 for grouped model')
    parser.add_argument(
        '--kernel_response_variant', choices=['serial', 'joint_residual'],
        default='serial',
        help='must match the checkpoint architecture; serial is the legacy model')
    parser.add_argument('--ddem_stage', type=int, choices=[1, 2], default=2)
    parser.add_argument('--ddem_dim', type=int, default=32)
    parser.add_argument('--ddem_blocks', type=int, default=2)
    parser.add_argument('--ddem_backbone', type=str,
                        choices=['local_global', 'transformer', 'conv'],
                        default='local_global')
    parser.add_argument('--ddem_heads', type=int, default=1)
    parser.add_argument('--ddem_win_size', type=int, default=8)
    parser.add_argument('--disable_z0_guidance', action='store_true', default=False,
                        help='inference ablation: force every Z0 stage adapter to identity')
    parser.add_argument('--disable_z1_guidance', action='store_true', default=False,
                        help='inference ablation: disable every Z1-guided local-feature transform')
    parser.add_argument('--query_embed', action='store_true', default=False, help='query embedding for the decoder')
    parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

    # args for vit
    parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
    parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
    parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
    parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
    parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
    parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
    parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
    parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

    parser.add_argument('--train_ps', type=int, default=256,
                        help='nominal model size; keep equal to the training patch size')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='overlapping inference tile size; 0 uses one full-image forward')
    parser.add_argument('--tile_overlap', type=int, default=64,
                        help='overlap between neighboring inference tiles')
    parser.add_argument(
        '--tile_mode', choices=['native', 'legacy_padded'], default='native',
        help=(
            'native tiles the original image and anchors the final tile at the '
            'real image edge; legacy_padded preserves the old globally padded '
            'inference path for reproducibility'))
    parser.add_argument(
        '--tile_blend', choices=['hann', 'uniform'], default='hann',
        help='weighting used to merge overlapping native-inference tiles')
    parser.add_argument('--max_images', type=int, default=0,
                        help='debug limit; 0 evaluates the complete test set')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers; use 0 for debugging')
    parser.add_argument('--existing_results', choices=['backup', 'error'], default='backup',
                        help='Back up a nonempty result directory before reusing its name, or stop')
    parser.add_argument('--check_only', action='store_true',
                        help='Validate dataset and strictly load weights on CPU; no output writes or inference')
    return parser


def parse_args(argv=None):
    parser = build_parser()
    options = parser.parse_args(argv)
    if options.batch_size != 1:
        parser.error('--batch_size must be 1 for variable-resolution paired images')
    if options.train_ps < 128 or options.train_ps % 128:
        parser.error('--train_ps must be a multiple of 128 and at least 128')
    if options.tile_size < 0:
        parser.error('--tile_size must be nonnegative')
    if options.tile_size:
        if options.tile_size < 128 or options.tile_size % 128:
            parser.error('--tile_size must be a multiple of 128 and at least 128')
        if not 0 <= options.tile_overlap < options.tile_size:
            parser.error('require 0 <= --tile_overlap < --tile_size')
    if options.max_images < 0 or options.num_workers < 0:
        parser.error('--max_images and --num_workers must be nonnegative')
    if not options.use_kernel_response and options.kernel_response_groups != 1:
        parser.error('--no_use_kernel_response requires --kernel_response_groups 1')
    if not options.use_kernel_response and options.kernel_response_variant != 'serial':
        parser.error('--no_use_kernel_response requires --kernel_response_variant serial')
    if options.use_kernel_response and not (
            options.use_ddem and options.use_local_branch
            and options.double_process and options.ddem_stage == 2):
        parser.error('kernel response requires local branch, double process and DDEM stage 2')
    gpu_ids = options.gpus.split(',')
    if not gpu_ids or any(not value.isdigit() for value in gpu_ids):
        parser.error('--gpus must be comma-separated nonnegative GPU IDs')
    for name in ('input_dir', 'weights', 'result_dir'):
        value = Path(getattr(options, name)).expanduser()
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        if name == 'result_dir' and value.is_symlink():
            parser.error('--result_dir must not be a symbolic link')
        setattr(options, name, str(value.resolve()))
    return options


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n',
                          encoding='utf-8')


def validate_inputs(options):
    """Read-only preflight before touching an old result directory."""
    from dataset.dataset_motiondeblur import get_validation_deblur_data

    data_root, weights, output = (
        Path(options.input_dir), Path(options.weights), Path(options.result_dir))
    if not data_root.is_dir():
        raise FileNotFoundError('Paired test dataset not found: ' + str(data_root))
    if not weights.is_file():
        raise FileNotFoundError('Checkpoint not found: ' + str(weights))
    if output == PROJECT_ROOT or output == Path(output.anchor):
        raise ValueError('Result directory must be a dedicated experiment directory')
    if output == data_root or output in data_root.parents or data_root in output.parents:
        raise ValueError('Result directory must be separate from the input dataset')
    if output == weights.parent or output in weights.parents:
        raise ValueError('Result directory must be separate from checkpoint storage')
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError('Result path must be an ordinary directory, not a file/symlink')
    if output.exists() and any(output.iterdir()) and options.existing_results == 'error':
        raise FileExistsError('Results already exist; choose --result_dir or --existing_results backup')
    dataset = get_validation_deblur_data(str(data_root))
    if not len(dataset):
        raise ValueError('The paired test dataset is empty')
    if output == Path(dataset.input_filenames[0]).parent or output == Path(dataset.target_filenames[0]).parent:
        raise ValueError('Results must not overwrite paired image directories')
    stems = [Path(name).stem for name in dataset.input_filenames]
    if len(set(stems)) != len(stems):
        raise ValueError('Input stems collide: restored PNG filenames would not be unique')
    return dataset


def load_evaluation_model(options):
    """Strictly load the requested state_dict on CPU; never guess EMA or architecture."""
    from utils.model_utils import get_arch

    checkpoint = torch.load(options.weights, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected a training checkpoint with top-level state_dict')
    checkpoint_variant = checkpoint.get('kernel_response_variant', 'serial')
    if options.use_kernel_response and checkpoint_variant != options.kernel_response_variant:
        raise ValueError(
            'Response controller architecture mismatch: checkpoint uses {!r}, '
            'but --kernel_response_variant is {!r}. Pass '
            '--kernel_response_variant {} to match this checkpoint.'.format(
                checkpoint_variant, options.kernel_response_variant, checkpoint_variant))
    model = get_arch(options)
    state = {
        key[7:] if key.startswith('module.') else key: value
        for key, value in checkpoint['state_dict'].items()
    }
    model.load_state_dict(state, strict=True)
    metadata = {
        'checkpoint_key': 'state_dict', 'strict_load': True,
        'epoch': checkpoint.get('epoch'),
        'global_iteration': checkpoint.get('global_iteration'),
        'max_iters': checkpoint.get('max_iters'),
        'kernel_response_variant': checkpoint_variant,
        'weights_sha256': sha256_file(options.weights),
    }
    return model.eval(), metadata


def prepare_result_directory(path, policy):
    """Rename old results on the same filesystem; never delete or overwrite them."""
    path = Path(path)
    if policy not in ('backup', 'error'):
        raise ValueError('Unknown existing-result policy')
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError('Result path must not be a file or symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists() and any(path.iterdir()):
        if policy == 'error':
            raise FileExistsError(path)
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        backup = path.with_name(path.name + '_backup_' + stamp)
        if backup.exists():
            raise FileExistsError(backup)
        path.rename(backup)
        print('Previous results preserved at:', backup, flush=True)
    path.mkdir(exist_ok=True)
    return str(backup) if backup else None


class TeeStream:
    """Write terminal output and the experiment log without a shell pipeline."""
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def inference_description(options):
    if options.tile_size:
        return 'tile_mode={}, tile_size={}, tile_overlap={}, tile_blend={}'.format(
            options.tile_mode, options.tile_size, options.tile_overlap,
            options.tile_blend if options.tile_mode == 'native' else 'uniform')
    return 'full_image_forward_with_reflect_padding'


def pad_to_multiple(timg, factor=128):
    """Reflect-pad height and width independently, preserving a rectangle."""
    _, _, h, w = timg.size()
    padded_h = int(math.ceil(h / float(factor)) * factor)
    padded_w = int(math.ceil(w / float(factor)) * factor)
    pad_h = padded_h - h
    pad_w = padded_w - w
    if pad_h or pad_w:
        timg = F.pad(timg, (0, pad_w, 0, pad_h), mode='reflect')
    return timg, (h, w)


def tile_starts(length, tile_size, overlap):
    """Cover an axis completely, anchoring the last tile at the far edge."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def pad_for_tiled_inference(timg, tile_size, factor=128):
    """Reflect-pad once so every tile is valid and the full image is covered."""
    _, _, h, w = timg.size()
    padded_h = max(tile_size, int(math.ceil(h / float(factor)) * factor))
    padded_w = max(tile_size, int(math.ceil(w / float(factor)) * factor))
    pad_h = padded_h - h
    pad_w = padded_w - w
    if pad_h or pad_w:
        timg = F.pad(timg, (0, pad_w, 0, pad_h), mode='reflect')
    return timg, (h, w)


def restore_tiled_legacy(input_image, target, tile_size, overlap, factor=128):
    """Preserve the old globally padded tiled inference for reproducibility."""
    padded_input, original_size = pad_for_tiled_inference(
        input_image, tile_size, factor=factor)
    original_h, original_w = original_size
    padded_h, padded_w = padded_input.shape[-2:]
    valid_mask = padded_input.new_zeros(1, 1, padded_h, padded_w)
    valid_mask[:, :, :original_h, :original_w] = 1

    padded_target = None
    if args.use_ddem and args.ddem_stage == 1:
        padded_target, target_size = pad_for_tiled_inference(
            target, tile_size, factor=factor)
        if target_size != original_size:
            raise ValueError(
                'Blur/GT sizes differ: {} vs {}'.format(
                    original_size, target_size))

    restored_sum = torch.zeros_like(padded_input, dtype=torch.float32)
    restored_weight = torch.zeros(
        1, 1, padded_h, padded_w, dtype=torch.float32)
    top_starts = tile_starts(padded_h, tile_size, overlap)
    left_starts = tile_starts(padded_w, tile_size, overlap)

    for top in top_starts:
        for left in left_starts:
            input_tile = padded_input[
                :, :, top:top + tile_size, left:left + tile_size].cuda()
            mask_tile = valid_mask[
                :, :, top:top + tile_size, left:left + tile_size].cuda()
            with torch.cuda.amp.autocast():
                if padded_target is not None:
                    target_tile = padded_target[
                        :, :, top:top + tile_size,
                        left:left + tile_size].cuda()
                    restored_tile = model_restoration(
                        input_tile, gt=target_tile, valid_mask=mask_tile)
                else:
                    restored_tile = model_restoration(
                        input_tile, valid_mask=mask_tile)
            restored_sum[
                :, :, top:top + tile_size, left:left + tile_size
            ] += restored_tile.float().cpu()
            restored_weight[
                :, :, top:top + tile_size, left:left + tile_size
            ] += 1.0
            del input_tile, mask_tile, restored_tile

    restored = restored_sum / restored_weight
    return restored[:, :, :original_h, :original_w]


def pad_to_minimum_tile(timg, tile_size):
    """Pad only dimensions smaller than a tile; normal test images stay intact."""
    _, _, h, w = timg.size()
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h or pad_w:
        # Reflection is preferred, but PyTorch requires every reflected pad to
        # be smaller than its corresponding input dimension.
        mode = 'reflect' if pad_h < h and pad_w < w else 'replicate'
        timg = F.pad(timg, (0, pad_w, 0, pad_h), mode=mode)
    return timg, (h, w)


def tile_blend_weight(tile_size, blend):
    """Return a positive 2-D weight map for seam-free overlap blending."""
    if blend == 'uniform':
        return torch.ones(1, 1, tile_size, tile_size, dtype=torch.float32)
    window = torch.hann_window(
        tile_size, periodic=False, dtype=torch.float32)
    # Keep a small positive floor. At an outer image boundary only one tile
    # contributes, so a true zero would leave that boundary uncovered.
    weight = torch.outer(window, window).clamp_min_(1.0e-3)
    return weight.unsqueeze(0).unsqueeze(0)


def restore_tiled_native(input_image, target, tile_size, overlap):
    """Restore the complete image without adding a padded right/bottom band.

    Every normal tile contains only real image pixels and has exactly the same
    all-valid geometry as a training patch. The final tile is anchored at the
    actual image edge. Padding is used only for an image dimension smaller than
    one tile, and reflected pixels are treated as ordinary context rather than
    introducing a test-only partial DDEM valid mask.
    """
    working_input, original_size = pad_to_minimum_tile(input_image, tile_size)
    original_h, original_w = original_size
    working_h, working_w = working_input.shape[-2:]

    working_target = None
    if args.use_ddem and args.ddem_stage == 1:
        working_target, target_size = pad_to_minimum_tile(target, tile_size)
        if target_size != original_size:
            raise ValueError(
                'Blur/GT sizes differ: {} vs {}'.format(
                    original_size, target_size))

    restored_sum = torch.zeros_like(working_input, dtype=torch.float32)
    restored_weight = torch.zeros(
        1, 1, working_h, working_w, dtype=torch.float32)
    blend_weight = tile_blend_weight(tile_size, args.tile_blend)
    top_starts = tile_starts(working_h, tile_size, overlap)
    left_starts = tile_starts(working_w, tile_size, overlap)

    for top in top_starts:
        for left in left_starts:
            input_tile = working_input[
                :, :, top:top + tile_size, left:left + tile_size].cuda()
            target_tile = None
            if working_target is not None:
                target_tile = working_target[
                    :, :, top:top + tile_size,
                    left:left + tile_size].cuda()
            with torch.cuda.amp.autocast():
                if target_tile is not None:
                    restored_tile = model_restoration(
                        input_tile, gt=target_tile)
                else:
                    restored_tile = model_restoration(input_tile)

            restored_sum[
                :, :, top:top + tile_size, left:left + tile_size
            ] += restored_tile.float().cpu() * blend_weight
            restored_weight[
                :, :, top:top + tile_size, left:left + tile_size
            ] += blend_weight
            del input_tile, target_tile, restored_tile

    if torch.any(restored_weight <= 0):
        raise RuntimeError('Tiled inference left uncovered output pixels')
    restored = restored_sum / restored_weight
    return restored[:, :, :original_h, :original_w]


def evaluate_dataset(test_loader):
    """Keep original inference, RGB metrics and arithmetic image-wise means."""
    from model import Z0StageAdapter, Z1StageAdapter
    from utils.image_utils import save_img

    if args.disable_z0_guidance:
        count = 0
        for module in model_restoration.modules():
            if isinstance(module, Z0StageAdapter):
                module.disable_guidance()
                count += 1
        print('===> Z0 guidance disabled in {} stage adapters'.format(count))
    if args.disable_z1_guidance:
        count = 0
        for module in model_restoration.modules():
            if isinstance(module, Z1StageAdapter):
                module.disable_guidance()
                count += 1
        print('===> Z1 guidance disabled in {} stage adapters'.format(count))

    psnr_values, ssim_values, rows = [], [], []
    csv_stream = None
    try:
        writer = None
        if args.save_per_image_metrics:
            csv_stream = (Path(args.result_dir) / 'psnr_ssim_per_image.csv').open(
                'x', newline='', encoding='utf-8')
            writer = csv.DictWriter(
                csv_stream, fieldnames=['filename', 'psnr', 'ssim', 'height', 'width'])
            writer.writeheader()
            csv_stream.flush()
        with torch.no_grad():
            for index, data_test in enumerate(tqdm(test_loader), 0):
                if args.max_images and index >= args.max_images:
                    break
                target, input_image, filenames = data_test
                if target.shape != input_image.shape:
                    raise ValueError('Blur/GT sizes differ: {} vs {}'.format(
                        tuple(input_image.shape), tuple(target.shape)))
                rgb_gt = target.numpy().squeeze().transpose((1, 2, 0))
                if args.tile_size:
                    if args.tile_mode == 'legacy_padded':
                        restored = restore_tiled_legacy(
                            input_image, target, args.tile_size, args.tile_overlap, factor=128)
                    else:
                        restored = restore_tiled_native(
                            input_image, target, args.tile_size, args.tile_overlap)
                else:
                    noisy, original_size = pad_to_multiple(input_image.cuda(), factor=128)
                    valid_mask = None
                    if args.use_ddem:
                        height, width = original_size
                        valid_mask = noisy.new_zeros(noisy.shape[0], 1, *noisy.shape[-2:])
                        valid_mask[:, :, :height, :width] = 1
                    with torch.cuda.amp.autocast():
                        if args.use_ddem and args.ddem_stage == 1:
                            padded_target, target_size = pad_to_multiple(target.cuda(), factor=128)
                            if target_size != original_size:
                                raise ValueError('Blur/GT sizes differ')
                            restored = model_restoration(
                                noisy, gt=padded_target, valid_mask=valid_mask)
                        else:
                            restored = model_restoration(noisy, valid_mask=valid_mask)
                    height, width = original_size
                    restored = restored[:, :, :height, :width]
                if not torch.isfinite(restored).all():
                    raise ValueError('Non-finite prediction: ' + filenames[0])
                rgb_restored = restored.clamp(0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
                psnr = psnr_loss(rgb_restored, rgb_gt)
                ssim = ssim_loss(rgb_restored, rgb_gt, channel_axis=-1, data_range=1.0)
                if np.isnan(psnr) or not np.isfinite(ssim):
                    raise ValueError('Invalid metrics: ' + filenames[0])
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                row = dict(filename=filenames[0], psnr=float(psnr), ssim=float(ssim),
                           height=int(rgb_restored.shape[0]), width=int(rgb_restored.shape[1]))
                if args.save_images:
                    save_img(str(Path(args.result_dir) / (filenames[0] + '.PNG')),
                             img_as_ubyte(rgb_restored))
                rows.append(row)
                if writer:
                    writer.writerow(row)
                    csv_stream.flush()
                print('PSNR:', psnr, ', SSIM:', ssim, filenames[0], rgb_restored.shape)
        if not rows:
            raise ValueError('No test images were processed')
        mean_psnr = sum(psnr_values) / len(psnr_values)
        mean_ssim = sum(ssim_values) / len(ssim_values)
        print('PSNR: %f, SSIM: %f ' % (mean_psnr, mean_ssim))
        with (Path(args.result_dir) / 'psnr_ssim.txt').open('x', encoding='utf-8') as stream:
            stream.write('Arch:' + args.arch + ', PSNR: %.4f, SSIM: %.4f] ' %
                         (mean_psnr, mean_ssim) + '\n')
            stream.write('Inference: ' + inference_description(args) + '\n')
        return rows, float(mean_psnr), float(mean_ssim)
    finally:
        if csv_stream:
            csv_stream.close()


def main(argv=None):
    global args, model_restoration
    args = parse_args(argv)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    dataset = validate_inputs(args)
    if not args.check_only and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; activate joint-cu118-test and check your GPU')
    model_restoration, checkpoint_metadata = load_evaluation_model(args)
    print('===>Testing using weights:', args.weights)
    print('===>Paired dataset:', args.input_dir, '({} images)'.format(len(dataset)))
    print('===>Results:', args.result_dir)
    print('===>Full-image inference:', inference_description(args))
    if args.check_only:
        print(json.dumps(dict(status='CHECK PASS', options=vars(args),
                              checkpoint=checkpoint_metadata, dataset_images=len(dataset)),
                         ensure_ascii=False, indent=2))
        print('Read-only check: no inference, output creation or old-result backup.')
        return 0

    model_restoration.cuda()
    test_loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        drop_last=False)
    backup = prepare_result_directory(args.result_dir, args.existing_results)
    output = Path(args.result_dir)
    metadata = dict(
        status='running', options=vars(args), checkpoint=checkpoint_metadata,
        input_images=len(dataset), diagnostic_subset=0 < args.max_images < len(dataset),
        processed_images=0, previous_results_backup=backup,
        precision='torch.cuda.amp.autocast fp16; float32 tile accumulation',
        metrics=dict(color='RGB', range='float [0,1]', crop_border=0,
                     psnr='skimage.metrics.peak_signal_noise_ratio (legacy inferred data_range)',
                     ssim='skimage.metrics.structural_similarity; channel_axis=-1; data_range=1.0',
                     gaussian_weights=False, win_size=7, use_sample_covariance=True,
                     aggregation='arithmetic mean of per-image values; before PNG quantization'),
        torch_version=torch.__version__, numpy_version=np.__version__,
        test_source_sha256=sha256_file(__file__))
    write_json(output / 'inference_config.json', metadata)
    started = time.monotonic()
    with (output / 'test.log').open('x', encoding='utf-8', buffering=1) as logfile:
        with redirect_stdout(TeeStream(sys.stdout, logfile)), redirect_stderr(TeeStream(sys.stderr, logfile)):
            try:
                print('===>Testing using weights:', args.weights)
                print('===>Checkpoint metadata:', checkpoint_metadata)
                print('===>Paired dataset:', args.input_dir, '({} images)'.format(len(dataset)))
                print('===>Results:', output)
                print('===>Previous results backup:', backup)
                print('===>Full-image inference:', inference_description(args))
                rows, psnr, ssim = evaluate_dataset(test_loader)
                metadata.update(status='complete', processed_images=len(rows),
                                mean_psnr=psnr, mean_ssim=ssim,
                                elapsed_seconds=time.monotonic() - started)
                write_json(output / 'inference_config.json', metadata)
            except BaseException as error:
                processed = 0
                csv_path = output / 'psnr_ssim_per_image.csv'
                if csv_path.is_file():
                    with csv_path.open(newline='', encoding='utf-8') as stream:
                        processed = sum(1 for _ in csv.DictReader(stream))
                metadata.update(status='failed', error=repr(error), processed_images=processed,
                                elapsed_seconds=time.monotonic() - started)
                write_json(output / 'inference_config.json', metadata)
                traceback.print_exc()
                raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

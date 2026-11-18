#!/usr/bin/env python3
"""Launch single-image online motion-deblurring training.

Training Blur/GT candidate pools are regenerated once per refresh block. Each
epoch selects one cached candidate per GT. Validation is generated once with a
fixed seed so PSNR remains comparable across epochs.
"""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MOTION_ROOT = PROJECT_ROOT.parent / "motion-flow-syn-python-flat"
MOTION_PYTHON = MOTION_ROOT / ".venv/bin/python"
GENERATOR = MOTION_ROOT / "generate_online_2d_epoch.py"
TRAIN_SCRIPT = PROJECT_ROOT / "train/train_motiondeblur.py"
DATASET_ROOT = PROJECT_ROOT / "dataset/UAV_single_image_pose_split"
FIXED_DATASET_ROOT = (
    DATASET_ROOT / "fixed_protocol_first_half_60_78_visdrone_all"
)
TRAIN_GT_ROOTS = {
    "full": FIXED_DATASET_ROOT / "train_full_7520",
    "fifth": FIXED_DATASET_ROOT / "train_fifth_1504",
}
VALIDATION_GT = FIXED_DATASET_ROOT / "validation_589"
VALIDATION_CACHE = FIXED_DATASET_ROOT / "validation_fixed_online_2d"
VALIDATION_DIR = VALIDATION_CACHE / "epoch_000000"
TRAIN_CACHE_ROOTS = {
    "full": PROJECT_ROOT / "online_2d_cache_fixed_full7520",
    "fifth": PROJECT_ROOT / "online_2d_cache_fixed_fifth1504",
}
DEFAULT_MAX_ITERS = 180_000
DEFAULT_WARMUP_ITERS = 2_820  # Fixed iteration count, independent of batch size.

EXPERIMENT_PREFIX = "_single_image_online_2d_patch256"

# Explicit architecture presets for incremental ablation experiments.
# z0/z1 are injected in the second process, so their preset requires it.
ABLATION_PRESETS = {
    'full': ('use_local_branch', 'use_channel_aware', 'double_process', 'use_ddem', 'use_kernel_response'),
    'uformer': (),
    'local': ('use_local_branch',),
    'local_cam': ('use_local_branch', 'use_channel_aware'),
    'local_cam_z0z1': ('use_local_branch', 'use_channel_aware', 'double_process', 'use_ddem'),
    'local_cam_double': ('use_local_branch', 'use_channel_aware', 'double_process'),
}


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read an integer environment variable and validate its lower bound."""
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def ensure_fixed_validation(
    workers: int,
    gpu_ids: str,
    cuda_batch: int,
    backend: str,
    generator_python: Path,
) -> None:
    manifest = VALIDATION_DIR / "pairs.tsv"
    if manifest.is_file() and manifest.stat().st_size > 0:
        with manifest.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
        expected = sum(
            path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
            for path in VALIDATION_GT.iterdir()
        )
        if (
            len(rows) == expected
            and all(Path(row["input_path"]).is_file() for row in rows)
            and all(Path(row["target_path"]).is_file() for row in rows)
        ):
            print(f"Using fixed validation set: {VALIDATION_DIR}", flush=True)
            return
        raise RuntimeError(
            f"Fixed validation cache is incomplete: {VALIDATION_DIR}. "
            "Inspect it, then remove that epoch directory before retrying."
        )
    command = [
        str(generator_python),
        str(GENERATOR),
        "--epoch", "0",
        "--base-seed", "20260810",
        "--gt-root", str(VALIDATION_GT),
        "--cache-root", str(VALIDATION_CACHE),
        "--patch-size", "256",
        "--cases", "all",
        "--strength-min", "1.0",
        "--strength-max", "1.80",
        "--workers", str(workers),
        "--backend", backend,
        "--gpu-indices", gpu_ids,
        "--cuda-subframe-batch", str(cuda_batch),
        "--variants-per-gt", "1",
    ]
    print("Generating the fixed validation set once...", flush=True)
    subprocess.run(command, cwd=str(MOTION_ROOT), check=True)
    if not manifest.is_file():
        raise RuntimeError(f"Validation generator did not create {manifest}")


def main(*, ema: bool = False, ablation: str | None = None) -> int:
    """Validate configuration, prepare validation data, and launch training."""
    # New runs use joint residual response control. Low-level model defaults
    # retain the serial controller for loading previous checkpoints explicitly.
    if ablation is not None and ablation not in ABLATION_PRESETS:
        raise ValueError('Unknown ablation preset: ' + str(ablation))
    model_flags = (
        ABLATION_PRESETS[ablation] if ablation is not None
        else ('use_local_branch', 'use_channel_aware', 'double_process', 'use_ddem')
    )
    kernel_response = (
        'use_kernel_response' in model_flags if ablation is not None
        else os.environ.get('KERNEL_RESPONSE', '1') == '1'
    )
    response_variant = os.environ.get(
        'RESPONSE_VARIANT', 'joint_residual' if kernel_response else 'serial')
    if ablation is not None and not kernel_response:
        response_variant = 'serial'
    if response_variant not in ('serial', 'joint_residual'):
        raise ValueError('RESPONSE_VARIANT must be serial or joint_residual')
    if not kernel_response and response_variant != 'serial':
        raise ValueError('RESPONSE_VARIANT=joint_residual requires KERNEL_RESPONSE=1')
    dry_run = os.environ.get('DRY_RUN', '0') == '1'
    gpu_ids = os.environ.get("GPUS", "0")
    batch_size = env_int("BATCH_SIZE", 4)
    epochs = env_int("EPOCHS", 500 if kernel_response else 450)
    max_iters = env_int("MAX_ITERS", DEFAULT_MAX_ITERS, minimum=0)
    warmup_iters = env_int("WARMUP_ITERS", DEFAULT_WARMUP_ITERS, minimum=0)
    if max_iters > 0 and warmup_iters >= max_iters:
        raise ValueError("WARMUP_ITERS must be smaller than MAX_ITERS")
    generator_workers = env_int("GEN_WORKERS", 24 if kernel_response else 16)
    cuda_batch = env_int("CUDA_SUBFRAME_BATCH", 32)
    blur_backend = os.environ.get("BLUR_BACKEND", "cpu").strip().lower()
    if blur_backend not in {"cpu", "cuda"}:
        raise ValueError("BLUR_BACKEND must be 'cpu' or 'cuda'")
    dataset_scope = os.environ.get("DATASET_SCOPE", "full").strip().lower()
    if dataset_scope not in {"full", "fifth"}:
        raise ValueError("DATASET_SCOPE must be 'full' or 'fifth'")
    pool_variants = env_int("POOL_VARIANTS", 12)
    pool_refresh_epochs = env_int("POOL_REFRESH_EPOCHS", 50)
    train_gt = TRAIN_GT_ROOTS[dataset_scope]
    train_cache = TRAIN_CACHE_ROOTS[dataset_scope]
    budget_suffix = f"_i{max_iters}" if max_iters > 0 else f"_e{epochs}"
    experiment = (
        f"{EXPERIMENT_PREFIX}_{dataset_scope}_k{pool_variants}"
        f"_r{pool_refresh_epochs}"
        f"{budget_suffix}"
    )
    if ablation is not None:
        ablation_suffix = f'_ablation_{ablation}_b{batch_size}'
        experiment += ablation_suffix
        train_cache = Path(str(train_cache) + ablation_suffix)
        if not kernel_response:
            run_tag = os.environ.get('RUN_TAG', '')
            if run_tag and not all(c.isalnum() or c in '_-' for c in run_tag):
                raise ValueError('RUN_TAG must contain letters, digits, underscores or hyphens')
            if run_tag:
                experiment += '_' + run_tag
                train_cache = Path(str(train_cache) + '_' + run_tag)
    if kernel_response:
        from kernel_response import label_spec
        response_groups = env_int('RESPONSE_GROUPS', 4)
        if response_groups not in (1, 4):
            raise ValueError('RESPONSE_GROUPS must be 1 (previous model) or 4')
        gate_log_interval = int(os.environ.get('RESPONSE_GATE_LOG_INTERVAL', '1' if response_groups > 1 else '0'))
        if gate_log_interval < 0:
            raise ValueError('RESPONSE_GATE_LOG_INTERVAL must be nonnegative')
        response_mode = os.environ.get('RESPONSE_LOSS_MODE', 'complex')
        response_weight = float(os.environ.get('RESPONSE_LOSS_WEIGHT', '0.01'))
        if response_mode not in ('complex', 'magnitude') or not 0 <= response_weight < float('inf'):
            raise ValueError('Invalid response loss configuration')
        schema_hash = label_spec().config_hash(blur_backend)[:10]
        version = 'v1' if response_groups == 1 else 'g4_v2'
        suffix = '_kr16_c64_all8_serial_{}_{}_w{}_{}'.format(version,
            response_mode, format(response_weight, 'g').replace('.', 'p'), schema_hash)
        if response_variant == 'joint_residual':
            suffix = '_kr16_c64_all8_joint_residual_v1_{}_{}_w{}_{}'.format(
                version, response_mode,
                format(response_weight, 'g').replace('.', 'p'), schema_hash)
        run_tag = os.environ.get('RUN_TAG', '')
        if run_tag and not all(c.isalnum() or c in '_-' for c in run_tag):
            raise ValueError('RUN_TAG must contain letters, digits, underscores or hyphens')
        suffix += '_' + run_tag if run_tag else ''
        experiment += suffix
        train_cache = Path(str(train_cache) + suffix)
    if ema:
        experiment += '_ema_e4_d0p999'
    models = PROJECT_ROOT / 'logs/motiondeblur/Online2D' / ('Uformer_B' + experiment) / 'models'
    if not dry_run and os.environ.get('RESUME', '0') != '1' and models.exists() and any(models.glob('*.pth')):
        raise FileExistsError('Existing experiment: use RESUME=1 or a new RUN_TAG')
    generator_python = (
        Path(sys.executable) if blur_backend == "cuda" else MOTION_PYTHON
    )
    for required in (
        MOTION_PYTHON,
        GENERATOR,
        TRAIN_SCRIPT,
        train_gt,
        VALIDATION_GT,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    source_count = sum(p.is_file() and p.suffix.lower() in image_extensions
                       for p in train_gt.iterdir())
    expected_count = {'full': 7520, 'fifth': 1504}[dataset_scope]
    if source_count != expected_count:
        raise ValueError('Expected {} GTs for {}, found {} in {}'.format(
            expected_count, dataset_scope, source_count, train_gt))
    if max_iters > 0:
        # The training loader drops incomplete batches and samples one pair per GT.
        steps_per_epoch = source_count // batch_size
        if steps_per_epoch < 1:
            raise ValueError('BATCH_SIZE exceeds training dataset size')
        epochs = (max_iters + steps_per_epoch - 1) // steps_per_epoch
    print('Training configuration: GTs={}, epochs={}, batch={}, CPU generation workers={}, '
          'backend={}, candidates/GT={}, refresh={} epochs, response_groups={}'.format(
              source_count, epochs, batch_size, generator_workers, blur_backend,
              pool_variants, pool_refresh_epochs, response_groups if kernel_response else 'off'), flush=True)
    budget_description = (
        f"{max_iters:,} iterations (up to {epochs} epochs)"
        if max_iters > 0 else f"{epochs} epochs"
    )
    print(f"Training budget: {budget_description}", flush=True)
    print(f"Response controller: {response_variant if kernel_response else 'off'}", flush=True)
    if ablation is not None:
        print(f"Ablation: {ablation}; enabled model flags: {', '.join(model_flags) or 'none'}", flush=True)
    print('GT source identity: {}'.format(train_gt), flush=True)
    print('Experiment: Uformer_B{}'.format(experiment), flush=True)
    if not dry_run:
        if kernel_response:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    'CUDA unavailable in the training environment; stopped before generating any cache. '
                    'Check nvidia-smi and the joint-cu118-test environment.')
        ensure_fixed_validation(generator_workers, gpu_ids, cuda_batch,
                                blur_backend, generator_python)

    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--arch", "Uformer_B", "--gpu", gpu_ids,
        "--dataset", "Online2D", "--mode", "deblur",
        "--ddem_stage", "2",
        "--ddem_kd_weight", "0", "--kd_grad_stat_interval", "0",
        "--ddem_dim", "32", "--ddem_blocks", "2",
        "--ddem_backbone", "local_global", "--ddem_heads", "1",
        "--ddem_win_size", "8", "--adapter_warmup_epochs", "0",
        "--backbone_lr_scale", "1.0", "--lr_initial", "0.0003",
        "--batch_size", str(batch_size), "--train_drop_last",
        "--train_ps", "256", "--train_workers", "4",
        "--val_ps", "256", "--val_dir", str(VALIDATION_DIR),
        "--eval_workers", "4", "--val_epoch_interval", "25",
        "--grad_clip", "1.0", "--checkpoint", "50",
        "--max_iters", str(max_iters), "--warmup_iters", str(warmup_iters),
        "--nepoch", str(epochs), "--warmup", "--warmup_epochs", "3",
        "--save_dir", str(PROJECT_ROOT / "logs"), "--env", experiment,
        "--online_2d", "--online_2d_project", str(MOTION_ROOT),
        "--online_2d_python", str(generator_python),
        "--online_2d_gt_root", str(train_gt),
        "--online_2d_cache_root", str(train_cache),
        "--online_2d_workers", str(generator_workers),
        "--online_2d_backend", blur_backend,
        "--online_2d_gpus", gpu_ids,
        "--online_2d_cuda_subframe_batch", str(cuda_batch),
        "--online_2d_seed", "1234", "--online_2d_cases", "all",
        "--online_2d_strength_min", "1.0",
        "--online_2d_strength_max", "1.80",
        "--online_2d_variants_per_gt", str(pool_variants),
        "--online_2d_pool_refresh_epochs", str(pool_refresh_epochs),
        "--online_2d_dataset_scope", dataset_scope,
        "--online_2d_preselected_gt_root",
    ]
    command.extend('--' + flag for flag in model_flags if flag != 'use_kernel_response')
    if 'use_ddem' in model_flags:
        command.append('--ddem_scratch_student')
    if ema:
        command.append("--ema")
    if os.environ.get("KEEP_CACHE", "0") == "1":
        command.append("--online_2d_keep_cache")
    if kernel_response:
        command.extend(['--use_kernel_response', '--kernel_response_groups', str(response_groups),
                        '--kernel_response_variant', response_variant,
                        '--response_gate_log_interval', str(gate_log_interval),
                        '--response_loss_mode', response_mode,
                        '--response_loss_weight', str(response_weight),
                        '--optimizer', 'adamw', '--weight_decay', '0.02'])
    if os.environ.get("RESUME", "0") == "1":
        latest = (
            PROJECT_ROOT / "logs/motiondeblur/Online2D"
            / f"Uformer_B{experiment}" / "models/model_latest.pth"
        )
        if not latest.is_file():
            raise FileNotFoundError(latest)
        command.extend(["--resume", "--pretrain_weights", str(latest)])

    print("Launching online single-image 2D training:", flush=True)
    print(shlex.join(command), flush=True)
    if dry_run:
        return 0
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

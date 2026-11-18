"""Reusable pool orchestration for online single-image 2D blur generation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _path(value):
    return Path(value).expanduser().resolve()


def _executable_path(value):
    """Return an absolute executable path without resolving venv symlinks.

    Python detects a virtual environment from the invoked ``.venv/bin/python``
    path. Resolving that symlink first can invoke the base interpreter instead
    and lose packages installed only in the virtual environment.
    """
    return Path(os.path.abspath(Path(value).expanduser()))


def resolve_paths(options):
    project = _path(options.online_2d_project)
    python = (
        _executable_path(options.online_2d_python)
        if options.online_2d_python
        else (
            _executable_path(sys.executable)
            if options.online_2d_backend == "cuda"
            else project / ".venv/bin/python"
        )
    )
    return {
        "project": project,
        "python": python,
        "generator": project / "generate_online_2d_epoch.py",
        "gt": _path(options.online_2d_gt_root),
        "cache": _path(options.online_2d_cache_root),
    }


def validate_options(options):
    paths = resolve_paths(options)
    for key in ("python", "generator"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    if not paths["gt"].is_dir():
        raise FileNotFoundError(paths["gt"])
    if options.online_2d_workers < 1:
        raise ValueError("--online_2d_workers must be positive")
    if options.online_2d_cuda_subframe_batch < 1:
        raise ValueError("--online_2d_cuda_subframe_batch must be positive")
    if options.online_2d_variants_per_gt < 1:
        raise ValueError("--online_2d_variants_per_gt must be positive")
    if options.online_2d_pool_refresh_epochs < 1:
        raise ValueError("--online_2d_pool_refresh_epochs must be positive")
    if not 0.0 < options.online_2d_strength_min <= options.online_2d_strength_max:
        raise ValueError(
            "Require 0 < --online_2d_strength_min <= --online_2d_strength_max"
        )
    if options.online_2d_max_images < 0:
        raise ValueError("--online_2d_max_images must be non-negative")
    return paths


def epoch_root(options, epoch):
    return resolve_paths(options)["cache"] / f"epoch_{int(epoch):06d}"


def epoch_manifest(options, epoch):
    return epoch_root(options, epoch) / "pairs.tsv"


def cache_epoch_for_training_epoch(options, training_epoch):
    """Return the first epoch of the reusable pool containing training_epoch."""
    refresh_epochs = int(options.online_2d_pool_refresh_epochs)
    training_epoch = int(training_epoch)
    return ((training_epoch - 1) // refresh_epochs) * refresh_epochs + 1


def cache_consumed_after_epoch(options, training_epoch, final_epoch):
    refresh_epochs = int(options.online_2d_pool_refresh_epochs)
    return (
        int(training_epoch) == int(final_epoch)
        or (int(training_epoch) - 1) % refresh_epochs == refresh_epochs - 1
    )


def generation_command(options, epoch):
    paths = validate_options(options)
    command = [
        str(paths["python"]),
        str(paths["generator"]),
        "--epoch",
        str(int(epoch)),
        "--base-seed",
        str(options.online_2d_seed),
        "--gt-root",
        str(paths["gt"]),
        "--cache-root",
        str(paths["cache"]),
        "--patch-size",
        str(options.train_ps),
        "--cases",
        options.online_2d_cases,
        "--strength-min",
        str(options.online_2d_strength_min),
        "--strength-max",
        str(options.online_2d_strength_max),
        "--workers",
        str(options.online_2d_workers),
        "--backend",
        options.online_2d_backend,
        "--gpu-indices",
        options.online_2d_gpus or options.gpu,
        "--cuda-subframe-batch",
        str(options.online_2d_cuda_subframe_batch),
        "--variants-per-gt",
        str(options.online_2d_variants_per_gt),
        "--dataset-scope",
        (
            "full"
            if getattr(options, "online_2d_preselected_gt_root", False)
            else options.online_2d_dataset_scope
        ),
        "--fixed-subframes",
        str(options.online_2d_fixed_subframes),
        "--interpolation-margin",
        str(options.online_2d_interpolation_margin),
        "--crop-trials",
        str(options.online_2d_crop_trials),
        "--max-images",
        str(options.online_2d_max_images),
    ]
    if not options.online_2d_adaptive_subframes:
        command.append("--no-adaptive-subframes")
    if getattr(options, 'use_kernel_response', False):
        command.append('--kernel-response')
    return command


def generate_epoch(options, epoch):
    training_epoch = int(epoch)
    cache_epoch = cache_epoch_for_training_epoch(options, training_epoch)
    manifest = epoch_manifest(options, cache_epoch)
    if manifest.is_file() and manifest.stat().st_size > 0:
        if getattr(options, 'use_kernel_response', False):
            # Validate once per pool/process, including a resumed process.
            # Generator checks the full request and every label, not just TSV.
            key = (str(manifest), manifest.stat().st_mtime_ns,
                   tuple(generation_command(options, cache_epoch)))
            checked = globals().setdefault('_checked_response_pools', set())
            if key not in checked:
                paths = validate_options(options)
                subprocess.run(generation_command(options, cache_epoch),
                               cwd=str(paths['project']), check=True)
                checked.add(key)
        print(
            "[Online 2D] Training epoch {} reuses cache epoch {}: {}".format(
                training_epoch, cache_epoch, manifest),
            flush=True,
        )
        return manifest
    paths = validate_options(options)
    start = time.time()
    print(
        "\n[Online 2D] Generating cache epoch {} for training epochs "
        "{}-{}: {} variants per GT...".format(
            cache_epoch,
            cache_epoch,
            cache_epoch + int(options.online_2d_pool_refresh_epochs) - 1,
            int(options.online_2d_variants_per_gt),
        ),
        flush=True,
    )
    subprocess.run(
        generation_command(options, cache_epoch),
        cwd=str(paths["project"]),
        check=True,
    )
    if not manifest.is_file() or manifest.stat().st_size == 0:
        raise RuntimeError(f"Online 2D generator did not create {manifest}")
    if getattr(options, 'use_kernel_response', False):
        globals().setdefault('_checked_response_pools', set()).add(
            (str(manifest), manifest.stat().st_mtime_ns,
             tuple(generation_command(options, cache_epoch))))
    print(
        "[Online 2D] Cache epoch {} ready in {:.1f}s: {}".format(
            cache_epoch, time.time() - start, manifest
        ),
        flush=True,
    )
    return manifest


def remove_epoch_cache(options, epoch):
    if options.online_2d_keep_cache:
        return
    cache_root = resolve_paths(options)["cache"].resolve()
    candidate = epoch_root(options, epoch)
    if not candidate.exists():
        return
    if (
        candidate.is_symlink()
        or not candidate.is_dir()
        or candidate.parent.resolve() != cache_root
        or not candidate.name.startswith("epoch_")
    ):
        raise RuntimeError(f"Refusing to remove invalid cache path: {candidate}")
    shutil.rmtree(candidate)
    print(f"[Online 2D] Removed consumed epoch cache: {candidate}", flush=True)


def remove_consumed_training_cache(options, training_epoch):
    cache_epoch = cache_epoch_for_training_epoch(options, training_epoch)
    remove_epoch_cache(options, cache_epoch)

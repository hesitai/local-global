#!/usr/bin/env python3
"""Generate one epoch of aligned single-image motion-blur training patches.

Each source GT receives a deterministic epoch-specific random motion case,
direction, strength and crop.  The output patch is rendered directly from the
full-resolution source image.  A crop is accepted only when every subframe
maps completely inside the source image, so no constant-zero warp boundary can
enter the saved blurry patch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

import script_uav_global_blur_multisubframe_no_depth as motion


@dataclass(frozen=True)
class PatchTask:
    source_index: int
    variant_index: int
    source_path: str
    staging_input_path: str
    staging_target_path: str
    output_name: str
    epoch: int
    base_seed: int
    patch_size: int
    cases: tuple[str, ...]
    strength_min: float
    strength_max: float
    adaptive_subframes: bool
    fixed_subframes: int
    interpolation_margin: int
    crop_trials: int
    kernel_response: bool = False
    response_backend: str = 'cpu'


@dataclass
class PreparedPatch:
    image_linear: np.ndarray
    target_patch: np.ndarray
    homographies: np.ndarray
    weights: np.ndarray
    center_index: int
    top: int
    bottom: int
    left: int
    right: int
    result: dict[str, Any]


MANIFEST_FIELDS = (
    "input_path",
    "target_path",
    "source_path",
    "sample_id",
    "epoch",
    "variant_index",
    "case_name",
    "strength",
    "strength_band",
    "strength_parameters",
    "direction_deg",
    "crop_top",
    "crop_bottom",
    "crop_left",
    "crop_right",
    "patch_size",
    "n_subframes",
    "seed",
)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument(
        "--cache-root", type=Path, default=root / "online_2d_cache"
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--strength-min", type=float, default=1.0)
    parser.add_argument("--strength-max", type=float, default=1.80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--gpu-indices",
        default="0",
        help="Comma-separated physical GPU IDs used by the CUDA backend.",
    )
    parser.add_argument("--cuda-subframe-batch", type=int, default=32)
    parser.add_argument("--variants-per-gt", type=int, default=1)
    parser.add_argument(
        "--dataset-scope",
        choices=("quarter", "full"),
        default="full",
        help="quarter deterministically keeps sorted source indices 0,4,8,...",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help=(
            "optional newline-delimited image manifest; when provided it "
            "defines the exact ordered source set instead of directory slicing"
        ),
    )
    parser.add_argument("--fixed-subframes", type=int, default=31)
    parser.add_argument(
        "--no-adaptive-subframes",
        action="store_true",
        help="Use --fixed-subframes instead of motion-aware temporal sampling.",
    )
    parser.add_argument("--interpolation-margin", type=int, default=2)
    parser.add_argument("--crop-trials", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--kernel-response", action="store_true",
                        help="Save versioned stride-16 kernel response supervision; default off")
    return parser


def parse_cases(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return tuple(motion.blur_cases)
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    if not requested:
        raise ValueError("--cases must be 'all' or a comma-separated case list")
    canonical = tuple(
        motion.LEGACY_CASE_ALIASES.get(item, item) for item in requested
    )
    unknown = sorted(set(canonical).difference(motion.blur_cases))
    if unknown:
        raise ValueError(f"Unknown motion cases: {unknown}")
    return canonical


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.epoch < 0:
        raise ValueError("--epoch must be non-negative")
    if arguments.patch_size < 8:
        raise ValueError("--patch-size must be at least 8")
    if arguments.workers < 1:
        raise ValueError("--workers must be positive")
    if arguments.cuda_subframe_batch < 1:
        raise ValueError("--cuda-subframe-batch must be positive")
    if arguments.variants_per_gt < 1:
        raise ValueError("--variants-per-gt must be positive")
    if arguments.fixed_subframes < 3 or arguments.fixed_subframes % 2 == 0:
        raise ValueError("--fixed-subframes must be odd and at least 3")
    if not 0.0 < arguments.strength_min <= arguments.strength_max:
        raise ValueError("Require 0 < --strength-min <= --strength-max")
    if arguments.interpolation_margin < 1:
        raise ValueError("--interpolation-margin must be at least 1")
    if arguments.crop_trials < 1:
        raise ValueError("--crop-trials must be positive")
    if arguments.max_images < 0:
        raise ValueError("--max-images must be non-negative")
    if not arguments.gt_root.is_dir():
        raise FileNotFoundError(arguments.gt_root)


def parse_gpu_indices(value: str) -> tuple[str, ...]:
    indices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not indices or any(not item.isdigit() for item in indices):
        raise ValueError("--gpu-indices must be comma-separated integers")
    return indices


def list_gt_images(root: Path) -> list[Path]:
    images = sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in motion.SUPPORTED_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No GT images found in {root}")
    return images


def list_manifest_images(manifest: Path) -> list[Path]:
    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    images = []
    seen = set()
    for line_number, raw_line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing source image at {manifest}:{line_number}: {path}"
            )
        if path.suffix.lower() not in motion.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported source image at {manifest}:{line_number}: {path}"
            )
        if path in seen:
            raise ValueError(f"Duplicate source image in {manifest}: {path}")
        seen.add(path)
        images.append(path)
    if not images:
        raise RuntimeError(f"No source images listed in {manifest}")
    return images


def rotate_xy(values: Iterable[float], angle_rad: float) -> tuple[float, float]:
    x_value, y_value = (float(value) for value in values)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return (
        cosine * x_value - sine * y_value,
        sine * x_value + cosine * y_value,
    )


def randomize_trajectory_geometry(
    trajectory: dict[str, Any],
    rng: np.random.Generator,
    strength_min: float,
    strength_max: float,
    direction_deg: float,
    prefix: str = "motion",
) -> dict[str, float]:
    """Independently scale every motion amplitude instead of one whole path.

    A single scalar preserves the original x/y ratio and curve shape.  Here,
    each trajectory coordinate, drift, vibration amplitude, rotation amplitude
    and attitude span receives its own factor.  The returned mapping is stored
    in the epoch manifest so every random decision remains inspectable.
    """
    factors: dict[str, float] = {}

    def draw(name: str) -> float:
        value = float(rng.uniform(strength_min, strength_max))
        factors[f"{prefix}.{name}"] = value
        return value

    def scaled(name: str, value: float) -> float:
        value = float(value)
        # A multiplier for a fixed zero coordinate has no physical effect and
        # must not dilute the reported mean strength.
        return value if abs(value) <= 1e-12 else draw(name) * value

    angle_rad = math.radians(direction_deg)
    projection = str(trajectory["projection_model"])
    if projection == "composite_homography":
        for index, component in enumerate(trajectory["components"]):
            component_name = str(component.get("case_name", f"component{index}"))
            factors.update(
                randomize_trajectory_geometry(
                    component,
                    rng,
                    strength_min,
                    strength_max,
                    direction_deg,
                    prefix=f"{prefix}.{index:02d}_{component_name}",
                )
            )
        return factors
    if projection == "pure_rotation_homography":
        attitude = trajectory["attitude"]
        attitude["target_span_px"] = (
            float(attitude["target_span_px"]) * draw("attitude_span")
        )
        if attitude["type"] == "smooth_axis_tilt":
            attitude["axis_angle_deg"] = (
                float(attitude["axis_angle_deg"]) + direction_deg
            ) % 180.0
        return factors

    path = trajectory["path"]
    path_type = str(path["type"])
    if path_type in {"line", "quadratic_bezier"}:
        randomized_points = []
        for point_index, point in enumerate(path["points"]):
            randomized_points.append(
                rotate_xy(
                    (
                        scaled(f"path_point_{point_index}_x", point[0]),
                        scaled(f"path_point_{point_index}_y", point[1]),
                    ),
                    angle_rad,
                )
            )
        path["points"] = tuple(randomized_points)
    elif path_type == "harmonic":
        for key in ("origin_xy", "drift_xy"):
            path[key] = rotate_xy(
                (
                    scaled(f"{key}_x", path[key][0]),
                    scaled(f"{key}_y", path[key][1]),
                ),
                angle_rad,
            )
        for harmonic_index, harmonic in enumerate(path["harmonics"]):
            amplitude = harmonic["amplitude_xy"]
            harmonic["amplitude_xy"] = rotate_xy(
                (
                    scaled(
                        f"harmonic_{harmonic_index}_amplitude_x", amplitude[0]
                    ),
                    scaled(
                        f"harmonic_{harmonic_index}_amplitude_y", amplitude[1]
                    ),
                ),
                angle_rad,
            )
    else:  # pragma: no cover - guarded by the upstream trajectory validator.
        raise ValueError(f"Unsupported path type: {path_type}")

    rotation = trajectory.get("rotation", {})
    for key in ("drift_deg", "amplitude_deg"):
        if key in rotation:
            rotation[key] = scaled(f"rotation_{key}", rotation[key])
    return factors


def build_homographies(
    trajectory: dict[str, Any],
    height: int,
    width: int,
    adaptive: bool,
    fixed_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    motion.prepare_trajectory_for_image(trajectory, height, width)
    if adaptive:
        tau_list, weights, metadata = motion.build_adaptive_subframe_schedule(
            trajectory, height, width
        )
    else:
        tau_list, weights, metadata = motion.build_fixed_subframe_schedule(
            fixed_count
        )
    homographies = np.stack(
        [
            motion.evaluate_forward_homography_at_time(
                trajectory, float(tau), height, width
            )
            for tau in tau_list
        ]
    )
    center_index = int(metadata["gt_reference_subframe_index_zero_based"])
    if not np.allclose(homographies[center_index], np.eye(3), atol=1e-8):
        raise RuntimeError("Center-reference homography is not identity")
    return homographies, np.asarray(weights, dtype=np.float64), center_index


def patch_grid(
    top: int, left: int, patch_size: int
) -> np.ndarray:
    x_values = np.linspace(left, left + patch_size - 1, 3)
    y_values = np.linspace(top, top + patch_size - 1, 3)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return np.stack(
        (x_grid.ravel(), y_grid.ravel(), np.ones(x_grid.size)), axis=0
    )


def patch_valid_for_all_subframes(
    top: int,
    left: int,
    patch_size: int,
    homography_inverses: np.ndarray,
    height: int,
    width: int,
    margin: int,
) -> bool:
    points = patch_grid(top, left, patch_size)
    for inverse in homography_inverses:
        mapped = inverse @ points
        denominator = mapped[2]
        if np.any(np.abs(denominator) <= 1e-10):
            return False
        x_input = mapped[0] / denominator
        y_input = mapped[1] / denominator
        if (
            np.any(x_input < margin)
            or np.any(x_input > width - 1 - margin)
            or np.any(y_input < margin)
            or np.any(y_input > height - 1 - margin)
        ):
            return False
    return True


def choose_valid_crop(
    rng: np.random.Generator,
    homographies: np.ndarray,
    height: int,
    width: int,
    patch_size: int,
    margin: int,
    trials: int,
) -> tuple[int, int]:
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Source {width}x{height} is smaller than patch {patch_size}"
        )
    inverses = np.linalg.inv(homographies)
    for _ in range(trials):
        top = int(rng.integers(0, height - patch_size + 1))
        left = int(rng.integers(0, width - patch_size + 1))
        if patch_valid_for_all_subframes(
            top, left, patch_size, inverses, height, width, margin
        ):
            return top, left

    # A deterministic center fallback makes failures reproducible and gives a
    # useful diagnostic when a user configures an excessively strong motion.
    top = (height - patch_size) // 2
    left = (width - patch_size) // 2
    if patch_valid_for_all_subframes(
        top, left, patch_size, inverses, height, width, margin
    ):
        return top, left
    raise RuntimeError(
        "No all-subframe-valid crop found; reduce motion strength or patch size"
    )


def strength_band(strength: float) -> str:
    if strength < 1.25:
        return "medium"
    if strength < 1.55:
        return "strong"
    return "very_strong"


def prepare_task(task: PatchTask) -> PreparedPatch:
    task_seed = (
        int(task.base_seed)
        + int(task.epoch) * 1_000_003
        + int(task.source_index) * 10_007
        + int(task.variant_index) * 97
    )
    rng = np.random.default_rng(task_seed)
    case_name = str(rng.choice(task.cases))
    direction_deg = float(rng.uniform(0.0, 360.0))
    case_seed = task_seed + motion.CASE_SEED_OFFSETS[case_name]
    trajectory = motion.make_uav_trajectory(
        case_name, case_seed, component_seed_base=task_seed
    )
    strength_parameters = randomize_trajectory_geometry(
        trajectory,
        rng,
        task.strength_min,
        task.strength_max,
        direction_deg,
    )
    strength = float(np.mean(tuple(strength_parameters.values())))

    image_srgb = motion.read_image_double(task.source_path)
    height, width = image_srgb.shape[:2]
    homographies, weights, center_index = build_homographies(
        trajectory,
        height,
        width,
        task.adaptive_subframes,
        task.fixed_subframes,
    )
    top, left = choose_valid_crop(
        rng,
        homographies,
        height,
        width,
        task.patch_size,
        task.interpolation_margin,
        task.crop_trials,
    )
    bottom = top + task.patch_size
    right = left + task.patch_size
    target_patch = image_srgb[top:bottom, left:right].copy()
    image_linear = motion.srgb_to_linear_uav(image_srgb).astype(np.float32)
    result = {
        "source_path": str(Path(task.source_path).resolve()),
        "sample_id": Path(task.output_name).stem,
        "epoch": task.epoch,
        "variant_index": task.variant_index,
        "case_name": case_name,
        "strength": f"{strength:.8f}",
        "strength_band": strength_band(strength),
        "strength_parameters": json.dumps(
            strength_parameters, sort_keys=True, separators=(",", ":")
        ),
        "direction_deg": f"{direction_deg:.8f}",
        "crop_top": top,
        "crop_bottom": bottom,
        "crop_left": left,
        "crop_right": right,
        "patch_size": task.patch_size,
        "n_subframes": len(weights),
        "seed": task_seed,
        "output_name": task.output_name,
    }
    return PreparedPatch(
        image_linear=image_linear,
        target_patch=target_patch,
        homographies=homographies,
        weights=weights,
        center_index=center_index,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        result=result,
    )


def finish_task(
    task: PatchTask, prepared: PreparedPatch, accumulator: np.ndarray
) -> dict[str, Any]:
    blurry_patch = motion.linear_to_srgb_uav(accumulator)
    motion.write_image_uint8(task.staging_input_path, blurry_patch)
    motion.write_image_uint8(task.staging_target_path, prepared.target_patch)
    if task.kernel_response:
        from kernel_response_labels import response_from_homographies, save_label, config_hash
        label_path = (Path(task.staging_input_path).parent.parent / 'kernel_response'
                      / (Path(task.output_name).stem + '.npz'))
        label = response_from_homographies(
            prepared.homographies, prepared.weights, prepared.top, prepared.left,
            task.patch_size, task.response_backend)
        save_label(label_path, label, prepared.result, task.response_backend)
        prepared.result['kernel_response_path'] = str(label_path)
        prepared.result['kernel_response_hash'] = config_hash(task.response_backend)
    return prepared.result


def render_task(task: PatchTask) -> dict[str, Any]:
    """CPU/OpenCV reference backend."""
    cv2.setNumThreads(1)
    prepared = prepare_task(task)
    image_linear = prepared.image_linear
    homographies = prepared.homographies
    weights = prepared.weights
    top, bottom = prepared.top, prepared.bottom
    left, right = prepared.left, prepared.right
    accumulator = np.zeros_like(prepared.target_patch, dtype=np.float64)
    to_patch = np.asarray(
        [[1.0, 0.0, -left], [0.0, 1.0, -top], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    for index, (homography, weight) in enumerate(zip(homographies, weights)):
        if index == prepared.center_index:
            frame = image_linear[top:bottom, left:right]
        else:
            frame = cv2.warpPerspective(
                image_linear,
                to_patch @ homography,
                (task.patch_size, task.patch_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        accumulator += float(weight) * np.asarray(frame, dtype=np.float64)
    return finish_task(task, prepared, accumulator)


def render_task_cuda(
    task: PatchTask, device: Any, subframe_batch: int
) -> dict[str, Any]:
    """Render one patch with batched PyTorch homography sampling on CUDA."""
    import torch
    import torch.nn.functional as torch_functional

    prepared = prepare_task(task)
    height, width = prepared.image_linear.shape[:2]
    patch_size = task.patch_size
    source = torch.from_numpy(prepared.image_linear).permute(2, 0, 1)
    source = source.unsqueeze(0).to(device=device, dtype=torch.float32)
    accumulator = torch.zeros(
        (3, patch_size, patch_size), device=device, dtype=torch.float32
    )
    y_patch, x_patch = torch.meshgrid(
        torch.arange(patch_size, device=device, dtype=torch.float32),
        torch.arange(patch_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(x_patch)
    output_points = torch.stack(
        (x_patch + prepared.left, y_patch + prepared.top, ones), dim=0
    ).reshape(3, -1)
    inverses = np.linalg.inv(prepared.homographies).astype(np.float32)

    with torch.inference_mode():
        for start in range(0, len(inverses), subframe_batch):
            end = min(start + subframe_batch, len(inverses))
            inverse_batch = torch.from_numpy(inverses[start:end]).to(device)
            mapped = torch.matmul(inverse_batch, output_points)
            denominator = mapped[:, 2].clamp_min(1e-8)
            x_input = mapped[:, 0] / denominator
            y_input = mapped[:, 1] / denominator
            grid = torch.stack(
                (
                    2.0 * x_input / float(width - 1) - 1.0,
                    2.0 * y_input / float(height - 1) - 1.0,
                ),
                dim=-1,
            ).reshape(end - start, patch_size, patch_size, 2)
            frames = torch_functional.grid_sample(
                source.expand(end - start, -1, -1, -1),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            batch_weights = torch.as_tensor(
                prepared.weights[start:end], device=device, dtype=torch.float32
            ).reshape(-1, 1, 1, 1)
            accumulator += (frames * batch_weights).sum(dim=0)
    result = accumulator.permute(1, 2, 0).cpu().numpy()
    return finish_task(task, prepared, result)


def render_cuda_chunk(
    gpu_index: str, tasks: list[PatchTask], subframe_batch: int
) -> list[tuple[int, int, dict[str, Any]]]:
    # One spawned process owns one physical GPU, matching the old joint 3DGS
    # worker isolation without importing CUDA in the orchestration process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in worker for GPU {gpu_index}")
    cv2.setNumThreads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    rendered = []
    for task in tasks:
        rendered.append(
            (
                task.source_index,
                task.variant_index,
                render_task_cuda(task, device, subframe_batch),
            )
        )
    torch.cuda.synchronize(device)
    return rendered


def valid_existing_epoch(epoch_root: Path, expected_count: int,
                         kernel_response=False, backend='cpu', request=None) -> bool:
    manifest = epoch_root / "pairs.tsv"
    if not manifest.is_file():
        return False
    with manifest.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    valid = (
        len(rows) == expected_count
        and all(Path(row["input_path"]).is_file() for row in rows)
        and all(Path(row["target_path"]).is_file() for row in rows)
    )
    if valid and kernel_response:
        from kernel_response_labels import load_label, config_hash
        try:
            summary = json.loads((epoch_root / 'summary.json').read_text())
            if summary.get('kernel_response_hash') != config_hash(backend):
                return False
            if request is not None and summary.get('generation_request') != request:
                return False
            for row in rows:
                load_label(row['kernel_response_path'], int(row['patch_size']), backend,
                           {k: row[k] for k in ('sample_id', 'seed', 'variant_index',
                            'crop_top', 'crop_left', 'source_path')})
        except (OSError, ValueError, KeyError):
            return False
    return valid


def main() -> int:
    arguments = build_parser().parse_args()
    validate_arguments(arguments)
    cases = parse_cases(arguments.cases)
    source_pool = list_gt_images(arguments.gt_root.resolve())
    source_pool_count = len(source_pool)
    if arguments.source_manifest is not None:
        source_images = list_manifest_images(arguments.source_manifest)
    else:
        source_images = source_pool
        if arguments.dataset_scope == "quarter":
            source_images = source_images[::4]
    if arguments.max_images:
        source_images = source_images[: arguments.max_images]
    expected_count = len(source_images) * arguments.variants_per_gt
    cache_root = arguments.cache_root.resolve()
    epoch_root = cache_root / f"epoch_{arguments.epoch:06d}"
    request = {k: str(v) for k, v in vars(arguments).items()
               if k not in ('cache_root', 'workers', 'gpu_indices', 'cuda_subframe_batch')}
    request['sources'] = [str(p) for p in source_images]
    if arguments.kernel_response:
        import hashlib
        request['motion_sha256'] = hashlib.sha256(Path(motion.__file__).read_bytes()).hexdigest()
        request['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        from kernel_response_labels import __file__ as response_source
        request['response_code_sha256'] = hashlib.sha256(Path(response_source).read_bytes()).hexdigest()
    if arguments.kernel_response and arguments.patch_size % 16:
        raise ValueError('Kernel response requires patch size divisible by 16')
    if valid_existing_epoch(epoch_root, expected_count, arguments.kernel_response,
                            arguments.backend, request):
        print(f"Reusing complete online 2D epoch: {epoch_root}")
        return 0
    if epoch_root.exists():
        raise FileExistsError(
            f"Incomplete output already exists; inspect it before retrying: {epoch_root}"
        )
    cache_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".epoch_{arguments.epoch:06d}.", dir=cache_root
    ) as temporary:
        staging = Path(temporary) / epoch_root.name
        staging_input = staging / "input"
        staging_target = staging / "target"
        staging_input.mkdir(parents=True)
        staging_target.mkdir(parents=True)
        tasks = []
        for source_index, source_path in enumerate(source_images):
            for variant_index in range(arguments.variants_per_gt):
                output_name = (
                    f"{source_index:06d}_{source_path.stem}_"
                    f"v{variant_index:02d}.png"
                )
                tasks.append(
                    PatchTask(
                        source_index=source_index,
                        variant_index=variant_index,
                        source_path=str(source_path),
                        staging_input_path=str(staging_input / output_name),
                        staging_target_path=str(staging_target / output_name),
                        output_name=output_name,
                        epoch=arguments.epoch,
                        base_seed=arguments.base_seed,
                        patch_size=arguments.patch_size,
                        cases=cases,
                        strength_min=arguments.strength_min,
                        strength_max=arguments.strength_max,
                        adaptive_subframes=not arguments.no_adaptive_subframes,
                        fixed_subframes=arguments.fixed_subframes,
                        interpolation_margin=arguments.interpolation_margin,
                        crop_trials=arguments.crop_trials,
                        kernel_response=arguments.kernel_response,
                        response_backend=arguments.backend,
                    )
                )
        if arguments.backend == "cuda":
            gpu_indices = parse_gpu_indices(arguments.gpu_indices)
            task_chunks = [tasks[index::len(gpu_indices)] for index in range(len(gpu_indices))]
            spawn_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=len(gpu_indices), mp_context=spawn_context
            ) as executor:
                futures = [
                    executor.submit(
                        render_cuda_chunk,
                        gpu_index,
                        chunk,
                        arguments.cuda_subframe_batch,
                    )
                    for gpu_index, chunk in zip(gpu_indices, task_chunks)
                    if chunk
                ]
                keyed_results = [item for future in futures for item in future.result()]
            keyed_results.sort(key=lambda item: (item[0], item[1]))
            results = [item[2] for item in keyed_results]
        elif arguments.workers == 1:
            results = [render_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
                results = list(executor.map(render_task, tasks))

        with (staging / "pairs.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as output:
            writer = csv.DictWriter(
                output, fieldnames=(MANIFEST_FIELDS + (
                    ('kernel_response_path', 'kernel_response_hash')
                    if arguments.kernel_response else ())), delimiter="\t"
            )
            writer.writeheader()
            for result in results:
                output_name = result.pop("output_name")
                if arguments.kernel_response:
                    result['kernel_response_path'] = str(
                        epoch_root / 'kernel_response' / (Path(output_name).stem + '.npz'))
                writer.writerow(
                    {
                        "input_path": str(epoch_root / "input" / output_name),
                        "target_path": str(epoch_root / "target" / output_name),
                        **result,
                    }
                )
        counts: dict[str, int] = {}
        for result in results:
            case_name = str(result["case_name"])
            counts[case_name] = counts.get(case_name, 0) + 1
        with (staging / "summary.json").open("w", encoding="utf-8") as output:
            json.dump(
                {
                    "pipeline": "online_single_image_2d_patch_v1",
                    "epoch": arguments.epoch,
                    "source_count": len(source_images),
                    "source_pool_count": source_pool_count,
                    "dataset_scope": arguments.dataset_scope,
                    "source_manifest": (
                        str(arguments.source_manifest.expanduser().resolve())
                        if arguments.source_manifest is not None
                        else None
                    ),
                    "variants_per_gt": arguments.variants_per_gt,
                    "pair_count": len(results),
                    "patch_size": arguments.patch_size,
                    "backend": arguments.backend,
                    "gpu_indices": (
                        list(parse_gpu_indices(arguments.gpu_indices))
                        if arguments.backend == "cuda"
                        else []
                    ),
                    "adaptive_subframes": not arguments.no_adaptive_subframes,
                    "strength_range": [
                        arguments.strength_min,
                        arguments.strength_max,
                    ],
                    "case_counts": counts,
                    **(dict(kernel_response_hash=__import__('kernel_response_labels').config_hash(arguments.backend),
                            kernel_response_spec=__import__('kernel_response_labels').specification(arguments.backend),
                            generation_request=request) if arguments.kernel_response else {}),
                },
                output,
                indent=2,
                ensure_ascii=False,
            )
            output.write("\n")
        staging.rename(epoch_root)
    print(
        f"Generated online 2D epoch {arguments.epoch}: "
        f"{expected_count} pairs at {epoch_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

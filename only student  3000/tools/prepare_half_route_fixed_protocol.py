#!/usr/bin/env python3
"""Build the fixed first-half UAV single-image protocol.

All percentages are relative to the first half of each complete flight:
train [0, 60), validation [75, 78), and test [87.5, 100).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT = Path(
    "/home/xtl/hst/joint/GS-Blur Confirm/data/my_uav_scene/input"
)
DATASET_ROOT = Path(
    "/home/xtl/hst/joint2D/only student  3000/dataset/"
    "UAV_single_image_pose_split"
)
OUTPUT_ROOT = DATASET_ROOT / "fixed_protocol_first_half_60_78"

TRAIN_TRANSLATION_M = 4.5
TRAIN_ROTATION_DEG = 3.0
EVAL_TRANSLATION_M = 8.0
EVAL_ROTATION_DEG = 4.0
DUPLICATE_POSITION_M = 4.0
DUPLICATE_ROTATION_DEG = 6.0
QUARTER_STRIDE = 4
BATCH_SIZE = 8


@dataclass(frozen=True)
class Frame:
    sequence: str
    source_index: int
    image_name: str
    source_path: Path
    position: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def euclidean(first: Frame, second: Frame) -> float:
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(first.position, second.position))
    )


def rotation_angle_deg(first: Frame, second: Frame) -> float:
    trace = sum(
        sum(first.rotation[k][i] * second.rotation[k][i] for k in range(3))
        for i in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def load_sequence(sequence_root: Path) -> list[Frame]:
    metadata_path = sequence_root / "sampleinfos_interpolated.json"
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = []
    for row in rows:
        transform = row["T4x4"]
        image_name = str(row["OriginalImageName"])
        source_path = (sequence_root / "interval1_CAM" / image_name).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        frames.append(
            Frame(
                sequence=sequence_root.name,
                source_index=int(row["SortedImageID"]),
                image_name=image_name,
                source_path=source_path,
                position=tuple(float(transform[i][3]) for i in range(3)),
                rotation=tuple(
                    tuple(float(transform[i][j]) for j in range(3))
                    for i in range(3)
                ),
            )
        )
    return sorted(frames, key=lambda frame: frame.source_index)


def select_sequential(
    frames: list[Frame], translation_m: float, rotation_deg: float
) -> list[Frame]:
    if not frames:
        return []
    selected = [frames[0]]
    anchor = frames[0]
    for frame in frames[1:]:
        if (
            euclidean(anchor, frame) >= translation_m
            or rotation_angle_deg(anchor, frame) >= rotation_deg
        ):
            selected.append(frame)
            anchor = frame
    return selected


def remove_pose_duplicates(frames: list[Frame]) -> list[Frame]:
    kept: list[Frame] = []
    cells: dict[tuple[int, int, int], list[Frame]] = defaultdict(list)
    for frame in frames:
        cell = tuple(
            math.floor(value / DUPLICATE_POSITION_M) for value in frame.position
        )
        duplicate = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for existing in cells.get(neighbor, ()):
                        if (
                            euclidean(frame, existing) < DUPLICATE_POSITION_M
                            and rotation_angle_deg(frame, existing)
                            < DUPLICATE_ROTATION_DEG
                        ):
                            duplicate = True
                            break
                    if duplicate:
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            kept.append(frame)
            cells[cell].append(frame)
    return kept


def conflicts_with(frame: Frame, references: list[Frame]) -> bool:
    return any(
        euclidean(frame, reference) < DUPLICATE_POSITION_M
        and rotation_angle_deg(frame, reference) < DUPLICATE_ROTATION_DEG
        for reference in references
    )


def sha256_text(lines: list[str]) -> str:
    return hashlib.sha256("".join(f"{line}\n" for line in lines).encode()).hexdigest()


def copy_split(output_root: Path, name: str, frames: list[Frame]) -> list[str]:
    destination = output_root / name
    destination.mkdir(parents=True)
    paths = []
    for frame in frames:
        target = destination / frame.image_name
        shutil.copy2(frame.source_path, target)
        paths.append(str(target.resolve()))
    return paths


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    train_full: list[Frame] = []
    validation: list[Frame] = []
    test: list[Frame] = []
    sequence_counts = {}

    sequence_roots = sorted(path for path in source_root.iterdir() if path.is_dir())
    for sequence_root in sequence_roots:
        frames = load_sequence(sequence_root)
        first_half_count = len(frames) // 2
        first_half = frames[:first_half_count]

        evaluation_pool = remove_pose_duplicates(
            select_sequential(
                first_half, EVAL_TRANSLATION_M, EVAL_ROTATION_DEG
            )
        )
        val_start = int(first_half_count * 0.75)
        val_end = int(first_half_count * 0.78)
        test_start = int(first_half_count * 0.875)
        sequence_validation = [
            frame
            for frame in evaluation_pool
            if val_start <= frame.source_index < val_end
        ]
        sequence_test = [
            frame
            for frame in evaluation_pool
            if test_start <= frame.source_index < first_half_count
        ]
        evaluation = sequence_validation + sequence_test

        train_end = int(first_half_count * 0.60)
        train_candidates = remove_pose_duplicates(
            select_sequential(
                first_half[:train_end],
                TRAIN_TRANSLATION_M,
                TRAIN_ROTATION_DEG,
            )
        )
        sequence_train = [
            frame
            for frame in train_candidates
            if not conflicts_with(frame, evaluation)
        ]

        train_full.extend(sequence_train)
        validation.extend(sequence_validation)
        test.extend(sequence_test)
        sequence_counts[sequence_root.name] = {
            "source": len(frames),
            "first_half": first_half_count,
            "train": len(sequence_train),
            "validation": len(sequence_validation),
            "test": len(sequence_test),
            "train_removed_for_eval_pose_conflict": (
                len(train_candidates) - len(sequence_train)
            ),
        }

    train_full.sort(key=lambda frame: (frame.sequence, frame.source_index))
    validation.sort(key=lambda frame: (frame.sequence, frame.source_index))
    test.sort(key=lambda frame: (frame.sequence, frame.source_index))
    train_quarter = train_full[::QUARTER_STRIDE]

    all_names = {
        "train_full": {frame.image_name for frame in train_full},
        "validation": {frame.image_name for frame in validation},
        "test": {frame.image_name for frame in test},
    }
    if len(all_names["train_full"]) != len(train_full):
        raise RuntimeError("Duplicate basename found in full training split")
    if all_names["train_full"] & all_names["validation"]:
        raise RuntimeError("Full training and validation splits overlap")
    if all_names["train_full"] & all_names["test"]:
        raise RuntimeError("Full training and test splits overlap")
    if all_names["validation"] & all_names["test"]:
        raise RuntimeError("Validation and test splits overlap")
    if len(train_full) <= 1000:
        raise RuntimeError(f"Full training split is too small: {len(train_full)}")
    if len(train_full) % (BATCH_SIZE * QUARTER_STRIDE):
        raise RuntimeError(
            "Full training count must be divisible by batch_size * quarter_stride"
        )
    if len(train_quarter) % BATCH_SIZE:
        raise RuntimeError("Quarter training count is not batch-size divisible")

    output_root.mkdir(parents=True)
    split_frames = {
        f"train_full_{len(train_full)}": train_full,
        f"train_quarter_{len(train_quarter)}": train_quarter,
        f"validation_{len(validation)}": validation,
        f"test_{len(test)}": test,
    }
    manifest_root = output_root / "manifests"
    manifest_root.mkdir()
    manifest_hashes = {}
    for name, split in split_frames.items():
        paths = copy_split(output_root, name, split)
        manifest_path = manifest_root / f"{name}.txt"
        manifest_path.write_text(
            "".join(f"{path}\n" for path in paths), encoding="utf-8"
        )
        manifest_hashes[name] = sha256_text(paths)

    with (manifest_root / "all_splits.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(
            ["split", "sequence", "source_index", "image_name", "source_path"]
        )
        for name, split in split_frames.items():
            for frame in split:
                writer.writerow(
                    [
                        name,
                        frame.sequence,
                        frame.source_index,
                        frame.image_name,
                        frame.source_path,
                    ]
                )

    summary = {
        "protocol": "first_half_train60_val75_78_test87.5_100_v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "percentages_relative_to_each_sequence_first_half": {
            "train": [0.0, 0.60],
            "buffer_train_validation": [0.60, 0.75],
            "validation": [0.75, 0.78],
            "buffer_validation_test": [0.78, 0.875],
            "test": [0.875, 1.0],
        },
        "selection": {
            "train_translation_m": TRAIN_TRANSLATION_M,
            "train_rotation_deg": TRAIN_ROTATION_DEG,
            "evaluation_translation_m": EVAL_TRANSLATION_M,
            "evaluation_rotation_deg": EVAL_ROTATION_DEG,
            "duplicate_position_m": DUPLICATE_POSITION_M,
            "duplicate_rotation_deg": DUPLICATE_ROTATION_DEG,
            "train_removed_if_conflicting_with_validation_or_test": True,
            "quarter_stride": QUARTER_STRIDE,
            "batch_size": BATCH_SIZE,
        },
        "counts": {
            "train_full": len(train_full),
            "train_quarter": len(train_quarter),
            "validation": len(validation),
            "test": len(test),
        },
        "sequence_counts": sequence_counts,
        "manifest_sha256": manifest_hashes,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

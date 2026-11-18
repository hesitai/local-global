#!/usr/bin/env python3
"""Freeze quarter/full training manifests and build a same-scene test GT set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SEED = 20260811
TRAIN_DISTANCE_M = 10.0
VALIDATION_DISTANCE_M = 10.0
TEST_DISTANCE_M = 8.0


@dataclass(frozen=True)
class Record:
    sequence: str
    source_index: int
    image_name: str
    source_path: Path
    pose: np.ndarray


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    dataset = project / "dataset/UAV_single_image_pose_split"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/home/xtl/hst/joint/GS-Blur Confirm/data/my_uav_scene/input"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=dataset)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def manifest_names(path: Path) -> set[str]:
    return {
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def manifest_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_records(source_root: Path) -> tuple[list[Record], dict[str, Record]]:
    records = []
    by_name = {}
    for sequence_root in sorted(path for path in source_root.iterdir() if path.is_dir()):
        metadata_path = sequence_root / "sampleinfos_interpolated.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for row in metadata:
            name = str(row["OriginalImageName"])
            source_path = (sequence_root / "interval1_CAM" / name).resolve()
            if source_path.suffix.lower() not in IMAGE_SUFFIXES or not source_path.is_file():
                raise FileNotFoundError(source_path)
            record = Record(
                sequence=sequence_root.name,
                source_index=int(row["SortedImageID"]),
                image_name=name,
                source_path=source_path,
                pose=np.asarray(row["T4x4"], dtype=np.float64)[:3, 3],
            )
            if name in by_name:
                raise RuntimeError(f"Duplicate image basename across sequences: {name}")
            records.append(record)
            by_name[name] = record
    return records, by_name


def selected_pool_names(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return {row["image_name"] for row in csv.DictReader(source, delimiter="\t")}


def euclidean(first: Record, second: Record) -> float:
    return float(np.linalg.norm(first.pose - second.pose))


def nearest_distance(record: Record, references: list[Record]) -> float:
    same_sequence = [item for item in references if item.sequence == record.sequence]
    if not same_sequence:
        return float("inf")
    reference_poses = np.stack([item.pose for item in same_sequence])
    return float(np.linalg.norm(reference_poses - record.pose, axis=1).min())


def nearest_distances(
    candidates: list[Record], references: list[Record]
) -> dict[str, float]:
    candidates_by_sequence = {}
    references_by_sequence = {}
    for record in candidates:
        candidates_by_sequence.setdefault(record.sequence, []).append(record)
    for record in references:
        references_by_sequence.setdefault(record.sequence, []).append(record)

    distances = {}
    for sequence, sequence_candidates in candidates_by_sequence.items():
        sequence_references = references_by_sequence.get(sequence, [])
        if not sequence_references:
            distances.update(
                (record.image_name, float("inf")) for record in sequence_candidates
            )
            continue
        reference_poses = np.stack([record.pose for record in sequence_references])
        for start in range(0, len(sequence_candidates), 1000):
            batch = sequence_candidates[start : start + 1000]
            candidate_poses = np.stack([record.pose for record in batch])
            values = np.linalg.norm(
                candidate_poses[:, None, :] - reference_poses[None, :, :], axis=2
            ).min(axis=1)
            distances.update(
                (record.image_name, float(value))
                for record, value in zip(batch, values)
            )
    return distances


def choose_additional(
    eligible: list[Record], count: int, seed: int
) -> list[Record]:
    by_sequence = {}
    for record in eligible:
        by_sequence.setdefault(record.sequence, []).append(record)
    sequences = sorted(by_sequence)
    rng = random.Random(seed)
    diverse_by_sequence = {}
    for sequence in sequences:
        source_candidates = list(by_sequence[sequence])
        best = []
        for _ in range(1000):
            candidates = list(source_candidates)
            rng.shuffle(candidates)
            attempt = []
            for record in candidates:
                if all(
                    euclidean(record, existing) >= TEST_DISTANCE_M
                    for existing in attempt
                ):
                    attempt.append(record)
            if len(attempt) > len(best):
                best = attempt
        diverse_by_sequence[sequence] = best

    base = count // len(sequences)
    quotas = {
        sequence: min(base, len(diverse_by_sequence[sequence]))
        for sequence in sequences
    }
    remaining = count - sum(quotas.values())
    while remaining:
        available = [
            sequence
            for sequence in sequences
            if quotas[sequence] < len(diverse_by_sequence[sequence])
        ]
        if not available:
            break
        sequence = min(
            available,
            key=lambda item: (
                quotas[item], -len(diverse_by_sequence[item]), item
            ),
        )
        quotas[sequence] += 1
        remaining -= 1

    selected = []
    for sequence in sequences:
        selected.extend(diverse_by_sequence[sequence][: quotas[sequence]])
    return sorted(selected, key=lambda record: (record.sequence, record.source_index))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifests = dataset_root / "manifests"
    fixed_root = dataset_root / "fixed_protocol"
    quarter_root = fixed_root / "train_quarter_536"
    full_root = fixed_root / "train_full_2000"
    validation_root = fixed_root / "validation_252"
    test_root = fixed_root / "test_fixed"

    train_manifest = manifests / "train.txt"
    quarter_manifest = manifests / "train_quarter_fixed.txt"
    validation_manifest = manifests / "validation.txt"
    former_test_manifest = manifests / "former_test.txt"
    selected_manifest = manifests / "source_selected_pool.tsv"

    train_lines = manifest_lines(train_manifest)
    train_names = {Path(line).name for line in train_lines}
    quarter_names = manifest_names(quarter_manifest)
    validation_names = manifest_names(validation_manifest)
    former_test_names = manifest_names(former_test_manifest)
    selected_names = selected_pool_names(selected_manifest)
    records, by_name = load_records(source_root)

    if len(train_names) != 2144 or len(quarter_names) != 536:
        raise RuntimeError(
            f"Unexpected training protocol sizes: full={len(train_names)}, "
            f"quarter={len(quarter_names)}"
        )
    if not quarter_names.issubset(train_names):
        raise RuntimeError("Fixed quarter manifest is not a subset of train.txt")

    old_unseen = sorted(
        (by_name[name] for name in former_test_names - quarter_names),
        key=lambda record: (record.sequence, record.source_index),
    )
    if len(old_unseen) != 147:
        raise RuntimeError(f"Expected 147 unseen former-test rows, got {len(old_unseen)}")
    quarter_records = [by_name[name] for name in quarter_names]
    old_distances = nearest_distances(old_unseen, quarter_records)
    dropped_old = sorted(
        old_unseen,
        key=lambda record: (
            old_distances[record.image_name], record.sequence, record.source_index
        ),
    )[:3]
    dropped_names = {record.image_name for record in dropped_old}
    old_test = [record for record in old_unseen if record.image_name not in dropped_names]
    if len(old_test) != 144:
        raise AssertionError(len(old_test))

    full_fixed_lines = [
        line for line in train_lines if Path(line).name not in {r.image_name for r in old_test}
    ]
    if len(full_fixed_lines) != 2000:
        raise RuntimeError(f"Expected fixed full training size 2000, got {len(full_fixed_lines)}")
    if not quarter_names.issubset({Path(line).name for line in full_fixed_lines}):
        raise RuntimeError("Removing old test rows changed the fixed quarter set")

    raw_unselected = [record for record in records if record.image_name not in selected_names]
    full_records = [by_name[name] for name in train_names]
    validation_records = [by_name[name] for name in validation_names]
    distance_to_train = nearest_distances(raw_unselected, full_records)
    distance_to_validation = nearest_distances(raw_unselected, validation_records)
    eligible = [
        record
        for record in raw_unselected
        if distance_to_train[record.image_name] >= TRAIN_DISTANCE_M
        and distance_to_validation[record.image_name] >= VALIDATION_DISTANCE_M
    ]
    additional_test = choose_additional(eligible, count=50, seed=SEED)
    final_test = sorted(
        [("old_holdout", record) for record in old_test]
        + [("additional_pose", record) for record in additional_test],
        key=lambda item: (item[0], item[1].sequence, item[1].source_index),
    )
    if len(final_test) != 144 + len(additional_test) or len(
            {record.image_name for _, record in final_test}) != len(final_test):
        raise RuntimeError("Final test set does not contain unique images")

    if args.copy_images:
        copy_groups = (
            (
                quarter_root,
                [(Path(line), Path(line).name) for line in manifest_lines(quarter_manifest)],
            ),
            (
                full_root,
                [(Path(line), Path(line).name) for line in full_fixed_lines],
            ),
            (
                validation_root,
                [(Path(line), Path(line).name) for line in manifest_lines(validation_manifest)],
            ),
            (
                test_root,
                [(record.source_path, record.image_name) for _, record in final_test],
            ),
        )
        for destination_root, source_rows in copy_groups:
            destination_root.mkdir(parents=True, exist_ok=True)
            expected_names = {name for _, name in source_rows}
            existing_names = {
                path.name for path in destination_root.iterdir() if path.is_file()
            }
            unexpected = sorted(existing_names - expected_names)
            if unexpected:
                raise RuntimeError(
                    "Refusing to reuse {} with unexpected files: {}".format(
                        destination_root, unexpected[:5]
                    )
                )
            for source_path, name in source_rows:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                shutil.copy2(source_path, destination_root / name)

    quarter_folder_manifest = manifests / "train_quarter_folder_fixed_536.txt"
    full_fixed_manifest = manifests / "train_full_folder_fixed_2000.txt"
    write_lines(
        quarter_folder_manifest,
        [str(quarter_root / Path(line).name) for line in manifest_lines(quarter_manifest)],
    )
    write_lines(
        full_fixed_manifest,
        [str(full_root / Path(line).name) for line in full_fixed_lines],
    )
    write_lines(
        manifests / "validation_folder_fixed_252.txt",
        [
            str(validation_root / Path(line).name)
            for line in manifest_lines(validation_manifest)
        ],
    )
    write_lines(
        manifests / "test_old_holdout_144.txt",
        [str(test_root / record.image_name) for record in old_test],
    )
    write_lines(
        manifests / "test_additional_pose_fixed.txt",
        [str(test_root / record.image_name) for record in additional_test],
    )
    write_lines(
        manifests / "test_final_fixed.txt",
        [str(test_root / record.image_name) for _, record in final_test],
    )
    write_lines(
        manifests / "former_test_unseen_dropped_3.txt",
        [record.image_name for record in dropped_old],
    )

    fields = (
        "test_group",
        "sequence",
        "source_index",
        "image_name",
        "source_path",
        "test_path",
        "pose_x",
        "pose_y",
        "pose_z",
        "nearest_train536_m",
        "nearest_train2144_m",
        "nearest_validation_m",
        "selection_reason",
    )
    all_test_records = [record for _, record in final_test]
    distance_to_quarter_final = nearest_distances(all_test_records, quarter_records)
    distance_to_full_final = nearest_distances(all_test_records, full_records)
    distance_to_validation_final = nearest_distances(all_test_records, validation_records)
    with (manifests / "test_final_fixed.tsv").open(
            "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for group, record in final_test:
            writer.writerow(
                {
                    "test_group": group,
                    "sequence": record.sequence,
                    "source_index": record.source_index,
                    "image_name": record.image_name,
                    "source_path": str(record.source_path),
                    "test_path": str(test_root / record.image_name),
                    "pose_x": f"{record.pose[0]:.9f}",
                    "pose_y": f"{record.pose[1]:.9f}",
                    "pose_z": f"{record.pose[2]:.9f}",
                    "nearest_train536_m": f"{distance_to_quarter_final[record.image_name]:.6f}",
                    "nearest_train2144_m": f"{distance_to_full_final[record.image_name]:.6f}",
                    "nearest_validation_m": f"{distance_to_validation_final[record.image_name]:.6f}",
                    "selection_reason": (
                        "former_test_not_in_fixed_quarter"
                        if group == "old_holdout"
                        else "never_selected_pose_filtered_balanced"
                    ),
                }
            )

    quarter_hash = sha256(quarter_manifest)
    protocol = {
        "protocol": "fixed_quarter536_full2000_test_v1",
        "seed": SEED,
        "quarter": {
            "count": 536,
            "source_manifest": str(quarter_manifest),
            "folder_manifest": str(quarter_folder_manifest),
            "gt_root": str(quarter_root),
            "sha256": quarter_hash,
        },
        "full": {
            "count": 2000,
            "manifest": str(full_fixed_manifest),
            "gt_root": str(full_root),
            "excluded_old_test_count": 144,
        },
        "validation": {
            "count": 252,
            "gt_root": str(validation_root),
            "manifest": str(manifests / "validation_folder_fixed_252.txt"),
        },
        "test": {
            "count": len(final_test),
            "old_holdout_count": 144,
            "additional_pose_count": len(additional_test),
            "manifest": str(manifests / "test_final_fixed.txt"),
            "gt_root": str(test_root),
            "additional_thresholds_m": {
                "train2144": TRAIN_DISTANCE_M,
                "validation252": VALIDATION_DISTANCE_M,
                "test_to_test": TEST_DISTANCE_M,
            },
        },
        "former_test_dropped": [record.image_name for record in dropped_old],
    }
    (manifests / "fixed_protocol_summary.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (manifests / "train_quarter_fixed.sha256").write_text(
        f"{quarter_hash}  {quarter_manifest.name}\n", encoding="utf-8"
    )

    print(f"fixed quarter: {len(quarter_names)}")
    print(f"fixed full: {len(full_fixed_lines)}")
    print(f"old holdout: {len(old_test)} (dropped nearest 3)")
    print(f"additional eligible: {len(eligible)}, selected: {len(additional_test)}")
    print(f"final test: {len(final_test)}")
    print(f"quarter sha256: {quarter_hash}")
    print(f"copied images: {args.copy_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

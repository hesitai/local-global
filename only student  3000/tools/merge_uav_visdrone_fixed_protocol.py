#!/usr/bin/env python3
"""Merge the fixed UAV protocol with all official VisDrone DET splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


DATASET_ROOT = Path(
    "/home/xtl/hst/joint2D/only student  3000/dataset/"
    "UAV_single_image_pose_split"
)
SOURCE_ROOT = DATASET_ROOT / "fixed_protocol_first_half_60_78"
OUTPUT_ROOT = DATASET_ROOT / "fixed_protocol_first_half_60_78_visdrone_all"
SEED = 20260811
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageRecord:
    source_group: str
    source_path: Path

    @property
    def name(self) -> str:
        return self.source_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def image_records(path: Path, source_group: str) -> list[ImageRecord]:
    records = [
        ImageRecord(source_group, item.resolve())
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not records:
        raise RuntimeError(f"No images found in {path}")
    return sorted(records, key=lambda record: record.name)


def assert_unique(records: list[ImageRecord], split: str) -> None:
    names = [record.name for record in records]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate basename in {split}")


def copy_split(
    output_root: Path, split_name: str, records: list[ImageRecord]
) -> list[str]:
    split_root = output_root / split_name
    split_root.mkdir(parents=True)
    destinations = []
    for record in records:
        destination = split_root / record.name
        shutil.copy2(record.source_path, destination)
        destinations.append(str(destination.resolve()))
    return destinations


def manifest_hash(lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    uav_train = image_records(source_root / "train_full_1056", "uav_train")
    uav_validation = image_records(source_root / "validation_41", "uav_validation")
    uav_test = image_records(source_root / "test_180", "uav_test")
    visdrone_root = source_root / "VisDrone2019-DET"
    visdrone_train = image_records(
        visdrone_root / "train/images", "visdrone_train"
    )
    visdrone_validation = image_records(
        visdrone_root / "val/images", "visdrone_validation"
    )
    visdrone_test = image_records(
        visdrone_root / "test-dev/images", "visdrone_test"
    )

    rng = random.Random(args.seed)
    dropped_uav_train = sorted(
        rng.sample(uav_train, 7), key=lambda record: record.name
    )
    dropped_names = {record.name for record in dropped_uav_train}
    retained_uav_train = [
        record for record in uav_train if record.name not in dropped_names
    ]

    train_full = sorted(
        retained_uav_train + visdrone_train,
        key=lambda record: record.name,
    )
    validation = sorted(
        uav_validation + visdrone_validation,
        key=lambda record: record.name,
    )
    test = sorted(uav_test + visdrone_test, key=lambda record: record.name)

    expected = {
        "train_full": 7520,
        "validation": 589,
        "test": 1790,
    }
    actual = {
        "train_full": len(train_full),
        "validation": len(validation),
        "test": len(test),
    }
    if actual != expected:
        raise RuntimeError(f"Unexpected split counts: {actual} != {expected}")
    for split, records in (
        ("train_full", train_full),
        ("validation", validation),
        ("test", test),
    ):
        assert_unique(records, split)
    split_names = {
        "train": {record.name for record in train_full},
        "validation": {record.name for record in validation},
        "test": {record.name for record in test},
    }
    if split_names["train"] & split_names["validation"]:
        raise RuntimeError("Train and validation overlap")
    if split_names["train"] & split_names["test"]:
        raise RuntimeError("Train and test overlap")
    if split_names["validation"] & split_names["test"]:
        raise RuntimeError("Validation and test overlap")
    if len(train_full) % 8:
        raise RuntimeError("Training split is not divisible by batch size 8")

    output_root.mkdir(parents=True)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir()
    split_records = {
        "train_full_7520": train_full,
        "validation_589": validation,
        "test_1790": test,
    }
    hashes = {}
    for split_name, records in split_records.items():
        destinations = copy_split(output_root, split_name, records)
        manifest_path = manifest_root / f"{split_name}.txt"
        manifest_path.write_text(
            "".join(f"{path}\n" for path in destinations), encoding="utf-8"
        )
        hashes[split_name] = manifest_hash(destinations)

    with (manifest_root / "all_splits.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(["split", "source_group", "image_name", "source_path"])
        for split_name, records in split_records.items():
            for record in records:
                writer.writerow(
                    [split_name, record.source_group, record.name, record.source_path]
                )

    summary = {
        "protocol": "fixed_uav_first_half_plus_all_visdrone_det_v1",
        "seed": args.seed,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "counts": actual,
        "components": {
            "train": {
                "uav_retained": len(retained_uav_train),
                "uav_randomly_dropped": len(dropped_uav_train),
                "visdrone": len(visdrone_train),
            },
            "validation": {
                "uav": len(uav_validation),
                "visdrone": len(visdrone_validation),
            },
            "test": {
                "uav": len(uav_test),
                "visdrone": len(visdrone_test),
            },
        },
        "dropped_uav_train": [record.name for record in dropped_uav_train],
        "manifest_sha256": hashes,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

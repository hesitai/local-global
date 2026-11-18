#!/usr/bin/env python3
"""Create the fixed every-fifth physical subset of the merged training GT."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(
    "/home/xtl/hst/joint2D/only student  3000/dataset/"
    "UAV_single_image_pose_split/"
    "fixed_protocol_first_half_60_78_visdrone_all"
)
SOURCE_MANIFEST = ROOT / "manifests/train_full_7520.txt"
DESTINATION = ROOT / "train_fifth_1504"
DESTINATION_MANIFEST = ROOT / "manifests/train_fifth_1504.txt"
ALL_SPLITS = ROOT / "manifests/all_splits.tsv"
SUMMARY = ROOT / "summary.json"


def main() -> int:
    if DESTINATION.exists() or DESTINATION_MANIFEST.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing fifth subset: {DESTINATION}"
        )
    source_paths = [
        Path(line.strip()).resolve()
        for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(source_paths) != 7520:
        raise RuntimeError(f"Expected 7520 full-training paths, got {len(source_paths)}")
    selected = source_paths[::5]
    if len(selected) != 1504:
        raise RuntimeError(f"Expected 1504 fifth-subset paths, got {len(selected)}")

    DESTINATION.mkdir()
    destination_paths = []
    for source in selected:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = DESTINATION / source.name
        shutil.copy2(source, target)
        destination_paths.append(str(target.resolve()))

    manifest_text = "".join(f"{path}\n" for path in destination_paths)
    DESTINATION_MANIFEST.write_text(manifest_text, encoding="utf-8")

    with ALL_SPLITS.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    source_group_by_name = {
        row["image_name"]: row["source_group"]
        for row in rows
        if row["split"] == "train_full_7520"
    }
    with ALL_SPLITS.open("a", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t")
        for source, destination in zip(selected, destination_paths):
            writer.writerow(
                [
                    "train_fifth_1504",
                    source_group_by_name[source.name],
                    source.name,
                    source,
                ]
            )

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary["counts"]["train_fifth"] = len(selected)
    summary["fifth_rule"] = (
        "train_full_7520 manifest entries at zero-based indices 0,5,10,..."
    )
    summary.setdefault("manifest_sha256", {})["train_fifth_1504"] = (
        hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    )
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created {len(selected)} images at {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

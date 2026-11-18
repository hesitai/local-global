"""CSV/JSON reporting and human MOS handling for no-reference evaluation.

MOS is a mean of human ratings, never an automatically predicted image score.
First average ratings within each image, then average image scores with equal
weight. Empty, invalid and failed-image scores are not replaced by zero.
This module has no third-party dependencies and can be imported without Torch.
"""

import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


OBJECTIVE_NAMES = ("NIQE", "BRISQUE", "LV", "MUSIQ", "CPBD", "Entropy")
ALL_METRIC_NAMES = ("NIQE", "BRISQUE", "LV", "MUSIQ", "CPBD", "MOS", "Entropy")
METRIC_DIRECTIONS = {
    name: "lower" if name in ("NIQE", "BRISQUE") else "higher"
    for name in ALL_METRIC_NAMES
}
PER_IMAGE_FIELDS = (
    "image_id", "input_path", "restored_path", "width", "height", "status",
    "error", "metric_errors", "NIQE", "BRISQUE", "LV", "MUSIQ", "CPBD",
    "MOS", "MOS_n", "Entropy",
)
SUMMARY_FIELDS = ("metric", "direction", "mean", "std", "valid_count", "missing_count")


def _finite_number(value):
    """Return a finite float, or None for missing/non-numerical values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _failed(row):
    status = str(row.get("status", "")).strip().lower()
    return status in {"failed", "error", "inference_error", "read_error"} or status.startswith(("error:", "failed:"))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path, rows, fieldnames):
    """Write an Excel-friendly UTF-8-BOM CSV; missing values are blank cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values = {}
            for name in fieldnames:
                value = row.get(name)
                if value is None or (isinstance(value, float) and not math.isfinite(value)):
                    value = ""
                elif isinstance(value, (dict, list, tuple)):
                    value = json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)
                values[name] = value
            writer.writerow(values)


def load_mos(path, expected_ids=None, score_min=1.0, score_max=5.0):
    """Read human scores: image_id,rater_id,score -> per-image MOS and count.

    Blank scores are unrated template entries and are ignored. Duplicate
    image/rater pairs, unknown images and nonfinite/out-of-range scores raise
    ValueError rather than silently changing a subjective experiment.
    """
    minimum, maximum = _finite_number(score_min), _finite_number(score_max)
    if minimum is None or maximum is None or minimum >= maximum:
        raise ValueError("MOS scale must have finite score_min < score_max")
    if path is None or str(path).strip() == "":
        return {}
    expected = None if expected_ids is None else set(expected_ids)
    ratings = defaultdict(list)
    seen = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"image_id", "rater_id", "score"}.issubset(reader.fieldnames):
            raise ValueError("MOS CSV must contain columns image_id,rater_id,score")
        for line_number, row in enumerate(reader, start=2):
            raw_score = str(row.get("score") or "").strip()
            if not raw_score:
                continue
            image_id = str(row.get("image_id") or "").strip()
            rater_id = str(row.get("rater_id") or "").strip()
            context = "MOS CSV line {}".format(line_number)
            if not image_id or not rater_id:
                raise ValueError(context + ": a scored row requires image_id and rater_id")
            if expected is not None and image_id not in expected:
                raise ValueError(context + ": unknown image_id {!r}".format(image_id))
            score = _finite_number(raw_score)
            if score is None or not minimum <= score <= maximum:
                raise ValueError(context + ": score must be finite and in [{}, {}]".format(minimum, maximum))
            pair = (image_id, rater_id)
            if pair in seen:
                raise ValueError(context + ": duplicate image/rater pair {!r}".format(pair))
            seen.add(pair)
            ratings[image_id].append(score)
    return {
        image_id: {"MOS": statistics.fmean(values), "MOS_n": len(values)}
        for image_id, values in ratings.items()
    }


def summarize_rows(rows, metric_names=ALL_METRIC_NAMES):
    """Calculate equally image-weighted means and population standard deviation.

    Every input row contributes to the denominator of missing_count. A failed
    inference row contributes no valid metric even if a stale score is present.
    """
    rows = list(rows)
    result = {}
    for name in metric_names:
        values = []
        for row in rows:
            value = None if _failed(row) else _finite_number(row.get(name))
            if value is not None:
                values.append(value)
        result[name] = {
            "direction": METRIC_DIRECTIONS.get(name, "higher"),
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if values else None,
            "valid_count": len(values),
            "missing_count": len(rows) - len(values),
        }
    return result


def write_reports(output_dir, rows, metadata, write_per_image=True):
    """Write per-image CSV and three aggregate report formats; return summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    metrics = summarize_rows(rows)
    summary = {
        "metadata": _json_safe(metadata),
        "image_count": len(rows),
        "status_counts": dict(Counter(str(row.get("status", "")) for row in rows)),
        "metrics": metrics,
    }
    if write_per_image:
        write_csv(output_dir / "per_image_metrics.csv", rows, PER_IMAGE_FIELDS)
    write_csv(
        output_dir / "summary.csv",
        [dict(metric=name, **metrics[name]) for name in ALL_METRIC_NAMES],
        SUMMARY_FIELDS,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "Real-image no-reference evaluation",
        "Images: {}".format(len(rows)),
        "Status counts: " + json.dumps(summary["status_counts"], ensure_ascii=False),
        "Mean: equal weight per image; std: population standard deviation.",
        "Missing/failed scores are excluded, never replaced by zero.",
        "MOS: human ratings averaged within each image, then across images.",
        "",
        "metric\tdirection\tmean\tstd\tvalid_count\tmissing_count",
    ]
    for name in ALL_METRIC_NAMES:
        item = metrics[name]
        mean = "N/A" if item["mean"] is None else "{:.8f}".format(item["mean"])
        std = "N/A" if item["std"] is None else "{:.8f}".format(item["std"])
        lines.append("{}\t{}\t{}\t{}\t{}\t{}".format(
            name, item["direction"], mean, std, item["valid_count"], item["missing_count"]))
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_mos_template(path, image_ids):
    """Create an unrated sheet without overwriting an existing rating sheet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", encoding="utf-8-sig", newline="")
    except FileExistsError:
        return False
    with stream:
        writer = csv.DictWriter(stream, fieldnames=("image_id", "rater_id", "score"))
        writer.writeheader()
        for image_id in image_ids:
            writer.writerow({"image_id": image_id, "rater_id": "", "score": ""})
    return True


def merge_mos_and_resummarize(result_dir, mos_csv, score_min=1.0, score_max=5.0):
    """Update human MOS and aggregates without re-running restoration/metrics."""
    result_dir = Path(result_dir)
    with (result_dir / "per_image_metrics.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "image_id" not in reader.fieldnames:
            raise ValueError("per_image_metrics.csv is missing the image_id column")
        rows = list(reader)
    image_ids = [row["image_id"] for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("per_image_metrics.csv has duplicate image_id entries")
    ratings = load_mos(mos_csv, image_ids, score_min, score_max)
    for row in rows:
        rating = None if _failed(row) else ratings.get(row["image_id"])
        row["MOS"] = rating["MOS"] if rating else None
        row["MOS_n"] = rating["MOS_n"] if rating else 0
    summary_path = result_dir / "summary.json"
    metadata = {}
    if summary_path.exists():
        metadata = json.loads(summary_path.read_text(encoding="utf-8")).get("metadata", {})
    metadata.update({
        "mos_csv": str(Path(mos_csv).expanduser().resolve()) if mos_csv else None,
        "mos_scale": {"min": float(score_min), "max": float(score_max)},
        "mos_aggregation": "mean of image MOS; each image MOS is the mean of human ratings",
    })
    return write_reports(result_dir, rows, metadata)

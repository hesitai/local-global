#!/usr/bin/env python3
"""真实图像无参考去模糊评测。配置入口：test/real_test/real_nr_config.json。

python test/real_test/evaluate_real_nr.py --help
python test/real_test/evaluate_real_nr.py --config test/real_test/real_nr_config.json
python test/real_test/evaluate_real_nr.py --summarize-only --output-dir results/real_nr --mos-csv ratings.csv

导入本文件不会加载模型、下载权重或启动评测。
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from importlib import metadata as import_metadata
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nr_results import (ALL_METRIC_NAMES, PER_IMAGE_FIELDS, merge_mos_and_resummarize, load_mos,
                        write_csv, write_mos_template, write_reports)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
OBJECTIVE_NAMES = ["NIQE", "BRISQUE", "LV", "MUSIQ", "CPBD", "Entropy"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "real_nr_config.json"))
    parser.add_argument("--input-dir", help="待恢复图目录；evaluate 模式填已有恢复图目录")
    parser.add_argument("--weights", help="模型 checkpoint；evaluate 模式无需填写")
    parser.add_argument("--output-dir", help="输出目录；相对路径以项目根目录为基准")
    parser.add_argument("--mode", choices=["restore", "restore_only", "evaluate"])
    parser.add_argument("--device", help="auto / cpu / cuda:0")
    parser.add_argument("--metric-device", help="指标设备，可与模型设备不同")
    parser.add_argument("--metrics", nargs="+", choices=OBJECTIVE_NAMES,
                        help="默认计算六个客观指标；MOS 通过评分 CSV 另外导入")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--tile-batch-size", type=int, help="一次并行推理的图块数，建议 2 或 4")
    parser.add_argument("--tile-size", type=int, help="覆盖 model.tile_size，0 为整图推理")
    parser.add_argument("--tile-overlap", type=int, help="覆盖 model.tile_overlap")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--amp", dest="amp", action="store_true", default=None,
                           help="恢复模型使用 CUDA FP16 混合精度；指标仍用原精度")
    precision.add_argument("--no-amp", dest="amp", action="store_false",
                           help="恢复模型使用原始 FP32 精度")
    parser.add_argument("--mos-csv", help="人工评分表 image_id,rater_id,score")
    parser.add_argument("--summarize-only", action="store_true",
                        help="只将 MOS 合并到已有逐图表并重新汇总，不推理也不计算客观指标")
    return parser


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(options):
    config_path = Path(options.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    for name in ("input_dir", "weights", "output_dir", "mode", "device", "metric_device",
                 "metrics", "max_images", "mos_csv"):
        value = getattr(options, name)
        if value is not None:
            config[name] = value
    config.setdefault("model", {})
    for name in ("tile_batch_size", "tile_size", "tile_overlap", "amp"):
        value = getattr(options, name)
        if value is not None:
            config["model"][name] = value
    config.setdefault("mode", "restore")
    config.setdefault("metrics", OBJECTIVE_NAMES[:])
    if config["mode"] not in ("restore", "restore_only", "evaluate"):
        raise ValueError("mode 必须为 restore、restore_only 或 evaluate。")
    if not config["metrics"] or any(name not in OBJECTIVE_NAMES for name in config["metrics"]):
        raise ValueError("metrics 必须为六个客观指标名称的非空列表。MOS 通过 mos_csv 导入。")
    if len(config["metrics"]) != len(set(config["metrics"])):
        raise ValueError("metrics 中有重复项。")
    if config.get("max_images", 0) < 0:
        raise ValueError("max_images 不能小于 0。")
    report_every = config.get("report_every", 50)
    if not isinstance(report_every, int) or isinstance(report_every, bool) or report_every < 1:
        raise ValueError("report_every 必须为正整数。")
    if not config.get("output_dir"):
        raise ValueError("请填写 output_dir。")
    config["output_dir"] = str(project_path(config["output_dir"]))
    for name in ("input_dir", "weights", "mos_csv", "model_adapter"):
        if config.get(name):
            config[name] = str(project_path(config[name]))
    if not (math.isfinite(config.get("mos_min", 1.0)) and
            math.isfinite(config.get("mos_max", 5.0)) and
            config.get("mos_min", 1.0) < config.get("mos_max", 5.0)):
        raise ValueError("mos_min 和 mos_max 必须有限且 mos_min < mos_max。")
    return config, config_path


def list_images(input_dir, recursive=True):
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted((p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                  key=lambda p: p.relative_to(input_dir).as_posix())


def read_rgb(path):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        # 不对 16 位/浮点遥感数据静默截断；先按固定规则转换成 8 位可视图再评测。
        if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
            raise ValueError("仅支持标准 8 位图像；当前模式 {}，请先统一转换。".format(image.mode))
        image = ImageOps.exif_transpose(image)
        if image.mode in ("LA", "RGBA") or (image.mode == "P" and "transparency" in image.info):
            raise ValueError("含透明通道，请先按统一背景合成 RGB 图像。")
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def choose_device(request):
    import torch

    if request == "auto":
        request = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无可用 CUDA；可配置 device=cpu。")
    return device


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("real_nr_user_model_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError("无法载入模型入口：{}".format(path))
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    for name in ("build_model", "predict"):
        if not callable(getattr(adapter, name, None)):
            raise TypeError("模型入口文件缺少函数 {}。".format(name))
    return adapter


def validate_inference_options(model_config):
    multiple = model_config.get("pad_multiple", 128)
    tile = model_config.get("tile_size", 0)
    overlap = model_config.get("tile_overlap", 32)
    batch_size = model_config.get("tile_batch_size", 1)
    if not isinstance(multiple, int) or isinstance(multiple, bool) or multiple < 1:
        raise ValueError("model.pad_multiple 必须为正整数。")
    if not isinstance(tile, int) or tile < 0:
        raise ValueError("model.tile_size 必须为非负整数。")
    if tile and (not isinstance(overlap, int) or not 0 <= overlap < tile):
        raise ValueError("分块推理需要 0 <= tile_overlap < tile_size。")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("model.tile_batch_size 必须为正整数。")
    if not isinstance(model_config.get("amp", False), bool):
        raise ValueError("model.amp 必须为 true 或 false。")


def tile_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def padded_predict(image, model, adapter, model_config):
    import torch
    import torch.nn.functional as functional

    height, width = image.shape[-2:]
    multiple = model_config.get("pad_multiple", 128)
    target_h = ((height + multiple - 1) // multiple) * multiple
    target_w = ((width + multiple - 1) // multiple) * multiple
    if model_config.get("square_pad", False):
        target_h = target_w = max(target_h, target_w)
    # 右、下复制边界补齐；支持尺寸为 1 的图像，推理后去除 padding。
    padded = functional.pad(image, (0, target_w - width, 0, target_h - height), mode="replicate")
    output = adapter.predict(model, padded, model_config)
    if not torch.is_tensor(output) or output.shape != padded.shape:
        raise ValueError("模型入口返回值须为与补齐后输入同尺寸的 RGB Tensor。")
    return output.float()[..., :height, :width]


def restore_image(rgb, model, adapter, model_config, device, runtime_stats=None):
    import numpy as np
    import torch

    image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0)
    image = image.to(device) / 255.0
    height, width = rgb.shape[:2]
    tile = model_config.get("tile_size", 0)
    amp_enabled = model_config.get("amp", False)
    if amp_enabled and device.type != "cuda":
        raise ValueError("--amp 需要 CUDA；CPU 请使用 --no-amp。")
    # 合并后统一检查有限值，避免每个图块都触发 GPU 到 CPU 的同步。
    runtime_config = dict(model_config, check_finite=False)
    stats = runtime_stats if runtime_stats is not None else {}
    stats.update(tile_count=1, tile_batches=1, tile_batch_size=1, amp=amp_enabled, oom_reduced=False)
    with torch.inference_mode():
        if not tile or (height <= tile and width <= tile):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                restored = padded_predict(image, model, adapter, runtime_config)
        else:
            restored = torch.zeros_like(image)
            counts = torch.zeros((1, 1, height, width), dtype=image.dtype, device=device)
            overlap = model_config.get("tile_overlap", 32)
            positions = [(top, left) for top in tile_starts(height, tile, overlap)
                         for left in tile_starts(width, tile, overlap)]
            batch_limit = min(model_config.get("tile_batch_size", 1), len(positions))
            cursor, batches = 0, 0
            while cursor < len(positions):
                current = positions[cursor:cursor + batch_limit]
                batch = outputs = None
                try:
                    batch = torch.cat([image[..., top:top + tile, left:left + tile]
                                       for top, left in current], dim=0)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        outputs = padded_predict(batch, model, adapter, runtime_config)
                except torch.cuda.OutOfMemoryError:
                    if device.type != "cuda" or len(current) == 1:
                        raise
                    batch = outputs = None
                    batch_limit = max(1, len(current) // 2)
                    stats["oom_reduced"] = True
                    torch.cuda.empty_cache()
                    print("显存不足，tile_batch_size 自动减至 {} 后重试。".format(batch_limit), flush=True)
                    continue
                ph, pw = batch.shape[-2:]
                # 与串行版本保持相同的图块位置、累加顺序和 FP32 等权融合。
                for offset, (top, left) in enumerate(current):
                    restored[..., top:top + ph, left:left + pw] += outputs[offset:offset + 1]
                    counts[..., top:top + ph, left:left + pw] += 1
                cursor += len(current)
                batches += 1
            stats.update(tile_count=len(positions), tile_batches=batches, tile_batch_size=batch_limit)
            restored /= counts
        if not torch.isfinite(restored).all().item():
            raise ValueError("模型输出包含 NaN 或 Inf。")
        restored = restored.clamp(0, 1)
        return (restored[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_versions():
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless", "scipy", "scikit-image",
                    "Pillow", "pyiqa", "timm"):
        try:
            versions[package] = import_metadata.version(package)
        except import_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run(config, config_path):
    import numpy as np
    import torch
    from PIL import Image

    if not config.get("input_dir"):
        raise ValueError("请在 real_nr_config.json 填写 input_dir，或传 --input-dir。")
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    if not input_dir.is_dir():
        raise NotADirectoryError("测试集目录不存在：{}".format(input_dir))
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("output_dir 必须放在 input_dir 之外，避免结果混入测试集。")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("输出目录非空，请更换 output_dir，以保留已有实验：{}".format(output_dir))
    paths = list_images(input_dir, config.get("recursive", True))
    if config.get("max_images", 0):
        paths = paths[:config["max_images"]]
    if not paths:
        raise ValueError("测试集中没有支持的图像文件。")
    image_ids = [path.relative_to(input_dir).as_posix() for path in paths]
    mos = load_mos(config.get("mos_csv"), expected_ids=image_ids,
                   score_min=config.get("mos_min", 1.0), score_max=config.get("mos_max", 5.0))
    random.seed(config.get("seed", 123))
    np.random.seed(config.get("seed", 123))
    torch.manual_seed(config.get("seed", 123))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get("seed", 123))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    mode = config["mode"]
    model = adapter = None
    model_device = choose_device(config.get("device", "auto"))
    metric_device = choose_device(config.get("metric_device", "auto"))
    model_config = config.get("model", {})
    inference_config = dict(model_config)
    if mode != "evaluate":
        validate_inference_options(model_config)
        if model_config.get("amp", False) and model_device.type != "cuda":
            raise ValueError("--amp 需要 CUDA；当前设备为 CPU，请使用 --no-amp。")
        adapter = load_adapter(config.get("model_adapter", str(SCRIPT_DIR / "nr_model_adapter.py")))
        model = adapter.build_model(config.get("weights", ""), model_device, model_config)
        model = model.to(model_device).eval()
    suite = None
    if mode != "restore_only":
        from nr_metrics import MetricSuite
        suite = MetricSuite(names=config["metrics"], device=str(metric_device))

    metadata = {
        "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "config_path": str(config_path), "config": config,
        "environment": environment_versions(),
        "model_device": str(model_device), "metric_device": str(metric_device),
        "weights_sha256": sha256_file(config["weights"]) if mode != "evaluate" else None,
        "model_adapter_sha256": sha256_file(config.get("model_adapter", SCRIPT_DIR / "nr_model_adapter.py"))
                                  if mode != "evaluate" else None,
        "metric_image": "saved 8-bit RGB PNG" if mode != "evaluate" else "input image converted to 8-bit RGB",
        "mos_scale": [config.get("mos_min", 1.0), config.get("mos_max", 5.0)],
        "averaging": "arithmetic mean over valid image scores; missing values excluded and counted",
        "evaluator_sha256": sha256_file(Path(__file__)),
        "metric_implementation_sha256": sha256_file(SCRIPT_DIR / "nr_metrics.py"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    write_mos_template(output_dir / "mos_ratings_template.csv", image_ids)
    write_csv(output_dir / "per_image_metrics.csv", [], PER_IMAGE_FIELDS)
    timing_fields = ("image_id", "input_load_seconds", "restoration_seconds", "save_seconds",
                     "metrics_seconds", "total_seconds", "tile_count", "tile_batches", "tile_batch_size", "amp") + tuple(
                     name + "_seconds" for name in OBJECTIVE_NAMES)
    write_csv(output_dir / "timings.csv", [], timing_fields)
    rows = []
    failure_log = output_dir / "errors.log"
    print("Images: {} | Mode: {} | Output: {}".format(len(paths), mode, output_dir))
    if mode != "evaluate":
        print("Inference: tile={} overlap={} batch={} AMP={}".format(
            model_config.get("tile_size", 0), model_config.get("tile_overlap", 32),
            model_config.get("tile_batch_size", 1), model_config.get("amp", False)), flush=True)
    processing_started = time.perf_counter()
    for index, (path, image_id) in enumerate(zip(paths, image_ids), 1):
        row = {name: None for name in ALL_METRIC_NAMES}
        row.update(image_id=image_id, input_path=str(path), restored_path="", width="", height="",
                   status="error", error="", metric_errors="", MOS_n=0)
        started = time.perf_counter()
        timing = dict(image_id=image_id, restoration_seconds=0.0, save_seconds=0.0, metrics_seconds=0.0)
        runtime_stats = {}
        try:
            try:
                rgb = read_rgb(path)
            finally:
                timing["input_load_seconds"] = time.perf_counter() - started
            row["height"], row["width"] = rgb.shape[:2]
            if mode == "evaluate":
                restored_path = path
                restored_rgb = rgb
            else:
                phase_started = time.perf_counter()
                try:
                    restored_rgb = restore_image(rgb, model, adapter, inference_config, model_device, runtime_stats)
                finally:
                    timing["restoration_seconds"] = time.perf_counter() - phase_started
                timing.update(runtime_stats)
                if runtime_stats["oom_reduced"]:
                    inference_config["tile_batch_size"] = runtime_stats["tile_batch_size"]
                    metadata.setdefault("inference_batch_adjustments", []).append({
                        "image_id": image_id, "effective_tile_batch_size": runtime_stats["tile_batch_size"]})
                # 保留原扩展名再追加 .png，避免 a.jpg 和 a.png 覆盖彼此。
                restored_path = output_dir / "restored" / (image_id + ".png")
                restored_path.parent.mkdir(parents=True, exist_ok=True)
                phase_started = time.perf_counter()
                try:
                    Image.fromarray(restored_rgb).save(restored_path, format="PNG")
                    # 从磁盘重读，确保计算的是交付的恢复图。
                    restored_rgb = read_rgb(restored_path)
                finally:
                    timing["save_seconds"] = time.perf_counter() - phase_started
            row["restored_path"] = str(restored_path)
            if suite is not None:
                phase_started = time.perf_counter()
                try:
                    row.update(suite(restored_rgb))
                finally:
                    timing["metrics_seconds"] = time.perf_counter() - phase_started
                timing.update({name + "_seconds": seconds for name, seconds in suite.last_timings.items()})
                if suite.last_errors:
                    row["metric_errors"] = json.dumps(suite.last_errors, ensure_ascii=False)
            if image_id in mos:
                row.update(mos[image_id])
            row["status"] = "partial" if row["metric_errors"] else "ok"
        except Exception as exc:
            row["error"] = "{}: {}".format(type(exc).__name__, exc)
            with failure_log.open("a", encoding="utf-8") as stream:
                stream.write("\n{}\n{}\n".format(image_id, traceback.format_exc()))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)
        timing["total_seconds"] = time.perf_counter() - started
        with (output_dir / "timings.csv").open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=timing_fields, extrasaction="ignore").writerow(timing)
        # 逐图追加并关闭文件；汇总定期刷新，避免对大测试集反复重写全部 CSV。
        with (output_dir / "per_image_metrics.csv").open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=PER_IMAGE_FIELDS, extrasaction="ignore").writerow(row)
        if index == 1 or index % config.get("report_every", 50) == 0:
            write_reports(output_dir, rows, metadata, write_per_image=False)
        values = " ".join("{}={:.5f}".format(name, row[name]) for name in ALL_METRIC_NAMES
                          if isinstance(row.get(name), (int, float)))
        elapsed = time.perf_counter() - processing_started
        eta = timedelta(seconds=int(elapsed / index * (len(paths) - index)))
        print("[{}/{}] {} | {} | {} | {:.2f}s (restore {:.2f}s, metrics {:.2f}s) | ETA {}".format(
            index, len(paths), image_id, row["status"], values, timing["total_seconds"],
            timing["restoration_seconds"], timing["metrics_seconds"], eta), flush=True)
        if row["error"] or row["metric_errors"]:
            print("  ", row["error"] or row["metric_errors"], flush=True)
    metadata["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    write_reports(output_dir, rows, metadata)
    print((output_dir / "summary.txt").read_text(encoding="utf-8"))
    if mos:
        print("已导入 {} 张图像的人工 MOS，详见逐图表和汇总有效数。".format(len(mos)))
    else:
        print("MOS 待人工评分；填写 mos_ratings_template.csv 后使用 --summarize-only 合并。")
    # 有失败图或指标缺失时提供非零返回码，同时保留可用结果与有效样本数。
    return 2 if any(row["status"] != "ok" for row in rows) else 0


def main(argv=None):
    options = build_parser().parse_args(argv)
    try:
        config, config_path = load_config(options)
        if options.summarize_only:
            if not config.get("mos_csv"):
                raise ValueError("--summarize-only 需要 --mos-csv 或配置 mos_csv。")
            merge_mos_and_resummarize(config["output_dir"], config["mos_csv"],
                                    config.get("mos_min", 1.0), config.get("mos_max", 5.0))
            print((Path(config["output_dir"]) / "summary.txt").read_text(encoding="utf-8"))
            return 0
        return run(config, config_path)
    except Exception as exc:
        print("评测启动失败：{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

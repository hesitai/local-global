#!/usr/bin/env python3
"""Scene-label the existing fixed test pairs without changing the source split.

Initial labels are zero-shot suggestions, not calibrated probabilities. Export
requires a visual-review file covering every sample. No restoration scores are
read by this program. Each exported image is a separate, byte-identical copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "UAV_single_image_pose_split" / "fixed_protocol_first_half_60_78_visdrone_all"
MODEL_ID = "google/siglip2-base-patch16-256"
MODEL_REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
RECORD_DIR = "test_scene_classification_records"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CATEGORIES = {
    "道路交通": {
        "description": "道路、路口、高速路、桥梁及停车场；交通或停车空间是主要场景。",
        "prompts": [
            "An aerial photograph of an urban street with traffic and roadside parking.",
            "An aerial photograph of a road intersection, roads, highways or an overpass.",
            "An aerial photograph of a parking lot with rows of parked cars.",
            "An aerial photograph of a street corridor, with buildings and trees along the road.",
        ],
    },
    "建筑区域": {
        "description": "住宅、楼宇、厂房及屋顶等建筑结构构成主要对象。",
        "prompts": [
            "An aerial photograph dominated by buildings and rooftops.",
            "An aerial photograph of a residential neighborhood or apartment buildings.",
            "An aerial photograph of industrial buildings, factories and warehouses.",
            "An aerial photograph showing the facades and structures of buildings.",
        ],
    },
    "自然植被": {
        "description": "树林、草地、农田或植被山地构成主要地表景观。",
        "prompts": [
            "An aerial photograph of a forest or wooded landscape dominated by trees.",
            "An aerial photograph of agricultural fields, farmland or crops.",
            "An aerial photograph of grassland, green fields or vegetated hills.",
            "An aerial photograph of a large wooded park, with scattered buildings and roads.",
        ],
    },
    "水域": {
        "description": "河流、湖泊、海面、港湾水面或可识别河道为主体；少水河道仍按河道场景记录。",
        "prompts": [
            "An aerial photograph of a river and its water surface.",
            "An aerial photograph of a lake or reservoir.",
            "An aerial photograph of the sea, a bay or coastal water.",
            "An aerial photograph of a harbor or port with a large water surface.",
        ],
    },
    "开阔场地": {
        "description": "广场、裸地、运动场、机场跑道或停机坪等开放场地为主体；停车场归道路交通。",
        "prompts": [
            "An aerial photograph of a large open paved plaza or public square.",
            "An aerial photograph of an open bare ground, dirt field or empty construction site.",
            "An aerial photograph of a sports field, athletic track or playground.",
            "An aerial photograph of an airport runway, taxiway or apron.",
        ],
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def load_pairs(root: Path) -> list[dict]:
    pairs = read_csv(root / "test_fixed_full/pairs.tsv", "\t")
    if len(pairs) != 1790:
        raise ValueError(f"Expected the unmodified 1790-pair test split, got {len(pairs)}")
    ids = [row["sample_id"] for row in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    for row in pairs:
        for field, folder in (("input_path", "input"), ("target_path", "target")):
            expected = root / "test_fixed_full" / folder / (row["sample_id"] + ".png")
            if not expected.is_file() or Path(row[field]).resolve() != expected.resolve():
                raise ValueError(f"Missing or mismatched {field}: {row['sample_id']}")
    for folder in ("input", "target"):
        names = {p.stem for p in (root / "test_fixed_full" / folder).glob("*.png")}
        if names != set(ids):
            raise ValueError(f"Extra/missing images in {folder}")
    return pairs


def infer(root: Path, records: Path, pairs: list[dict], batch_size: int, device: str) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor

    output = records / "initial_predictions.csv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite initial predictions: {output}")
    torch.manual_seed(1234)
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).eval().to(device)
    if device.startswith("cuda"):
        model = model.half()
    names = list(CATEGORIES)
    texts = [text for info in CATEGORIES.values() for text in info["prompts"]]
    text_inputs = processor(text=texts, padding="max_length", truncation=True, max_length=model.config.text_config.max_position_embeddings, return_tensors="pt").to(device)
    with torch.inference_mode():
        text_features = F.normalize(model.get_text_features(**text_inputs).float(), dim=-1)

    def read_image(row: dict) -> Image.Image:
        with Image.open(row["target_path"]) as im:
            im = im.convert("RGB")
            im.thumbnail((768, 768), Image.Resampling.LANCZOS)
            return im

    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool, torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            group = pairs[start:start + batch_size]
            images = list(pool.map(read_image, group))
            inputs = processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device=device, dtype=next(model.parameters()).dtype)
            image_features = F.normalize(model.get_image_features(pixel_values=pixels).float(), dim=-1)
            similarity = image_features @ text_features.T
            # Four equal-weight descriptive prompts per class; average best two
            # matches so the presence of a small incidental object is insufficient.
            scores = similarity.reshape(len(group), len(names), 4).topk(2, dim=-1).values.mean(-1)
            values, ranks = scores.topk(2, dim=-1)
            for row, score, val, rank in zip(group, scores.cpu().tolist(), values.cpu().tolist(), ranks.cpu().tolist()):
                rows.append({
                    "sample_id": row["sample_id"], "source_index": row["source_index"],
                    "initial_category": names[rank[0]], "runner_up": names[rank[1]],
                    "similarity_margin": f"{val[0] - val[1]:.6f}",
                    **{f"similarity_{name}": f"{value:.6f}" for name, value in zip(names, score)},
                })
            print(f"Initial labels: {min(start + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    save_csv(output, rows, list(rows[0]))
    save_json(records / "classification_config.json", {
        "model": MODEL_ID, "revision": MODEL_REVISION, "seed": 1234,
        "categories": CATEGORIES, "scores_are_calibrated_probabilities": False,
        "image_basis": "existing cropped sharp target; no restoration results or PSNR/SSIM used",
        "pairs_sha256": digest(root / "test_fixed_full/pairs.tsv"),
    })


def make_previews(records: Path, pairs: list[dict]) -> None:
    suggestions = {r["sample_id"]: r for r in read_csv(records / "initial_predictions.csv")}
    preview_dir = records / "review_contact_sheets"
    preview_dir.mkdir(exist_ok=True)
    font = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 14)
    columns, per_page = 8, 64
    cell_w, cell_h = 244, 178
    for start in range(0, len(pairs), per_page):
        group = pairs[start:start + per_page]
        sheet = Image.new("RGB", (columns * cell_w, 38 + math.ceil(len(group) / columns) * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 5), f"GT场景复核 {start:04d}–{start + len(group) - 1:04d}  候选标签仅作辅助", font=font, fill="black")
        for offset, row in enumerate(group):
            x, y = (offset % columns) * cell_w, 38 + (offset // columns) * cell_h
            with Image.open(row["target_path"]) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (240, 137), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x + (240 - thumb.width) // 2, y))
            prediction = suggestions[row["sample_id"]]
            draw.text((x + 2, y + 138), f"{int(row['source_index']):04d} {prediction['initial_category']}", font=font, fill="black")
            draw.text((x + 2, y + 160), f"次选:{prediction['runner_up']} Δ{prediction['similarity_margin']}", font=small, fill="#555555")
        path = preview_dir / f"page_{start // per_page + 1:02d}_{start:04d}_{start + len(group) - 1:04d}.jpg"
        sheet.save(path, quality=91)
    print(f"Contact sheets saved: {preview_dir}", flush=True)


def load_review(records: Path, pairs: list[dict]) -> dict[str, dict]:
    review = json.loads((records / "visual_review.json").read_text(encoding="utf-8"))
    labels = {}
    for entry in review["ranges"]:
        start, end, category, reason = entry
        if category not in CATEGORIES and category != "待复核":
            raise ValueError(f"Unknown review category: {category}")
        for index in range(start, end + 1):
            if index < 0 or index >= len(pairs) or int(pairs[index]["source_index"]) != index:
                raise ValueError(f"Invalid source index {index}")
            key = pairs[index]["sample_id"]
            if key in labels:
                raise ValueError(f"Overlapping review ranges: {key}")
            labels[key] = {"category": category, "review_note": reason}
    if set(labels) != {row["sample_id"] for row in pairs}:
        raise ValueError("Visual review must cover every test pair exactly once")
    return labels


def make_final_previews(records: Path, pairs: list[dict]) -> None:
    labels = load_review(records, pairs)
    font = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 14)
    columns, per_page = 8, 64
    cell_w, cell_h = 244, 178
    for category in list(CATEGORIES) + ["待复核"]:
        selected = [row for row in pairs if labels[row["sample_id"]]["category"] == category]
        if not selected:
            continue
        destination = records / "final_scene_previews" / category
        destination.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(selected), per_page):
            group = selected[start:start + per_page]
            sheet = Image.new("RGB", (columns * cell_w, 38 + math.ceil(len(group) / columns) * cell_h), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 5), f"{category}  第{start // per_page + 1}/{math.ceil(len(selected) / per_page)}页  共{len(selected)}对  视觉复核后标签", font=font, fill="black")
            for offset, row in enumerate(group):
                x, y = (offset % columns) * cell_w, 38 + (offset // columns) * cell_h
                with Image.open(row["target_path"]) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (240, 137), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x + (240 - thumb.width) // 2, y))
                draw.text((x + 2, y + 138), f"{int(row['source_index']):04d} {category}", font=font, fill="black")
                # Stable source index maps directly to the full sample ID in CSV.
                draw.text((x + 2, y + 160), row["sample_id"][:29], font=small, fill="#555555")
            sheet.save(destination / f"page_{start // per_page + 1:02d}.jpg", quality=91)
        print(f"Final previews: {category} ({len(selected)} pairs)", flush=True)


def export(root: Path, records: Path, pairs: list[dict]) -> None:
    config = json.loads((records / "classification_config.json").read_text(encoding="utf-8"))
    pairs_hash = digest(root / "test_fixed_full/pairs.tsv")
    if config["pairs_sha256"] != pairs_hash:
        raise ValueError("Original pair manifest changed since initial classification")
    labels = load_review(records, pairs)
    counts = Counter(entry["category"] for entry in labels.values())
    names = list(CATEGORIES) + (["待复核"] if counts["待复核"] else [])
    for name in names:
        if (root / name).exists():
            raise FileExistsError(f"Refusing to overwrite an existing scene folder: {root / name}")
    initial = {row["sample_id"]: row for row in read_csv(records / "initial_predictions.csv")}
    for name in names:
        for kind in ("input", "gt"):
            (root / name / kind).mkdir(parents=True, exist_ok=False)

    def copy_pair(row: dict) -> dict:
        key = row["sample_id"]
        category = labels[key]["category"]
        record = {
            "sample_id": key, "source_index": row["source_index"],
            "category": category, "review_status": "assistant_visual_review" if category != "待复核" else "needs_user_review",
            "review_note": labels[key]["review_note"],
            "initial_category": initial[key]["initial_category"],
            "original_input": row["input_path"], "original_gt": row["target_path"],
            "source_image": row["source_path"], "case_name": row["case_name"],
        }
        for field, kind in (("input_path", "input"), ("target_path", "gt")):
            source = Path(row[field])
            destination = root / category / kind / source.name
            before = digest(source)
            shutil.copy2(source, destination)
            if digest(destination) != before or digest(source) != before:
                raise ValueError(f"Copy/source integrity check failed: {source}")
            record[f"scene_{kind}"] = str(destination)
            record[f"sha256_{kind}"] = before
        return record

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(copy_pair, pairs), start=1):
            rows.append(row)
            if index % 250 == 0 or index == len(pairs):
                print(f"Copied and SHA-256 verified: {index}/{len(pairs)} pairs", flush=True)
    if digest(root / "test_fixed_full/pairs.tsv") != pairs_hash:
        raise ValueError("Original manifest unexpectedly modified")
    for category in names:
        inp = {p.name for p in (root / category / "input").glob("*.png")}
        gt = {p.name for p in (root / category / "gt").glob("*.png")}
        expected = {row["sample_id"] + ".png" for row in rows if row["category"] == category}
        if inp != gt or inp != expected:
            raise ValueError(f"Export pair mismatch: {category}")
    save_csv(records / "scene_labels.csv", rows, list(rows[0]))
    save_csv(records / "scene_counts.csv", [
        {"category": name, "count": counts[name], "percentage": f"{100 * counts[name] / len(rows):.4f}"}
        for name in names
    ], ["category", "count", "percentage"])
    for name in names:
        chosen = [row for row in rows if row["category"] == name]
        save_csv(records / f"{name}_pairs.csv", chosen, list(rows[0]))
    save_json(records / "summary.json", {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_split": "test_fixed_full", "total_pairs": len(rows),
        "category_counts": {name: counts[name] for name in names},
        "pending_review": counts["待复核"],
        "reviewer": "AI assistant visual review; not independently human-annotated ground truth",
        "method": "zero-shot suggestions followed by source-ordered contact-sheet visual review and full-image checks for ambiguous samples",
        "labels_use_restoration_metrics": False,
        "copy_mode": "independent copies, not symlinks or hardlinks",
        "every_exported_file_sha256_matches_source": True,
        "original_pair_manifest_sha256": pairs_hash,
        "initial_predictions_sha256": digest(records / "initial_predictions.csv"),
        "visual_review_sha256": digest(records / "visual_review.json"),
        "scene_labels_sha256": digest(records / "scene_labels.csv"),
    })
    description = "\n".join(f"| {name} | {counts[name]} | {CATEGORIES[name]['description']} |" for name in CATEGORIES)
    readme = f"""# 固定测试集的场景分类

本目录记录对现有1790对测试样本的单标签场景分类，不重新划分训练/验证/测试集，不重新生成模糊，不改变原始测试图像。

| 场景文件夹 | 图像对数 | 归类依据 |
|---|---:|---|
{description}

正式场景仅有五类。“混合场所”不作为第六类；确实难判定的样本如有，放入“待复核”文件夹，本次待复核数：{counts['待复核']}。

## 文件组织

五个中文场景文件夹直接位于本目录的上一级，每个场景下有input（原模糊输入）和gt（原清晰参考）子目录。原文件名、分辨率、图像内容及配对关系保持不变。文件是独立副本，全部逐文件SHA-256校验通过；原test_fixed_full保持不变。

scene_labels.csv是最终样本级分类记录，包含原始路径、分类后路径、复核说明、初筛类别和文件SHA-256。各场景_pairs.csv提供相应测试清单。scene_counts.csv和summary.json保存数量统计。visual_review.json保存逐段复核决定；final_scene_previews按最终场景类别保存总览；review_contact_sheets仅保存初筛阶段总览，其中候选标签不应当作最终标签。

## 分类规则与来源

- 根据现有清晰参考图的主要场景及功能判定；不根据是否出现某个物体或单纯像素面积判定。天空不作为地表场景类别。
- 街道及停车空间为主体时归道路交通；建筑主体近景或建筑群为主体时归建筑区域；大范围连续植被主导的远景归自然植被；水面主导归水域；非停车用途的开放场地归开阔场地。
- 桥梁及桥面车辆本身是观察主体时归道路交通，不因河面面积大自动归水域。可识别的河槽、岸坡为主体时归水域，即使河槽水量很少。大面积裸地、稀疏植被空地归开阔场地。
- 同一序列逐张检查，不将一个序列直接全部赋予相同标签。每张图只计入一类，不记录次要标签，不根据类别数量调比例。
- 先用固定版本SigLIP2和固定文字描述作零样本初筛，再由AI助手逐页视觉复核，并对边界样本查看原图。相似度不是校准后的类别概率。此结果不是经过独立人工标注者一致性验证的标准标签；正式发表前建议研究者复核边界样本。
- 分类过程没有读取任何恢复模型的预测图像或PSNR/SSIM结果，不据恢复效果调整标签。
- 用户已讨论的示例：索引149按城市街道与停车功能归道路交通；索引1761按连续树林主导的园区远景归自然植被。
- 待复核不是第六种场景。归属未确定前，应单独说明其数量，不能在分场景评价时无说明地丢弃；完整测试集仍为1790对。

## 复现

脚本位于项目tools/classify_test_scenes.py。infer生成初筛，preview生成初筛复核图，final-preview生成最终类别总览，export读取visual_review.json后复制与校验。export拒绝覆盖已有场景文件夹。不要在未完成视觉复核时直接使用初筛类别进行论文统计。
"""
    (records / "README_分类说明.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"total": len(rows), "counts": dict(counts)}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["infer", "preview", "final-preview", "export", "check"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    root = args.root.resolve()
    pairs = load_pairs(root)
    records = root / RECORD_DIR
    records.mkdir(exist_ok=True)
    if args.command == "infer":
        infer(root, records, pairs, args.batch_size, args.device)
    elif args.command == "preview":
        make_previews(records, pairs)
    elif args.command == "final-preview":
        make_final_previews(records, pairs)
    elif args.command == "export":
        export(root, records, pairs)
    else:
        labels = load_review(records, pairs)
        print(Counter(entry["category"] for entry in labels.values()))


if __name__ == "__main__":
    main()

"""GroundingDINO + SAM2：从 UAV GT 图像自动生成实例与联合二值 mask。

默认流程：
    GT -> GroundingDINO 文本检测 -> 框过滤/NMS -> SAM2 框提示分割
       -> mask 质量与面积过滤 -> 保留所有合格实例 -> 保存实例与联合 0/255 PNG mask

本脚本只负责自动生成 mask，不执行背景补全和运动模糊。模型只在启动时加载
一次，既支持 ``--image`` 单张测试，也支持 ``--image-dir`` 批量处理。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# =========================================================================
# 1. 用户配置区
# =========================================================================
script_root = Path(__file__).resolve().parent
workspace_root = script_root.parent

image_dir = script_root / "dataset/image/2"
mask_output_dir = script_root / "output/mask"
instance_mask_output_dir = script_root / "output/mask_instances"
debug_output_dir = script_root / "output/mask_auto_debug"

groundingdino_root = workspace_root / "GroundingDINO"
dino_config_path = (
    groundingdino_root
    / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
dino_checkpoint_path = (
    groundingdino_root / "weights/groundingdino_swint_ogc.pth"
)

sam2_root = workspace_root / "sam2"
sam2_model_config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_checkpoint_path = sam2_root / "checkpoints/sam2.1_hiera_base_plus.pt"

text_prompt = "car. boat. ship."
box_threshold = 0.20
text_threshold = 0.20
nms_iou_threshold = 0.50
min_box_area_ratio = 0.002
# 不限制检测框的最大面积；1.0 表示允许检测框覆盖整张图像。
max_box_area_ratio = 1.0
border_margin_ratio = 0.01
min_mask_area_ratio = 0.002
max_mask_area_ratio = 0.25
min_sam_score = 0.80
max_candidates = 20
# 0 表示保留所有通过检测、NMS 和 SAM2 质量筛选的目标，并合并为联合 mask。
max_objects = 0
use_largest_component_only = True
save_debug = True
offline_mode = True
device_name = "cuda"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================================================================
# 2. 模型加载
# =========================================================================
def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description}不存在：{path}")


def resolve_device(requested_device: str) -> str:
    import torch

    requested_device = requested_device.strip().lower()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "指定了 CUDA，但 torch.cuda.is_available() 为 False。"
            "请先在当前终端运行 nvidia-smi 和 PyTorch CUDA 检查。"
        )
    return requested_device


def load_models(
    dino_config: Path,
    dino_checkpoint: Path,
    sam_config: str,
    sam_checkpoint: Path,
    device: str,
) -> tuple[Any, Any]:
    require_file(dino_config, "GroundingDINO 配置")
    require_file(dino_checkpoint, "GroundingDINO 权重")
    require_file(sam_checkpoint, "SAM2 权重")

    import torch
    from groundingdino.util.inference import load_model
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Load GroundingDINO on {device} ...")
    dino_model = load_model(str(dino_config), str(dino_checkpoint), device=device)

    print(f"Load SAM2 on {device} ...")
    sam_model = build_sam2(
        sam_config,
        str(sam_checkpoint),
        device=device,
    )
    sam_predictor = SAM2ImagePredictor(sam_model)
    print("Models loaded.")
    return dino_model, sam_predictor


# =========================================================================
# 3. 图像、检测框和 mask 处理
# =========================================================================
def list_input_images(single_image: Path | None, input_dir: Path) -> list[Path]:
    if single_image is not None:
        if not single_image.is_file():
            raise FileNotFoundError(f"输入图像不存在：{single_image}")
        if single_image.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的图像格式：{single_image}")
        return [single_image]

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入图像目录不存在：{input_dir}")
    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"目录中没有图像：{input_dir}")
    return images


def boxes_cxcywh_to_xyxy(
    normalized_boxes: Any, width: int, height: int
) -> Any:
    import torch
    from torchvision.ops import box_convert

    boxes = box_convert(normalized_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
    boxes = boxes * scale
    boxes[:, 0::2].clamp_(0.0, float(width - 1))
    boxes[:, 1::2].clamp_(0.0, float(height - 1))
    return boxes


def filter_dino_candidates(
    boxes: Any,
    scores: Any,
    phrases: list[str],
    width: int,
    height: int,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    margin_ratio: float,
    nms_threshold: float,
    candidate_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    from torchvision.ops import nms

    raw_candidates: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    image_area = float(width * height)
    margin_x = float(width) * margin_ratio
    margin_y = float(height) * margin_ratio

    for index, (box_tensor, score_tensor, phrase) in enumerate(
        zip(boxes, scores, phrases)
    ):
        box = [float(value) for value in box_tensor.tolist()]
        x1, y1, x2, y2 = box
        area_ratio = max(x2 - x1, 0.0) * max(y2 - y1, 0.0) / image_area
        reasons: list[str] = []
        if x2 <= x1 or y2 <= y1:
            reasons.append("invalid_box")
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            reasons.append("box_area")
        if (
            x1 < margin_x
            or y1 < margin_y
            or x2 > width - 1 - margin_x
            or y2 > height - 1 - margin_y
        ):
            reasons.append("near_boundary")

        raw_candidates.append(
            {
                "raw_index": index,
                "phrase": phrase,
                "dino_score": float(score_tensor),
                "box_xyxy": box,
                "box_area_ratio": float(area_ratio),
                "accepted_before_nms": not reasons,
                "rejection_reasons": reasons,
            }
        )
        if not reasons:
            valid_indices.append(index)

    if not valid_indices:
        return raw_candidates, []

    valid_index_tensor = torch.tensor(valid_indices, dtype=torch.long)
    kept_local = nms(
        boxes[valid_index_tensor], scores[valid_index_tensor], nms_threshold
    ).tolist()
    kept_indices = [valid_indices[index] for index in kept_local]
    kept_indices = kept_indices[:candidate_limit]
    selected_candidates = [dict(raw_candidates[index]) for index in kept_indices]
    for candidate in selected_candidates:
        candidate["passed_nms"] = True
    return raw_candidates, selected_candidates


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    if count <= 2:
        return mask.astype(bool)
    largest_label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def segment_candidates(
    predictor: Any,
    image_rgb: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    minimum_sam_score: float,
    largest_component_only: bool,
    object_limit: int,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    predictor.set_image(np.ascontiguousarray(image_rgb))
    image_area = float(image_rgb.shape[0] * image_rgb.shape[1])
    accepted: list[tuple[float, np.ndarray, dict[str, Any]]] = []

    for candidate in candidates:
        box = np.asarray(candidate["box_xyxy"], dtype=np.float32)
        masks, sam_scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )
        best_index = int(np.argmax(sam_scores))
        mask = np.asarray(masks[best_index], dtype=bool)
        if largest_component_only:
            mask = keep_largest_component(mask)

        mask_area = int(mask.sum())
        mask_area_ratio = mask_area / image_area
        sam_score = float(sam_scores[best_index])
        combined_score = float(candidate["dino_score"] * sam_score)
        result = dict(candidate)
        result.update(
            sam_score=sam_score,
            combined_score=combined_score,
            mask_area=mask_area,
            mask_area_ratio=float(mask_area_ratio),
        )
        rejection_reasons: list[str] = []
        if sam_score < minimum_sam_score:
            rejection_reasons.append("sam_score")
        if not min_area_ratio <= mask_area_ratio <= max_area_ratio:
            rejection_reasons.append("mask_area")
        result["mask_rejection_reasons"] = rejection_reasons
        if not rejection_reasons:
            accepted.append((combined_score, mask, result))

    accepted.sort(key=lambda item: item[0], reverse=True)
    if object_limit > 0:
        accepted = accepted[:object_limit]
    if not accepted:
        return np.zeros(image_rgb.shape[:2], dtype=bool), [], []

    union_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    instance_masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for instance_id, (_, mask, result) in enumerate(accepted, start=1):
        union_mask |= mask
        instance_masks.append(mask)
        result["instance_id"] = instance_id
        metadata.append(result)
    return union_mask, instance_masks, metadata


# =========================================================================
# 4. 输出 mask、诊断图和元数据
# =========================================================================
def write_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = mask.astype(np.uint8) * 255
    if not cv2.imwrite(str(path), data):
        raise OSError(f"无法写入 mask：{path}")


def safe_label_for_filename(label: str) -> str:
    """把 GroundingDINO phrase 转成稳定、便于查找的文件名片段。"""
    normalized = "".join(
        character.lower()
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in label.strip()
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "object"


def clear_instance_masks(instance_root: Path, image_stem: str) -> Path:
    """清理同一图像上一次生成的实例 mask，防止目标数减少后读到旧文件。"""
    image_instance_dir = instance_root / image_stem
    if image_instance_dir.is_dir():
        for old_mask in image_instance_dir.glob("obj_*.png"):
            old_mask.unlink()
    image_instance_dir.mkdir(parents=True, exist_ok=True)
    return image_instance_dir


def save_instance_masks(
    instance_root: Path,
    image_stem: str,
    instance_masks: list[np.ndarray],
    selected_objects: list[dict[str, Any]],
) -> list[Path]:
    """逐实例保存 mask，并把实际路径写回对应目标元数据。"""
    if len(instance_masks) != len(selected_objects):
        raise ValueError("实例 mask 数量与目标元数据数量不一致")
    image_instance_dir = clear_instance_masks(instance_root, image_stem)
    saved_paths: list[Path] = []
    for instance_id, (mask, selected) in enumerate(
        zip(instance_masks, selected_objects), start=1
    ):
        label = safe_label_for_filename(str(selected.get("phrase", "object")))
        mask_path = image_instance_dir / f"obj_{instance_id:03d}_{label}.png"
        write_binary_mask(mask_path, mask)
        selected["instance_id"] = instance_id
        selected["instance_mask"] = str(mask_path.resolve())
        saved_paths.append(mask_path)
    return saved_paths


def make_debug_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    raw_candidates: list[dict[str, Any]],
    selected_objects: list[dict[str, Any]],
) -> np.ndarray:
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = np.ascontiguousarray(overlay.copy())

    for candidate in raw_candidates:
        x1, y1, x2, y2 = [int(round(value)) for value in candidate["box_xyxy"]]
        color = (0, 180, 255) if candidate["accepted_before_nms"] else (80, 80, 220)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        label = f"{candidate['phrase']} {candidate['dino_score']:.2f}"
        cv2.putText(
            overlay,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    color_layer = np.zeros_like(overlay)
    color_layer[mask] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.42, 0.0)

    for selected in selected_objects:
        x1, y1, x2, y2 = [int(round(value)) for value in selected["box_xyxy"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return overlay


def save_debug_outputs(
    debug_dir: Path,
    image_stem: str,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    raw_candidates: list[dict[str, Any]],
    selected_objects: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = make_debug_overlay(
        image_rgb, mask, raw_candidates, selected_objects
    )
    overlay_path = debug_dir / f"{image_stem}_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"无法写入诊断图：{overlay_path}")
    metadata_path = debug_dir / f"{image_stem}_meta.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# =========================================================================
# 5. 主流程
# =========================================================================
def run_generation(arguments: argparse.Namespace) -> None:
    if arguments.offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    from groundingdino.util.inference import load_image, predict

    device = resolve_device(arguments.device)
    images = list_input_images(arguments.image, arguments.image_dir)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    arguments.instance_output_dir.mkdir(parents=True, exist_ok=True)
    if arguments.save_debug:
        arguments.debug_dir.mkdir(parents=True, exist_ok=True)

    dino_model, sam_predictor = load_models(
        arguments.dino_config,
        arguments.dino_checkpoint,
        arguments.sam_config,
        arguments.sam_checkpoint,
        device,
    )

    print(f"Found {len(images)} input image(s).")
    success_count = 0
    for image_index, image_path in enumerate(images, start=1):
        print(f"\n[{image_index}/{len(images)}] {image_path}")
        image_rgb, image_tensor = load_image(str(image_path))
        image_rgb = np.asarray(image_rgb, dtype=np.uint8).copy()
        height, width = image_rgb.shape[:2]

        boxes_normalized, dino_scores, phrases = predict(
            model=dino_model,
            image=image_tensor,
            caption=arguments.text_prompt,
            box_threshold=arguments.box_threshold,
            text_threshold=arguments.text_threshold,
            device=device,
        )
        boxes_xyxy = boxes_cxcywh_to_xyxy(
            boxes_normalized, width, height
        )
        raw_candidates, dino_candidates = filter_dino_candidates(
            boxes_xyxy,
            dino_scores,
            phrases,
            width,
            height,
            min_area_ratio=arguments.min_box_area_ratio,
            max_area_ratio=arguments.max_box_area_ratio,
            margin_ratio=arguments.border_margin_ratio,
            nms_threshold=arguments.nms_iou_threshold,
            candidate_limit=arguments.max_candidates,
        )
        mask, instance_masks, selected_objects = segment_candidates(
            sam_predictor,
            image_rgb,
            dino_candidates,
            min_area_ratio=arguments.min_mask_area_ratio,
            max_area_ratio=arguments.max_mask_area_ratio,
            minimum_sam_score=arguments.min_sam_score,
            largest_component_only=arguments.largest_component_only,
            object_limit=arguments.max_objects,
        )

        metadata: dict[str, Any] = {
            "image": str(image_path.resolve()),
            "image_height": height,
            "image_width": width,
            "text_prompt": arguments.text_prompt,
            "box_threshold": arguments.box_threshold,
            "text_threshold": arguments.text_threshold,
            "nms_iou_threshold": arguments.nms_iou_threshold,
            "raw_detections": raw_candidates,
            "dino_candidates_after_filter_and_nms": dino_candidates,
            "selected_objects": selected_objects,
            "final_mask_area": int(mask.sum()),
            "final_mask_area_ratio": float(mask.mean()),
        }

        if not mask.any():
            # 当前运行没有合格目标时清理旧实例，避免后续模糊脚本误读历史结果。
            clear_instance_masks(arguments.instance_output_dir, image_path.stem)
            stale_union_mask = arguments.output_dir / f"{image_path.stem}.png"
            if stale_union_mask.is_file():
                stale_union_mask.unlink()
            print(
                "  Warning: no object passed all filters; mask was not written. "
                "See debug metadata or lower thresholds."
            )
            if arguments.save_debug:
                save_debug_outputs(
                    arguments.debug_dir,
                    image_path.stem,
                    image_rgb,
                    mask,
                    raw_candidates,
                    selected_objects,
                    metadata,
                )
            continue

        mask_path = arguments.output_dir / f"{image_path.stem}.png"
        write_binary_mask(mask_path, mask)
        instance_paths = save_instance_masks(
            arguments.instance_output_dir,
            image_path.stem,
            instance_masks,
            selected_objects,
        )
        metadata["selected_objects"] = selected_objects
        metadata["instance_mask_dir"] = str(
            (arguments.instance_output_dir / image_path.stem).resolve()
        )
        if arguments.save_debug:
            save_debug_outputs(
                arguments.debug_dir,
                image_path.stem,
                image_rgb,
                mask,
                raw_candidates,
                selected_objects,
                metadata,
            )
        success_count += 1
        best = selected_objects[0]
        print(
            f"  Union mask saved: {mask_path} | objects={len(instance_paths)} "
            f"| area={int(mask.sum())} "
            f"({mask.mean():.4%}) | DINO={best['dino_score']:.4f} "
            f"| SAM2={best['sam_score']:.4f}"
        )
        print(
            "  Instance masks: "
            f"{arguments.instance_output_dir / image_path.stem}"
        )

    print(
        f"\nDone: {success_count}/{len(images)} mask(s) generated. "
        f"Output: {arguments.output_dir}"
    )

    del dino_model, sam_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image",
        type=Path,
        default=None,
        help="单张 GT 图像；提供后忽略 --image-dir。",
    )
    input_group.add_argument(
        "--image-dir",
        type=Path,
        default=image_dir,
        help="批量 GT 图像目录。",
    )
    parser.add_argument("--output-dir", type=Path, default=mask_output_dir)
    parser.add_argument(
        "--instance-output-dir",
        type=Path,
        default=instance_mask_output_dir,
        help="逐实例 mask 根目录；每张图像保存到一个同 stem 子目录。",
    )
    parser.add_argument("--debug-dir", type=Path, default=debug_output_dir)
    parser.add_argument("--text-prompt", default=text_prompt)
    parser.add_argument("--box-threshold", type=float, default=box_threshold)
    parser.add_argument("--text-threshold", type=float, default=text_threshold)
    parser.add_argument(
        "--nms-iou-threshold", type=float, default=nms_iou_threshold
    )
    parser.add_argument(
        "--min-box-area-ratio", type=float, default=min_box_area_ratio
    )
    parser.add_argument(
        "--max-box-area-ratio", type=float, default=max_box_area_ratio
    )
    parser.add_argument(
        "--border-margin-ratio", type=float, default=border_margin_ratio
    )
    parser.add_argument(
        "--min-mask-area-ratio", type=float, default=min_mask_area_ratio
    )
    parser.add_argument(
        "--max-mask-area-ratio", type=float, default=max_mask_area_ratio
    )
    parser.add_argument("--min-sam-score", type=float, default=min_sam_score)
    parser.add_argument("--max-candidates", type=int, default=max_candidates)
    parser.add_argument(
        "--max-objects",
        type=int,
        default=max_objects,
        help="最终保留目标数；0 表示保留所有合格目标。",
    )
    parser.add_argument(
        "--largest-component-only",
        action=argparse.BooleanOptionalAction,
        default=use_largest_component_only,
    )
    parser.add_argument(
        "--save-debug",
        action=argparse.BooleanOptionalAction,
        default=save_debug,
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=offline_mode,
    )
    parser.add_argument("--device", default=device_name)
    parser.add_argument("--dino-config", type=Path, default=dino_config_path)
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=dino_checkpoint_path
    )
    parser.add_argument("--sam-config", default=sam2_model_config)
    parser.add_argument(
        "--sam-checkpoint", type=Path, default=sam2_checkpoint_path
    )
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    ratio_names = (
        "box_threshold",
        "text_threshold",
        "nms_iou_threshold",
        "min_box_area_ratio",
        "max_box_area_ratio",
        "border_margin_ratio",
        "min_mask_area_ratio",
        "max_mask_area_ratio",
        "min_sam_score",
    )
    for name in ratio_names:
        value = float(getattr(arguments, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须在 [0, 1] 内")
    if arguments.min_box_area_ratio > arguments.max_box_area_ratio:
        raise ValueError("min box area ratio 不能大于 max box area ratio")
    if arguments.min_mask_area_ratio > arguments.max_mask_area_ratio:
        raise ValueError("min mask area ratio 不能大于 max mask area ratio")
    if arguments.max_candidates < 1:
        raise ValueError("--max-candidates 必须至少为 1")
    if arguments.max_objects < 0:
        raise ValueError("--max-objects 不能小于 0")


def main() -> None:
    arguments = build_parser().parse_args()
    validate_arguments(arguments)
    run_generation(arguments)


if __name__ == "__main__":
    main()

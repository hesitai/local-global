"""自动生成独立目标运动方案 A 的第一轮测试数据。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.matplotlib_cn import configure_chinese_font

configure_chinese_font()


def make_test_data(
    root: str | Path = "dataset/object_motion_A",
    height: int = 384,
    width: int = 512,
    show: bool = True,
) -> tuple[Path, Path, Path]:
    root = Path(root)
    image_dir = root / "image"
    mask_dir = root / "mask"
    background_dir = root / "background_gt"
    for folder in (image_dir, mask_dir, background_dir):
        folder.mkdir(parents=True, exist_ok=True)

    x, y = np.meshgrid(
        np.arange(1, width + 1, dtype=np.float64),
        np.arange(1, height + 1, dtype=np.float64),
    )
    background = np.zeros((height, width, 3), dtype=np.float64)
    background[..., 0] = 0.22 + 0.18 * x / width
    background[..., 1] = 0.28 + 0.15 * y / height
    background[..., 2] = 0.35 + 0.10 * (x + y) / (width + height)
    for x_position in range(32, width + 1, 32):
        start, stop = max(0, x_position - 2), min(width, x_position + 1)
        background[:, start:stop, :] *= 0.85
    for y_position in range(32, height + 1, 32):
        start, stop = max(0, y_position - 2), min(height, y_position + 1)
        background[start:stop, :, :] *= 0.85
    background = np.clip(background, 0.0, 1.0)

    center_x = round(width * 0.50)
    center_y = round(height * 0.52)
    object_width, object_height, corner_radius = 92, 54, 12
    x1 = center_x - object_width // 2
    x2 = center_x + object_width // 2
    y1 = center_y - object_height // 2
    y2 = center_y + object_height // 2
    rectangle_mask = (
        (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    ).astype(np.uint8)
    kernel_size = 2 * corner_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    object_mask = cv2.morphologyEx(rectangle_mask, cv2.MORPH_OPEN, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel).astype(bool)

    foreground = np.zeros((height, width, 3), dtype=np.float64)
    foreground[..., 0] = 0.92 * object_mask
    foreground[..., 1] = 0.24 * object_mask
    foreground[..., 2] = 0.10 * object_mask
    stripe_mask = (
        (y >= center_y - 5)
        & (y <= center_y + 5)
        & (x >= x1 + 12)
        & (x <= x2 - 12)
        & object_mask
    )
    foreground[stripe_mask] = (1.00, 0.75, 0.20)
    marker_radius = 7
    marker_x = center_x + round(object_width * 0.25)
    marker_mask = (
        ((x - marker_x) ** 2 + (y - center_y) ** 2 <= marker_radius**2)
        & object_mask
    )
    foreground[marker_mask] = (0.10, 0.45, 0.95)

    gt = np.clip(background * (1.0 - object_mask[..., None]) + foreground, 0.0, 1.0)
    gt_path = image_dir / "0001.png"
    mask_path = mask_dir / "0001.png"
    background_path = background_dir / "0001.png"
    write_rgb(gt_path, gt)
    cv2.imwrite(str(mask_path), object_mask.astype(np.uint8) * 255)
    write_rgb(background_path, background)

    if show:
        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(gt)
        axes[0].set_title("自动生成的清晰 GT")
        axes[1].imshow(object_mask, cmap="gray")
        axes[1].set_title("自动生成的单目标 mask")
        axes[2].imshow(background)
        axes[2].set_title("真实背景（未放置目标）")
        for axis in axes:
            axis.axis("off")
        plt.show()

    print("\n测试数据生成完成：")
    print(f"GT：         {gt_path}")
    print(f"Mask：       {mask_path}")
    print(f"真实背景：   {background_path}")
    print("\n接下来可直接运行 script_uav_object_motion_blur_framework_A.py。")
    return gt_path, mask_path, background_path


def write_rgb(path: Path, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(
        np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    if not cv2.imwrite(str(path), bgr):
        raise OSError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/object_motion_A"))
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    make_test_data(arguments.data_root, arguments.height, arguments.width, arguments.show)


if __name__ == "__main__":
    main()

"""对应 ``demo_flow_global_blur_no_mex.m`` 的单图曝光积分演示。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from gen_mf import MotionParameters, gen_mf
from utils.matplotlib_cn import configure_chinese_font

configure_chinese_font()


def run_demo(
    image_path: str | Path,
    mode: int = 3,
    flow_scale: float = 1.0,
    num_steps: int = 21,
    output: str | Path | None = None,
    show: bool = True,
) -> Path:
    image = read_image_double(image_path)
    height, width = image.shape[:2]
    if mode == 1:
        parameters = MotionParameters(14, 5, 0, 0, 0, (height / 2, width / 2), 0)
        default_name = "demo_flow_blur_translation.png"
    elif mode == 2:
        parameters = MotionParameters(8, 3, 0.15, 0.15, 0, (height / 2, width / 2), np.pi / 180)
        default_name = "demo_flow_blur_translation_rotation.png"
    elif mode == 3:
        parameters = MotionParameters(0, 0, 0, 0, 0.0008, (height / 2, width / 2), 0)
        default_name = "demo_flow_blur_radial.png"
    else:
        raise ValueError("mode must be 1, 2, or 3")

    motion_field = gen_mf(height, width, parameters)
    motion_field.mu *= flow_scale
    motion_field.mv *= flow_scale
    x, y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    accumulator = np.zeros_like(image, dtype=np.float64)
    valid_count = np.zeros((height, width), dtype=np.float64)
    for alpha in np.linspace(-0.5, 0.5, num_steps):
        sample_x = x + alpha * motion_field.mu
        sample_y = y + alpha * motion_field.mv
        sampled = cv2.remap(
            image,
            sample_x.astype(np.float32),
            sample_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        accumulator += sampled
        valid_count += (
            (sample_x >= 0)
            & (sample_x <= width - 1)
            & (sample_y >= 0)
            & (sample_y <= height - 1)
        )
    if accumulator.ndim == 3:
        blur = accumulator / np.maximum(valid_count[..., None], 1.0)
    else:
        blur = accumulator / np.maximum(valid_count, 1.0)
    blur = np.clip(blur, 0.0, 1.0)
    flow_magnitude = np.sqrt(motion_field.mu**2 + motion_field.mv**2)

    output_path = Path(default_name if output is None else output)
    write_image_uint8(output_path, blur)
    if show:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes[0].set_title("原始清晰图")
        axes[1].imshow(blur, cmap="gray" if blur.ndim == 2 else None)
        axes[1].set_title("全局平台运动模糊")
        heatmap = axes[2].imshow(flow_magnitude)
        axes[2].set_title("Motion Flow 强度")
        figure.colorbar(heatmap, ax=axes[2])
        for axis in axes:
            axis.axis("off")
        plt.show()
    print(f"完成！已经生成：{output_path}")
    return output_path


def read_image_double(path: str | Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def write_image_uint8(path: str | Path, image: np.ndarray) -> None:
    output = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), output):
        raise OSError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("dataset/image/train/8143.jpg"))
    parser.add_argument("--mode", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--flow-scale", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=21)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    run_demo(arguments.image, arguments.mode, arguments.flow_scale, arguments.num_steps, arguments.output, arguments.show)


if __name__ == "__main__":
    main()

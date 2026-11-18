"""原 ``script_data_gen.m`` 的 Python 版本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat

from gen_blurry_img import gen_blurry_img
from gen_mf_para import MotionOptions, gen_mf_para
from utils.mfuv2mo import mfuv2mo


# =========================================================================
# 用户配置区
# =========================================================================
is_show = False
data_root = Path("dataset")
image_path = data_root / "image"
output_path = data_root / "data_syn"
splits = ("train", "test", "val")
q_type = "uv_basic"
noise_level = 0.001  # 原 MATLAB 在 [0,255] 数值域中使用的标准差
is_crop_boundary = True
motion_scale = 1.0
num_mf = 50
num_mf_m = 0


def run_generation(
    input_root: str | Path = image_path,
    output_root: str | Path = output_path,
    split_names: tuple[str, ...] = splits,
    *,
    label_type: str = q_type,
    number_of_flows: int = num_mf,
    number_of_mild_flows: int = num_mf_m,
    scale: float = motion_scale,
    crop_boundary: bool = is_crop_boundary,
    noise_sigma_255: float = noise_level,
    seed: int | None = None,
    show: bool = is_show,
) -> None:
    """每张图生成 ``number_of_flows`` 个模糊样本和一个清晰样本。"""
    strong_options = MotionOptions()
    mild_options = MotionOptions(
        tx_max=19.0,
        tx_acc_max=0.25,
        ty_max=19.0,
        ty_acc_max=0.25,
        tz_max=2e-3,
        cen_z_shift_max=5.0,
        rot_z_max=np.pi / 120.0,
        isblur=True,
    )
    rng = np.random.default_rng(seed)
    input_root = Path(input_root)
    output_root = Path(output_root)

    for split_index, split_name in enumerate(split_names, start=1):
        input_dir = input_root / split_name
        output_dir = output_root / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        # 原脚本只读取 jpg；这里扩展到常见图像格式，不改变算法。
        image_files = list_image_files(input_dir)
        for file_index, image_file in enumerate(image_files, start=1):
            image = read_image_double(image_file)
            height, width = image.shape[:2]
            for flow_index in range(1, number_of_flows + 2):
                print(
                    f"dir={split_index}, img={file_index}/{len(image_files)}, "
                    f"blur_idx={flow_index}/{number_of_flows + 1}"
                )
                if flow_index <= number_of_mild_flows:
                    options = mild_options
                elif flow_index <= number_of_flows:
                    options = strong_options
                else:
                    options = MotionOptions(isblur=False)

                motion_field, _ = gen_mf_para(height, width, options, rng)
                motion_field.mu = np.rint(motion_field.mu * scale)
                motion_field.mv = np.rint(motion_field.mv * scale)
                motion_field.mag, motion_field.ori = mfuv2mo(
                    -motion_field.mv, motion_field.mu
                )
                blurred = gen_blurry_img(image, motion_field)
                if noise_sigma_255 > 0:
                    blurred = np.clip(
                        blurred
                        + rng.normal(
                            0.0, noise_sigma_255 / 255.0, size=blurred.shape
                        ),
                        0.0,
                        1.0,
                    )

                mfmap = build_mfmap(motion_field, label_type)
                gt_image = image
                if crop_boundary:
                    bound_h = int(np.floor(np.max(np.abs(motion_field.mv)) / 2.0))
                    bound_w = int(np.floor(np.max(np.abs(motion_field.mu)) / 2.0))
                    blurred = crop_array(blurred, bound_h, bound_w)
                    mfmap = crop_array(mfmap, bound_h, bound_w)
                    gt_image = crop_array(gt_image, bound_h, bound_w)

                prefix = output_dir / f"{image_file.stem}_{flow_index:03d}"
                write_image_uint8(
                    prefix.with_name(prefix.name + "_gtimg.png"), gt_image
                )
                write_image_uint8(
                    prefix.with_name(prefix.name + "_blurryimg.png"), blurred
                )
                savemat(
                    prefix.with_name(prefix.name + "_mfmap.mat"),
                    {"mfmap": mfmap},
                    do_compression=True,
                )
                if show:
                    show_result(gt_image, blurred, motion_field.mu, motion_field.mv)


def list_image_files(folder: str | Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in suffixes)


def read_image_double(path: str | Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(path)
    if data.ndim == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA if data.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    data = data.astype(np.float64)
    return data / 255.0 if data.max(initial=0.0) > 1.0 else data


def write_image_uint8(path: str | Path, image: np.ndarray) -> None:
    output = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if output.ndim == 3:
        output = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA if output.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"Cannot write image: {path}")


def build_mfmap(motion_field, label_type: str) -> np.ndarray:
    mu, mv = motion_field.mu, motion_field.mv
    if label_type in {"uv_original", "uv_basic"}:
        return np.stack([mu, mv], axis=-1)
    if label_type == "uv_half":
        output_u, output_v = mu.copy(), mv.copy()
        negative = output_u < 0
        output_u[negative] *= -1
        output_v[negative] *= -1
        return np.stack([output_u, output_v], axis=-1)
    if label_type == "uv_half_multi":
        output_u, output_v = mu.copy(), mv.copy()
        negative = output_u < 0
        output_u[negative] *= -1
        output_v[negative] *= -1
        return np.stack(
            [
                output_u,
                output_v,
                output_u - np.mod(output_u, 2),
                output_v - np.mod(output_v, 2),
                output_u - np.mod(output_u - 1, 2),
                output_v - np.mod(output_v - 1, 2),
            ],
            axis=-1,
        )
    if label_type == "mo":
        return np.stack([motion_field.mag, motion_field.ori], axis=-1)
    raise ValueError(f"Wrong q type: {label_type}")


def crop_array(array: np.ndarray, bound_h: int, bound_w: int) -> np.ndarray:
    height, width = array.shape[:2]
    if bound_h * 2 >= height or bound_w * 2 >= width:
        return array
    rows = slice(bound_h, height - bound_h if bound_h else height)
    columns = slice(bound_w, width - bound_w if bound_w else width)
    return array[rows, columns, ...]


def show_result(gt: np.ndarray, blur: np.ndarray, mu: np.ndarray, mv: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt, cmap="gray" if gt.ndim == 2 else None)
    axes[0].set_title("GT")
    axes[1].imshow(blur, cmap="gray" if blur.ndim == 2 else None)
    axes[1].set_title("Blur")
    axes[2].imshow(np.sqrt(mu * mu + mv * mv))
    axes[2].set_title("Motion magnitude")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=image_path)
    parser.add_argument("--output-root", type=Path, default=output_path)
    parser.add_argument("--splits", default=",".join(splits))
    parser.add_argument("--q-type", default=q_type, choices=("uv_original", "uv_basic", "uv_half", "uv_half_multi", "mo"))
    parser.add_argument("--num-mf", type=int, default=num_mf)
    parser.add_argument("--num-mf-m", type=int, default=num_mf_m)
    parser.add_argument("--motion-scale", type=float, default=motion_scale)
    parser.add_argument("--noise-level", type=float, default=noise_level)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--crop-boundary", action=argparse.BooleanOptionalAction, default=is_crop_boundary)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=is_show)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    run_generation(
        arguments.input_root,
        arguments.output_root,
        tuple(item.strip() for item in arguments.splits.split(",") if item.strip()),
        label_type=arguments.q_type,
        number_of_flows=arguments.num_mf,
        number_of_mild_flows=arguments.num_mf_m,
        scale=arguments.motion_scale,
        crop_boundary=arguments.crop_boundary,
        noise_sigma_255=arguments.noise_level,
        seed=arguments.seed,
        show=arguments.show,
    )


if __name__ == "__main__":
    main()


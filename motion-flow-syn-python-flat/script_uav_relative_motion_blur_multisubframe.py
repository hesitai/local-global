"""UAV 平台运动 + 多实例目标自身运动的同步多子帧曝光积分。

输入仍是当前工程中的单张清晰 GT 和 GroundingDINO + SAM2 实例 mask。
每个曝光子时刻 ``tau`` 只构造一张完整的瞬时清晰图：

    background_linear_k = warp(background_linear, H_global(tau))
    object_linear_ik = warp(object_linear_i, H_global(tau) @ H_local_i(tau))
    frame_linear_k = composite(background_linear_k, object_linear_1k, ...)
    blurry_srgb = linear_to_srgb(mean(frame_linear_k))

矩阵乘法表示同一时刻的坐标变换复合，不是先做一次局部模糊、再做一次
全局模糊。目标的 RGB（预乘 alpha）和 mask 被打包成 RGBA 后一起 warp，
因此每个目标每个子帧只插值一次，且目标与 mask 始终同步。

本脚本复用 ``script_uav_object_motion_blur_framework_A.py`` 中已经验证过的：
图像/mask 读取、目标直线/曲线轨迹、mask 修正和背景补全。相对运动的核心
渲染流程完整保留在本文件中，原有 A 脚本和全局平台脚本不会被修改。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import savemat

import script_uav_object_motion_blur_framework_A as object_model
import uav_sensor_degradation as sensor_model


# =========================================================================
# 1. 用户配置区
# =========================================================================
script_root = Path(__file__).resolve().parent
base_seed = 20260721

image_dir = script_root / "dataset/image/2"
mask_dir = script_root / "output/mask"
instance_mask_dir = script_root / "output/mask_instances"
background_dir: Path | None = None
output_dir = script_root / "output/image/relative"

# 默认只生成一种平台匀速直线运动，避免一次运行默认生成五倍数据。
# 可用 --platform-cases 选择下方任意一种或多种；00_static 用于退化验证。
platform_cases = ("01_linear_uniform",)
SUPPORTED_PLATFORM_CASES = (
    "00_static",
    "01_linear_uniform",
    "02_linear_variable_speed",
    "03_hover_jitter_nonuniform",
    "04_maneuver_spatial_field",
    "05_high_frequency_vibration",
)

# 保持当前 A 脚本的目标模型：整批实例约 70% 直线、30% 曲线，每个实例
# 独立采样方向和 3%—10% 最长边的运动距离。
object_generation_case = "01_obj_independent_random"
linear_motion_probability = 0.70
linear_distance_ratio_range = (0.03, 0.10)
curved_path_ratio_range = (0.03, 0.10)
curved_turn_angle_deg_range = (2.0, 10.0)

# 固定 21 个子帧在速度与平滑度之间更可控。自适应采样仍可显式开启，且设置
# 上限，避免大图旋转时出现上百个全分辨率子帧。
N_subframes = 21
use_adaptive_subframes = False
adaptive_sample_step_px = 1.5
max_adaptive_subframes = 41

# GT 是曝光中心，曝光响应为均匀矩形快门。
gt_reference = "center"
save_debug_subframes = True
use_soft_mask = True
soft_mask_sigma = object_model.soft_mask_sigma
global_border_mode = "constant"
# 默认裁掉所有平台子帧不共同有效的区域，不再把镜像/黑色补边写入训练图。
is_crop_boundary = True
# 在最大公共有效矩形内部再收缩的安全像素，补偿双线性插值边缘误差。
crop_margin = 1
# 有效覆盖率达到该阈值，才认为像素完全来自真实输入背景。
valid_coverage_threshold = 1.0 - 1e-6
# 保留原 ``_blurryimg.png``，并额外输出同一曝光的传感器退化版本。
generate_sensor_variant = True  # True额外输出 ``_blurryimg_sensor.png``。
sensor_saturation_highlights = True  # 每个完整线性子帧恢复部分饱和高光能量。
sensor_saturation_threshold_low = 0.95  # 最大线性RGB通道达到该值后开始软增强。
sensor_saturation_threshold_high = 0.99  # 达到该值后使用完整高光增益。
sensor_saturation_gain_range = (0.25, 1.25)  # 一条曝光轨迹只采样一次。
sensor_poisson_gaussian_noise = True  # 曝光平均后只添加一次shot/read noise。
sensor_shot_noise_scale_range = (1.0e-5, 1.0e-4)  # Poisson光子噪声尺度。
sensor_read_noise_sigma_range = (3.0e-6, 5.0e-5)  # 线性伪传感器域读取噪声。
sensor_seed_offset = 8_000_000  # 与平台/目标运动seed分离并保持可复现。


# =========================================================================
# 2. 同一时刻的齐次矩阵与平台轨迹
# =========================================================================
def similarity_forward_matrix(
    dx: float,
    dy: float,
    theta_deg: float,
    scale_value: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """返回 3x3 前向矩阵：绕 ``(cx, cy)`` 缩放/旋转后再平移。

    图像坐标采用 x 向右、y 向下；正角在图像中视觉上为顺时针。
    OpenCV ``warpAffine`` 接收此前向矩阵，并在内部完成逆向采样。
    """
    if scale_value <= 0.0:
        raise ValueError("scale_value 必须大于 0")
    theta = float(np.deg2rad(theta_deg))
    cosine = float(np.cos(theta)) * scale_value
    sine = float(np.sin(theta)) * scale_value
    linear = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    center = np.asarray([cx, cy], dtype=np.float64)
    translation = center + np.asarray([dx, dy]) - linear @ center
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = translation
    return matrix


def make_centered_platform_trajectory(
    case_name: str,
    case_seed: int,
    strength_scale: float = 1.0,
) -> dict[str, Any]:
    """建立以 tau=0 为 GT 的平台图像运动轨迹。

    参数描述的是场景在成像平面上的运动，而不是无人机世界坐标位姿；因此若
    已知真实相机位姿，需先投影成图像平面变换再替换这里的轨迹。
    """
    if strength_scale < 0.0:
        raise ValueError("platform strength scale 不能小于 0")
    trajectory: dict[str, Any] = {
        "case_name": case_name,
        "case_seed": int(case_seed),
        "strength_scale": float(strength_scale),
    }
    scale = float(strength_scale)
    if case_name == "00_static":
        trajectory.update(description="static platform")
    elif case_name == "01_linear_uniform":
        trajectory.update(
            description="center-referenced constant-speed platform translation",
            total_dx=18.0 * scale,
            total_dy=-7.0 * scale,
        )
    elif case_name == "02_linear_variable_speed":
        trajectory.update(
            description="center-referenced variable-speed platform translation",
            total_dx=-22.0 * scale,
            total_dy=8.0 * scale,
            beta=2.0,
        )
    elif case_name == "03_hover_jitter_nonuniform":
        trajectory.update(
            description="platform drift plus low-frequency jitter",
            drift_dx=3.0 * scale,
            drift_dy=2.0 * scale,
            amp_dx=2.0 * scale,
            amp_dy=2.0 * scale,
            freq_dx=1.25,
            freq_dy=1.75,
            theta_amp_deg=0.6 * scale,
            theta_freq=1.0,
        )
    elif case_name == "04_maneuver_spatial_field":
        trajectory.update(
            description="platform translation, rotation and zoom",
            total_dx=8.0 * scale,
            total_dy=-5.0 * scale,
            total_theta_deg=8.0 * scale,
            total_zoom=0.08 * scale,
        )
    elif case_name == "05_high_frequency_vibration":
        trajectory.update(
            description="high-frequency vertical platform vibration",
            vib_amp=3.0 * scale,
            vib_amp2=0.8 * scale,
            vib_freq=3.0,
            vib_freq2=5.0,
        )
    else:
        raise ValueError(
            f"未知平台运动 {case_name}；可选值：{SUPPORTED_PLATFORM_CASES}"
        )
    return trajectory


def evaluate_platform_trajectory_at_time(
    trajectory: dict[str, Any], tau: float
) -> tuple[float, float, float, float]:
    """计算中心参考平台轨迹；tau=-0.5/0/0.5 为曝光首/中/末。"""
    case_name = str(trajectory["case_name"])
    if case_name == "00_static":
        return 0.0, 0.0, 0.0, 1.0
    if case_name == "01_linear_uniform":
        return (
            float(trajectory["total_dx"] * tau),
            float(trajectory["total_dy"] * tau),
            0.0,
            1.0,
        )
    if case_name == "02_linear_variable_speed":
        beta = float(trajectory["beta"])
        # 这是关于 tau=0 的奇函数。分母使 f(0.5)-f(-0.5)=1，因而
        # total_dx/total_dy 仍表示曝光首帧到末帧的总位移。
        normalized_time = (tau + beta * tau**3) / (1.0 + 0.25 * beta)
        return (
            float(trajectory["total_dx"] * normalized_time),
            float(trajectory["total_dy"] * normalized_time),
            0.0,
            1.0,
        )
    if case_name == "03_hover_jitter_nonuniform":
        dx = trajectory["drift_dx"] * tau + trajectory["amp_dx"] * np.sin(
            2.0 * np.pi * trajectory["freq_dx"] * tau
        )
        dy = trajectory["drift_dy"] * tau + trajectory["amp_dy"] * np.sin(
            2.0 * np.pi * trajectory["freq_dy"] * tau
        )
        theta = trajectory["theta_amp_deg"] * np.sin(
            2.0 * np.pi * trajectory["theta_freq"] * tau
        )
        return float(dx), float(dy), float(theta), 1.0
    if case_name == "04_maneuver_spatial_field":
        return (
            float(trajectory["total_dx"] * tau),
            float(trajectory["total_dy"] * tau),
            float(trajectory["total_theta_deg"] * tau),
            float(1.0 + trajectory["total_zoom"] * tau),
        )
    if case_name == "05_high_frequency_vibration":
        dy = trajectory["vib_amp"] * np.sin(
            2.0 * np.pi * trajectory["vib_freq"] * tau
        )
        dy += trajectory["vib_amp2"] * np.sin(
            2.0 * np.pi * trajectory["vib_freq2"] * tau
        )
        return 0.0, float(dy), 0.0, 1.0
    raise ValueError(f"未知平台运动：{case_name}")


def platform_matrix_at_time(
    trajectory: dict[str, Any],
    tau: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    parameters = evaluate_platform_trajectory_at_time(trajectory, tau)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return similarity_forward_matrix(*parameters, cx, cy), parameters


def local_object_matrix_at_time(
    trajectory: dict[str, Any], tau: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    parameters = object_model.evaluate_object_trajectory_at_time(trajectory, tau)
    cx = float(trajectory["object_centroid"][0] - 1.0)
    cy = float(trajectory["object_centroid"][1] - 1.0)
    return similarity_forward_matrix(*parameters, cx, cy), parameters


# =========================================================================
# 3. mask 与采样数量辅助函数
# =========================================================================
def split_union_mask_into_instances(mask: np.ndarray) -> list[np.ndarray]:
    """没有逐实例文件时，把联合 mask 的连通域拆成独立目标。"""
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    components: list[tuple[int, np.ndarray]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > 0:
            components.append((area, labels == label))
    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in components]


def load_instance_masks(
    image_file: Path,
    union_mask_root: Path,
    instance_mask_root: Path,
    mask_mode: str,
) -> tuple[list[np.ndarray], list[str], str]:
    instance_paths = object_model.list_instance_mask_paths(
        instance_mask_root, image_file
    )
    if instance_paths:
        masks = [
            object_model.read_binary_mask(path, True) for path in instance_paths
        ]
        return masks, [str(path) for path in instance_paths], "instance_masks"

    union_path = object_model.build_mask_path(
        union_mask_root, image_file, mask_mode
    )
    if not union_path.is_file():
        return [], [], "missing"
    union_mask = object_model.read_binary_mask(union_path, False)
    components = split_union_mask_into_instances(union_mask)
    sources = [f"{union_path}#component_{index:03d}" for index in range(1, len(components) + 1)]
    return components, sources, "union_mask_connected_components"


def count_available_instances(
    image_files: list[Path],
    union_mask_root: Path,
    instance_mask_root: Path,
    mask_mode: str,
) -> int:
    count = 0
    for image_file in image_files:
        instance_paths = object_model.list_instance_mask_paths(
            instance_mask_root, image_file
        )
        if instance_paths:
            count += len(instance_paths)
            continue
        union_path = object_model.build_mask_path(
            union_mask_root, image_file, mask_mode
        )
        if union_path.is_file():
            union = object_model.read_binary_mask(union_path, False)
            count += len(split_union_mask_into_instances(union))
    return count


def transform_points(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    homogeneous = np.column_stack(
        [points_xy, np.ones(points_xy.shape[0], dtype=np.float64)]
    )
    transformed = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def estimate_max_composed_path_length(
    width: int,
    height: int,
    platform_trajectory: dict[str, Any],
    object_trajectories: list[dict[str, Any]],
    sample_count: int = 33,
) -> float:
    """估计画面角点和目标框角点在曝光期间的最大路径长度。"""
    tau_list = np.linspace(-0.5, 0.5, sample_count, dtype=np.float64)
    image_corners = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]]
    )
    point_sets: list[tuple[np.ndarray, dict[str, Any] | None]] = [
        (image_corners, None)
    ]
    for trajectory in object_trajectories:
        x, y, box_width, box_height = np.asarray(
            trajectory["object_bbox"], dtype=np.float64
        )
        # object_bbox 是 MATLAB 风格像素边界坐标，转换成近似像素中心角点。
        points = np.asarray(
            [
                [x - 0.5, y - 0.5],
                [x + box_width - 1.5, y - 0.5],
                [x - 0.5, y + box_height - 1.5],
                [x + box_width - 1.5, y + box_height - 1.5],
            ]
        )
        point_sets.append((points, trajectory))

    maximum_path = 0.0
    for points, object_trajectory in point_sets:
        previous: np.ndarray | None = None
        path_lengths = np.zeros(points.shape[0], dtype=np.float64)
        for tau in tau_list:
            global_matrix, _ = platform_matrix_at_time(
                platform_trajectory, float(tau), width, height
            )
            if object_trajectory is None:
                composed = global_matrix
            else:
                local_matrix, _ = local_object_matrix_at_time(
                    object_trajectory, float(tau)
                )
                composed = global_matrix @ local_matrix
            current = transform_points(composed, points)
            if previous is not None:
                path_lengths += np.linalg.norm(current - previous, axis=1)
            previous = current
        maximum_path = max(maximum_path, float(path_lengths.max(initial=0.0)))
    return maximum_path


def choose_subframe_count(
    requested_count: int,
    adaptive: bool,
    max_path_length: float,
    sample_step_px: float,
    maximum_count: int,
) -> int:
    if requested_count < 3:
        raise ValueError("--n-subframes 不能小于 3")
    count = requested_count
    if adaptive:
        if sample_step_px <= 0.0:
            raise ValueError("--adaptive-sample-step-px 必须大于 0")
        count = max(3, int(np.ceil(max_path_length / sample_step_px)) + 1)
        count = min(count, maximum_count)
    # center GT 需要唯一的 tau=0 样本，因此强制奇数。达到上限时向下取奇数。
    if count % 2 == 0:
        count = count + 1 if count < maximum_count or not adaptive else count - 1
    return max(3, count)


# =========================================================================
# 4. 相对运动核心：同一 tau 组合矩阵，只做一次时间平均
# =========================================================================
def cv_border_mode(name: str) -> int:
    mapping = {
        "reflect101": cv2.BORDER_REFLECT_101,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
        "constant": cv2.BORDER_CONSTANT,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"未知 border mode：{name}") from error


def warp_affine_forward(
    source: np.ndarray,
    forward_matrix: np.ndarray,
    output_size: tuple[int, int],
    *,
    border_mode: int,
) -> np.ndarray:
    """使用前向矩阵只进行一次双线性重采样。"""
    return cv2.warpAffine(
        np.asarray(source, dtype=np.float32),
        np.asarray(forward_matrix[:2, :], dtype=np.float64),
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=0.0,
    )


def largest_valid_rectangle(
    valid_mask: np.ndarray,
    *,
    safety_margin: int = 0,
) -> tuple[int, int, int, int]:
    """寻找公共有效mask中面积最大的非对称轴对齐矩形。

    返回零基半开坐标 ``(top, bottom, left, right)``。面积相同时优先保留
    更接近原图中心的矩形，避免无意义地把主体裁到画面边缘。
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 2 or not valid_mask.any():
        raise ValueError("平台运动的公共有效区域为空")
    if safety_margin < 0:
        raise ValueError("crop safety margin 不能小于0")

    height, width = valid_mask.shape
    histogram = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_center_distance = np.inf
    best_rectangle = (0, 0, 0, 0)
    image_center_y = (height - 1.0) / 2.0
    image_center_x = (width - 1.0) / 2.0

    for bottom_inclusive in range(height):
        histogram = np.where(
            valid_mask[bottom_inclusive], histogram + 1, 0
        )
        stack: list[tuple[int, int]] = []
        for column in range(width + 1):
            current_height = int(histogram[column]) if column < width else 0
            start_column = column
            while stack and stack[-1][1] > current_height:
                left, rectangle_height = stack.pop()
                start_column = left
                top = bottom_inclusive - rectangle_height + 1
                bottom = bottom_inclusive + 1
                right = column
                area = rectangle_height * (right - left)
                center_y = (top + bottom - 1.0) / 2.0
                center_x = (left + right - 1.0) / 2.0
                center_distance = (
                    (center_y - image_center_y) ** 2
                    + (center_x - image_center_x) ** 2
                )
                if area > best_area or (
                    area == best_area
                    and center_distance < best_center_distance
                ):
                    best_area = area
                    best_center_distance = center_distance
                    best_rectangle = (top, bottom, left, right)
            if (
                current_height > 0
                and (not stack or stack[-1][1] < current_height)
            ):
                stack.append((start_column, current_height))

    top, bottom, left, right = best_rectangle
    top += safety_margin
    bottom -= safety_margin
    left += safety_margin
    right -= safety_margin
    if top >= bottom or left >= right:
        raise ValueError(
            "公共有效矩形应用crop_margin后为空；请减小平台运动或crop_margin"
        )
    if not valid_mask[top:bottom, left:right].all():
        raise RuntimeError("内部错误：公共有效裁剪框中仍包含无效像素")
    return top, bottom, left, right


def crop_rectangle(
    array: np.ndarray,
    rectangle: tuple[int, int, int, int],
) -> np.ndarray:
    """按零基半开矩形裁剪二维或多通道数组。"""
    top, bottom, left, right = rectangle
    return array[top:bottom, left:right, ...]


def transform_matrices_to_cropped_coordinates(
    matrices: np.ndarray,
    rectangle: tuple[int, int, int, int],
) -> np.ndarray:
    """执行 ``H_crop=C@H_full@inv(C)``，将矩阵改到裁后坐标系。"""
    top, _, left, _ = rectangle
    crop_translation = np.asarray(
        [[1.0, 0.0, -left], [0.0, 1.0, -top], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    inverse_crop_translation = np.asarray(
        [[1.0, 0.0, left], [0.0, 1.0, top], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrices = np.asarray(matrices, dtype=np.float64)
    return np.einsum(
        "ij,...jk,kl->...il",
        crop_translation,
        matrices,
        inverse_crop_translation,
    )


def add_cropped_coordinate_metadata(
    metadata: dict[str, Any],
    rectangle: tuple[int, int, int, int],
) -> None:
    """保留全图矩阵，并额外记录可直接用于裁剪输出的矩阵。"""
    metadata["H_global_cropped"] = transform_matrices_to_cropped_coordinates(
        metadata["H_global"], rectangle
    )
    for object_metadata in metadata["objects"]:
        object_metadata["H_local_cropped"] = (
            transform_matrices_to_cropped_coordinates(
                object_metadata["H_local"], rectangle
            )
        )
        object_metadata["H_composed_cropped"] = (
            transform_matrices_to_cropped_coordinates(
                object_metadata["H_composed_global_at_local"], rectangle
            )
        )


def synthesize_relative_motion_blur(
    reference_image: np.ndarray,
    background: np.ndarray,
    foregrounds: list[np.ndarray],
    object_masks_soft: list[np.ndarray],
    object_trajectories: list[dict[str, Any]],
    platform_trajectory: dict[str, Any],
    n_subframes: int,
    *,
    border_mode_name: str = global_border_mode,
    crop_safety_margin: int = crop_margin,
    valid_threshold: float = valid_coverage_threshold,
    sensor_parameters: sensor_model.SensorSampleParameters | None = None,
) -> tuple[Any, ...]:
    """在线性光强空间同步渲染平台和目标运动，并做一次均匀平均。

    公开输入和输出仍是 sRGB；alpha/mask 不做 gamma 转换。
    """
    if n_subframes < 3 or n_subframes % 2 == 0:
        raise ValueError("中心参考曝光要求子帧数是不小于 3 的奇数")
    if crop_safety_margin < 0:
        raise ValueError("crop_safety_margin 不能小于0")
    if not 0.0 < valid_threshold <= 1.0:
        raise ValueError("valid_threshold 必须位于 (0,1]")
    object_count = len(object_trajectories)
    if not (
        len(foregrounds) == len(object_masks_soft) == object_count
    ):
        raise ValueError("foreground、mask 和目标轨迹数量不一致")

    reference_srgb = np.asarray(reference_image, dtype=np.float64)
    background_srgb = np.asarray(background, dtype=np.float64)
    if reference_srgb.shape != background_srgb.shape:
        raise ValueError("GT 与背景尺寸不一致")
    reference = object_model.srgb_to_linear_uav(reference_srgb).astype(
        np.float32
    )
    background_source = object_model.srgb_to_linear_uav(
        background_srgb
    ).astype(np.float32)
    height, width = reference.shape[:2]
    output_size = (width, height)
    border_mode = cv_border_mode(border_mode_name)

    # 输入 RGB 是 sRGB 预乘 alpha。必须先解除预乘、转线性、再重新预乘；
    # 随后与 alpha 合并成 RGBA，只 warp 一次，避免 RGB/mask 边缘错位。
    object_rgba_sources = [
        np.dstack(
            [
                object_model.premultiplied_srgb_to_linear_uav(
                    foreground, mask
                ).astype(np.float32),
                np.asarray(mask, dtype=np.float32),
            ]
        )
        for foreground, mask in zip(foregrounds, object_masks_soft)
    ]

    tau_list = np.linspace(-0.5, 0.5, n_subframes, dtype=np.float64)
    exposure_weights = np.full(
        n_subframes, 1.0 / n_subframes, dtype=np.float64
    )
    middle_index = n_subframes // 2
    accumulator = np.zeros_like(reference, dtype=np.float64)
    sensor_accumulator = (
        np.zeros_like(reference, dtype=np.float64)
        if sensor_parameters is not None
        else None
    )
    valid_source = np.ones((height, width), dtype=np.float32)
    pack_background_and_valid = border_mode == cv2.BORDER_CONSTANT
    background_with_valid = (
        np.dstack((background_source, valid_source))
        if pack_background_and_valid
        else None
    )
    common_valid_coverage = np.ones((height, width), dtype=np.float32)
    debug_frames: dict[str, np.ndarray] = {}

    global_parameters = np.zeros((n_subframes, 4), dtype=np.float64)
    global_matrices = np.zeros((n_subframes, 3, 3), dtype=np.float64)
    object_parameters = [
        np.zeros((n_subframes, 4), dtype=np.float64)
        for _ in object_trajectories
    ]
    local_matrices = [
        np.zeros((n_subframes, 3, 3), dtype=np.float64)
        for _ in object_trajectories
    ]
    composed_matrices = [
        np.zeros((n_subframes, 3, 3), dtype=np.float64)
        for _ in object_trajectories
    ]

    for subframe_index, tau in enumerate(tau_list):
        global_matrix, global_parameter = platform_matrix_at_time(
            platform_trajectory, float(tau), width, height
        )
        global_parameters[subframe_index] = global_parameter
        global_matrices[subframe_index] = global_matrix

        if pack_background_and_valid:
            assert background_with_valid is not None
            warped_background_with_valid = warp_affine_forward(
                background_with_valid,
                global_matrix,
                output_size,
                border_mode=cv2.BORDER_CONSTANT,
            )
            background_now = warped_background_with_valid[..., :3]
            valid_now = warped_background_with_valid[..., 3]
        else:
            background_now = warp_affine_forward(
                background_source,
                global_matrix,
                output_size,
                border_mode=border_mode,
            )
            # 有效性图始终使用常量0边界，与用户选择的显示补边模式无关。
            valid_now = warp_affine_forward(
                valid_source,
                global_matrix,
                output_size,
                border_mode=cv2.BORDER_CONSTANT,
            )
        frame = background_now

        # 先 local 后 global 只体现在同一时刻的矩阵乘法 G @ T。随后直接用
        # 组合后的矩阵对原始目标 RGBA 做一次 warp，没有第二次子帧平均。
        for object_index, (rgba_source, trajectory) in enumerate(
            zip(object_rgba_sources, object_trajectories)
        ):
            local_matrix, local_parameter = local_object_matrix_at_time(
                trajectory, float(tau)
            )
            composed_matrix = global_matrix @ local_matrix
            object_parameters[object_index][subframe_index] = local_parameter
            local_matrices[object_index][subframe_index] = local_matrix
            composed_matrices[object_index][subframe_index] = composed_matrix

            rgba_now = warp_affine_forward(
                rgba_source,
                composed_matrix,
                output_size,
                border_mode=cv2.BORDER_CONSTANT,
            )
            alpha_now = np.clip(rgba_now[..., 3], 0.0, 1.0)
            frame = (
                (1.0 - alpha_now[..., None]) * frame
                + rgba_now[..., :3]
            )

        # tau=0 时 G=I 且所有 T_i=I。直接采用实测 GT，既消除 soft mask
        # 边缘的二次混合误差，也明确保证训练 GT 就是曝光中心清晰帧。
        if subframe_index == middle_index:
            frame = reference.copy()
            valid_now = valid_source

        frame = np.clip(frame, 0.0, 1.0)
        accumulator += exposure_weights[subframe_index] * frame
        if sensor_accumulator is not None:
            assert sensor_parameters is not None
            sensor_frame = (
                sensor_model.restore_saturated_highlights_linear(
                    frame, sensor_parameters
                )
            )
            sensor_accumulator += (
                exposure_weights[subframe_index] * sensor_frame
            )
        common_valid_coverage = np.minimum(
            common_valid_coverage, valid_now
        )

        if subframe_index == 0:
            debug_frames["first"] = frame.copy()
        elif subframe_index == middle_index:
            debug_frames["mid"] = frame.copy()
        elif subframe_index == n_subframes - 1:
            debug_frames["last"] = frame.copy()

    object_metadata: list[dict[str, Any]] = []
    for index, trajectory in enumerate(object_trajectories):
        object_metadata.append(
            {
                "instance_id": int(trajectory.get("instance_id", index + 1)),
                "instance_mask_file": str(
                    trajectory.get("instance_mask_file", "")
                ),
                "case_name": str(trajectory["case_name"]),
                "case_seed": int(trajectory["case_seed"]),
                "motion_strength_ratio": float(
                    trajectory["motion_strength_ratio"]
                ),
                "object_bbox": trajectory["object_bbox"],
                "object_centroid": trajectory["object_centroid"],
                "local_parameters_dx_dy_theta_scale": object_parameters[index],
                "H_local": local_matrices[index],
                "H_composed_global_at_local": composed_matrices[index],
            }
        )

    common_valid_mask = common_valid_coverage >= valid_threshold
    applied_crop_margin = (
        int(crop_safety_margin) if not common_valid_mask.all() else 0
    )
    crop_rectangle_value = largest_valid_rectangle(
        common_valid_mask,
        safety_margin=applied_crop_margin,
    )
    crop_top, crop_bottom, crop_left, crop_right = crop_rectangle_value
    cropped_coverage = common_valid_coverage[
        crop_top:crop_bottom, crop_left:crop_right
    ]

    metadata: dict[str, Any] = {
        "scheme_name": "relative_motion_single_exposure_integration",
        "formation_model": (
            "frame_linear(t)=composite(W(B_linear,G(t)), "
            "W(F_linear_i,G(t)@T_i(t))); "
            "B_srgb=g(uniform_mean_t(frame_linear(t)))"
        ),
        "gt_reference": gt_reference,
        "exposure_model": "uniform_rectangular_shutter",
        "N_subframes": int(n_subframes),
        "tau_list": tau_list[:, None],
        "exposure_weights": exposure_weights[:, None],
        "platform_case_name": str(platform_trajectory["case_name"]),
        "platform_parameters_dx_dy_theta_scale": global_parameters,
        "H_global": global_matrices,
        "object_count": int(object_count),
        "objects": object_metadata,
        "matrix_convention": "forward; column vector; H_object=H_global@H_local",
        "warp_count_per_subframe": int(
            1 + object_count
            if pack_background_and_valid
            else 2 + object_count
        ),
        "background_validity_packed_with_rgb": bool(
            pack_background_and_valid
        ),
        "temporal_average_count": 1,
        "radiometric_model": "linear_srgb_exposure_integration",
        "input_transfer_function": "IEC_61966_2_1_sRGB",
        "temporal_integration_space": "linear_rgb",
        "output_transfer_function": "IEC_61966_2_1_sRGB",
        "output_same_size_as_gt": True,
        "pre_crop_same_size_as_input": True,
        "crop_rectangle_top_bottom_left_right": np.asarray(
            crop_rectangle_value, dtype=np.int32
        ),
        "crop_rectangle_coordinate_convention": (
            "zero_based_half_open_[top,bottom,left,right]"
        ),
        "crop_safety_margin_requested": int(crop_safety_margin),
        "crop_safety_margin_applied": int(applied_crop_margin),
        "common_valid_threshold": float(valid_threshold),
        "common_valid_area_before_rectangle": int(common_valid_mask.sum()),
        "full_image_height": int(height),
        "full_image_width": int(width),
        "cropped_height": int(crop_bottom - crop_top),
        "cropped_width": int(crop_right - crop_left),
        "common_valid_min_coverage_inside_crop": float(
            cropped_coverage.min()
        ),
        "boundary_policy": (
            "global_background_valid_intersection_then_largest_rectangle"
        ),
        "validity_motion_scope": "global_platform_background_only",
        "global_border_mode": border_mode_name,
    }
    blur_srgb = object_model.linear_to_srgb_uav(accumulator)
    sensor_blur_srgb = (
        sensor_model.finalize_sensor_variant_srgb(
            sensor_accumulator, sensor_parameters
        )
        if sensor_accumulator is not None
        and sensor_parameters is not None
        else None
    )
    debug_frames_srgb = {
        name: object_model.linear_to_srgb_uav(frame)
        for name, frame in debug_frames.items()
    }
    if sensor_blur_srgb is not None:
        return (
            blur_srgb,
            sensor_blur_srgb,
            metadata,
            debug_frames_srgb,
        )
    return blur_srgb, metadata, debug_frames_srgb


# =========================================================================
# 5. 数据处理主流程
# =========================================================================
def run_generation(
    input_image_dir: str | Path = image_dir,
    input_mask_dir: str | Path = mask_dir,
    input_instance_mask_dir: str | Path = instance_mask_dir,
    result_dir: str | Path = output_dir,
    case_names: tuple[str, ...] = platform_cases,
    *,
    input_background_dir: str | Path | None = background_dir,
    seed: int = base_seed,
    n_subframes: int = N_subframes,
    adaptive_subframes: bool = use_adaptive_subframes,
    adaptive_step_px: float = adaptive_sample_step_px,
    adaptive_max_count: int = max_adaptive_subframes,
    platform_strength_scale: float = 1.0,
    linear_probability: float = linear_motion_probability,
    linear_ratio_range: tuple[float, float] = linear_distance_ratio_range,
    curved_ratio_range: tuple[float, float] = curved_path_ratio_range,
    turn_angle_range: tuple[float, float] = curved_turn_angle_deg_range,
    soft_mask: bool = use_soft_mask,
    mask_sigma: float = soft_mask_sigma,
    background_visibility_threshold_value: float = (
        object_model.background_visibility_threshold
    ),
    background_safety_ratio: float = (
        object_model.background_safety_dilation_ratio
    ),
    background_safety_min: int = (
        object_model.background_safety_dilation_min_px
    ),
    background_safety_max: int = (
        object_model.background_safety_dilation_max_px
    ),
    save_debug: bool = save_debug_subframes,
    border_mode_name: str = global_border_mode,
    crop_boundary: bool = is_crop_boundary,
    crop_safety_margin: int = crop_margin,
    valid_threshold: float = valid_coverage_threshold,
    mask_mode: str = object_model.mask_name_mode,
    max_images: int = 0,
    sensor_variant: bool = generate_sensor_variant,
    saturation_highlights: bool = sensor_saturation_highlights,
    poisson_gaussian_noise: bool = sensor_poisson_gaussian_noise,
) -> None:
    input_image_dir = Path(input_image_dir)
    input_mask_dir = Path(input_mask_dir)
    input_instance_mask_dir = Path(input_instance_mask_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    sensor_config = sensor_model.SensorDegradationConfig(
        saturation_enabled=saturation_highlights,
        saturation_threshold_low=sensor_saturation_threshold_low,
        saturation_threshold_high=sensor_saturation_threshold_high,
        saturation_gain_range=sensor_saturation_gain_range,
        poisson_gaussian_enabled=poisson_gaussian_noise,
        shot_noise_scale_range=sensor_shot_noise_scale_range,
        read_noise_sigma_range=sensor_read_noise_sigma_range,
    )
    if sensor_variant:
        sensor_model.validate_sensor_config(sensor_config)
    known_background_dir = (
        Path(input_background_dir) if input_background_dir is not None else None
    )
    if known_background_dir is not None and not known_background_dir.is_dir():
        raise FileNotFoundError(f"真实背景目录不存在：{known_background_dir}")

    unknown_cases = set(case_names) - set(SUPPORTED_PLATFORM_CASES)
    if unknown_cases:
        raise ValueError(f"未知平台运动类型：{sorted(unknown_cases)}")
    image_files = object_model.list_image_files_uav(input_image_dir)
    if max_images > 0:
        image_files = image_files[:max_images]
    if not image_files:
        raise FileNotFoundError(f"在 {input_image_dir} 中没有找到输入图像")

    total_object_count = count_available_instances(
        image_files,
        input_mask_dir,
        input_instance_mask_dir,
        mask_mode,
    )
    assignments = object_model.make_balanced_motion_case_assignments(
        total_object_count, linear_probability, seed + 1000
    )
    linear_count = assignments.count("01_obj_linear_motion")
    print(
        f"Found {len(image_files)} image(s), {total_object_count} object(s). "
        f"Object motion allocation: linear={linear_count}, "
        f"curved={len(assignments) - linear_count}."
    )
    assignment_position = 0

    for file_index, image_file in enumerate(image_files, start=1):
        raw_masks, mask_sources, mask_source_type = load_instance_masks(
            image_file,
            input_mask_dir,
            input_instance_mask_dir,
            mask_mode,
        )
        if not raw_masks:
            print(f"Warning: skip {image_file.name}; no usable mask")
            continue

        image = object_model.read_image_double_rgb(image_file)
        for mask, source in zip(raw_masks, mask_sources):
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Mask 与图像尺寸不一致：{source}")

        refined = [
            object_model.refine_render_mask(
                mask,
                object_model.render_mask_refine_ratio,
                object_model.render_mask_refine_min_px,
                object_model.render_mask_refine_max_px,
            )
            for mask in raw_masks
        ]
        object_masks = [item[0] for item in refined]
        refine_radii = [int(item[1]) for item in refined]
        object_masks_soft = [
            object_model.smooth_mask(mask, mask_sigma)
            if soft_mask
            else mask.astype(np.float64)
            for mask in object_masks
        ]
        foregrounds = [
            image * alpha[..., None] for alpha in object_masks_soft
        ]
        union_mask = np.logical_or.reduce(object_masks)
        union_mask_soft = np.maximum.reduce(object_masks_soft)

        forced_cases = assignments[
            assignment_position : assignment_position + len(object_masks)
        ]
        assignment_position += len(object_masks)
        image_seed = seed + file_index * 10000
        object_trajectories: list[dict[str, Any]] = []
        for object_index, (mask, source, forced_case) in enumerate(
            zip(object_masks, mask_sources, forced_cases), start=1
        ):
            object_seed = image_seed + object_index * 100
            trajectory = object_model.make_independent_object_trajectory(
                object_generation_case,
                object_seed,
                mask,
                forced_motion_case=forced_case,
                linear_probability=linear_probability,
                linear_ratio_range=linear_ratio_range,
                curved_ratio_range=curved_ratio_range,
                turn_angle_range=turn_angle_range,
            )
            trajectory["instance_id"] = object_index
            trajectory["instance_mask_file"] = source
            object_model.validate_center_referenced_trajectory(trajectory)
            object_trajectories.append(trajectory)

        # 先确定每个平台 case 的实际子帧时间。背景与目标之间的相对可见性
        # 只由 local trajectory 决定，但必须使用与最终渲染相同的 tau 样本。
        # 多个平台 case 的子帧数不同时，取所有 tau 的并集，背景只构造一次。
        platform_jobs: list[
            tuple[str, dict[str, Any], float, int]
        ] = []
        background_tau_parts: list[np.ndarray] = []
        for platform_index, case_name in enumerate(case_names, start=1):
            platform_trajectory = make_centered_platform_trajectory(
                case_name,
                image_seed + platform_index * 1000,
                platform_strength_scale,
            )
            max_path = estimate_max_composed_path_length(
                image.shape[1],
                image.shape[0],
                platform_trajectory,
                object_trajectories,
            )
            case_n_subframes = choose_subframe_count(
                n_subframes,
                adaptive_subframes,
                max_path,
                adaptive_step_px,
                adaptive_max_count,
            )
            platform_jobs.append(
                (case_name, platform_trajectory, max_path, case_n_subframes)
            )
            background_tau_parts.append(
                np.linspace(
                    -0.5, 0.5, case_n_subframes, dtype=np.float64
                )
            )
        background_tau_list = np.unique(
            np.concatenate(background_tau_parts)
        )

        background_path: Path | None = None
        if known_background_dir is not None:
            background_path = object_model.find_matching_background_path(
                known_background_dir, image_file
            )
            background = object_model.read_image_double_rgb(background_path)
            if background.shape != image.shape:
                raise ValueError(
                    f"背景尺寸不一致：{background_path.name}"
                )
            provided_inpaint_masks_and_radii = [
                object_model.dilate_inpaint_mask(
                    object_mask,
                    float(trajectory["object_scale"]),
                    object_model.inpaint_dilation_ratio,
                    object_model.inpaint_dilation_min_px,
                    object_model.inpaint_dilation_max_px,
                )
                for object_mask, trajectory in zip(
                    object_masks, object_trajectories
                )
            ]
            provided_inpaint_masks = [
                item[0] for item in provided_inpaint_masks_and_radii
            ]
            inpaint_union_mask = np.logical_or.reduce(
                provided_inpaint_masks
            )
            (
                provided_needed_alpha_masks,
                provided_needed_masks,
                provided_write_support_masks,
                provided_write_alpha_masks,
                provided_safety_radii,
            ) = object_model.compute_soft_background_visibility_model(
                object_masks_soft,
                object_trajectories,
                provided_inpaint_masks,
                background_tau_list,
                background_visibility_threshold_value,
                background_safety_ratio,
                background_safety_min,
                background_safety_max,
            )
            needed_background_mask = np.logical_or.reduce(
                provided_needed_masks
            )
            provided_write_support = np.logical_or.reduce(
                provided_write_support_masks
            )
            background_write_alpha = np.maximum.reduce(
                provided_write_alpha_masks
            )
            background_metadata: dict[str, Any] = {
                "method": "provided_background",
                "inpaint_mask_area": int(inpaint_union_mask.sum()),
                "needed_background_area": int(needed_background_mask.sum()),
                "background_write_mask_area": int(
                    provided_write_support.sum()
                ),
                "background_write_alpha_sum": float(
                    background_write_alpha.sum()
                ),
                "safety_dilation_radii_px": np.asarray(
                    provided_safety_radii, dtype=np.int32
                ),
            }
            background_source = "provided"
        else:
            (
                background,
                inpaint_union_mask,
                needed_background_mask,
                background_write_alpha,
                background_metadata,
            ) = object_model.build_trajectory_aware_background(
                image,
                object_masks,
                object_trajectories,
                object_masks_soft=object_masks_soft,
                exposure_tau_list=background_tau_list,
                method=object_model.inpaint_method,
                dilation_ratio=object_model.inpaint_dilation_ratio,
                dilation_min_px=object_model.inpaint_dilation_min_px,
                dilation_max_px=object_model.inpaint_dilation_max_px,
                motion_samples=object_model.background_motion_samples,
                boundary_feather_ratio=object_model.background_boundary_feather_ratio,
                boundary_feather_min_px=object_model.background_boundary_feather_min_px,
                boundary_feather_max_px=object_model.background_boundary_feather_max_px,
                tone_ring_ratio=object_model.background_tone_ring_ratio,
                tone_ring_min_px=object_model.background_tone_ring_min_px,
                tone_ring_max_px=object_model.background_tone_ring_max_px,
                tone_gain_range=object_model.background_tone_gain_range,
                visibility_threshold=(
                    background_visibility_threshold_value
                ),
                safety_dilation_ratio=background_safety_ratio,
                safety_dilation_min_px=background_safety_min,
                safety_dilation_max_px=background_safety_max,
            )
            background_source = (
                "soft_needed_safety_alpha_"
                f"{object_model.inpaint_method}"
            )

        print(
            f"\n[{file_index}/{len(image_files)}] {image_file.name}: "
            f"objects={len(object_masks)}, masks={mask_source_type}, "
            f"background={background_source}"
        )
        for index, trajectory in enumerate(object_trajectories, start=1):
            print(
                f"  object {index}: {trajectory['case_name']}, "
                f"distance={object_model.trajectory_motion_distance(trajectory):.2f}px, "
                f"ratio={trajectory['motion_strength_ratio']:.3f}"
            )

        for (
            case_name,
            platform_trajectory,
            max_path,
            case_n_subframes,
        ) in platform_jobs:
            print(
                f"  platform={case_name}, subframes={case_n_subframes}, "
                f"estimated_max_path={max_path:.2f}px"
            )

            sensor_parameters = (
                sensor_model.sample_sensor_parameters(
                    sensor_config,
                    int(platform_trajectory["case_seed"])
                    + sensor_seed_offset,
                )
                if sensor_variant
                else None
            )
            synthesis_result = synthesize_relative_motion_blur(
                image,
                background,
                foregrounds,
                object_masks_soft,
                object_trajectories,
                platform_trajectory,
                case_n_subframes,
                border_mode_name=border_mode_name,
                crop_safety_margin=crop_safety_margin,
                valid_threshold=valid_threshold,
                sensor_parameters=sensor_parameters,
            )
            if sensor_parameters is None:
                blur_image, metadata, debug_frames = synthesis_result
                sensor_blur_image = None
            else:
                (
                    blur_image,
                    sensor_blur_image,
                    metadata,
                    debug_frames,
                ) = synthesis_result
                metadata["sensor_degradation"] = (
                    sensor_model.sensor_metadata_dict(sensor_parameters)
                )
            metadata.update(
                {
                    "input_image": str(image_file),
                    "mask_source": mask_source_type,
                    "instance_mask_files": mask_sources,
                    "render_mask_refine_radii_px": np.asarray(refine_radii),
                    "background_source": background_source,
                    "background_file": (
                        str(background_path) if background_path is not None else ""
                    ),
                    "background_completion": background_metadata,
                    "platform_trajectory": platform_trajectory,
                    "object_trajectories": object_trajectories,
                    "adaptive_subframes": bool(adaptive_subframes),
                    "estimated_max_composed_path_px": float(max_path),
                }
            )

            recommended_rectangle = tuple(
                int(value)
                for value in np.asarray(
                    metadata["crop_rectangle_top_bottom_left_right"]
                ).ravel()
            )
            full_rectangle = (0, image.shape[0], 0, image.shape[1])
            applied_rectangle = (
                recommended_rectangle if crop_boundary else full_rectangle
            )
            crop_top, crop_bottom, crop_left, crop_right = (
                applied_rectangle
            )
            add_cropped_coordinate_metadata(
                metadata, applied_rectangle
            )
            metadata.update(
                {
                    "crop_enabled": bool(crop_boundary),
                    "applied_crop_rectangle_top_bottom_left_right": np.asarray(
                        applied_rectangle, dtype=np.int32
                    ),
                    "crop_offset_x": int(crop_left),
                    "crop_offset_y": int(crop_top),
                    "output_height": int(crop_bottom - crop_top),
                    "output_width": int(crop_right - crop_left),
                    "output_same_size_as_input": bool(
                        applied_rectangle == full_rectangle
                    ),
                    "matrix_coordinate_note": (
                        "H_* use full-image coordinates; "
                        "H_*_cropped use saved-output coordinates"
                    ),
                }
            )

            output_prefix = result_dir / f"{image_file.stem}_{case_name}_relative"
            outputs = {
                "_gtimg.png": crop_rectangle(image, applied_rectangle),
                "_blurryimg.png": crop_rectangle(
                    blur_image, applied_rectangle
                ),
                "_background_reference.png": crop_rectangle(
                    background, applied_rectangle
                ),
                "_objmask.png": crop_rectangle(
                    union_mask_soft, applied_rectangle
                ),
                "_inpaintmask.png": crop_rectangle(
                    inpaint_union_mask.astype(np.float64),
                    applied_rectangle,
                ),
                "_neededbackgroundmask.png": crop_rectangle(
                    needed_background_mask.astype(np.float64),
                    applied_rectangle,
                ),
                "_backgroundwritealpha.png": crop_rectangle(
                    background_write_alpha, applied_rectangle
                ),
            }
            if sensor_blur_image is not None:
                outputs["_blurryimg_sensor.png"] = crop_rectangle(
                    sensor_blur_image, applied_rectangle
                )
            for suffix, output_image in outputs.items():
                object_model.write_image_uint8(
                    output_prefix.with_name(output_prefix.name + suffix),
                    output_image,
                )
            if save_debug:
                for name, frame in debug_frames.items():
                    object_model.write_image_uint8(
                        output_prefix.with_name(
                            output_prefix.name + f"_subframe_{name}.png"
                        ),
                        crop_rectangle(frame, applied_rectangle),
                    )
            if sensor_parameters is not None:
                sensor_model.save_sensor_metadata_npz(
                    output_prefix.with_name(
                        output_prefix.name + "_sensor_metadata.npz"
                    ),
                    sensor_parameters,
                )
            savemat(
                output_prefix.with_name(output_prefix.name + "_meta.mat"),
                {"meta": metadata},
                do_compression=True,
                long_field_names=True,
            )

    print(f"\nDone. Results are saved to: {result_dir}")


# =========================================================================
# 6. 命令行
# =========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=image_dir)
    parser.add_argument("--mask-dir", type=Path, default=mask_dir)
    parser.add_argument(
        "--instance-mask-dir", type=Path, default=instance_mask_dir
    )
    parser.add_argument("--background-dir", type=Path, default=background_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--platform-cases",
        default=",".join(platform_cases),
        help="逗号分隔的平台运动类型；00_static 可用于仅验证目标运动。",
    )
    parser.add_argument("--platform-strength-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=base_seed)
    parser.add_argument("--n-subframes", type=int, default=N_subframes)
    parser.add_argument(
        "--adaptive-subframes",
        action=argparse.BooleanOptionalAction,
        default=use_adaptive_subframes,
    )
    parser.add_argument(
        "--adaptive-sample-step-px",
        type=float,
        default=adaptive_sample_step_px,
    )
    parser.add_argument(
        "--max-adaptive-subframes", type=int, default=max_adaptive_subframes
    )
    parser.add_argument(
        "--linear-probability", type=float, default=linear_motion_probability
    )
    parser.add_argument(
        "--linear-distance-ratio-min",
        type=float,
        default=linear_distance_ratio_range[0],
    )
    parser.add_argument(
        "--linear-distance-ratio-max",
        type=float,
        default=linear_distance_ratio_range[1],
    )
    parser.add_argument(
        "--curved-path-ratio-min",
        type=float,
        default=curved_path_ratio_range[0],
    )
    parser.add_argument(
        "--curved-path-ratio-max",
        type=float,
        default=curved_path_ratio_range[1],
    )
    parser.add_argument(
        "--curved-turn-angle-min",
        type=float,
        default=curved_turn_angle_deg_range[0],
    )
    parser.add_argument(
        "--curved-turn-angle-max",
        type=float,
        default=curved_turn_angle_deg_range[1],
    )
    parser.add_argument(
        "--soft-mask",
        action=argparse.BooleanOptionalAction,
        default=use_soft_mask,
    )
    parser.add_argument("--soft-mask-sigma", type=float, default=soft_mask_sigma)
    parser.add_argument(
        "--background-visibility-threshold",
        type=float,
        default=object_model.background_visibility_threshold,
        help="soft mask 相对 GT 覆盖率下降超过该值时计入 needed 核心。",
    )
    parser.add_argument(
        "--background-safety-dilation-ratio",
        type=float,
        default=object_model.background_safety_dilation_ratio,
    )
    parser.add_argument(
        "--background-safety-dilation-min-px",
        type=int,
        default=object_model.background_safety_dilation_min_px,
    )
    parser.add_argument(
        "--background-safety-dilation-max-px",
        type=int,
        default=object_model.background_safety_dilation_max_px,
    )
    parser.add_argument(
        "--save-debug-subframes",
        action=argparse.BooleanOptionalAction,
        default=save_debug_subframes,
    )
    parser.add_argument(
        "--global-border-mode",
        choices=("reflect101", "reflect", "replicate", "constant"),
        default=global_border_mode,
        help=(
            "临时图外填充值；默认constant。开启裁剪后，保存区域不会使用"
            "这些填充值。"
        ),
    )
    parser.add_argument(
        "--crop-boundary",
        action=argparse.BooleanOptionalAction,
        default=is_crop_boundary,
        help="按所有平台子帧的公共有效mask最大矩形同步裁剪全部输出。",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=crop_margin,
        help="在最大公共有效矩形内部额外收缩的安全像素。",
    )
    parser.add_argument(
        "--valid-coverage-threshold",
        type=float,
        default=valid_coverage_threshold,
        help="像素被判定为完全来自真实输入背景所需的最低覆盖率。",
    )
    parser.add_argument(
        "--mask-name-mode",
        choices=("same_stem", "same_name", "stem_png"),
        default=object_model.mask_name_mode,
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 表示全部处理；正数只处理前 N 张，便于快速验证。",
    )
    parser.add_argument(
        "--sensor-variant",
        action=argparse.BooleanOptionalAction,
        default=generate_sensor_variant,
        help="额外保存带高光饱和和曝光后传感器噪声的模糊图",
    )
    parser.add_argument(
        "--sensor-saturation-highlights",
        action=argparse.BooleanOptionalAction,
        default=sensor_saturation_highlights,
        help="在各完整线性RGB子帧中恢复一部分饱和高光能量",
    )
    parser.add_argument(
        "--sensor-poisson-gaussian-noise",
        action=argparse.BooleanOptionalAction,
        default=sensor_poisson_gaussian_noise,
        help="曝光积分完成后添加一次Poisson-Gaussian传感器噪声",
    )
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not 0.0 <= arguments.linear_probability <= 1.0:
        raise ValueError("--linear-probability 必须在 [0, 1] 内")
    ranges = (
        (
            arguments.linear_distance_ratio_min,
            arguments.linear_distance_ratio_max,
            "linear-distance-ratio",
        ),
        (
            arguments.curved_path_ratio_min,
            arguments.curved_path_ratio_max,
            "curved-path-ratio",
        ),
        (
            arguments.curved_turn_angle_min,
            arguments.curved_turn_angle_max,
            "curved-turn-angle",
        ),
    )
    for minimum, maximum, name in ranges:
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(f"--{name}-min/max 必须满足 0 < min <= max")
    if arguments.n_subframes < 3:
        raise ValueError("--n-subframes 不能小于 3")
    if arguments.max_adaptive_subframes < 3:
        raise ValueError("--max-adaptive-subframes 不能小于 3")
    if arguments.adaptive_sample_step_px <= 0.0:
        raise ValueError("--adaptive-sample-step-px 必须大于 0")
    if arguments.platform_strength_scale < 0.0:
        raise ValueError("--platform-strength-scale 不能小于 0")
    if arguments.crop_margin < 0:
        raise ValueError("--crop-margin 不能小于 0")
    if not 0.0 < arguments.valid_coverage_threshold <= 1.0:
        raise ValueError(
            "--valid-coverage-threshold 必须位于 (0,1]"
        )
    if not 0.0 <= arguments.background_visibility_threshold < 1.0:
        raise ValueError(
            "--background-visibility-threshold 必须在 [0, 1) 内"
        )
    if arguments.background_safety_dilation_ratio < 0.0:
        raise ValueError(
            "--background-safety-dilation-ratio 不能小于 0"
        )
    if (
        arguments.background_safety_dilation_min_px < 0
        or arguments.background_safety_dilation_max_px
        < arguments.background_safety_dilation_min_px
    ):
        raise ValueError(
            "--background-safety-dilation-min/max-px "
            "必须满足 0 <= min <= max"
        )


def main() -> None:
    arguments = build_parser().parse_args()
    validate_arguments(arguments)
    cases = tuple(
        item.strip()
        for item in arguments.platform_cases.split(",")
        if item.strip()
    )
    if not cases:
        raise ValueError("--platform-cases 不能为空")
    run_generation(
        arguments.image_dir,
        arguments.mask_dir,
        arguments.instance_mask_dir,
        arguments.output_dir,
        cases,
        input_background_dir=arguments.background_dir,
        seed=arguments.seed,
        n_subframes=arguments.n_subframes,
        adaptive_subframes=arguments.adaptive_subframes,
        adaptive_step_px=arguments.adaptive_sample_step_px,
        adaptive_max_count=arguments.max_adaptive_subframes,
        platform_strength_scale=arguments.platform_strength_scale,
        linear_probability=arguments.linear_probability,
        linear_ratio_range=(
            arguments.linear_distance_ratio_min,
            arguments.linear_distance_ratio_max,
        ),
        curved_ratio_range=(
            arguments.curved_path_ratio_min,
            arguments.curved_path_ratio_max,
        ),
        turn_angle_range=(
            arguments.curved_turn_angle_min,
            arguments.curved_turn_angle_max,
        ),
        soft_mask=arguments.soft_mask,
        mask_sigma=arguments.soft_mask_sigma,
        background_visibility_threshold_value=(
            arguments.background_visibility_threshold
        ),
        background_safety_ratio=(
            arguments.background_safety_dilation_ratio
        ),
        background_safety_min=(
            arguments.background_safety_dilation_min_px
        ),
        background_safety_max=(
            arguments.background_safety_dilation_max_px
        ),
        save_debug=arguments.save_debug_subframes,
        border_mode_name=arguments.global_border_mode,
        crop_boundary=arguments.crop_boundary,
        crop_safety_margin=arguments.crop_margin,
        valid_threshold=arguments.valid_coverage_threshold,
        mask_mode=arguments.mask_name_mode,
        max_images=arguments.max_images,
        sensor_variant=arguments.sensor_variant,
        saturation_highlights=arguments.sensor_saturation_highlights,
        poisson_gaussian_noise=(
            arguments.sensor_poisson_gaussian_noise
        ),
    )


if __name__ == "__main__":
    main()

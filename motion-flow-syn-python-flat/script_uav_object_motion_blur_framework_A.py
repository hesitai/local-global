"""方案 A：多实例独立运动、静止背景、动态 mask、多子帧积分。

本文件对应 ``script_uav_object_motion_blur_framework_A.m``。用户配置、主循环、
mask 处理、轨迹、warp、合成和裁剪全部放在同一个 Python 文件中。每个实例
独立、可复现地选择直线或曲线运动，并独立采样运动方向与强度。所有瞬时
子帧在线性 RGB 中完成 warp、alpha 合成和曝光积分，最后转换回 sRGB。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from skimage.restoration import inpaint_biharmonic

import uav_sensor_degradation as sensor_model


# =========================================================================
# 1. 用户配置区
# =========================================================================
base_seed = 202600716
script_root = Path(__file__).resolve().parent
image_dir = script_root / "dataset/image/2"
# 联合 mask 用作旧数据兼容；新流程优先读取下面的逐实例 mask 目录。
mask_dir = script_root / "output/mask"
instance_mask_dir = script_root / "output/mask_instances"
# 默认仍使用 inpaint；只有命令行显式传入 --background-dir 时才读取真实背景。
background_dir: Path | None = None
output_dir = script_root / "output/image/2"
N_subframes = 15
# 默认根据本张图中最长目标轨迹自动设置子帧数，使相邻采样约为 1 像素。
use_adaptive_subframes = True
# 默认保留输入 GT 的原始尺寸；如需去除边界，可显式传入 --crop-boundary。
is_crop_boundary = False
crop_margin = 4
is_save_debug_subframes = True
is_show = False
use_soft_mask = True
soft_mask_sigma = 2.0
# SAM2 的自动 mask 偶尔会漏掉目标最外侧 1—4 像素，导致车漆、船沿等颜色
# 被误当成背景留在原位置。先闭运算补小缺口，再做很小的尺度自适应膨胀；
# 该 mask 仍同时用于前景提取和运动渲染，避免背景清理 mask 与目标错位。
render_mask_refine_ratio = 0.002
render_mask_refine_min_px = 1
render_mask_refine_max_px = 5
# 大目标使用 Telea 生成候选背景，但只把实际露出的 needed mask 写回图像。
inpaint_method = "telea"
inpaint_dilation_ratio = 0.01
inpaint_dilation_min_px = 5
inpaint_dilation_max_px = 30
background_motion_samples = 17
# 复制的真实邻域纹理只在 needed mask 内使用。其边缘与完整 inpaint 候选
# 背景做窄带渐变，消除硬切换线，同时保留中间区域的真实道路/水面纹理。
background_boundary_feather_ratio = 0.004
background_boundary_feather_min_px = 4
background_boundary_feather_max_px = 12
# 复制纹理可能来自目标附近较亮/较暗的另一处。用目标外侧的真实背景环估计
# 低频亮度，只调整亮度增益而不改变纹理方向；例如阴影中的车辆不会再露出
# 一圈明显更亮的灰色路面。
background_tone_ring_ratio = 0.02
background_tone_ring_min_px = 20
background_tone_ring_max_px = 60
background_tone_gain_range = (0.55, 1.45)
# soft needed background 与最终目标渲染使用相同的双线性 soft mask。只有目标
# 覆盖率相对 GT 至少下降以下阈值时，才认为该像素会露出新的背景。
background_visibility_threshold = 1.0 / 255.0
# needed 核心外增加很小的安全圈，补偿 SAM2 边缘误差、亚像素插值和有限
# 时间采样；安全圈只允许位于完整 inpaint/source-exclusion mask 内。
background_safety_dilation_ratio = 0.002
background_safety_dilation_min_px = 2
background_safety_dilation_max_px = 8
# 默认模拟曝光期间快门响应恒定的矩形曝光：所有子帧等权。
# 非均匀终点增强仍可通过命令行参数开启，但不再作为训练数据默认模型。
exposure_start_weight = 1.0
exposure_terminal_fraction = 0.0
# 在不改变原 ``_blurryimg.png`` 的前提下，额外生成更接近传感器成像的版本。
generate_sensor_variant = True  # True额外输出 ``_blurryimg_sensor.png``。
sensor_saturation_highlights = True  # 每个完整线性子帧恢复部分饱和高光能量。
sensor_saturation_threshold_low = 0.95  # 最大线性RGB通道达到该值后开始软增强。
sensor_saturation_threshold_high = 0.99  # 达到该值后使用完整高光增益。
sensor_saturation_gain_range = (0.25, 1.25)  # 每张图只采样一次并用于全部子帧。
sensor_poisson_gaussian_noise = True  # 曝光平均后只添加一次shot/read noise。
sensor_shot_noise_scale_range = (1.0e-5, 1.0e-4)  # Poisson光子噪声尺度。
sensor_read_noise_sigma_range = (3.0e-6, 5.0e-5)  # 线性伪传感器域读取噪声。
sensor_seed_offset = 8_000_000  # 与目标运动seed分开，且保持输出可复现。
motion_cases = ("01_obj_independent_random",)
# 多目标随机轨迹配置。距离均相对于各实例 mask 外接框的最长边。
# 整批实例按最接近 70%/30% 的数量随机分配；每个实例的强度和方向仍独立。
linear_motion_probability = 0.70
linear_distance_ratio_range = (0.03, 0.10)
curved_path_ratio_range = (0.03, 0.10)
curved_turn_angle_deg_range = (2.0, 10.0)
# 只凭单张 mask 估计车辆/船的前后轴：有向外接矩形长宽比达到该值时，
# 认为是俯视/侧上方的细长轮廓，运动沿长轴；低于该值时认为更接近正面或
# 背面视角，图像中的车身长度被透视压缩，运动改沿短轴，避免车辆横向平移。
front_back_axis_aspect_ratio_threshold = 1.50
# 默认按不含扩展名的文件名匹配：image/1.jpg 可自动读取 mask/1.png。
mask_name_mode = "same_stem"
use_same_case_seed_for_all_images = False
use_largest_connected_component_only = True
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================================================================
# 2—4. 创建目录、读取列表、主处理流程
# =========================================================================
def run_generation(
    input_image_dir: str | Path = image_dir,
    input_mask_dir: str | Path = mask_dir,
    result_dir: str | Path = output_dir,
    case_names: tuple[str, ...] = motion_cases,
    *,
    input_instance_mask_dir: str | Path = instance_mask_dir,
    input_background_dir: str | Path | None = background_dir,
    seed: int = base_seed,
    n_subframes: int = N_subframes,
    adaptive_subframes: bool = use_adaptive_subframes,
    crop_boundary: bool = is_crop_boundary,
    extra_crop_margin: int = crop_margin,
    save_debug_subframes: bool = is_save_debug_subframes,
    show: bool = is_show,
    soft_mask: bool = use_soft_mask,
    mask_sigma: float = soft_mask_sigma,
    render_refine_ratio: float = render_mask_refine_ratio,
    render_refine_min: int = render_mask_refine_min_px,
    render_refine_max: int = render_mask_refine_max_px,
    background_method: str = inpaint_method,
    inpaint_dilation_ratio_value: float = inpaint_dilation_ratio,
    inpaint_dilation_min: int = inpaint_dilation_min_px,
    inpaint_dilation_max: int = inpaint_dilation_max_px,
    background_samples: int = background_motion_samples,
    background_feather_ratio: float = background_boundary_feather_ratio,
    background_feather_min: int = background_boundary_feather_min_px,
    background_feather_max: int = background_boundary_feather_max_px,
    background_ring_ratio: float = background_tone_ring_ratio,
    background_ring_min: int = background_tone_ring_min_px,
    background_ring_max: int = background_tone_ring_max_px,
    background_gain_range: tuple[float, float] = background_tone_gain_range,
    background_visibility_threshold_value: float = background_visibility_threshold,
    background_safety_ratio: float = background_safety_dilation_ratio,
    background_safety_min: int = background_safety_dilation_min_px,
    background_safety_max: int = background_safety_dilation_max_px,
    start_exposure_weight: float = exposure_start_weight,
    terminal_exposure_fraction: float = exposure_terminal_fraction,
    mask_mode: str = mask_name_mode,
    same_case_seed: bool = use_same_case_seed_for_all_images,
    largest_component_only: bool = use_largest_connected_component_only,
    linear_probability: float = linear_motion_probability,
    linear_ratio_range: tuple[float, float] = linear_distance_ratio_range,
    curved_ratio_range: tuple[float, float] = curved_path_ratio_range,
    turn_angle_range: tuple[float, float] = curved_turn_angle_deg_range,
    front_back_axis_threshold: float = (
        front_back_axis_aspect_ratio_threshold
    ),
    sensor_variant: bool = generate_sensor_variant,
    saturation_highlights: bool = sensor_saturation_highlights,
    poisson_gaussian_noise: bool = sensor_poisson_gaussian_noise,
) -> None:
    input_image_dir = Path(input_image_dir)
    input_mask_dir = Path(input_mask_dir)
    input_instance_mask_dir = Path(input_instance_mask_dir)
    known_background_dir = (
        Path(input_background_dir) if input_background_dir is not None else None
    )
    if known_background_dir is not None and not known_background_dir.is_dir():
        raise FileNotFoundError(f"真实背景目录不存在：{known_background_dir}")
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
    image_files = list_image_files_uav(input_image_dir)
    if not image_files:
        raise FileNotFoundError(f"在目录 {input_image_dir} 中未找到任何图像文件。")
    print(f"Found {len(image_files)} image files in {input_image_dir}")

    # 为小批量数据也维持接近设定比例，先统计本次实际可处理的实例数量。
    total_object_count = 0
    for image_file in image_files:
        paths = list_instance_mask_paths(input_instance_mask_dir, image_file)
        if paths:
            total_object_count += len(paths)
        elif build_mask_path(input_mask_dir, image_file, mask_mode).is_file():
            total_object_count += 1
    balanced_case_assignments: dict[str, tuple[str, ...]] = {}
    balanced_case_positions: dict[str, int] = {}
    for case_index, case_name in enumerate(case_names, start=1):
        if case_name == "01_obj_independent_random":
            balanced_case_assignments[case_name] = (
                make_balanced_motion_case_assignments(
                    total_object_count,
                    linear_probability,
                    seed + case_index * 1000,
                )
            )
            balanced_case_positions[case_name] = 0
            assignments = balanced_case_assignments[case_name]
            linear_count = assignments.count("01_obj_linear_motion")
            print(
                "Independent-random motion allocation: "
                f"linear={linear_count}, curved={len(assignments) - linear_count}, "
                f"total={len(assignments)}"
            )

    for file_index, image_file in enumerate(image_files, start=1):
        instance_mask_paths = list_instance_mask_paths(
            input_instance_mask_dir, image_file
        )
        if instance_mask_paths:
            object_masks = [
                read_binary_mask(path, largest_component_only)
                for path in instance_mask_paths
            ]
            mask_source = "instance_masks"
        else:
            mask_path = build_mask_path(input_mask_dir, image_file, mask_mode)
            if not mask_path.exists():
                print(
                    f"Warning: skip {image_file.name}; no instance masks in "
                    f"{input_instance_mask_dir / image_file.stem} and union mask "
                    f"does not exist: {mask_path}"
                )
                continue
            object_masks = [
                read_binary_mask(mask_path, largest_component_only)
            ]
            instance_mask_paths = [mask_path]
            mask_source = "union_mask_fallback"
        print(f"\nProcessing image {file_index}/{len(image_files)}: {image_file.name}")

        image = read_image_double_rgb(image_file)
        for object_mask, current_mask_path in zip(
            object_masks, instance_mask_paths
        ):
            if object_mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Mask 尺寸与图像不一致：{current_mask_path.name}"
                )
        raw_mask_areas = [int(mask.sum()) for mask in object_masks]
        refined_masks_and_radii = [
            refine_render_mask(
                object_mask,
                render_refine_ratio,
                render_refine_min,
                render_refine_max,
            )
            for object_mask in object_masks
        ]
        object_masks = [
            refined_mask for refined_mask, _ in refined_masks_and_radii
        ]
        render_refine_radii = [
            radius for _, radius in refined_masks_and_radii
        ]
        object_masks_soft = [
            smooth_mask(object_mask, mask_sigma)
            if soft_mask
            else object_mask.astype(np.float64)
            for object_mask in object_masks
        ]
        foregrounds = [
            image * object_mask_soft[..., None]
            for object_mask_soft in object_masks_soft
        ]
        union_mask = np.logical_or.reduce(object_masks)
        union_mask_soft = np.maximum.reduce(object_masks_soft)
        foreground = image * union_mask_soft[..., None]
        print(
            f"  Masks: {mask_source}, objects={len(object_masks)}, "
            f"render_refine_px={render_refine_radii}"
        )
        background_path: Path | None = None
        provided_background: np.ndarray | None = None
        if known_background_dir is not None:
            background_path = find_matching_background_path(
                known_background_dir, image_file
            )
            provided_background = read_image_double_rgb(background_path)
            if provided_background.shape != image.shape:
                raise ValueError(
                    "真实背景尺寸与图像不一致："
                    f"{background_path.name} {provided_background.shape[:2]} != "
                    f"{image_file.name} {image.shape[:2]}"
                )
            print(f"  Background: provided ({background_path})")

        for case_index, case_name in enumerate(case_names, start=1):
            case_seed = (
                seed + case_index * 1000
                if same_case_seed
                else seed + file_index * 10000 + case_index * 1000
            )
            print(f"  Generate case: {case_name}")
            trajectories: list[dict[str, Any]] = []
            for instance_index, (object_mask, current_mask_path) in enumerate(
                zip(object_masks, instance_mask_paths), start=1
            ):
                instance_seed = case_seed + instance_index * 100
                forced_motion_case: str | None = None
                if case_name == "01_obj_independent_random":
                    assignment_position = balanced_case_positions[case_name]
                    forced_motion_case = balanced_case_assignments[case_name][
                        assignment_position
                    ]
                    balanced_case_positions[case_name] = assignment_position + 1
                trajectory = make_independent_object_trajectory(
                    case_name,
                    instance_seed,
                    object_mask,
                    forced_motion_case=forced_motion_case,
                    linear_probability=linear_probability,
                    linear_ratio_range=linear_ratio_range,
                    curved_ratio_range=curved_ratio_range,
                    turn_angle_range=turn_angle_range,
                    front_back_axis_threshold=front_back_axis_threshold,
                )
                trajectory["instance_id"] = instance_index
                trajectory["instance_mask_file"] = str(current_mask_path)
                trajectories.append(trajectory)
                print(
                    f"    object {instance_index}: "
                    f"{trajectory['case_name']}, "
                    f"strength={trajectory['motion_strength_ratio']:.3f}, "
                    f"distance={trajectory_motion_distance(trajectory):.2f}px, "
                    f"front_back_axis="
                    f"{trajectory['front_back_axis_deg']:.1f}deg, "
                    f"axis_rule={trajectory['front_back_axis_rule']}, "
                    f"seed={instance_seed}"
                )
            for trajectory in trajectories:
                validate_center_referenced_trajectory(trajectory)
            print("    time reference=center, centered trajectory=validated")
            max_motion_distance = max(
                trajectory_motion_distance(trajectory)
                for trajectory in trajectories
            )
            case_n_subframes = (
                max(2, int(np.ceil(max_motion_distance)) + 1)
                if adaptive_subframes
                else n_subframes
            )
            # 中心参考曝光必须包含唯一的 tau=0 中间帧，因此子帧数取奇数。
            if case_n_subframes % 2 == 0:
                case_n_subframes += 1
            print(
                f"    subframes={case_n_subframes}, "
                f"max_distance={max_motion_distance:.2f}px, "
                f"sample_step≈{max_motion_distance / (case_n_subframes - 1):.3f}px"
            )
            if provided_background is None:
                (
                    background,
                    inpaint_union_mask,
                    needed_background_mask,
                    background_write_alpha,
                    background_metadata,
                ) = build_trajectory_aware_background(
                    image,
                    object_masks,
                    trajectories,
                    object_masks_soft=object_masks_soft,
                    exposure_tau_list=np.linspace(
                        -0.5,
                        0.5,
                        case_n_subframes,
                        dtype=np.float64,
                    ),
                    method=background_method,
                    dilation_ratio=inpaint_dilation_ratio_value,
                    dilation_min_px=inpaint_dilation_min,
                    dilation_max_px=inpaint_dilation_max,
                    motion_samples=background_samples,
                    boundary_feather_ratio=background_feather_ratio,
                    boundary_feather_min_px=background_feather_min,
                    boundary_feather_max_px=background_feather_max,
                    tone_ring_ratio=background_ring_ratio,
                    tone_ring_min_px=background_ring_min,
                    tone_ring_max_px=background_ring_max,
                    tone_gain_range=background_gain_range,
                    visibility_threshold=(
                        background_visibility_threshold_value
                    ),
                    safety_dilation_ratio=background_safety_ratio,
                    safety_dilation_min_px=background_safety_min,
                    safety_dilation_max_px=background_safety_max,
                )
                background_source = (
                    f"soft_needed_safety_alpha_{background_method}"
                )
                print(
                    "    background fill: "
                    f"needed={background_metadata['needed_background_area']} px, "
                    f"write_support={background_metadata['background_write_mask_area']} px, "
                    f"nearby_texture={background_metadata['nearby_texture_filled_area']} px, "
                    f"inpaint_fallback={background_metadata['needed_inpaint_area']} px, "
                    "write_scope=needed_core_plus_soft_safety_ring"
                )
            else:
                background = provided_background
                provided_inpaint_masks_and_radii = [
                    dilate_inpaint_mask(
                        object_mask,
                        float(trajectory["object_scale"]),
                        inpaint_dilation_ratio_value,
                        inpaint_dilation_min,
                        inpaint_dilation_max,
                    )
                    for object_mask, trajectory in zip(
                        object_masks, trajectories
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
                ) = compute_soft_background_visibility_model(
                    object_masks_soft,
                    trajectories,
                    provided_inpaint_masks,
                    np.linspace(
                        -0.5,
                        0.5,
                        case_n_subframes,
                        dtype=np.float64,
                    ),
                    background_visibility_threshold_value,
                    background_safety_ratio,
                    background_safety_min,
                    background_safety_max,
                )
                needed_background_mask = np.logical_or.reduce(
                    provided_needed_masks
                )
                background_write_alpha = np.maximum.reduce(
                    provided_write_alpha_masks
                )
                provided_write_support = np.logical_or.reduce(
                    provided_write_support_masks
                )
                background_metadata = {
                    "method": "provided_background",
                    "inpaint_mask_area": int(inpaint_union_mask.sum()),
                    "needed_background_area": int(
                        needed_background_mask.sum()
                    ),
                    "background_write_mask_area": int(
                        provided_write_support.sum()
                    ),
                    "background_write_alpha_sum": float(
                        background_write_alpha.sum()
                    ),
                    "visibility_threshold": float(
                        background_visibility_threshold_value
                    ),
                    "safety_dilation_radii_px": np.asarray(
                        provided_safety_radii, dtype=np.int32
                    ),
                    "directional_filled_area": 0,
                    "nearest_filled_area": 0,
                    "fallback_area": 0,
                    "objects": [],
                }
                background_source = "provided"

            if show:
                show_layers(image, union_mask_soft, foreground, background)

            sensor_parameters = (
                sensor_model.sample_sensor_parameters(
                    sensor_config, case_seed + sensor_seed_offset
                )
                if sensor_variant
                else None
            )
            synthesis_result = synthesize_multi_object_motion_blur_A(
                foregrounds,
                background,
                object_masks_soft,
                trajectories,
                case_n_subframes,
                extra_crop_margin,
                generation_case_name=case_name,
                reference_image=image,
                start_exposure_weight=start_exposure_weight,
                terminal_exposure_fraction=terminal_exposure_fraction,
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
            metadata["mask_source"] = mask_source
            metadata["instance_mask_files"] = [
                str(path) for path in instance_mask_paths
            ]
            metadata["background_source"] = background_source
            metadata["background_file"] = (
                str(background_path) if background_path is not None else ""
            )
            metadata["background_completion"] = background_metadata
            metadata["render_mask_refinement"] = [
                {
                    "instance_id": int(instance_index),
                    "radius_px": int(radius),
                    "raw_area": int(raw_area),
                    "refined_area": int(refined_mask.sum()),
                }
                for instance_index, (
                    radius,
                    raw_area,
                    refined_mask,
                ) in enumerate(
                    zip(
                        render_refine_radii,
                        raw_mask_areas,
                        object_masks,
                    ),
                    start=1,
                )
            ]
            metadata["adaptive_subframes"] = bool(adaptive_subframes)
            metadata["center_reference_validated"] = True
            metadata["max_motion_distance_px"] = float(max_motion_distance)
            metadata["motion_sample_step_px"] = float(
                max_motion_distance / (case_n_subframes - 1)
            )

            gt_to_save = image
            background_to_save = background
            foreground_to_save = foreground
            mask_to_save = union_mask_soft
            inpaint_mask_to_save = inpaint_union_mask.astype(np.float64)
            needed_mask_to_save = needed_background_mask.astype(np.float64)
            background_write_alpha_to_save = background_write_alpha
            blur_to_save = blur_image
            sensor_blur_to_save = sensor_blur_image
            if crop_boundary:
                bound_h = int(metadata["crop_bound_h"])
                bound_w = int(metadata["crop_bound_w"])
                gt_to_save, _ = crop_any_array(gt_to_save, bound_h, bound_w)
                background_to_save, _ = crop_any_array(
                    background_to_save, bound_h, bound_w
                )
                foreground_to_save, _ = crop_any_array(
                    foreground_to_save, bound_h, bound_w
                )
                mask_to_save, _ = crop_any_array(mask_to_save, bound_h, bound_w)
                inpaint_mask_to_save, _ = crop_any_array(
                    inpaint_mask_to_save, bound_h, bound_w
                )
                needed_mask_to_save, _ = crop_any_array(
                    needed_mask_to_save, bound_h, bound_w
                )
                background_write_alpha_to_save, _ = crop_any_array(
                    background_write_alpha_to_save, bound_h, bound_w
                )
                blur_to_save, _ = crop_any_array(blur_to_save, bound_h, bound_w)
                if sensor_blur_to_save is not None:
                    sensor_blur_to_save, _ = crop_any_array(
                        sensor_blur_to_save, bound_h, bound_w
                    )
                if save_debug_subframes:
                    debug_frames = {
                        name: crop_any_array(frame, bound_h, bound_w)[0]
                        for name, frame in debug_frames.items()
                    }

            output_prefix = result_dir / f"{image_file.stem}_{case_name}"
            outputs = {
                "_gtimg.png": gt_to_save,
                "_blurryimg.png": blur_to_save,
                "_background.png": background_to_save,
                "_foreground.png": foreground_to_save,
                "_objmask.png": mask_to_save,
                "_inpaintmask.png": inpaint_mask_to_save,
                "_neededbackgroundmask.png": needed_mask_to_save,
                "_backgroundwritealpha.png": background_write_alpha_to_save,
            }
            if sensor_blur_to_save is not None:
                outputs["_blurryimg_sensor.png"] = sensor_blur_to_save
            for suffix, output_image in outputs.items():
                write_image_uint8(
                    output_prefix.with_name(output_prefix.name + suffix), output_image
                )
            if save_debug_subframes:
                for name, frame in debug_frames.items():
                    write_image_uint8(
                        output_prefix.with_name(
                            output_prefix.name + f"_subframe_{name}.png"
                        ),
                        frame,
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
                {
                    "meta": metadata,
                    "traj": {
                        "generation_case_name": case_name,
                        "objects": trajectories,
                    },
                },
                do_compression=True,
                long_field_names=True,
            )

    print(f"\nDone. Results are saved to: {result_dir}")


# =========================================================================
# 5. 本地函数区
# =========================================================================
def list_image_files_uav(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def list_instance_mask_paths(
    folder: str | Path, image_file: Path
) -> list[Path]:
    """读取 mask 生成脚本保存的 ``<stem>/obj_*.png`` 实例文件。"""
    image_instance_dir = Path(folder) / image_file.stem
    if not image_instance_dir.is_dir():
        return []
    return sorted(
        path
        for path in image_instance_dir.glob("obj_*.png")
        if path.is_file()
    )


def build_mask_path(folder: str | Path, image_file: Path, mode_name: str) -> Path:
    folder = Path(folder)
    if mode_name == "same_name":
        return folder / image_file.name
    if mode_name == "stem_png":
        return folder / f"{image_file.stem}.png"
    if mode_name == "same_stem":
        # 二值 mask 推荐保存为 PNG，因此同 stem 时优先选择 PNG。
        png_path = folder / f"{image_file.stem}.png"
        if png_path.is_file():
            return png_path

        # 没有 PNG 时再接受与 GT 完全同名的 mask。
        exact_path = folder / image_file.name
        if exact_path.is_file():
            return exact_path

        candidates = sorted(
            path
            for path in folder.glob(f"{image_file.stem}.*")
            if path.is_file()
            and path.stem == image_file.stem
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(
                f"同 stem 的 mask 存在多个候选：{image_file.stem} -> {names}"
            )

        # 返回期望的 PNG 路径，让主流程给出统一、清晰的缺失提示。
        return png_path
    raise ValueError(f"Unsupported mask_name_mode: {mode_name}")


def find_matching_background_path(folder: str | Path, image_file: Path) -> Path:
    """查找与输入图同名/同 stem 的真实背景，优先完全同名和 PNG。"""
    folder = Path(folder)
    exact_path = folder / image_file.name
    if exact_path.is_file():
        return exact_path

    png_path = folder / f"{image_file.stem}.png"
    if png_path.is_file():
        return png_path

    candidates = sorted(
        path
        for path in folder.glob(f"{image_file.stem}.*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"真实背景存在多个同 stem 候选文件：{image_file.stem} -> {names}"
        )
    raise FileNotFoundError(
        f"找不到 {image_file.name} 对应的真实背景；已检查目录：{folder}"
    )


def read_image_double_rgb(path: str | Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    elif data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)
    else:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float64)
        if data.max(initial=0.0) > 1.0:
            data /= 255.0
    return clip01_uav(data)


def srgb_to_linear_uav(image_srgb: np.ndarray) -> np.ndarray:
    """将 [0, 1] 标准 sRGB 转为线性光强，供 warp、合成和曝光积分使用。"""
    image_srgb = clip01_uav(np.asarray(image_srgb, dtype=np.float64))
    return np.where(
        image_srgb <= 0.04045,
        image_srgb / 12.92,
        ((image_srgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb_uav(image_linear: np.ndarray) -> np.ndarray:
    """将 [0, 1] 线性光强转换回标准 sRGB，供显示和文件保存使用。"""
    image_linear = clip01_uav(np.asarray(image_linear, dtype=np.float64))
    return clip01_uav(
        np.where(
            image_linear <= 0.0031308,
            12.92 * image_linear,
            1.055 * image_linear ** (1.0 / 2.4) - 0.055,
        )
    )


def premultiplied_srgb_to_linear_uav(
    foreground_srgb_premultiplied: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """把 ``sRGB * alpha`` 正确转换为 ``linear RGB * alpha``。

    gamma 变换与 alpha 乘法不可交换，因此不能直接对预乘前景调用
    ``srgb_to_linear_uav``。这里先安全地解除预乘、转换颜色空间，再重新
    预乘；alpha 本身始终保持在线性覆盖率空间。
    """
    foreground = np.asarray(foreground_srgb_premultiplied, dtype=np.float64)
    alpha = clip01_uav(np.asarray(alpha, dtype=np.float64))
    if foreground.shape[:2] != alpha.shape or foreground.ndim != 3:
        raise ValueError("预乘前景与 alpha 的尺寸不一致")
    alpha_channel = alpha[..., None]
    straight_srgb = np.divide(
        foreground,
        alpha_channel,
        out=np.zeros_like(foreground, dtype=np.float64),
        where=alpha_channel > 1e-12,
    )
    return srgb_to_linear_uav(straight_srgb) * alpha_channel


def write_image_uint8(path: str | Path, image: np.ndarray) -> None:
    array = np.rint(clip01_uav(np.asarray(image)) * 255.0).astype(np.uint8)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Cannot write image: {path}")


def read_binary_mask(path: str | Path, largest_component_only: bool) -> np.ndarray:
    mask_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    mask = mask_image > 0
    if not mask.any():
        raise ValueError(f"Mask is empty: {path}")
    if largest_component_only:
        count, labels, statistics, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count > 2:
            largest_label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
            mask = labels == largest_label
    return mask


def smooth_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    return clip01_uav(gaussian_filter(mask.astype(np.float64), sigma=sigma))


def refine_render_mask(
    mask: np.ndarray,
    refine_ratio: float,
    refine_min_px: int,
    refine_max_px: int,
) -> tuple[np.ndarray, int]:
    """修补自动分割的细小缺口，并覆盖容易遗漏的目标最外缘。

    这里的膨胀远小于用于背景补全的 inpaint mask 膨胀。前者会参与目标
    warp，目的是防止原位置留下彩色轮廓；后者只负责排除不可靠背景来源。
    """
    if refine_ratio < 0.0:
        raise ValueError("render mask refine ratio 不能小于 0")
    if refine_min_px < 0 or refine_max_px < refine_min_px:
        raise ValueError("render mask refine 必须满足 0 <= min_px <= max_px")
    _, _, _, object_scale = _object_geometry(mask)
    radius = int(
        np.clip(
            np.ceil(float(object_scale) * float(refine_ratio)),
            int(refine_min_px),
            int(refine_max_px),
        )
    )
    if radius <= 0:
        return mask.astype(bool).copy(), 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    closed = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    )
    refined = cv2.dilate(closed, kernel, iterations=1)
    return refined > 0, radius


def inpaint_background_rgb(
    image: np.ndarray, mask: np.ndarray, method: str = "biharmonic"
) -> np.ndarray:
    """补全目标遮挡区域；biharmonic 更接近 MATLAB regionfill 的 PDE 思路。"""
    method = method.lower()
    if method == "biharmonic":
        return clip01_uav(inpaint_biharmonic(image, mask, channel_axis=-1))
    if method == "telea":
        rgb_uint8 = np.rint(clip01_uav(image) * 255.0).astype(np.uint8)
        bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(
            bgr_uint8,
            mask.astype(np.uint8) * 255,
            3.0,
            cv2.INPAINT_TELEA,
        )
        return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    raise ValueError("inpaint_method must be 'biharmonic' or 'telea'")


def warp_binary_mask_fast(
    mask: np.ndarray,
    dx: float,
    dy: float,
    theta_deg: float,
    scale_value: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """用与 ``warp_frame_similarity`` 一致的几何变换快速移动二值 mask。"""
    height, width = mask.shape
    # OpenCV 的正角方向与本脚本的图像坐标角度相反，因此传入 -theta。
    transform = cv2.getRotationMatrix2D(
        (float(cx), float(cy)), -float(theta_deg), float(scale_value)
    )
    transform[0, 2] += float(dx)
    transform[1, 2] += float(dy)
    moved = cv2.warpAffine(
        mask.astype(np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return moved > 0


def compute_needed_background_mask(
    object_mask: np.ndarray,
    trajectory: dict[str, Any],
    motion_samples: int,
) -> np.ndarray:
    """返回曝光期间会被目标让出的原位置区域。

    该区域只位于初始 render mask 内。若一个像素在所有采样时刻都仍被目标
    覆盖，就无需依赖它的背景值；只要有一个时刻露出，就标为 needed。
    """
    if motion_samples < 2:
        raise ValueError("background motion_samples must be at least 2")
    cx = float(trajectory["object_centroid"][0] - 1.0)
    cy = float(trajectory["object_centroid"][1] - 1.0)
    always_covered = object_mask.astype(bool).copy()
    for tau in np.linspace(-0.5, 0.5, motion_samples, dtype=np.float64):
        dx, dy, theta, scale_value = evaluate_object_trajectory_at_time(
            trajectory, float(tau)
        )
        moved_mask = warp_binary_mask_fast(
            object_mask, dx, dy, theta, scale_value, cx, cy
        )
        always_covered &= moved_mask
    return object_mask.astype(bool) & ~always_covered


def compute_union_needed_background_mask(
    object_masks: list[np.ndarray],
    trajectories: list[dict[str, Any]],
    motion_samples: int,
) -> np.ndarray:
    if len(object_masks) != len(trajectories):
        raise ValueError("mask 与 trajectory 的实例数量必须一致")
    needed_masks = [
        compute_needed_background_mask(mask, trajectory, motion_samples)
        for mask, trajectory in zip(object_masks, trajectories)
    ]
    return np.logical_or.reduce(needed_masks)


def dilate_inpaint_mask(
    render_mask: np.ndarray,
    object_scale: float,
    dilation_ratio: float,
    dilation_min_px: int,
    dilation_max_px: int,
) -> tuple[np.ndarray, int]:
    """膨胀 render mask，避免补全算法从目标边缘继续采到目标颜色。"""
    radius = int(
        np.clip(
            np.ceil(float(object_scale) * float(dilation_ratio)),
            int(dilation_min_px),
            int(dilation_max_px),
        )
    )
    if radius <= 0:
        return render_mask.astype(bool).copy(), 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    dilated = cv2.dilate(
        render_mask.astype(np.uint8), kernel, iterations=1
    )
    return dilated > 0, radius


def warp_soft_mask_fast(
    mask_soft: np.ndarray,
    dx: float,
    dy: float,
    theta_deg: float,
    scale_value: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """用与最终目标渲染相同的双线性插值移动 soft mask。"""
    height, width = mask_soft.shape
    transform = cv2.getRotationMatrix2D(
        (float(cx), float(cy)), -float(theta_deg), float(scale_value)
    )
    transform[0, 2] += float(dx)
    transform[1, 2] += float(dy)
    moved = cv2.warpAffine(
        np.asarray(mask_soft, dtype=np.float32),
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(moved.astype(np.float64), 0.0, 1.0)


def compute_soft_needed_background_alpha(
    object_mask_soft: np.ndarray,
    trajectory: dict[str, Any],
    tau_list: np.ndarray,
) -> np.ndarray:
    """计算目标相对 GT 减少覆盖时所需的新背景权重。

    使用 ``max_tau(max(M_ref - M_tau, 0))``，而不是 ``1-M_tau``。
    因此目标静止时结果严格为零；只有相对于参考时刻新露出的部分才需要
    伪造背景。平台运动没有出现在这里，因为背景和目标共同受到平台矩阵，
    相对可见性只由目标自身运动决定。
    """
    reference_alpha = np.clip(
        np.asarray(object_mask_soft, dtype=np.float64), 0.0, 1.0
    )
    if reference_alpha.ndim != 2:
        raise ValueError("object_mask_soft 必须是二维数组")
    tau_values = np.asarray(tau_list, dtype=np.float64).reshape(-1)
    if tau_values.size < 2:
        raise ValueError("soft needed background 至少需要两个曝光时刻")
    cx = float(trajectory["object_centroid"][0] - 1.0)
    cy = float(trajectory["object_centroid"][1] - 1.0)
    needed_alpha = np.zeros_like(reference_alpha, dtype=np.float64)
    for tau in tau_values:
        dx, dy, theta, scale_value = evaluate_object_trajectory_at_time(
            trajectory, float(tau)
        )
        moved_alpha = warp_soft_mask_fast(
            reference_alpha, dx, dy, theta, scale_value, cx, cy
        )
        needed_alpha = np.maximum(
            needed_alpha, np.maximum(reference_alpha - moved_alpha, 0.0)
        )
    return np.clip(needed_alpha, 0.0, 1.0)


def build_background_write_alpha(
    needed_core_mask: np.ndarray,
    inpaint_mask: np.ndarray,
    object_scale: float,
    safety_ratio: float,
    safety_min_px: int,
    safety_max_px: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """在 needed 核心外建立受 inpaint mask 限制的渐变安全圈。

    核心区域 alpha 恒为 1；安全圈内从接近 1 平滑下降到 0；圈外为 0。
    这样不会削弱真正会露出的背景，又避免候选背景和原图之间硬切换。
    """
    if safety_ratio < 0.0:
        raise ValueError("background safety dilation ratio 不能小于 0")
    if safety_min_px < 0 or safety_max_px < safety_min_px:
        raise ValueError(
            "background safety dilation 必须满足 0 <= min_px <= max_px"
        )
    core = np.asarray(needed_core_mask, dtype=bool)
    source_exclusion = np.asarray(inpaint_mask, dtype=bool)
    radius = int(
        np.clip(
            np.ceil(float(object_scale) * float(safety_ratio)),
            int(safety_min_px),
            int(safety_max_px),
        )
    )
    if not core.any():
        return (
            np.zeros_like(core, dtype=np.float64),
            np.zeros_like(core, dtype=bool),
            radius,
        )

    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        support = cv2.dilate(
            core.astype(np.uint8), kernel, iterations=1
        ) > 0
    else:
        support = core.copy()
    support &= source_exclusion
    support |= core

    write_alpha = np.zeros_like(core, dtype=np.float64)
    write_alpha[core] = 1.0
    ring = support & ~core
    if radius > 0 and ring.any():
        # distanceTransform 对非零像素计算到最近零像素的距离；把 core 设为
        # 零即可得到安全圈像素到 needed 核心的距离。
        distance_to_core = cv2.distanceTransform(
            (~core).astype(np.uint8), cv2.DIST_L2, 5
        )
        write_alpha[ring] = np.clip(
            (radius + 1.0 - distance_to_core[ring]) / (radius + 1.0),
            0.0,
            1.0,
        )
    return write_alpha, support, radius


def compute_soft_background_visibility_model(
    object_masks_soft: list[np.ndarray],
    trajectories: list[dict[str, Any]],
    inpaint_masks: list[np.ndarray],
    tau_list: np.ndarray,
    visibility_threshold: float,
    safety_ratio: float,
    safety_min_px: int,
    safety_max_px: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[int],
]:
    """为每个实例生成 soft可见度、needed核心、安全圈和写回alpha。"""
    if not (
        len(object_masks_soft) == len(trajectories) == len(inpaint_masks)
    ):
        raise ValueError("soft mask、trajectory 与 inpaint mask 数量必须一致")
    if not 0.0 <= visibility_threshold < 1.0:
        raise ValueError("background visibility threshold 必须在 [0, 1) 内")

    needed_alpha_masks: list[np.ndarray] = []
    needed_core_masks: list[np.ndarray] = []
    write_support_masks: list[np.ndarray] = []
    write_alpha_masks: list[np.ndarray] = []
    safety_radii: list[int] = []
    for mask_soft, trajectory, inpaint_mask_now in zip(
        object_masks_soft, trajectories, inpaint_masks
    ):
        needed_alpha = compute_soft_needed_background_alpha(
            mask_soft, trajectory, tau_list
        )
        needed_core = needed_alpha > float(visibility_threshold)
        write_alpha, write_support, safety_radius = (
            build_background_write_alpha(
                needed_core,
                inpaint_mask_now,
                float(trajectory["object_scale"]),
                safety_ratio,
                safety_min_px,
                safety_max_px,
            )
        )
        needed_alpha_masks.append(needed_alpha)
        needed_core_masks.append(needed_core)
        write_support_masks.append(write_support)
        write_alpha_masks.append(write_alpha)
        safety_radii.append(safety_radius)
    return (
        needed_alpha_masks,
        needed_core_masks,
        write_support_masks,
        write_alpha_masks,
        safety_radii,
    )


def copy_background_from_motion_opposite_side(
    background: np.ndarray,
    source_image: np.ndarray,
    needed_mask: np.ndarray,
    reliable_source_mask: np.ndarray,
    final_dx: float,
    final_dy: float,
) -> np.ndarray:
    """从运动反方向复制可靠邻域，返回实际成功复制的像素 mask。

    对目标最终位移 ``(dx, dy)``，目标原位置的尾部会露出；目标像素 ``(x, y)``
    优先采用 ``(x-dx, y-dy)`` 附近的原图背景。若最近候选仍在目标或越界，
    再逐渐扩大取样距离；仍无法覆盖的位置交给最近可靠背景/Telea 兜底。
    """
    filled_mask = np.zeros_like(needed_mask, dtype=bool)
    if not needed_mask.any() or float(np.hypot(final_dx, final_dy)) < 1e-6:
        return filled_mask

    rows, columns = np.nonzero(needed_mask)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    roi_height, roi_width = y1 - y0, x1 - x0
    grid_x = np.broadcast_to(
        np.arange(x0, x1, dtype=np.float32)[None, :],
        (roi_height, roi_width),
    ).copy()
    grid_y = np.broadcast_to(
        np.arange(y0, y1, dtype=np.float32)[:, None],
        (roi_height, roi_width),
    ).copy()

    needed_roi = needed_mask[y0:y1, x0:x1]
    remaining = needed_roi.copy()
    background_roi = background[y0:y1, x0:x1]
    reliable_uint8 = reliable_source_mask.astype(np.uint8)
    motion_norm = float(np.hypot(final_dx, final_dy))
    trailing_x = -float(final_dx) / motion_norm
    trailing_y = -float(final_dy) / motion_norm

    # 优先使用目标运动后方、紧邻目标的真实背景带。若后方空间比待补区域
    # 短，则沿运动方向对这条纹理带做镜像往返采样，避免从远处复制一整块
    # 亮度不同的道路，也避免逐像素最近邻产生放射纹。
    projection = grid_x * trailing_x + grid_y * trailing_y
    trailing_boundary = float(projection[needed_roi].max())
    image_corners = (
        0.0,
        (source_image.shape[1] - 1) * trailing_x,
        (source_image.shape[0] - 1) * trailing_y,
        (source_image.shape[1] - 1) * trailing_x
        + (source_image.shape[0] - 1) * trailing_y,
    )
    border_safety = 4.0
    available_trailing_length = (
        max(image_corners) - trailing_boundary - 2.0 * border_safety
    )
    if available_trailing_length >= 16.0:
        distance_to_trailing_edge = np.maximum(
            trailing_boundary - projection, 0.0
        )
        period = 2.0 * available_trailing_length
        reflected_distance = np.mod(distance_to_trailing_edge, period)
        reflected_distance = np.where(
            reflected_distance <= available_trailing_length,
            reflected_distance,
            period - reflected_distance,
        )
        source_projection = (
            trailing_boundary + border_safety + reflected_distance
        )
        projection_shift = source_projection - projection
        trailing_source_x = (
            grid_x + np.float32(trailing_x) * projection_shift
        ).astype(np.float32)
        trailing_source_y = (
            grid_y + np.float32(trailing_y) * projection_shift
        ).astype(np.float32)
        trailing_is_reliable = cv2.remap(
            reliable_uint8,
            trailing_source_x,
            trailing_source_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        use_trailing_strip = remaining & trailing_is_reliable
        if use_trailing_strip.any():
            trailing_candidate = cv2.remap(
                source_image,
                trailing_source_x,
                trailing_source_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            # 镜像往返在折返点处位置连续但梯度反向，细纹理上可能形成一条
            # 对称接缝；只对折返点附近几像素做轻微平滑。
            seam_band = (
                np.minimum(
                    np.abs(reflected_distance),
                    np.abs(reflected_distance - available_trailing_length),
                )
                <= 4.0
            )
            if (seam_band & use_trailing_strip).any():
                smoothed_candidate = cv2.GaussianBlur(
                    trailing_candidate, (0, 0), sigmaX=2.0, sigmaY=2.0
                )
                trailing_candidate[seam_band] = smoothed_candidate[seam_band]
            trailing_candidate = cv2.GaussianBlur(
                trailing_candidate, (0, 0), sigmaX=0.6, sigmaY=0.6
            )
            background_roi[use_trailing_strip] = trailing_candidate[
                use_trailing_strip
            ]
            filled_mask[y0:y1, x0:x1] |= use_trailing_strip
            remaining &= ~use_trailing_strip

    # 先评估每个候选整体能覆盖多少待补区域，再优先使用覆盖率最高的一整块
    # 纹理。旧实现逐像素优先选择最近来源，会把不同方向/距离的像素拼成明显
    # 的放射纹和分块接缝。
    candidate_factors = (
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        -1.0,
        -1.25,
        -1.5,
        -2.0,
        -2.5,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -8.0,
        -10.0,
    )
    scored_factors: list[tuple[int, float]] = []
    for distance_factor in candidate_factors:
        source_x = grid_x - np.float32(final_dx * distance_factor)
        source_y = grid_y - np.float32(final_dy * distance_factor)
        source_is_reliable = cv2.remap(
            reliable_uint8,
            source_x,
            source_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        coverage = int((needed_roi & source_is_reliable).sum())
        if coverage > 0:
            scored_factors.append((coverage, float(distance_factor)))
    scored_factors.sort(key=lambda item: (-item[0], abs(item[1]), item[1] < 0))

    for _, distance_factor in scored_factors:
        if not remaining.any():
            break
        source_x = grid_x - np.float32(final_dx * distance_factor)
        source_y = grid_y - np.float32(final_dy * distance_factor)
        candidate = cv2.remap(
            source_image,
            source_x,
            source_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        source_is_reliable = cv2.remap(
            reliable_uint8,
            source_x,
            source_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        use_candidate = remaining & source_is_reliable
        background_roi[use_candidate] = candidate[use_candidate]
        filled_mask[y0:y1, x0:x1] |= use_candidate
        remaining &= ~use_candidate
    return filled_mask


def feather_background_patch_to_candidate(
    background: np.ndarray,
    candidate_background: np.ndarray,
    filled_mask: np.ndarray,
    feather_px: int,
    candidate_gain: float,
) -> None:
    """将复制纹理的硬边缘渐变到连续的候选背景中。

    只修改 ``filled_mask`` 内部。渐变两端分别是复制来的真实背景纹理和
    已经排除了目标颜色的完整 inpaint 候选背景，不会把原目标像素混回来。
    """
    if not filled_mask.any():
        return
    feather_px = max(1, int(feather_px))
    distance_inside = cv2.distanceTransform(
        filled_mask.astype(np.uint8), cv2.DIST_L2, 5
    )
    normalized_distance = np.clip(
        distance_inside / float(feather_px), 0.0, 1.0
    )
    # smoothstep 比线性 alpha 在过渡带两端更平滑，不容易形成新的亮度折线。
    texture_alpha = (
        normalized_distance
        * normalized_distance
        * (3.0 - 2.0 * normalized_distance)
    )
    local_alpha = texture_alpha[filled_mask, None]
    copied_values = background[filled_mask]
    candidate_values = clip01_uav(
        candidate_background[filled_mask] * float(candidate_gain)
    )
    background[filled_mask] = (
        local_alpha * copied_values
        + (1.0 - local_alpha) * candidate_values
    )


def match_background_patch_luminance(
    background: np.ndarray,
    source_image: np.ndarray,
    needed_mask: np.ndarray,
    inpaint_mask: np.ndarray,
    source_exclusion_mask: np.ndarray,
    object_scale: float,
    ring_ratio: float,
    ring_min_px: int,
    ring_max_px: int,
    gain_range: tuple[float, float],
) -> tuple[float, float, float, int, int]:
    """将补丁的低频亮度匹配到目标外侧的真实背景环。

    使用中位亮度可以避开少量高光、路面裂纹和噪声。这里只做统一 RGB
    增益，因此会保留复制纹理的颗粒、颜色比例和空间结构。
    """
    if not needed_mask.any():
        return 1.0, 0.0, 0.0, 0, 0
    ring_width = int(
        np.clip(
            np.ceil(float(object_scale) * float(ring_ratio)),
            int(ring_min_px),
            int(ring_max_px),
        )
    )
    if ring_width <= 0:
        return 1.0, 0.0, 0.0, 0, 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_width + 1, 2 * ring_width + 1)
    )
    reference_ring = (
        cv2.dilate(inpaint_mask.astype(np.uint8), kernel, iterations=1) > 0
    ) & ~source_exclusion_mask
    source_count = int(needed_mask.sum())
    reference_count = int(reference_ring.sum())
    if source_count < 64 or reference_count < 64:
        return 1.0, 0.0, 0.0, source_count, reference_count

    luminance_weights = np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float64
    )
    source_luminance = background[needed_mask] @ luminance_weights
    reference_luminance = source_image[reference_ring] @ luminance_weights
    source_median = float(np.median(source_luminance))
    reference_median = float(np.median(reference_luminance))
    gain_min, gain_max = gain_range
    gain = float(
        np.clip(
            reference_median / max(source_median, 1e-6),
            float(gain_min),
            float(gain_max),
        )
    )
    background[needed_mask] = clip01_uav(
        background[needed_mask] * gain
    )
    return (
        gain,
        source_median,
        reference_median,
        source_count,
        reference_count,
    )


def build_trajectory_aware_background(
    image: np.ndarray,
    object_masks: list[np.ndarray],
    trajectories: list[dict[str, Any]],
    *,
    object_masks_soft: list[np.ndarray] | None = None,
    exposure_tau_list: np.ndarray | None = None,
    method: str,
    dilation_ratio: float,
    dilation_min_px: int,
    dilation_max_px: int,
    motion_samples: int,
    boundary_feather_ratio: float,
    boundary_feather_min_px: int,
    boundary_feather_max_px: int,
    tone_ring_ratio: float,
    tone_ring_min_px: int,
    tone_ring_max_px: int,
    tone_gain_range: tuple[float, float],
    visibility_threshold: float = background_visibility_threshold,
    safety_dilation_ratio: float = background_safety_dilation_ratio,
    safety_dilation_min_px: int = background_safety_dilation_min_px,
    safety_dilation_max_px: int = background_safety_dilation_max_px,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """构造仅供分层渲染使用的静止背景缓存。

    ``render mask`` 始终保持原始精确轮廓，只用于移动/合成目标；
    膨胀后的轮廓只用于排除目标本体及边缘残色的取样；
    ``needed mask`` 使用与最终渲染相同的 soft mask、双线性插值和曝光时刻，
    表示运动过程中相对 GT 新露出的目标下方背景区域。

    补全算法在完整膨胀区域上求解候选背景，但不把整块结果写回。needed
    核心完整写入，核心外只有一个很小的安全圈，并通过连续 alpha 衰减到
    原图。目标中曝光期间始终被覆盖的部分仍保留原 GT。
    """
    if not object_masks or len(object_masks) != len(trajectories):
        raise ValueError("至少需要一组数量一致的 mask 与 trajectory")
    if dilation_ratio < 0.0:
        raise ValueError("inpaint dilation ratio 不能小于 0")
    if dilation_min_px < 0 or dilation_max_px < dilation_min_px:
        raise ValueError("inpaint dilation 必须满足 0 <= min_px <= max_px")
    if motion_samples < 2:
        raise ValueError("background motion_samples must be at least 2")
    if boundary_feather_ratio < 0.0:
        raise ValueError("background boundary feather ratio 不能小于 0")
    if (
        boundary_feather_min_px < 0
        or boundary_feather_max_px < boundary_feather_min_px
    ):
        raise ValueError(
            "background boundary feather 必须满足 0 <= min_px <= max_px"
        )
    if tone_ring_ratio < 0.0:
        raise ValueError("background tone ring ratio 不能小于 0")
    if tone_ring_min_px < 0 or tone_ring_max_px < tone_ring_min_px:
        raise ValueError(
            "background tone ring 必须满足 0 <= min_px <= max_px"
        )
    if (
        tone_gain_range[0] <= 0.0
        or tone_gain_range[1] < tone_gain_range[0]
    ):
        raise ValueError("background tone gain 必须满足 0 < min <= max")

    if object_masks_soft is None:
        object_masks_soft = [
            mask.astype(np.float64) for mask in object_masks
        ]
    if len(object_masks_soft) != len(object_masks):
        raise ValueError("object_masks_soft 与 object_masks 数量必须一致")
    for mask, mask_soft in zip(object_masks, object_masks_soft):
        if np.asarray(mask_soft).shape != np.asarray(mask).shape:
            raise ValueError("soft mask 与 binary mask 尺寸必须一致")
    if exposure_tau_list is None:
        exposure_tau_list = np.linspace(
            -0.5, 0.5, motion_samples, dtype=np.float64
        )
    else:
        exposure_tau_list = np.asarray(
            exposure_tau_list, dtype=np.float64
        ).reshape(-1)
    if exposure_tau_list.size < 2:
        raise ValueError("exposure_tau_list 至少需要两个时刻")

    render_union_mask = np.logical_or.reduce(object_masks)
    inpaint_masks: list[np.ndarray] = []
    dilation_radii: list[int] = []
    for object_mask, trajectory in zip(object_masks, trajectories):
        inpaint_mask_now, radius = dilate_inpaint_mask(
            object_mask,
            float(trajectory["object_scale"]),
            dilation_ratio,
            dilation_min_px,
            dilation_max_px,
        )
        inpaint_masks.append(inpaint_mask_now)
        dilation_radii.append(radius)
    inpaint_union_mask = np.logical_or.reduce(inpaint_masks)

    (
        needed_alpha_masks,
        needed_masks,
        write_support_masks,
        write_alpha_masks,
        safety_radii,
    ) = compute_soft_background_visibility_model(
        object_masks_soft,
        trajectories,
        inpaint_masks,
        exposure_tau_list,
        visibility_threshold,
        safety_dilation_ratio,
        safety_dilation_min_px,
        safety_dilation_max_px,
    )
    needed_union_mask = np.logical_or.reduce(needed_masks)
    write_support_union_mask = np.logical_or.reduce(write_support_masks)
    background_write_alpha = np.maximum.reduce(write_alpha_masks)

    # 求解时屏蔽完整目标及其边缘，避免目标颜色成为背景来源。该结果只是
    # 候选背景；真正写回只发生在 needed 核心和小范围渐变安全圈内。
    candidate_background = inpaint_background_rgb(
        image, inpaint_union_mask, method
    )
    background = image.copy()
    background[write_support_union_mask] = candidate_background[
        write_support_union_mask
    ]

    # 在写回支撑区域内优先使用运动后方的真实二维纹理块。整块平移复制会
    # 保留道路/水面的局部纹理，不会像 PDE/Telea 那样把边缘颜色放射状
    # 延伸到内部。复制失败的少量像素继续保留上面的 inpaint 兜底结果。
    reliable_source_mask = ~inpaint_union_mask
    nearby_filled_union = np.zeros_like(write_support_union_mask, dtype=bool)
    nearby_filled_masks: list[np.ndarray] = []
    boundary_feather_radii: list[int] = []
    for write_support_mask, trajectory in zip(
        write_support_masks, trajectories
    ):
        start_dx, start_dy, _, _ = evaluate_object_trajectory_at_time(
            trajectory, -0.5
        )
        end_dx, end_dy, _, _ = evaluate_object_trajectory_at_time(
            trajectory, 0.5
        )
        span_dx = end_dx - start_dx
        span_dy = end_dy - start_dy
        center_x = float(trajectory["object_centroid"][0] - 1.0)
        center_y = float(trajectory["object_centroid"][1] - 1.0)
        needed_rows, needed_columns = np.nonzero(write_support_mask)
        signed_progress = (
            (needed_columns - center_x) * span_dx
            + (needed_rows - center_y) * span_dy
        )
        start_side_needed = np.zeros_like(write_support_mask, dtype=bool)
        start_side_needed[
            needed_rows[signed_progress < 0.0],
            needed_columns[signed_progress < 0.0],
        ] = True
        end_side_needed = write_support_mask & ~start_side_needed

        # 中心参考会在 GT 两侧都露出背景：起点侧从运动反方向取样，
        # 终点侧从运动正方向取样，避免把同一侧纹理拉过整个目标。
        start_side_filled = copy_background_from_motion_opposite_side(
            background,
            image,
            start_side_needed,
            reliable_source_mask,
            span_dx,
            span_dy,
        )
        end_side_filled = copy_background_from_motion_opposite_side(
            background,
            image,
            end_side_needed,
            reliable_source_mask,
            -span_dx,
            -span_dy,
        )
        nearby_filled = start_side_filled | end_side_filled
        nearby_filled_union |= nearby_filled
        nearby_filled_masks.append(nearby_filled)
        feather_radius = int(
            np.clip(
                np.ceil(
                    float(trajectory["object_scale"])
                    * float(boundary_feather_ratio)
                ),
                int(boundary_feather_min_px),
                int(boundary_feather_max_px),
            )
        )
        boundary_feather_radii.append(feather_radius)

    # 先用不含任何目标像素的外侧真实背景环匹配低频亮度，再在复制区域内部
    # 的窄带过渡到同样经过亮度匹配的候选背景。这样同时消除整块亮/暗补丁
    # 和复制块内部边界；稍后再用 background_write_alpha 处理写回区域与原图
    # 之间的外边界。
    tone_match_results: list[tuple[float, float, float, int, int]] = []
    for (
        write_support_mask,
        inpaint_mask_now,
        nearby_filled,
        feather_radius,
        trajectory,
    ) in zip(
        write_support_masks,
        inpaint_masks,
        nearby_filled_masks,
        boundary_feather_radii,
        trajectories,
    ):
        tone_result = match_background_patch_luminance(
            background,
            image,
            write_support_mask,
            inpaint_mask_now,
            inpaint_union_mask,
            float(trajectory["object_scale"]),
            tone_ring_ratio,
            tone_ring_min_px,
            tone_ring_max_px,
            tone_gain_range,
        )
        tone_match_results.append(tone_result)
        tone_gain = tone_result[0]
        if feather_radius > 0:
            feather_background_patch_to_candidate(
                background,
                candidate_background,
                nearby_filled,
                feather_radius,
                tone_gain,
            )

    # 膨胀环的大部分和不需要露出的目标内部都只作为不可信取样区，不写入。
    cleanup_ring_mask = inpaint_union_mask & ~render_union_mask

    changed_outside_support = (
        np.max(np.abs(background - image), axis=2) > (0.5 / 255.0)
    ) & ~write_support_union_mask
    if changed_outside_support.any():
        raise RuntimeError(
            "背景补全越出了 background write support mask。"
        )

    # needed 核心 alpha=1；安全圈内连续衰减；圈外 alpha=0。候选背景与
    # 原图因此没有二值硬切换，同时真正会露出的核心不会混回目标颜色。
    background = (
        background_write_alpha[..., None] * background
        + (1.0 - background_write_alpha[..., None]) * image
    )

    object_metadata: list[dict[str, Any]] = []
    for instance_index, (
        needed_mask,
        needed_alpha,
        write_support_mask,
        write_alpha,
        trajectory,
        radius,
        safety_radius,
        nearby_filled,
        feather_radius,
        tone_result,
    ) in enumerate(
        zip(
            needed_masks,
            needed_alpha_masks,
            write_support_masks,
            write_alpha_masks,
            trajectories,
            dilation_radii,
            safety_radii,
            nearby_filled_masks,
            boundary_feather_radii,
            tone_match_results,
        ),
        start=1,
    ):
        start_dx, start_dy, _, _ = evaluate_object_trajectory_at_time(
            trajectory, -0.5
        )
        end_dx, end_dy, _, _ = evaluate_object_trajectory_at_time(
            trajectory, 0.5
        )
        needed_area = int(needed_mask.sum())
        write_support_area = int(write_support_mask.sum())
        nearby_area = int((nearby_filled & write_support_mask).sum())
        fallback_area = write_support_area - nearby_area
        (
            tone_gain,
            patch_luminance_median,
            reference_luminance_median,
            tone_source_area,
            tone_reference_area,
        ) = tone_result
        object_metadata.append(
            {
                "instance_id": int(
                    trajectory.get("instance_id", instance_index)
                ),
                "inpaint_dilation_radius_px": int(radius),
                "background_safety_dilation_radius_px": int(safety_radius),
                "background_boundary_feather_px": int(feather_radius),
                "background_tone_gain": float(tone_gain),
                "patch_luma_median_before": float(
                    patch_luminance_median
                ),
                "reference_luma_median": float(
                    reference_luminance_median
                ),
                "tone_source_area": int(tone_source_area),
                "tone_reference_area": int(tone_reference_area),
                "needed_background_area": needed_area,
                "soft_needed_alpha_sum": float(needed_alpha.sum()),
                "soft_needed_alpha_max": float(needed_alpha.max(initial=0.0)),
                "background_write_mask_area": write_support_area,
                "background_write_alpha_sum": float(write_alpha.sum()),
                "nearby_texture_filled_area": nearby_area,
                "needed_inpaint_area": fallback_area,
                "directional_filled_area": nearby_area,
                "cleanup_ring_directional_area": 0,
                "nearest_filled_area": 0,
                "fallback_area": fallback_area,
                "start_dx": float(start_dx),
                "start_dy": float(start_dy),
                "end_dx": float(end_dx),
                "end_dy": float(end_dy),
                "span_dx": float(end_dx - start_dx),
                "span_dy": float(end_dy - start_dy),
            }
        )

    needed_area = int(needed_union_mask.sum())
    write_support_area = int(write_support_union_mask.sum())
    nearby_area = int(
        (nearby_filled_union & write_support_union_mask).sum()
    )
    fallback_area = write_support_area - nearby_area
    changed_in_safety_ring = int(
        (
            (np.max(np.abs(background - image), axis=2) > (0.5 / 255.0))
            & write_support_union_mask
            & ~needed_union_mask
        ).sum()
    )
    metadata: dict[str, Any] = {
        "method": (
            "soft_visibility_needed_with_safety_alpha_"
            f"nearby_texture_and_{method}_fallback"
        ),
        "render_mask_area": int(render_union_mask.sum()),
        # inpaint_mask 是生成候选背景和排除不可靠源像素时使用的完整膨胀
        # 区域；真正写回背景的支撑范围由 background_write_mask_area 记录。
        "inpaint_mask_area": int(inpaint_union_mask.sum()),
        "background_write_mask_area": write_support_area,
        "background_write_alpha_sum": float(background_write_alpha.sum()),
        "background_write_alpha_max": float(
            background_write_alpha.max(initial=0.0)
        ),
        "source_exclusion_mask_area": int(inpaint_union_mask.sum()),
        "cleanup_ring_area": int(cleanup_ring_mask.sum()),
        "cleanup_ring_directional_area": 0,
        "changed_outside_render_area": 0,
        "changed_outside_needed_area": changed_in_safety_ring,
        "changed_outside_write_support_area": 0,
        "needed_background_area": needed_area,
        "nearby_texture_filled_area": nearby_area,
        "needed_inpaint_area": fallback_area,
        "directional_filled_area": nearby_area,
        "nearest_filled_area": 0,
        "fallback_area": fallback_area,
        "inpaint_dilation_ratio": float(dilation_ratio),
        "inpaint_dilation_min_px": int(dilation_min_px),
        "inpaint_dilation_max_px": int(dilation_max_px),
        "motion_samples": int(exposure_tau_list.size),
        "exposure_tau_list": exposure_tau_list[:, None],
        "needed_visibility_model": "max_positive_alpha_drop_from_reference",
        "visibility_threshold": float(visibility_threshold),
        "safety_dilation_ratio": float(safety_dilation_ratio),
        "safety_dilation_min_px": int(safety_dilation_min_px),
        "safety_dilation_max_px": int(safety_dilation_max_px),
        "boundary_feather_ratio": float(boundary_feather_ratio),
        "boundary_feather_min_px": int(boundary_feather_min_px),
        "boundary_feather_max_px": int(boundary_feather_max_px),
        "tone_ring_ratio": float(tone_ring_ratio),
        "tone_ring_min_px": int(tone_ring_min_px),
        "tone_ring_max_px": int(tone_ring_max_px),
        "tone_gain_min": float(tone_gain_range[0]),
        "tone_gain_max": float(tone_gain_range[1]),
        "objects": object_metadata,
    }
    return (
        clip01_uav(background),
        inpaint_union_mask,
        needed_union_mask,
        background_write_alpha,
        metadata,
    )


def _object_geometry(
    object_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    rows, columns = np.nonzero(object_mask)
    if columns.size == 0:
        raise ValueError("Object mask is empty")
    x_min, x_max = int(columns.min()), int(columns.max())
    y_min, y_max = int(rows.min()), int(rows.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    # 按 MATLAB regionprops 坐标语义保存：BoundingBox 左上角位于像素边界，
    # Centroid 使用 1-based 像素中心坐标。
    bounding_box = np.asarray(
        [x_min + 0.5, y_min + 0.5, width, height], dtype=np.float64
    )
    centroid = np.asarray(
        [columns.mean() + 1.0, rows.mean() + 1.0], dtype=np.float64
    )
    area = int(columns.size)
    return bounding_box, centroid, area, float(max(width, height))


def estimate_object_principal_axis_deg(object_mask: np.ndarray) -> float:
    """估计目标长轴方向；图像坐标中 x 向右、y 向下、正角为顺时针。"""
    rows, columns = np.nonzero(object_mask)
    if columns.size < 2:
        return 0.0
    coordinates = np.column_stack(
        (columns - columns.mean(), rows - rows.mean())
    ).astype(np.float64)
    covariance = coordinates.T @ coordinates / float(coordinates.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[-1]) <= np.finfo(np.float64).eps:
        return 0.0
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle_deg = float(np.rad2deg(np.arctan2(axis[1], axis[0])))
    # 长轴存在 180 度二义性。统一到 [-90, 90)，保证结果可复现；对于以曝光
    # 中点为参考的对称直线模糊，选择相反方向不会改变最终模糊。
    return float((angle_deg + 90.0) % 180.0 - 90.0)


def normalize_undirected_axis_deg(angle_deg: float) -> float:
    """把具有180度二义性的轴方向统一到 ``[-90, 90)``。"""
    return float((float(angle_deg) + 90.0) % 180.0 - 90.0)


def estimate_object_front_back_axis(
    object_mask: np.ndarray,
    aspect_ratio_threshold: float = (
        front_back_axis_aspect_ratio_threshold
    ),
) -> tuple[float, dict[str, Any]]:
    """从单个实例 mask 自动估计车辆/船在图像中的前后运动轴。

    对明显细长的俯视轮廓使用有向外接矩形长轴；对接近方形的正面/背面
    轮廓使用短轴，因为车辆物理长度在这种视角下会被透视压缩。返回的轴
    只有180度方向，不尝试从 mask 判断车头和车尾。
    """
    if aspect_ratio_threshold <= 1.0:
        raise ValueError("front/back axis aspect ratio threshold 必须大于 1")
    mask_u8 = np.asarray(object_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [contour for contour in contours if contour.shape[0] >= 3]
    if not contours:
        raise ValueError("Object mask does not contain a usable contour")
    contour = max(contours, key=cv2.contourArea)
    (_, _), (rectangle_width, rectangle_height), raw_angle_deg = (
        cv2.minAreaRect(contour)
    )
    rectangle_width = float(rectangle_width)
    rectangle_height = float(rectangle_height)
    short_side = min(rectangle_width, rectangle_height)
    long_side = max(rectangle_width, rectangle_height)
    if short_side <= np.finfo(np.float64).eps:
        raise ValueError("Object oriented bounding rectangle is degenerate")

    long_axis_deg = float(raw_angle_deg)
    if rectangle_height > rectangle_width:
        long_axis_deg += 90.0
    long_axis_deg = normalize_undirected_axis_deg(long_axis_deg)
    short_axis_deg = normalize_undirected_axis_deg(long_axis_deg + 90.0)
    oriented_aspect_ratio = float(long_side / short_side)
    use_long_axis = (
        oriented_aspect_ratio >= float(aspect_ratio_threshold)
    )
    front_back_axis_deg = (
        long_axis_deg if use_long_axis else short_axis_deg
    )
    metadata: dict[str, Any] = {
        "front_back_axis_deg": float(front_back_axis_deg),
        "oriented_long_axis_deg": float(long_axis_deg),
        "oriented_short_axis_deg": float(short_axis_deg),
        "oriented_bbox_long_side_px": long_side,
        "oriented_bbox_short_side_px": short_side,
        "oriented_bbox_aspect_ratio": oriented_aspect_ratio,
        "front_back_axis_aspect_ratio_threshold": float(
            aspect_ratio_threshold
        ),
        "front_back_axis_rule": (
            "oriented_bbox_long_axis_for_elongated_top_view"
            if use_long_axis
            else "oriented_bbox_short_axis_for_front_or_rear_view"
        ),
        "front_back_direction_ambiguity": (
            "mask_only_axis_has_180_degree_head_tail_ambiguity"
        ),
    }
    return float(front_back_axis_deg), metadata


def make_object_trajectory(
    case_name: str,
    case_seed: int,
    object_mask: np.ndarray,
    *,
    front_back_axis_threshold: float = (
        front_back_axis_aspect_ratio_threshold
    ),
) -> dict[str, Any]:
    bounding_box, centroid, area, object_scale = _object_geometry(object_mask)
    principal_axis_deg = estimate_object_principal_axis_deg(object_mask)
    front_back_axis_deg, axis_metadata = estimate_object_front_back_axis(
        object_mask, front_back_axis_threshold
    )
    initial_heading_deg = front_back_axis_deg
    trajectory: dict[str, Any] = {
        "case_name": case_name,
        "case_seed": int(case_seed),
        "object_bbox": bounding_box,
        "object_centroid": centroid,
        "object_area": area,
        "object_scale": object_scale,
        "initial_heading_deg": initial_heading_deg,
        "motion_heading_deg": initial_heading_deg,
        "motion_axis_constraint": (
            "center_path_tangent_parallel_to_estimated_front_back_axis"
        ),
        "mask_pca_principal_axis_deg": principal_axis_deg,
        **axis_metadata,
    }
    if case_name == "01_obj_linear_motion":
        # 目标沿自动估计的前后轴做匀速直线运动；曝光中心与 GT 对齐。
        total_distance = 0.15* object_scale
        heading_rad = np.deg2rad(initial_heading_deg)
        trajectory.update(
            description="constant-speed straight motion along estimated front-back axis",
            total_distance=total_distance,
            total_dx=total_distance * np.cos(heading_rad),
            total_dy=total_distance * np.sin(heading_rad),
            total_theta_deg=0.0,
            total_zoom=0.0,
        )
    elif case_name == "02_obj_curved_motion":
        # 圆弧总弧长与总转角均指曝光开始到曝光结束，GT 位于圆弧中点。
        # 目标自身旋转角始终等于轨迹切线方向的变化。
        total_path_length = 0.2 * object_scale
        turn_sign = 1.0 if case_seed % 2 == 0 else -1.0
        total_turn_angle_deg = turn_sign * 4.0
        total_turn_angle_rad = float(np.deg2rad(total_turn_angle_deg))
        trajectory.update(
            description="circular-arc center motion with tangent-aligned self rotation",
            total_path_length=total_path_length,
            total_turn_angle_deg=total_turn_angle_deg,
            turn_radius=total_path_length / abs(total_turn_angle_rad),
            total_theta_deg=total_turn_angle_deg,
            total_zoom=0.0,
        )
    else:
        raise ValueError(f"Unknown object motion case: {case_name}")
    return trajectory


def make_independent_object_trajectory(
    generation_case_name: str,
    instance_seed: int,
    object_mask: np.ndarray,
    *,
    forced_motion_case: str | None = None,
    linear_probability: float = linear_motion_probability,
    linear_ratio_range: tuple[float, float] = linear_distance_ratio_range,
    curved_ratio_range: tuple[float, float] = curved_path_ratio_range,
    turn_angle_range: tuple[float, float] = curved_turn_angle_deg_range,
    front_back_axis_threshold: float = (
        front_back_axis_aspect_ratio_threshold
    ),
) -> dict[str, Any]:
    """为实例选择运动类型和强度，并把中心切线约束到目标前后轴。"""
    rng = np.random.default_rng(instance_seed)
    if forced_motion_case is not None:
        if forced_motion_case not in {
            "01_obj_linear_motion",
            "02_obj_curved_motion",
        }:
            raise ValueError(f"Unknown forced motion case: {forced_motion_case}")
        selected_case = forced_motion_case
    elif generation_case_name == "01_obj_independent_random":
        selected_case = (
            "01_obj_linear_motion"
            if float(rng.random()) < linear_probability
            else "02_obj_curved_motion"
        )
    elif generation_case_name in {
        "01_obj_linear_motion",
        "02_obj_curved_motion",
    }:
        # 保留旧 --cases 用法：可强制所有实例使用同一种运动，但强度仍独立。
        selected_case = generation_case_name
    else:
        raise ValueError(f"Unknown object motion case: {generation_case_name}")

    trajectory = make_object_trajectory(
        selected_case,
        instance_seed,
        object_mask,
        front_back_axis_threshold=front_back_axis_threshold,
    )
    direction_sign = 1.0 if int(rng.integers(0, 2)) == 0 else -1.0
    front_back_axis_deg = float(trajectory["front_back_axis_deg"])
    heading_deg = front_back_axis_deg
    if direction_sign < 0:
        heading_deg += 180.0
    trajectory["initial_heading_deg"] = heading_deg
    trajectory["motion_heading_deg"] = heading_deg
    trajectory["generation_case_name"] = generation_case_name
    trajectory["direction_sign"] = direction_sign
    trajectory["motion_axis_constraint"] = (
        "center_path_tangent_parallel_to_estimated_front_back_axis"
    )

    object_scale = float(trajectory["object_scale"])
    if selected_case == "01_obj_linear_motion":
        strength_ratio = float(rng.uniform(*linear_ratio_range))
        total_distance = strength_ratio * object_scale
        heading_rad = float(np.deg2rad(heading_deg))
        trajectory.update(
            total_distance=total_distance,
            total_dx=total_distance * np.cos(heading_rad),
            total_dy=total_distance * np.sin(heading_rad),
            motion_strength_ratio=strength_ratio,
        )
    else:
        strength_ratio = float(rng.uniform(*curved_ratio_range))
        total_path_length = strength_ratio * object_scale
        turn_sign = 1.0 if int(rng.integers(0, 2)) == 0 else -1.0
        turn_angle_magnitude = float(rng.uniform(*turn_angle_range))
        total_turn_angle_deg = turn_sign * turn_angle_magnitude
        total_turn_angle_rad = float(np.deg2rad(total_turn_angle_deg))
        trajectory.update(
            total_path_length=total_path_length,
            total_turn_angle_deg=total_turn_angle_deg,
            turn_radius=total_path_length / abs(total_turn_angle_rad),
            total_theta_deg=total_turn_angle_deg,
            motion_strength_ratio=strength_ratio,
        )
    return trajectory


def make_balanced_motion_case_assignments(
    object_count: int,
    linear_probability: float,
    assignment_seed: int,
) -> tuple[str, ...]:
    """按整批目标数生成接近指定比例、顺序随机且可复现的类型列表。"""
    if object_count < 0:
        raise ValueError("object_count 不能小于 0")
    if not 0.0 <= linear_probability <= 1.0:
        raise ValueError("linear_probability 必须在 [0, 1] 内")
    linear_count = int(np.floor(object_count * linear_probability + 0.5))
    assignments = np.asarray(
        ["01_obj_linear_motion"] * linear_count
        + ["02_obj_curved_motion"] * (object_count - linear_count),
        dtype=object,
    )
    np.random.default_rng(assignment_seed).shuffle(assignments)
    return tuple(str(value) for value in assignments.tolist())


def trajectory_motion_distance(trajectory: dict[str, Any]) -> float:
    """返回曝光开始到结束的轨迹长度，单位为像素。"""
    case_name = str(trajectory["case_name"])
    if case_name == "01_obj_linear_motion":
        return float(trajectory["total_distance"])
    if case_name == "02_obj_curved_motion":
        return float(trajectory["total_path_length"])
    raise ValueError(f"Unknown object motion case: {case_name}")


def evaluate_object_trajectory_at_time(
    trajectory: dict[str, Any], tau: float
) -> tuple[float, float, float, float]:
    """计算中心参考轨迹；tau=-0.5/0/0.5 对应曝光首/中/末帧。"""
    case_name = trajectory["case_name"]
    if case_name == "01_obj_linear_motion":
        return (
            float(trajectory["total_dx"] * tau),
            float(trajectory["total_dy"] * tau),
            0.0,
            1.0,
        )
    if case_name == "02_obj_curved_motion":
        path_length = float(trajectory["total_path_length"])
        total_turn_rad = float(
            np.deg2rad(trajectory["total_turn_angle_deg"])
        )
        heading_center_rad = float(
            np.deg2rad(trajectory["initial_heading_deg"])
        )
        if path_length <= 0.0:
            raise ValueError("Curved trajectory path length must be positive")

        curvature = total_turn_rad / path_length
        arc_length_from_center = path_length * tau
        heading_now_rad = (
            heading_center_rad + curvature * arc_length_from_center
        )
        if abs(curvature) < 1e-12:
            dx = arc_length_from_center * np.cos(heading_center_rad)
            dy = arc_length_from_center * np.sin(heading_center_rad)
        else:
            dx = (
                np.sin(heading_now_rad) - np.sin(heading_center_rad)
            ) / curvature
            dy = (
                np.cos(heading_center_rad) - np.cos(heading_now_rad)
            ) / curvature
        theta_deg = float(np.rad2deg(heading_now_rad - heading_center_rad))
        return float(dx), float(dy), theta_deg, 1.0
    raise ValueError(f"Unknown object motion case: {case_name}")


def validate_center_referenced_trajectory(
    trajectory: dict[str, Any],
    sample_count: int = 33,
) -> None:
    """验证中间时刻与 GT 对齐，且轨迹从曝光开始到结束不折返。"""
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")
    if sample_count % 2 == 0:
        sample_count += 1
    tau_samples = np.linspace(-0.5, 0.5, sample_count)
    samples = np.asarray(
        [
            evaluate_object_trajectory_at_time(trajectory, float(tau))[:3]
            for tau in tau_samples
        ],
        dtype=np.float64,
    )
    middle_index = sample_count // 2
    if not np.allclose(samples[middle_index], 0.0, atol=1e-9):
        raise RuntimeError("目标轨迹在曝光中心没有与 GT 的零位移对齐。")

    endpoint_displacement = samples[-1, :2] - samples[0, :2]
    endpoint_distance = float(np.linalg.norm(endpoint_displacement))
    if endpoint_distance <= 1e-9:
        raise RuntimeError("目标曝光首末位置相同，无法生成有效运动模糊。")
    forward_axis = endpoint_displacement / endpoint_distance
    forward_progress = samples[:, :2] @ forward_axis
    if np.any(np.diff(forward_progress) < -1e-6):
        raise RuntimeError("目标中心轨迹在曝光期间发生折返。")

    rotation_steps = np.diff(samples[:, 2])
    if np.any(rotation_steps > 1e-6) and np.any(rotation_steps < -1e-6):
        raise RuntimeError("目标旋转轨迹在曝光期间发生折返。")

    # 首末连线不适合验证曲线的中心方向，因此在 tau=0 两侧取极小时间差，
    # 检查中心轨迹切线是否与自动估计的目标前后轴平行。
    if "front_back_axis_deg" in trajectory:
        epsilon = 1.0e-4
        position_before = np.asarray(
            evaluate_object_trajectory_at_time(trajectory, -epsilon)[:2],
            dtype=np.float64,
        )
        position_after = np.asarray(
            evaluate_object_trajectory_at_time(trajectory, epsilon)[:2],
            dtype=np.float64,
        )
        center_tangent = position_after - position_before
        tangent_norm = float(np.linalg.norm(center_tangent))
        if tangent_norm <= 1e-12:
            raise RuntimeError("目标曝光中心的运动切线无效。")
        axis_rad = float(
            np.deg2rad(trajectory["front_back_axis_deg"])
        )
        front_back_axis = np.asarray(
            [np.cos(axis_rad), np.sin(axis_rad)], dtype=np.float64
        )
        lateral_fraction = abs(
            float(
                center_tangent[0] * front_back_axis[1]
                - center_tangent[1] * front_back_axis[0]
            )
        ) / tangent_norm
        if lateral_fraction > 1.0e-6:
            raise RuntimeError(
                "目标曝光中心运动方向没有沿估计的前后轴。"
            )
        trajectory["center_tangent_lateral_fraction"] = float(
            lateral_fraction
        )


def warp_frame_similarity(
    image: np.ndarray,
    dx: float,
    dy: float,
    theta_deg: float,
    scale_value: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    x_out, y_out = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    theta = np.deg2rad(theta_deg)
    x_centered = x_out - cx - dx
    y_centered = y_out - cy - dy
    x_input = (
        np.cos(theta) * x_centered + np.sin(theta) * y_centered
    ) / scale_value + cx
    y_input = (
        -np.sin(theta) * x_centered + np.cos(theta) * y_centered
    ) / scale_value + cy
    output = cv2.remap(
        np.asarray(image, dtype=np.float64),
        x_input.astype(np.float32),
        y_input.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return clip01_uav(output)


def make_exposure_weights(
    tau_list: np.ndarray,
    start_weight: float,
    terminal_fraction: float = exposure_terminal_fraction,
) -> np.ndarray:
    """生成总和为 1 的曝光权重；默认参数返回严格均匀曝光。"""
    if not 0.0 < start_weight <= 1.0:
        raise ValueError("start_exposure_weight 必须满足 0 < value <= 1")
    if not 0.0 <= terminal_fraction < 1.0:
        raise ValueError(
            "terminal_exposure_fraction 必须满足 0 <= value < 1"
        )
    tau_list = np.asarray(tau_list, dtype=np.float64)
    tau_span = float(tau_list.max() - tau_list.min())
    if tau_span <= 0.0:
        raise ValueError("tau_list 至少需要两个不同的曝光时刻")
    normalized_tau = (tau_list - tau_list.min()) / tau_span
    raw_weights = np.exp(
        np.log(start_weight) * (1.0 - normalized_tau)
    )
    weights = raw_weights / raw_weights.sum()
    weights *= 1.0 - terminal_fraction
    weights[-1] += terminal_fraction
    return weights


def synthesize_multi_object_motion_blur_A(
    foregrounds: list[np.ndarray],
    background: np.ndarray,
    object_masks_soft: list[np.ndarray],
    trajectories: list[dict[str, Any]],
    n_subframes: int,
    extra_crop_margin: int,
    *,
    generation_case_name: str = "01_obj_independent_random",
    reference_image: np.ndarray | None = None,
    start_exposure_weight: float = exposure_start_weight,
    terminal_exposure_fraction: float = exposure_terminal_fraction,
    sensor_parameters: sensor_model.SensorSampleParameters | None = None,
) -> tuple[Any, ...]:
    """在线性光强空间渲染多个独立实例并进行单次曝光时间积分。

    本函数的公开输入和输出仍是 sRGB；内部会将背景、GT 和预乘前景转换为
    线性 RGB。这样旧调用点无需改变，同时确保 warp、alpha 合成和时间平均
    都不再直接作用于 gamma 编码的 sRGB 数值。
    """
    object_count = len(trajectories)
    if object_count == 0:
        raise ValueError("At least one object trajectory is required")
    if len(foregrounds) != object_count or len(object_masks_soft) != object_count:
        raise ValueError("foreground、mask 与 trajectory 的实例数量必须一致")
    if n_subframes < 3 or n_subframes % 2 == 0:
        raise ValueError("中心参考曝光要求 N_subframes 为不小于 3 的奇数")

    background_srgb = np.asarray(background, dtype=np.float64)
    background = srgb_to_linear_uav(background_srgb)
    foregrounds = [
        premultiplied_srgb_to_linear_uav(foreground, object_mask_soft)
        for foreground, object_mask_soft in zip(
            foregrounds, object_masks_soft
        )
    ]
    if reference_image is not None:
        reference_image_srgb = np.asarray(reference_image, dtype=np.float64)
        if reference_image_srgb.shape != background.shape:
            raise ValueError("reference_image 与 background 尺寸必须一致")
        reference_image = srgb_to_linear_uav(reference_image_srgb)
    tau_list = np.linspace(-0.5, 0.5, n_subframes, dtype=np.float64)
    exposure_weights = make_exposure_weights(
        tau_list, start_exposure_weight, terminal_exposure_fraction
    )
    exposure_model = (
        "uniform_rectangular_shutter"
        if np.allclose(
            exposure_weights,
            np.full(n_subframes, 1.0 / n_subframes, dtype=np.float64),
        )
        else "centered_exponential_with_terminal_emphasis"
    )
    blur_accumulator = np.zeros_like(background, dtype=np.float64)
    sensor_accumulator = (
        np.zeros_like(background, dtype=np.float64)
        if sensor_parameters is not None
        else None
    )
    debug_frames: dict[str, np.ndarray] = {}
    middle_index = (n_subframes - 1) // 2
    max_abs_x = 0.0
    max_abs_y = 0.0

    dx_lists = [np.zeros(n_subframes) for _ in trajectories]
    dy_lists = [np.zeros(n_subframes) for _ in trajectories]
    theta_lists = [np.zeros(n_subframes) for _ in trajectories]
    scale_lists = [np.zeros(n_subframes) for _ in trajectories]

    for subframe_index, tau in enumerate(tau_list):
        frame = background.copy()
        # 固定按实例编号合成；mask 重叠时，编号较大的实例位于较上层。
        for object_index, (
            foreground,
            object_mask_soft,
            trajectory,
        ) in enumerate(zip(foregrounds, object_masks_soft, trajectories)):
            cx = float(trajectory["object_centroid"][0] - 1.0)
            cy = float(trajectory["object_centroid"][1] - 1.0)
            dx, dy, theta, scale_value = evaluate_object_trajectory_at_time(
                trajectory, float(tau)
            )
            dx_lists[object_index][subframe_index] = dx
            dy_lists[object_index][subframe_index] = dy
            theta_lists[object_index][subframe_index] = theta
            scale_lists[object_index][subframe_index] = scale_value

            foreground_now = warp_frame_similarity(
                foreground, dx, dy, theta, scale_value, cx, cy
            )
            mask_now = warp_frame_similarity(
                object_mask_soft, dx, dy, theta, scale_value, cx, cy
            )
            frame = clip01_uav(
                (1.0 - mask_now[..., None]) * frame + foreground_now
            )
            max_abs_x = max(max_abs_x, abs(dx))
            max_abs_y = max(max_abs_y, abs(dy))

        # GT 定义为曝光中心时刻；中间帧显式采用原 GT，确保 mid 与输入图
        # 逐像素一致。
        if subframe_index == middle_index and reference_image is not None:
            frame = reference_image.copy()
        blur_accumulator += exposure_weights[subframe_index] * frame
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
        if subframe_index == 0:
            debug_frames["first"] = frame
        if subframe_index == middle_index:
            debug_frames["mid"] = frame
        if subframe_index == n_subframes - 1:
            debug_frames["last"] = frame

    object_metadata: list[dict[str, Any]] = []
    for object_index, trajectory in enumerate(trajectories):
        object_metadata.append(
            {
                "instance_id": int(trajectory.get("instance_id", object_index + 1)),
                "instance_mask_file": str(
                    trajectory.get("instance_mask_file", "")
                ),
                "case_name": trajectory["case_name"],
                "case_seed": int(trajectory["case_seed"]),
                "motion_strength_ratio": float(
                    trajectory["motion_strength_ratio"]
                ),
                "object_bbox": trajectory["object_bbox"],
                "object_centroid": trajectory["object_centroid"],
                "object_scale": float(trajectory["object_scale"]),
                "initial_heading_deg": float(
                    trajectory["initial_heading_deg"]
                ),
                "front_back_axis_deg": float(
                    trajectory["front_back_axis_deg"]
                ),
                "motion_heading_deg": float(
                    trajectory["motion_heading_deg"]
                ),
                "front_back_axis_rule": str(
                    trajectory["front_back_axis_rule"]
                ),
                "oriented_bbox_aspect_ratio": float(
                    trajectory["oriented_bbox_aspect_ratio"]
                ),
                "center_tangent_lateral_fraction": float(
                    trajectory.get(
                        "center_tangent_lateral_fraction", np.nan
                    )
                ),
                "dx_list": dx_lists[object_index][:, None],
                "dy_list": dy_lists[object_index][:, None],
                "theta_list": theta_lists[object_index][:, None],
                "scale_list": scale_lists[object_index][:, None],
                "weighted_mean_dx": float(
                    exposure_weights @ dx_lists[object_index]
                ),
                "weighted_mean_dy": float(
                    exposure_weights @ dy_lists[object_index]
                ),
            }
        )

    blur_image = linear_to_srgb_uav(blur_accumulator)
    sensor_blur_image = (
        sensor_model.finalize_sensor_variant_srgb(
            sensor_accumulator, sensor_parameters
        )
        if sensor_accumulator is not None
        and sensor_parameters is not None
        else None
    )
    debug_frames = {
        name: linear_to_srgb_uav(frame)
        for name, frame in debug_frames.items()
    }
    metadata: dict[str, Any] = {
        "scheme_name": "A_multi_object_center_referenced_multisubframe",
        "formation_model": (
            "B_srgb=g(sum_t(w_t*composite_linear("
            "B_linear,W(F_linear*alpha,T_t),W(alpha,T_t))))"
        ),
        "gt_reference": "center",
        "generation_case_name": generation_case_name,
        "object_count": object_count,
        "N_subframes": int(n_subframes),
        "tau_list": tau_list,
        "weighted_mean_tau": float(exposure_weights @ tau_list),
        "exposure_model": exposure_model,
        "exposure_start_weight": float(start_exposure_weight),
        "terminal_exposure_fraction": float(terminal_exposure_fraction),
        "exposure_end_to_start_ratio": float(
            exposure_weights[-1] / exposure_weights[0]
        ),
        "exposure_weights": exposure_weights[:, None],
        "radiometric_model": "linear_srgb_exposure_integration",
        "input_transfer_function": "IEC_61966_2_1_sRGB",
        "temporal_integration_space": "linear_rgb",
        "output_transfer_function": "IEC_61966_2_1_sRGB",
        "objects": object_metadata,
        "crop_margin": int(extra_crop_margin),
        "crop_bound_h": int(np.ceil(max_abs_y)) + int(extra_crop_margin),
        "crop_bound_w": int(np.ceil(max_abs_x)) + int(extra_crop_margin),
    }
    if sensor_blur_image is not None:
        return blur_image, sensor_blur_image, metadata, debug_frames
    return blur_image, metadata, debug_frames


def synthesize_object_motion_blur_A(
    foreground: np.ndarray,
    background: np.ndarray,
    object_mask_soft: np.ndarray,
    trajectory: dict[str, Any],
    n_subframes: int,
    extra_crop_margin: int,
    *,
    reference_image: np.ndarray | None = None,
    start_exposure_weight: float = exposure_start_weight,
    terminal_exposure_fraction: float = exposure_terminal_fraction,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """在线性光强空间渲染单个目标；公开输入和输出均为 sRGB。"""
    if n_subframes < 3 or n_subframes % 2 == 0:
        raise ValueError("中心参考曝光要求 N_subframes 为不小于 3 的奇数")
    background = srgb_to_linear_uav(background)
    foreground = premultiplied_srgb_to_linear_uav(
        foreground, object_mask_soft
    )
    # 输入 GT 固定为曝光中心时刻，首末帧位于 GT 的两侧。
    tau_list = np.linspace(-0.5, 0.5, n_subframes, dtype=np.float64)
    if reference_image is not None:
        reference_image_srgb = np.asarray(reference_image, dtype=np.float64)
        if reference_image_srgb.shape != background.shape:
            raise ValueError("reference_image 与 background 尺寸必须一致")
        reference_image = srgb_to_linear_uav(reference_image_srgb)
    exposure_weights = make_exposure_weights(
        tau_list, start_exposure_weight, terminal_exposure_fraction
    )
    exposure_model = (
        "uniform_rectangular_shutter"
        if np.allclose(
            exposure_weights,
            np.full(n_subframes, 1.0 / n_subframes, dtype=np.float64),
        )
        else "centered_exponential_with_terminal_emphasis"
    )
    # trajectory 中按 MATLAB 保存 1-based centroid；warp 使用 0-based 坐标。
    cx = float(trajectory["object_centroid"][0] - 1.0)
    cy = float(trajectory["object_centroid"][1] - 1.0)
    blur_accumulator = np.zeros_like(background, dtype=np.float64)
    dx_list = np.zeros(n_subframes)
    dy_list = np.zeros(n_subframes)
    theta_list = np.zeros(n_subframes)
    scale_list = np.zeros(n_subframes)
    max_abs_x = 0.0
    max_abs_y = 0.0
    debug_frames: dict[str, np.ndarray] = {}
    middle_index = (n_subframes - 1) // 2

    for index, tau in enumerate(tau_list):
        dx, dy, theta, scale_value = evaluate_object_trajectory_at_time(
            trajectory, float(tau)
        )
        dx_list[index], dy_list[index] = dx, dy
        theta_list[index], scale_list[index] = theta, scale_value
        foreground_now = warp_frame_similarity(
            foreground, dx, dy, theta, scale_value, cx, cy
        )
        mask_now = warp_frame_similarity(
            object_mask_soft, dx, dy, theta, scale_value, cx, cy
        )
        frame = clip01_uav(
            (1.0 - mask_now[..., None]) * background + foreground_now
        )
        if index == middle_index and reference_image is not None:
            frame = reference_image.copy()
        blur_accumulator += exposure_weights[index] * frame
        max_abs_x = max(max_abs_x, abs(dx))
        max_abs_y = max(max_abs_y, abs(dy))
        if index == 0:
            debug_frames["first"] = frame
        if index == middle_index:
            debug_frames["mid"] = frame
        if index == n_subframes - 1:
            debug_frames["last"] = frame

    blur_image = linear_to_srgb_uav(blur_accumulator)
    debug_frames = {
        name: linear_to_srgb_uav(frame)
        for name, frame in debug_frames.items()
    }
    metadata: dict[str, Any] = {
        "scheme_name": "A_single_object_center_referenced_multisubframe",
        "formation_model": (
            "B_srgb=g(sum_t(w_t*composite_linear("
            "B_linear,W(F_linear*alpha,T_t),W(alpha,T_t))))"
        ),
        "gt_reference": "center",
        "case_name": trajectory["case_name"],
        "case_seed": trajectory["case_seed"],
        "N_subframes": int(n_subframes),
        "tau_list": tau_list,
        "weighted_mean_tau": float(exposure_weights @ tau_list),
        "exposure_model": exposure_model,
        "exposure_start_weight": float(start_exposure_weight),
        "terminal_exposure_fraction": float(terminal_exposure_fraction),
        "exposure_end_to_start_ratio": float(
            exposure_weights[-1] / exposure_weights[0]
        ),
        "exposure_weights": exposure_weights[:, None],
        "radiometric_model": "linear_srgb_exposure_integration",
        "input_transfer_function": "IEC_61966_2_1_sRGB",
        "temporal_integration_space": "linear_rgb",
        "output_transfer_function": "IEC_61966_2_1_sRGB",
        "dx_list": dx_list[:, None],
        "dy_list": dy_list[:, None],
        "theta_list": theta_list[:, None],
        "scale_list": scale_list[:, None],
        "weighted_mean_dx": float(exposure_weights @ dx_list),
        "weighted_mean_dy": float(exposure_weights @ dy_list),
        "object_bbox": trajectory["object_bbox"],
        "object_centroid": trajectory["object_centroid"],
        "object_scale": trajectory["object_scale"],
        "crop_margin": int(extra_crop_margin),
        "crop_bound_h": int(np.ceil(max_abs_y)) + int(extra_crop_margin),
        "crop_bound_w": int(np.ceil(max_abs_x)) + int(extra_crop_margin),
    }
    return blur_image, metadata, debug_frames


def crop_any_array(
    array: np.ndarray, bound_h: int, bound_w: int
) -> tuple[np.ndarray, bool]:
    height, width = array.shape[:2]
    if bound_h * 2 >= height or bound_w * 2 >= width:
        print("Warning: crop too large; skip one array.")
        return array, False
    rows = slice(bound_h, height - bound_h if bound_h else height)
    columns = slice(bound_w, width - bound_w if bound_w else width)
    return array[rows, columns, ...], True


def clip01_uav(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0.0, 1.0)


def show_layers(
    image: np.ndarray,
    mask: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, data, title in zip(
        axes,
        (image, mask, foreground, background),
        ("GT", "Mask", "Foreground", "Background"),
    ):
        axis.imshow(data, cmap="gray" if data.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=image_dir)
    parser.add_argument("--mask-dir", type=Path, default=mask_dir)
    parser.add_argument(
        "--instance-mask-dir",
        type=Path,
        default=instance_mask_dir,
        help="逐实例 mask 根目录；存在 <stem>/obj_*.png 时优先于联合 mask。",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=background_dir,
        help="已知真实背景目录；提供后跳过 inpaint，按原图同名或同 stem 查找背景。",
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--cases", default=",".join(motion_cases))
    parser.add_argument("--seed", type=int, default=base_seed)
    parser.add_argument(
        "--n-subframes",
        type=int,
        default=N_subframes,
        help="关闭自适应子帧时使用的固定数量。",
    )
    parser.add_argument(
        "--adaptive-subframes",
        action=argparse.BooleanOptionalAction,
        default=use_adaptive_subframes,
        help="按最长轨迹设置 N=ceil(distance_px)+1，使采样间隔不超过约1像素。",
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
        "--front-back-axis-aspect-threshold",
        type=float,
        default=front_back_axis_aspect_ratio_threshold,
        help=(
            "有向框长宽比达到该值时沿长轴运动；否则按正/背面视角沿短轴运动。"
        ),
    )
    parser.add_argument("--crop-margin", type=int, default=crop_margin)
    parser.add_argument(
        "--crop-boundary", action=argparse.BooleanOptionalAction, default=is_crop_boundary
    )
    parser.add_argument(
        "--save-debug-subframes",
        action=argparse.BooleanOptionalAction,
        default=is_save_debug_subframes,
    )
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=is_show)
    parser.add_argument(
        "--soft-mask", action=argparse.BooleanOptionalAction, default=use_soft_mask
    )
    parser.add_argument("--soft-mask-sigma", type=float, default=soft_mask_sigma)
    parser.add_argument(
        "--render-mask-refine-ratio",
        type=float,
        default=render_mask_refine_ratio,
        help="自动 mask 边缘修正半径占目标最长边的比例。",
    )
    parser.add_argument(
        "--render-mask-refine-min-px",
        type=int,
        default=render_mask_refine_min_px,
        help="渲染 mask 边缘修正的最小半径（像素）。",
    )
    parser.add_argument(
        "--render-mask-refine-max-px",
        type=int,
        default=render_mask_refine_max_px,
        help="渲染 mask 边缘修正的最大半径（像素）。",
    )
    parser.add_argument(
        "--inpaint-method", choices=("biharmonic", "telea"), default=inpaint_method
    )
    parser.add_argument(
        "--inpaint-dilation-ratio",
        type=float,
        default=inpaint_dilation_ratio,
        help="补全 mask 的膨胀半径占目标最长边的比例。",
    )
    parser.add_argument(
        "--inpaint-dilation-min-px",
        type=int,
        default=inpaint_dilation_min_px,
        help="补全 mask 的最小膨胀半径（像素）。",
    )
    parser.add_argument(
        "--inpaint-dilation-max-px",
        type=int,
        default=inpaint_dilation_max_px,
        help="补全 mask 的最大膨胀半径（像素）。",
    )
    parser.add_argument(
        "--background-motion-samples",
        type=int,
        default=background_motion_samples,
        help="计算目标运动后实际露出区域时的轨迹采样数。",
    )
    parser.add_argument(
        "--background-boundary-feather-ratio",
        type=float,
        default=background_boundary_feather_ratio,
        help="复制背景纹理的边缘渐变宽度占目标最长边的比例。",
    )
    parser.add_argument(
        "--background-boundary-feather-min-px",
        type=int,
        default=background_boundary_feather_min_px,
        help="背景边界渐变的最小宽度（像素）。",
    )
    parser.add_argument(
        "--background-boundary-feather-max-px",
        type=int,
        default=background_boundary_feather_max_px,
        help="背景边界渐变的最大宽度（像素）。",
    )
    parser.add_argument(
        "--background-tone-ring-ratio",
        type=float,
        default=background_tone_ring_ratio,
        help="用于背景亮度匹配的外侧参考环宽度占目标最长边的比例。",
    )
    parser.add_argument(
        "--background-tone-ring-min-px",
        type=int,
        default=background_tone_ring_min_px,
        help="背景亮度参考环的最小宽度（像素）。",
    )
    parser.add_argument(
        "--background-tone-ring-max-px",
        type=int,
        default=background_tone_ring_max_px,
        help="背景亮度参考环的最大宽度（像素）。",
    )
    parser.add_argument(
        "--background-tone-gain-min",
        type=float,
        default=background_tone_gain_range[0],
        help="背景亮度匹配允许的最小增益。",
    )
    parser.add_argument(
        "--background-tone-gain-max",
        type=float,
        default=background_tone_gain_range[1],
        help="背景亮度匹配允许的最大增益。",
    )
    parser.add_argument(
        "--background-visibility-threshold",
        type=float,
        default=background_visibility_threshold,
        help="soft mask 相对 GT 覆盖率下降超过该值时计入 needed 核心。",
    )
    parser.add_argument(
        "--background-safety-dilation-ratio",
        type=float,
        default=background_safety_dilation_ratio,
        help="needed 核心外安全圈半径占目标最长边的比例。",
    )
    parser.add_argument(
        "--background-safety-dilation-min-px",
        type=int,
        default=background_safety_dilation_min_px,
    )
    parser.add_argument(
        "--background-safety-dilation-max-px",
        type=int,
        default=background_safety_dilation_max_px,
    )
    parser.add_argument(
        "--exposure-start-weight",
        type=float,
        default=exposure_start_weight,
        help=(
            "起始子帧相对末帧的曝光权重；默认 1.0，所有子帧等权，"
            "对应曝光期间响应恒定的矩形快门。小于 1 才启用终点增强。"
        ),
    )
    parser.add_argument(
        "--terminal-exposure-fraction",
        type=float,
        default=exposure_terminal_fraction,
        help=(
            "额外分配给曝光末帧的能量比例；默认 0，不制造“清晰终点 + "
            "半透明长尾”。仅在模拟末端闪光/非均匀快门时设为正数。"
        ),
    )
    parser.add_argument(
        "--mask-name-mode",
        choices=("same_stem", "same_name", "stem_png"),
        default=mask_name_mode,
        help=(
            "mask 文件匹配方式；默认 same_stem，忽略 GT 与 mask 的扩展名差异，"
            "并优先读取同 stem 的 PNG。"
        ),
    )
    parser.add_argument(
        "--same-case-seed",
        action=argparse.BooleanOptionalAction,
        default=use_same_case_seed_for_all_images,
    )
    parser.add_argument(
        "--largest-component-only",
        action=argparse.BooleanOptionalAction,
        default=use_largest_connected_component_only,
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


def main() -> None:
    arguments = build_parser().parse_args()
    if not 0.0 <= arguments.linear_probability <= 1.0:
        raise ValueError("--linear-probability 必须在 [0, 1] 内")
    if arguments.front_back_axis_aspect_threshold <= 1.0:
        raise ValueError(
            "--front-back-axis-aspect-threshold 必须大于 1"
        )
    range_pairs = (
        (
            "linear-distance-ratio",
            arguments.linear_distance_ratio_min,
            arguments.linear_distance_ratio_max,
        ),
        (
            "curved-path-ratio",
            arguments.curved_path_ratio_min,
            arguments.curved_path_ratio_max,
        ),
        (
            "curved-turn-angle",
            arguments.curved_turn_angle_min,
            arguments.curved_turn_angle_max,
        ),
    )
    for range_name, minimum, maximum in range_pairs:
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(
                f"--{range_name}-min/max 必须满足 0 < min <= max"
            )
    if arguments.inpaint_dilation_ratio < 0.0:
        raise ValueError("--inpaint-dilation-ratio 不能小于 0")
    if (
        arguments.inpaint_dilation_min_px < 0
        or arguments.inpaint_dilation_max_px
        < arguments.inpaint_dilation_min_px
    ):
        raise ValueError(
            "--inpaint-dilation-min/max-px 必须满足 0 <= min <= max"
        )
    if arguments.background_motion_samples < 2:
        raise ValueError("--background-motion-samples 不能小于 2")
    if arguments.render_mask_refine_ratio < 0.0:
        raise ValueError("--render-mask-refine-ratio 不能小于 0")
    if (
        arguments.render_mask_refine_min_px < 0
        or arguments.render_mask_refine_max_px
        < arguments.render_mask_refine_min_px
    ):
        raise ValueError(
            "--render-mask-refine-min/max-px 必须满足 0 <= min <= max"
        )
    if arguments.background_boundary_feather_ratio < 0.0:
        raise ValueError("--background-boundary-feather-ratio 不能小于 0")
    if (
        arguments.background_boundary_feather_min_px < 0
        or arguments.background_boundary_feather_max_px
        < arguments.background_boundary_feather_min_px
    ):
        raise ValueError(
            "--background-boundary-feather-min/max-px "
            "必须满足 0 <= min <= max"
        )
    if arguments.background_tone_ring_ratio < 0.0:
        raise ValueError("--background-tone-ring-ratio 不能小于 0")
    if (
        arguments.background_tone_ring_min_px < 0
        or arguments.background_tone_ring_max_px
        < arguments.background_tone_ring_min_px
    ):
        raise ValueError(
            "--background-tone-ring-min/max-px 必须满足 0 <= min <= max"
        )
    if (
        arguments.background_tone_gain_min <= 0.0
        or arguments.background_tone_gain_max
        < arguments.background_tone_gain_min
    ):
        raise ValueError(
            "--background-tone-gain-min/max 必须满足 0 < min <= max"
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
    if not 0.0 < arguments.exposure_start_weight <= 1.0:
        raise ValueError("--exposure-start-weight 必须满足 0 < value <= 1")
    if not 0.0 <= arguments.terminal_exposure_fraction < 1.0:
        raise ValueError(
            "--terminal-exposure-fraction 必须满足 0 <= value < 1"
        )
    run_generation(
        arguments.image_dir,
        arguments.mask_dir,
        arguments.output_dir,
        tuple(item.strip() for item in arguments.cases.split(",") if item.strip()),
        input_instance_mask_dir=arguments.instance_mask_dir,
        input_background_dir=arguments.background_dir,
        seed=arguments.seed,
        n_subframes=arguments.n_subframes,
        adaptive_subframes=arguments.adaptive_subframes,
        crop_boundary=arguments.crop_boundary,
        extra_crop_margin=arguments.crop_margin,
        save_debug_subframes=arguments.save_debug_subframes,
        show=arguments.show,
        soft_mask=arguments.soft_mask,
        mask_sigma=arguments.soft_mask_sigma,
        render_refine_ratio=arguments.render_mask_refine_ratio,
        render_refine_min=arguments.render_mask_refine_min_px,
        render_refine_max=arguments.render_mask_refine_max_px,
        background_method=arguments.inpaint_method,
        inpaint_dilation_ratio_value=arguments.inpaint_dilation_ratio,
        inpaint_dilation_min=arguments.inpaint_dilation_min_px,
        inpaint_dilation_max=arguments.inpaint_dilation_max_px,
        background_samples=arguments.background_motion_samples,
        background_feather_ratio=(
            arguments.background_boundary_feather_ratio
        ),
        background_feather_min=(
            arguments.background_boundary_feather_min_px
        ),
        background_feather_max=(
            arguments.background_boundary_feather_max_px
        ),
        background_ring_ratio=arguments.background_tone_ring_ratio,
        background_ring_min=arguments.background_tone_ring_min_px,
        background_ring_max=arguments.background_tone_ring_max_px,
        background_gain_range=(
            arguments.background_tone_gain_min,
            arguments.background_tone_gain_max,
        ),
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
        start_exposure_weight=arguments.exposure_start_weight,
        terminal_exposure_fraction=arguments.terminal_exposure_fraction,
        mask_mode=arguments.mask_name_mode,
        same_case_seed=arguments.same_case_seed,
        largest_component_only=arguments.largest_component_only,
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
        front_back_axis_threshold=(
            arguments.front_back_axis_aspect_threshold
        ),
        sensor_variant=arguments.sensor_variant,
        saturation_highlights=arguments.sensor_saturation_highlights,
        poisson_gaussian_noise=(
            arguments.sensor_poisson_gaussian_noise
        ),
    )


if __name__ == "__main__":
    main()

"""十五类 UAV 全局平台运动模糊：多子帧积分、无深度图版本。

本文件逐段对应 ``script_uav_global_blur_multisubframe_no_depth.m``。为方便
对照 MATLAB，用户配置、主处理流程和全部本地函数都保留在这一个文件中。
输入 sRGB 先转换为线性 RGB，所有子帧 warp 和曝光积分完成后再编码为 sRGB。

前七类是基础运动：直线匀速、直线变速、转弯匀速、转弯变速、低频悬停
抖动、俯仰/横滚透视空间场和三轴高频微振动。后八类在同一个曝光时刻
组合基础路径、姿态机动和高频振动，只进行一次warp和一次曝光积分。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import savemat

import uav_sensor_degradation as sensor_model


class _BooleanOptionalAction(argparse.Action):
    """Python 3.8 backport of argparse.BooleanOptionalAction."""

    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        expanded_option_strings = []
        for option_string in option_strings:
            expanded_option_strings.append(option_string)
            if option_string.startswith("--"):
                expanded_option_strings.append("--no-" + option_string[2:])
        super().__init__(
            option_strings=expanded_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        del parser, values
        setattr(namespace, self.dest, not option_string.startswith("--no-"))


BooleanOptionalAction = getattr(
    argparse, "BooleanOptionalAction", _BooleanOptionalAction
)


def strict_zip(*iterables):
    """Backport zip(strict=True) while preserving length validation."""

    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("zip() arguments have different lengths")
        yield values


# =========================================================================
# 1. 用户配置区（可以直接修改，也可以通过命令行参数覆盖）
# =========================================================================
base_seed = 20260109  # 随机种子；控制第6/7类方向、强度、相位以及可选输出噪声。
q_type = "uv_basic"  # 运动场标签："uv_basic" 为全位移，"uv_half" 为一半位移。
N_subframes = 21  # 关闭自适应采样时使用的固定子帧数；必须是不小于3的奇数。
adaptive_subframe_sampling = True  # True按屏幕运动距离自动选择子帧，默认启用。
SUBFRAME_DISTANCE_STEP_PX = 1.5  # 全图移动最快位置每累计约1.5像素选取一个子帧。
RECIPROCAL_SAMPLES_PER_CYCLE = 10  # 往复运动每个最高频周期至少选取10个子帧区间。
ADAPTIVE_PREVIEW_SAMPLES = 4001  # 仅计算变换、不渲染图像的轨迹预采样数；必须为奇数。
ADAPTIVE_SAMPLING_GRID_SIZE = 9  # 透视运动按9×9网格估计全图最快像素，而不只看四角。
GT_REFERENCE_TAU = 0.5  # GT对应曝光中间时刻；轨迹在该时刻归一化为恒等变换。
is_crop_boundary = True  # True默认裁掉所有子帧不共同有效的真实内容缺失区域。
crop_margin = 1  # 在最大公共有效矩形内再向内收缩的安全像素，防止插值边缘泄漏。
valid_coverage_threshold = 1.0 - 1e-6  # 判定像素未使用任何图外补充值的覆盖率阈值。
noise_sigma_255 = 0.0  # 输出域高斯噪声标准差（0—255尺度）；0 表示不加噪声。
# 在不改变原 ``_blurryimg.png`` 的前提下，额外保存传感器退化版本。
generate_sensor_variant = True  # True额外输出 ``_blurryimg_sensor.png``。
sensor_saturation_highlights = True  # 在线性子帧中恢复一部分JPEG饱和高光能量。
sensor_saturation_threshold_low = 0.95  # 线性RGB最大通道达到该值后开始软增强。
sensor_saturation_threshold_high = 0.99  # 达到该值后使用完整高光增益。
sensor_saturation_gain_range = (0.25, 1.25)  # 每张图只采样一次，所有子帧共用。
sensor_poisson_gaussian_noise = True  # 曝光积分后只添加一次shot/read noise。
sensor_shot_noise_scale_range = (1.0e-5, 1.0e-4)  # Poisson光子噪声尺度。
sensor_read_noise_sigma_range = (3.0e-6, 5.0e-5)  # 线性伪传感器域读取噪声标准差。
sensor_seed_offset = 8_000_000  # 与运动轨迹seed分离，同时保持结果完全可复现。
is_show = False  # 是否弹出 GT、模糊图和运动统计调试窗口。
script_root = Path(__file__).resolve().parent  # 当前脚本所在目录，通常不需要修改。
image_path = script_root / "dataset/image/5"  # 默认输入清晰 GT 图像目录。
output_path = Path("/home/xtl/hst/joint2D/image_use/motivation/1.8")  # 默认模糊图和运动标签输出目录。
splits: tuple[str, ...] = ()  # 空元组直接处理输入目录；也可填 ("train", "test")。
ARC_LENGTH_LOOKUP_SAMPLES = 2001  # Bézier弧长查找表精度；越大越准，首次计算略慢。

# 第6、7类只输入一张GT、没有真实标定参数时使用的虚拟相机内参。
# 主点默认位于图像中心，fx=fy由水平视场角自动换算。若以后得到真实相机
# 内参，应优先把 build_virtual_camera_intrinsics() 改为读取真实fx/fy/cx/cy。
VIRTUAL_CAMERA_HORIZONTAL_FOV_DEG = 70.0  # 估计水平视场角；越小表示焦距越长。
ATTITUDE_CALIBRATION_TIME_SAMPLES = 301  # 标定姿态振幅时的时间采样点数。
ATTITUDE_CALIBRATION_GRID_SIZE = 9  # 用于把角度振幅换算成像素模糊跨度的网格密度。

# 第6类：曝光期间相机有效俯仰/横滚发生一次平滑姿态机动。
# 运动轴角0°近似相机绕水平轴转动（上下透视变化），90°近似绕竖直轴
# 转动（左右透视变化）；中间角度表示两者耦合。程序为每张图随机选择方向，
# 并自动反求角度，使全图最快位置的模糊跨度落在指定像素范围。
ATTITUDE_TILT_AXIS_ANGLE_DEG_RANGE = (0.0, 180.0)  # 俯仰/横滚耦合方向范围。
ATTITUDE_TILT_TARGET_SPAN_PX_RANGE = (
    14.4,
    25.2,
)  # 类型6基础姿态跨度按上一版放大1.8倍。

# 第7类：三轴小幅高频姿态振动。x/y主要产生透视倾斜，z产生图像平面转动；
# 三轴频率、幅值比例和相位故意不同，避免不真实的完全同步振动。
ATTITUDE_VIBRATION_FREQUENCIES = (3.0, 4.0, 5.0)  # x/y/z每次曝光内的周期数。
ATTITUDE_VIBRATION_AMPLITUDE_RATIOS = (1.0, 0.8, 0.45)  # x/y/z相对角振幅。
ATTITUDE_VIBRATION_TARGET_SPAN_PX_RANGE = (
    14.4,
    25.2,
)  # 类型7基础振动跨度按上一版放大1.8倍。

# 航向直线匀速和航向直线变速也共用同一条空间路径，仅改变时间映射。
HEADING_LINEAR_PATH_POINTS = (
    (0.0, 0.0),  # 直线起点位移(dx,dy)，单位像素。
    (25.2, -9.8),  # 类型1/2基础平移按上一版放大1.4倍。
)
HEADING_LINEAR_VARIABLE_SPEED_GAMMA = 2.0  # 直线变速指数；>1为先慢后快。

# 航向转弯匀速和航向转弯变速必须使用完全相同的空间路径，才能只比较
# 速度规律带来的差异。P0/P1/P2是二次Bézier曲线的起点、控制点和终点。
TURNING_PATH_POINTS = (
    (0.0, 0.0),  # P0：曝光开始时的平台图像位移，单位像素。
    (12.6, -16.8),  # 类型3/4曲线控制点按上一版放大1.4倍。
    (25.2, -9.8),  # 类型3/4曲线终点按上一版放大1.4倍。
)
TURNING_VARIABLE_SPEED_GAMMA = 2.0  # 转弯变速指数；>1为先慢后快，越大加速越明显。

# 时间映射 type 可选：
# "uniform"（匀速）、"power_accelerate"（加速）、
# "power_decelerate"（减速）、"smoothstep"（平滑启停）。
# path type 可选：
# "line"（直线）、"quadratic_bezier"（二次 Bézier）、
# "harmonic"（漂移与若干正弦分量叠加）。
# rotation type 可选：
# "constant"、"linear"、"harmonic"、"follow_path_tangent"。
# scale type 可选："constant"、"linear"、"log_linear"。
PLATFORM_MOTION_CONFIGS: dict[str, dict[str, Any]] = {
    "01_heading_linear_uniform": {
        "projection_model": "similarity_2d",  # 2D平移/旋转/尺度相似变换。
        "sampling": {
            "type": "screen_distance",  # 全图最大屏幕累计位移每达到1.5像素取一帧。
        },
        "path": {
            "type": "line",  # 航向保持不变，平台图像沿直线移动。
            "points": HEADING_LINEAR_PATH_POINTS,  # 与02共用相同直线路径。
        },
        "time_mapping": {
            "type": "uniform",  # s(t)=t，沿直线路径匀速推进。
            "gamma": 1.0,  # uniform 不使用；切换为 power 类型时表示速度曲线指数。
        },
        "rotation": {
            "type": "constant",  # 曝光期间不发生平台旋转。
            "value_deg": 0.0,  # 原始姿态路径的固定旋转角，单位度。
        },
        "scale": {
            "type": "constant",  # 曝光期间不发生平台缩放。
            "value": 1.0,  # 固定缩放比例；1表示原始尺寸。
        },
    },
    "02_heading_linear_variable_speed": {
        "projection_model": "similarity_2d",  # 与01使用相同成像模型。
        "sampling": {
            "type": "screen_distance",  # 时间点可不均匀，积分时使用对应曝光时间权重。
        },
        "path": {
            "type": "line",  # 航向保持不变，空间路径仍然是直线。
            "points": HEADING_LINEAR_PATH_POINTS,  # 与01共用路径，只改变速度规律。
        },
        "time_mapping": {
            "type": "power_accelerate",  # s(t)=t^gamma，先慢后快。
            "gamma": HEADING_LINEAR_VARIABLE_SPEED_GAMMA,  # >1；越大后期加速越明显。
        },
        "rotation": {
            "type": "constant",  # 只验证变速平移，不叠加旋转。
            "value_deg": 0.0,  # 固定旋转角，单位度。
        },
        "scale": {
            "type": "constant",  # 不缩放。
            "value": 1.0,  # 固定缩放比例。
        },
    },
    "03_heading_turning_uniform": {
        "projection_model": "similarity_2d",  # 整幅图沿曲线平移，不绕图像中心旋转。
        "sampling": {
            "type": "screen_distance",  # 按整图曲线平移累计距离自动选择子帧。
        },
        "path": {
            "type": "quadratic_bezier",  # 平台中心沿二次Bézier曲线转弯。
            "points": TURNING_PATH_POINTS,  # 与04完全相同，保证只比较速度规律。
            "arc_length_parameterized": True,  # 按弧长采样，使uniform真正表示空间匀速。
            "arc_length_samples": ARC_LENGTH_LOOKUP_SAMPLES,  # 弧长反查表采样数。
        },
        "time_mapping": {
            "type": "uniform",  # s(t)=t，沿整条曲线保持近似恒定的空间速度。
            "gamma": 1.0,  # uniform不使用，仅保留统一配置格式。
        },
        "rotation": {
            "type": "constant",  # 不叠加图像中心旋转，避免把曲线模糊变成旋转模糊。
            "value_deg": 0.0,  # 固定为0°，所有像素共享同一个曲线平移量。
        },
        "scale": {
            "type": "constant",  # 单独验证转弯，不叠加缩放。
            "value": 1.0,  # 固定缩放比例。
        },
    },
    "04_heading_turning_variable_speed": {
        "projection_model": "similarity_2d",  # 与03共用纯曲线平移，只改变速度规律。
        "sampling": {
            "type": "screen_distance",  # 按屏幕运动选帧，并用时间权重保留先慢后快。
        },
        "path": {
            "type": "quadratic_bezier",  # 与03使用同一条二次Bézier转弯路径。
            "points": TURNING_PATH_POINTS,  # 不改变空间轨迹，只改变到达各位置的时刻。
            "arc_length_parameterized": True,  # 路径进度s仍定义为归一化弧长。
            "arc_length_samples": ARC_LENGTH_LOOKUP_SAMPLES,  # 弧长反查表采样数。
        },
        "time_mapping": {
            "type": "power_accelerate",  # s(t)=t^gamma，曝光前期慢、后期快。
            "gamma": TURNING_VARIABLE_SPEED_GAMMA,  # 转弯加速强度，必须大于1。
        },
        "rotation": {
            "type": "constant",  # 与03一样不绕图像中心旋转。
            "value_deg": 0.0,  # 固定为0°，保证只比较曲线上的时间速度分布。
        },
        "scale": {
            "type": "constant",  # 单独比较转弯速度，不叠加缩放。
            "value": 1.0,  # 固定缩放比例。
        },
    },
    "05_hover_jitter_nonuniform": {
        "projection_model": "similarity_2d",  # 低频平移抖动叠加小幅平面角抖动。
        "sampling": {
            "type": "screen_distance_or_period",  # 距离或1/10最高频周期，先达到者取帧。
        },
        "path": {
            "type": "harmonic",  # 线性漂移上叠加水平、垂直低频正弦抖动。
            "origin_xy": (0.0, 0.0),  # 原始路径在曝光开始时的位移。
            "drift_xy": (5.4, 3.6),  # 类型5基础漂移按上一版放大1.8倍。
            "harmonics": (
                {
                    "amplitude_xy": (3.6, 0.0),  # 类型5水平振幅放大1.8倍。
                    "frequency": 1.25,  # 一次完整路径进度内的水平抖动周期数。
                    "phase_deg": 0.0,  # 水平抖动初相位。
                },
                {
                    "amplitude_xy": (0.0, 3.6),  # 类型5垂直振幅放大1.8倍。
                    "frequency": 1.75,  # 一次完整路径进度内的垂直抖动周期数。
                    "phase_deg": 0.0,  # 垂直抖动初相位。
                },
            ),
        },
        "time_mapping": {
            "type": "uniform",  # 按物理相位均匀推进，保留正弦振动的时间规律。
            "gamma": 1.0,  # uniform 不使用。
        },
        "rotation": {
            "type": "harmonic",  # 平台进行小幅周期性角抖动。
            "base_deg": 0.0,  # 原始姿态路径的基础旋转角。
            "drift_deg": 0.0,  # 一次路径进度内额外积累的旋转角。
            "amplitude_deg": 0.54,  # 类型5旋转振幅按上一版放大1.8倍。
            "frequency": 1.0,  # 一次完整路径进度内的旋转振动周期数。
            "phase_deg": 0.0,  # 旋转振动初相位。
        },
        "scale": {
            "type": "constant",  # 悬停抖动不模拟尺度变化。
            "value": 1.0,  # 固定缩放比例。
        },
    },
    "06_camera_tilt_perspective_spatial_field": {
        "projection_model": "pure_rotation_homography",  # H(t)=K R(t) K^-1。
        "sampling": {
            "type": "screen_distance",  # 每累计约1.5像素的最大屏幕运动取一帧。
        },
        "attitude": {
            "type": "smooth_axis_tilt",  # 一次平滑俯仰/横滚耦合姿态机动。
            "time_mapping": {
                "type": "smoothstep",  # 曝光首末速度较平滑，中心时刻严格为GT。
                "gamma": 1.0,  # smoothstep不使用，保留统一字段。
            },
            "axis_angle_deg_range": ATTITUDE_TILT_AXIS_ANGLE_DEG_RANGE,
            "target_span_px_range": ATTITUDE_TILT_TARGET_SPAN_PX_RANGE,
            "rotation_axis_definition": (
                "camera_xy_axis:0deg=about_camera_x,90deg=about_camera_y"
            ),
        },
    },
    "07_high_frequency_micro_vibration": {
        "projection_model": "pure_rotation_homography",  # H(t)=K R(t) K^-1。
        "sampling": {
            "type": "screen_distance_or_period",  # 同时满足1.5像素和每周期至少10段。
        },
        "attitude": {
            "type": "harmonic_euler",  # 相机x/y/z三轴小幅高频往复角振动。
            "frequencies": ATTITUDE_VIBRATION_FREQUENCIES,
            "amplitude_ratios": ATTITUDE_VIBRATION_AMPLITUDE_RATIOS,
            "target_span_px_range": ATTITUDE_VIBRATION_TARGET_SPAN_PX_RANGE,
            "rotation_order": "Rz_Ry_Rx",  # R=Rz(z)@Ry(y)@Rx(x)。
        },
    },
}

# 八种复合运动。component_case_names 按“源GT依次经历的变换”排列：
# 主路径 → 低频姿态机动 → 高频姿态振动。对同一个子时刻先组合矩阵，
# 再对原始GT做一次warp；绝不把已经模糊的图片再次模糊。
COMPOSITE_MOTION_CONFIGS: dict[str, dict[str, Any]] = {
    "08_heading_linear_uniform_plus_micro_vibration": {
        "component_case_names": (
            "01_heading_linear_uniform",
            "07_high_frequency_micro_vibration",
        ),
        "description": "1+7: straight uniform motion plus micro vibration",
    },
    "09_heading_linear_variable_speed_plus_attitude": {
        "component_case_names": (
            "02_heading_linear_variable_speed",
            "06_camera_tilt_perspective_spatial_field",
        ),
        "description": "2+6: variable-speed line plus attitude maneuver",
    },
    "10_heading_linear_variable_speed_plus_attitude_micro_vibration": {
        "component_case_names": (
            "02_heading_linear_variable_speed",
            "06_camera_tilt_perspective_spatial_field",
            "07_high_frequency_micro_vibration",
        ),
        "description": (
            "2+6+7: variable-speed line, attitude maneuver and vibration"
        ),
    },
    "11_heading_turning_uniform_plus_attitude": {
        "component_case_names": (
            "03_heading_turning_uniform",
            "06_camera_tilt_perspective_spatial_field",
        ),
        "description": "3+6: uniform curved translation plus attitude maneuver",
    },
    "12_heading_turning_variable_speed_plus_attitude": {
        "component_case_names": (
            "04_heading_turning_variable_speed",
            "06_camera_tilt_perspective_spatial_field",
        ),
        "description": (
            "4+6: variable-speed curved translation plus attitude maneuver"
        ),
    },
    "13_heading_turning_variable_speed_plus_attitude_micro_vibration": {
        "component_case_names": (
            "04_heading_turning_variable_speed",
            "06_camera_tilt_perspective_spatial_field",
            "07_high_frequency_micro_vibration",
        ),
        "description": (
            "4+6+7: variable-speed curve, attitude maneuver and vibration"
        ),
    },
    "14_hover_jitter_plus_micro_vibration": {
        "component_case_names": (
            "05_hover_jitter_nonuniform",
            "07_high_frequency_micro_vibration",
        ),
        "description": "5+7: low-frequency hover jitter plus micro vibration",
    },
    "15_attitude_plus_micro_vibration": {
        "component_case_names": (
            "06_camera_tilt_perspective_spatial_field",
            "07_high_frequency_micro_vibration",
        ),
        "description": "6+7: smooth attitude maneuver plus micro vibration",
    },
}

# 复合类型中的同一基础分量使用与单独类型相同的seed。例如09和10中的
# 第2/6分量完全一致，10只额外加入第7分量，便于做受控对照实验。
COMPONENT_SEED_OFFSETS = {
    case_name: case_index * 1000
    for case_index, case_name in enumerate(
        PLATFORM_MOTION_CONFIGS.keys(), start=1
    )
}

# 旧名称仍可用于 --cases，避免已有命令突然失效。默认输出采用上面的15个名称。
LEGACY_CASE_ALIASES = {
    "01_linear_uniform": "01_heading_linear_uniform",
    "02_linear_variable_speed": "02_heading_linear_variable_speed",
    "03_hover_jitter_nonuniform": "05_hover_jitter_nonuniform",
    "04_maneuver_spatial_field": "06_camera_tilt_perspective_spatial_field",
    "06_maneuver_spatial_field": "06_camera_tilt_perspective_spatial_field",
    "05_high_frequency_vibration": "07_high_frequency_micro_vibration",
}

blur_cases = (
    "01_heading_linear_uniform",  # 航向保持不变：直线路径 × 匀速。
    "02_heading_linear_variable_speed",  # 航向保持不变：直线路径 × 加速。
    "03_heading_turning_uniform",  # 航向随曲线改变：曲线路径 × 匀速。
    "04_heading_turning_variable_speed",  # 同一曲线路径 × 加速。
    "05_hover_jitter_nonuniform",  # 低频漂移 + 非均匀悬停抖动。
    "06_camera_tilt_perspective_spatial_field",  # 俯仰/横滚耦合的透视空间变化。
    "07_high_frequency_micro_vibration",  # 小幅三轴高频姿态往复振动。
    "08_heading_linear_uniform_plus_micro_vibration",  # 1+7。
    "09_heading_linear_variable_speed_plus_attitude",  # 2+6。
    "10_heading_linear_variable_speed_plus_attitude_micro_vibration",  # 2+6+7。
    "11_heading_turning_uniform_plus_attitude",  # 3+6。
    "12_heading_turning_variable_speed_plus_attitude",  # 4+6。
    "13_heading_turning_variable_speed_plus_attitude_micro_vibration",  # 4+6+7。
    "14_hover_jitter_plus_micro_vibration",  # 5+7。
    "15_attitude_plus_micro_vibration",  # 6+7。
)
# 无论 --cases 选择了哪些类型或如何排序，同一图片的同一类型都使用固定seed。
# 这样单独生成第7类与一次生成全部15类时，第7类的轨迹不会发生变化。
CASE_SEED_OFFSETS = {
    case_name: case_index * 1000
    for case_index, case_name in enumerate(blur_cases, start=1)
}
SUPPORTED_EXTENSIONS = {  # 输入图像允许的扩展名，通常不需要修改。
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


# =========================================================================
# 2. 主处理流程
# =========================================================================
def run_generation(
    input_root: str | Path = image_path,
    output_root: str | Path = output_path,
    split_names: tuple[str, ...] = splits,
    case_names: tuple[str, ...] = blur_cases,
    *,
    seed: int = base_seed,
    motion_strength: float = 1.0,
    n_subframes: int = N_subframes,
    use_adaptive_sampling: bool = adaptive_subframe_sampling,
    crop_boundary: bool = is_crop_boundary,
    extra_crop_margin: int = crop_margin,
    label_type: str = q_type,
    noise_sigma: float = noise_sigma_255,
    sensor_variant: bool = generate_sensor_variant,
    saturation_highlights: bool = sensor_saturation_highlights,
    poisson_gaussian_noise: bool = sensor_poisson_gaussian_noise,
    show: bool = is_show,
) -> None:
    """直接遍历输入目录，或按指定 split 遍历，为每张图生成多子帧模糊。"""
    if motion_strength <= 0.0:
        raise ValueError("motion_strength必须大于0")
    input_root = Path(input_root)
    output_root = Path(output_root)
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

    processing_splits = split_names if split_names else ("",)
    for split_index, split_name in enumerate(processing_splits, start=1):
        input_dir = input_root / split_name if split_name else input_root
        output_dir = output_root / split_name if split_name else output_root
        output_dir.mkdir(parents=True, exist_ok=True)
        image_files = list_image_files_uav(input_dir)
        if not image_files:
            print(f"Warning: no image files found in {input_dir}")
            continue

        for file_index, image_file in enumerate(image_files, start=1):
            image = read_image_double(image_file)
            image_seed = (
                seed + split_index * 1_000_000 + file_index * 10_000
            )
            print(
                f"split={split_name or 'root'} "
                f"({split_index}/{len(processing_splits)}), "
                f"image={file_index}/{len(image_files)}, name={image_file.name}"
            )

            for case_index, case_name in enumerate(case_names, start=1):
                canonical_case_name = LEGACY_CASE_ALIASES.get(
                    case_name, case_name
                )
                case_seed = image_seed + CASE_SEED_OFFSETS.get(
                    canonical_case_name, case_index * 1000
                )
                trajectory = make_uav_trajectory(
                    case_name,
                    case_seed,
                    component_seed_base=image_seed,
                )
                scale_trajectory_motion_amplitudes(
                    trajectory, motion_strength
                )
                projection_model = str(trajectory["projection_model"])
                if projection_model == "similarity_2d":
                    print(
                        f"  blur case={case_name}, "
                        f"projection={projection_model}, "
                        f"path={trajectory['path']['type']}, "
                        f"time={trajectory['time_mapping']['type']}, "
                        f"rotation={trajectory['rotation']['type']}, "
                        f"scale={trajectory['scale']['type']}"
                    )
                elif projection_model == "pure_rotation_homography":
                    print(
                        f"  blur case={case_name}, "
                        f"projection={projection_model}, "
                        f"attitude={trajectory['attitude']['type']}, "
                        f"target_span="
                        f"{trajectory['attitude']['target_span_px']:.2f}px"
                    )
                else:
                    print(
                        f"  blur case={case_name}, "
                        f"projection={projection_model}, "
                        "components="
                        f"{'+'.join(trajectory['component_case_names'])}"
                    )
                sensor_parameters = (
                    sensor_model.sample_sensor_parameters(
                        sensor_config, case_seed + sensor_seed_offset
                    )
                    if sensor_variant
                    else None
                )
                synthesis_result = synthesize_blur_by_multisubframe_v2(
                    image,
                    trajectory,
                    n_subframes,
                    extra_crop_margin,
                    use_adaptive_sampling=use_adaptive_sampling,
                    sensor_parameters=sensor_parameters,
                )
                if sensor_parameters is None:
                    blur_image, motion_field, parameters, motion_summary = (
                        synthesis_result
                    )
                    sensor_blur_image = None
                else:
                    (
                        blur_image,
                        sensor_blur_image,
                        motion_field,
                        parameters,
                        motion_summary,
                    ) = synthesis_result
                    parameters["sensor_degradation"] = (
                        sensor_model.sensor_metadata_dict(sensor_parameters)
                    )
                print(
                    f"    subframes={parameters['N_subframes_actual']}, "
                    f"sampling={parameters['sampling_type']}, "
                    f"weight_sum={parameters['temporal_weight_sum']:.6f}"
                )

                if noise_sigma > 0:
                    noise_rng = np.random.default_rng(seed + file_index * 100 + case_index)
                    blur_image = clip01_uav(
                        blur_image
                        + noise_rng.normal(0.0, noise_sigma / 255.0, blur_image.shape)
                    )

                mfmap = build_mfmap_uav(motion_field, label_type)
                gt_image = image
                if crop_boundary:
                    crop_rectangle = tuple(
                        int(value)
                        for value in np.asarray(
                            parameters[
                                "crop_rectangle_top_bottom_left_right"
                            ]
                        ).ravel()
                    )
                    blur_image, mfmap, gt_image = (
                        crop_valid_rectangle_uav(
                            blur_image,
                            mfmap,
                            gt_image,
                            crop_rectangle,
                        )
                    )
                    motion_summary = crop_motion_summary_rectangle(
                        motion_summary, crop_rectangle
                    )
                    if sensor_blur_image is not None:
                        sensor_blur_image = _crop_rectangle(
                            sensor_blur_image, crop_rectangle
                        )

                output_prefix = output_dir / f"{image_file.stem}_{case_name}"
                write_image_uint8(
                    output_prefix.with_name(output_prefix.name + "_blurryimg.png"),
                    blur_image,
                )
                write_image_uint8(
                    output_prefix.with_name(output_prefix.name + "_gtimg.png"),
                    gt_image,
                )
                if sensor_blur_image is not None:
                    write_image_uint8(
                        output_prefix.with_name(
                            output_prefix.name + "_blurryimg_sensor.png"
                        ),
                        sensor_blur_image,
                    )
                    assert sensor_parameters is not None
                    sensor_model.save_sensor_metadata_npz(
                        output_prefix.with_name(
                            output_prefix.name + "_sensor_metadata.npz"
                        ),
                        sensor_parameters,
                    )
                savemat(
                    output_prefix.with_name(output_prefix.name + "_mfmap.mat"),
                    {"mfmap": mfmap},
                    do_compression=True,
                )
                savemat(
                    output_prefix.with_name(output_prefix.name + "_mf_para.mat"),
                    {
                        "para": parameters,
                        "mf": motion_field,
                        "traj": trajectory,
                        "motion_summary": motion_summary,
                    },
                    do_compression=True,
                    long_field_names=True,
                )

                if show:
                    show_debug_result(
                        gt_image,
                        blur_image,
                        mfmap,
                        motion_summary,
                        case_name,
                    )

    print(f"Done. Results are saved to: {output_root}")


# =========================================================================
# 3. 本地函数区
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


def read_image_double(path: str | Path) -> np.ndarray:
    """对应 ``im2double(imread(...))``，颜色顺序统一为 RGB。"""
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if data.ndim == 3:
        if data.shape[2] == 4:
            data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
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
    """将 [0, 1] 标准 sRGB 转为用于曝光积分的线性光强。"""
    image_srgb = clip01_uav(np.asarray(image_srgb, dtype=np.float64))
    return np.where(
        image_srgb <= 0.04045,
        image_srgb / 12.92,
        ((image_srgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb_uav(image_linear: np.ndarray) -> np.ndarray:
    """将线性曝光结果转换回 [0, 1] 标准 sRGB。"""
    image_linear = clip01_uav(np.asarray(image_linear, dtype=np.float64))
    return clip01_uav(
        np.where(
            image_linear <= 0.0031308,
            12.92 * image_linear,
            1.055 * image_linear ** (1.0 / 2.4) - 0.055,
        )
    )


def write_image_uint8(path: str | Path, image: np.ndarray) -> None:
    array = np.rint(clip01_uav(np.asarray(image)) * 255.0).astype(np.uint8)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
        else:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Cannot write image: {path}")


def scale_trajectory_motion_amplitudes(
    trajectory: dict[str, Any], strength: float
) -> None:
    """Apply one fixed comparison strength to every motion amplitude.

    This entry point is intended for controlled visual comparisons such as
    strength=1.0 versus 1.8. Online training uses independently sampled
    per-parameter strengths in ``generate_online_2d_epoch.py``.
    """
    strength = float(strength)
    if strength <= 0.0:
        raise ValueError("strength必须大于0")
    projection = str(trajectory["projection_model"])
    if projection == "composite_homography":
        for component in trajectory["components"]:
            scale_trajectory_motion_amplitudes(component, strength)
        return
    if projection == "pure_rotation_homography":
        trajectory["attitude"]["target_span_px"] = (
            float(trajectory["attitude"]["target_span_px"]) * strength
        )
        return

    path = trajectory["path"]
    if path["type"] in {"line", "quadratic_bezier"}:
        path["points"] = tuple(
            (strength * float(point[0]), strength * float(point[1]))
            for point in path["points"]
        )
    elif path["type"] == "harmonic":
        for key in ("origin_xy", "drift_xy"):
            path[key] = tuple(strength * float(value) for value in path[key])
        for harmonic in path["harmonics"]:
            harmonic["amplitude_xy"] = tuple(
                strength * float(value)
                for value in harmonic["amplitude_xy"]
            )
    else:
        raise ValueError(f"Unsupported path type: {path['type']}")

    rotation = trajectory.get("rotation", {})
    for key in ("drift_deg", "amplitude_deg"):
        if key in rotation:
            rotation[key] = float(rotation[key]) * strength


def make_uav_trajectory(
    case_name: str,
    case_seed: int,
    *,
    component_seed_base: int | None = None,
) -> dict[str, Any]:
    """从顶部配置构建一条确定轨迹；同一case_seed可完全复现。"""
    requested_case_name = case_name
    case_name = LEGACY_CASE_ALIASES.get(case_name, case_name)

    if case_name in COMPOSITE_MOTION_CONFIGS:
        composite_config = deepcopy(COMPOSITE_MOTION_CONFIGS[case_name])
        component_case_names = tuple(
            str(name)
            for name in composite_config["component_case_names"]
        )
        component_seed_root = (
            int(case_seed)
            if component_seed_base is None
            else int(component_seed_base)
        )
        component_seeds = tuple(
            component_seed_root + COMPONENT_SEED_OFFSETS[name]
            for name in component_case_names
        )
        components = [
            make_uav_trajectory(name, component_seed)
            for name, component_seed in strict_zip(
                component_case_names, component_seeds
            )
        ]
        contains_periodic_motion = any(
            maximum_periodic_frequency(component) > 0.0
            for component in components
        )
        trajectory = {
            "case_name": case_name,
            "requested_case_name": requested_case_name,
            "case_seed": int(case_seed),
            "projection_model": "composite_homography",
            "sampling": {
                "type": (
                    "screen_distance_or_period"
                    if contains_periodic_motion
                    else "screen_distance"
                ),
            },
            "description": str(composite_config["description"]),
            "component_case_names": component_case_names,
            "component_seeds": component_seeds,
            "components": components,
            "composition_order": (
                "H_total=H_last@...@H_first; source GT experiences "
                "components in component_case_names order"
            ),
            "total_dx": np.nan,
            "total_dy": np.nan,
            "total_theta_deg": np.nan,
            "total_zoom": np.nan,
        }
        validate_trajectory_config(trajectory)
        return trajectory

    if case_name not in PLATFORM_MOTION_CONFIGS:
        raise ValueError(f"Unknown blur case: {case_name}")
    trajectory: dict[str, Any] = {
        "case_name": case_name,
        "requested_case_name": requested_case_name,
        "case_seed": int(case_seed),
        **deepcopy(PLATFORM_MOTION_CONFIGS[case_name]),
    }

    projection_model = str(
        trajectory.get("projection_model", "similarity_2d")
    )
    if projection_model == "pure_rotation_homography":
        # 只随机“方向、目标像素跨度和相位”，随后根据实际输入图尺寸/FOV
        # 自动反求角度。这样不同分辨率仍得到可比较的模糊强度。
        rng = np.random.default_rng(int(case_seed))
        attitude = trajectory["attitude"]
        attitude_type = str(attitude["type"])
        span_range = np.asarray(
            attitude["target_span_px_range"], dtype=np.float64
        )
        attitude["target_span_px"] = float(
            rng.uniform(span_range[0], span_range[1])
        )
        if attitude_type == "smooth_axis_tilt":
            axis_range = np.asarray(
                attitude["axis_angle_deg_range"], dtype=np.float64
            )
            attitude["axis_angle_deg"] = float(
                rng.uniform(axis_range[0], axis_range[1])
            )
        elif attitude_type == "harmonic_euler":
            attitude["phase_deg"] = tuple(
                float(value) for value in rng.uniform(0.0, 360.0, size=3)
            )

    validate_trajectory_config(trajectory)
    if projection_model == "similarity_2d":
        start = evaluate_trajectory_at_time(trajectory, 0.0)
        middle = evaluate_trajectory_at_time(
            trajectory, GT_REFERENCE_TAU
        )
        end = evaluate_trajectory_at_time(trajectory, 1.0)
        if not np.allclose(
            middle, (0.0, 0.0, 0.0, 1.0), atol=1e-10
        ):
            raise ValueError(
                f"{case_name} 使用 GT=center，因此 t={GT_REFERENCE_TAU} "
                f"必须为恒等变换；当前为 {middle}"
            )

        # 兼容已有MAT查看器；实际轨迹仍由模块化配置计算。
        trajectory["total_dx"] = float(end[0] - start[0])
        trajectory["total_dy"] = float(end[1] - start[1])
        trajectory["total_theta_deg"] = float(end[2] - start[2])
        trajectory["total_zoom"] = float(end[3] - start[3])
    else:
        # 姿态角要在知道图像尺寸和虚拟K后才能按目标像素跨度标定。
        trajectory["total_dx"] = np.nan
        trajectory["total_dy"] = np.nan
        trajectory["total_theta_deg"] = np.nan
        trajectory["total_zoom"] = np.nan
    return trajectory


def validate_trajectory_config(trajectory: dict[str, Any]) -> None:
    """尽早检查顶部配置，避免运行到一半才因参数错误退出。"""
    sampling_type = str(trajectory["sampling"]["type"])
    if sampling_type not in {
        "screen_distance",
        "screen_distance_or_period",
    }:
        raise ValueError(f"Unsupported sampling type: {sampling_type}")
    if SUBFRAME_DISTANCE_STEP_PX <= 0.0:
        raise ValueError("SUBFRAME_DISTANCE_STEP_PX 必须大于0")
    if RECIPROCAL_SAMPLES_PER_CYCLE < 2:
        raise ValueError("RECIPROCAL_SAMPLES_PER_CYCLE 必须不小于2")
    if ADAPTIVE_PREVIEW_SAMPLES < 101 or ADAPTIVE_PREVIEW_SAMPLES % 2 == 0:
        raise ValueError("ADAPTIVE_PREVIEW_SAMPLES 必须是不小于101的奇数")

    projection_model = str(
        trajectory.get("projection_model", "similarity_2d")
    )
    if projection_model not in {
        "similarity_2d",
        "pure_rotation_homography",
        "composite_homography",
    }:
        raise ValueError(f"Unsupported projection model: {projection_model}")

    if projection_model == "composite_homography":
        components = trajectory.get("components", ())
        component_names = tuple(
            str(name)
            for name in trajectory.get("component_case_names", ())
        )
        component_seeds = tuple(
            int(seed)
            for seed in trajectory.get("component_seeds", ())
        )
        if len(components) < 2:
            raise ValueError("复合运动必须至少包含两个基础运动分量")
        if (
            len(component_names) != len(components)
            or len(component_seeds) != len(components)
        ):
            raise ValueError(
                "复合运动的names/seeds/components数量必须完全一致"
            )
        for expected_name, component in strict_zip(
            component_names, components
        ):
            if str(component["case_name"]) != expected_name:
                raise ValueError("复合运动分量名称与分量轨迹不一致")
            if str(component["projection_model"]) == "composite_homography":
                raise ValueError("当前不允许复合运动中再次嵌套复合运动")
            validate_trajectory_config(component)
        return

    if projection_model == "pure_rotation_homography":
        if not 1.0 < VIRTUAL_CAMERA_HORIZONTAL_FOV_DEG < 179.0:
            raise ValueError("虚拟相机水平FOV必须位于(1,179)度")
        if (
            ATTITUDE_CALIBRATION_TIME_SAMPLES < 101
            or ATTITUDE_CALIBRATION_TIME_SAMPLES % 2 == 0
        ):
            raise ValueError("姿态振幅标定时间点必须是不小于101的奇数")
        if ATTITUDE_CALIBRATION_GRID_SIZE < 3:
            raise ValueError("姿态振幅标定网格边长必须不小于3")
        attitude = trajectory["attitude"]
        attitude_type = str(attitude["type"])
        target_range = np.asarray(
            attitude["target_span_px_range"], dtype=np.float64
        )
        if (
            target_range.shape != (2,)
            or target_range[0] <= 0.0
            or target_range[1] < target_range[0]
        ):
            raise ValueError("target_span_px_range必须是正的(min,max)")
        target_span = float(attitude["target_span_px"])
        if not target_range[0] <= target_span <= target_range[1]:
            raise ValueError("随机target_span_px超出配置范围")
        if attitude_type == "smooth_axis_tilt":
            axis_range = np.asarray(
                attitude["axis_angle_deg_range"], dtype=np.float64
            )
            if axis_range.shape != (2,) or axis_range[1] < axis_range[0]:
                raise ValueError("axis_angle_deg_range必须是(min,max)")
            if not axis_range[0] <= float(
                attitude["axis_angle_deg"]
            ) <= axis_range[1]:
                raise ValueError("随机axis_angle_deg超出配置范围")
            time_type = str(attitude["time_mapping"]["type"])
            if time_type not in {
                "uniform",
                "power_accelerate",
                "power_decelerate",
                "smoothstep",
            }:
                raise ValueError(
                    f"Unsupported attitude time mapping: {time_type}"
                )
        elif attitude_type == "harmonic_euler":
            frequencies = np.asarray(
                attitude["frequencies"], dtype=np.float64
            )
            amplitude_ratios = np.asarray(
                attitude["amplitude_ratios"], dtype=np.float64
            )
            phases = np.asarray(attitude["phase_deg"], dtype=np.float64)
            if (
                frequencies.shape != (3,)
                or amplitude_ratios.shape != (3,)
                or phases.shape != (3,)
            ):
                raise ValueError(
                    "三轴振动frequencies/amplitude_ratios/phase_deg"
                    "都必须含3个数"
                )
            if np.any(frequencies <= 0.0):
                raise ValueError("三轴振动频率必须全部大于0")
            if np.any(amplitude_ratios < 0.0) or not np.any(
                amplitude_ratios > 0.0
            ):
                raise ValueError("三轴振动幅值比例必须非负且不能全为0")
        else:
            raise ValueError(f"Unsupported attitude type: {attitude_type}")
        return

    path = trajectory["path"]
    path_type = str(path["type"])
    if path_type == "line":
        points = np.asarray(path["points"], dtype=np.float64)
        if points.shape != (2, 2):
            raise ValueError("line path 的 points 必须是两个二维点")
    elif path_type == "quadratic_bezier":
        points = np.asarray(path["points"], dtype=np.float64)
        if points.shape != (3, 2):
            raise ValueError("quadratic_bezier 的 points 必须是 P0/P1/P2")
        samples = int(path.get("arc_length_samples", ARC_LENGTH_LOOKUP_SAMPLES))
        if samples < 101:
            raise ValueError("Bézier arc_length_samples 建议且必须不小于101")
    elif path_type == "harmonic":
        if np.asarray(path["origin_xy"], dtype=np.float64).shape != (2,):
            raise ValueError("harmonic path 的 origin_xy 必须是二维向量")
        if np.asarray(path["drift_xy"], dtype=np.float64).shape != (2,):
            raise ValueError("harmonic path 的 drift_xy 必须是二维向量")
        for harmonic in path["harmonics"]:
            if np.asarray(
                harmonic["amplitude_xy"], dtype=np.float64
            ).shape != (2,):
                raise ValueError("harmonic amplitude_xy 必须是二维向量")
            if float(harmonic["frequency"]) < 0.0:
                raise ValueError("harmonic frequency 不能小于0")
    else:
        raise ValueError(f"Unsupported path type: {path_type}")

    time_mapping = trajectory["time_mapping"]
    time_type = str(time_mapping["type"])
    supported_time_types = {
        "uniform",
        "power_accelerate",
        "power_decelerate",
        "smoothstep",
    }
    if time_type not in supported_time_types:
        raise ValueError(f"Unsupported time mapping: {time_type}")
    if time_type.startswith("power_") and float(time_mapping["gamma"]) <= 1.0:
        raise ValueError(f"{time_type} 的 gamma 必须大于1")

    # 通过首末点评估同时验证 rotation/scale 类型以及缩放始终为正。
    for progress in np.linspace(0.0, 1.0, 101):
        theta = evaluate_rotation_path(
            trajectory["rotation"], path, float(progress)
        )
        scale_value = evaluate_scale_path(
            trajectory["scale"], float(progress)
        )
        if not np.isfinite(theta) or not np.isfinite(scale_value):
            raise ValueError("rotation/scale 配置产生了非有限数值")
        if scale_value <= 0.0:
            raise ValueError("整个曝光期间的 scale 必须始终大于0")


def evaluate_time_mapping(time_mapping: dict[str, Any], tau: float) -> float:
    """把物理曝光时间 ``tau`` 映射为归一化路径进度 ``s``。"""
    if not -1e-12 <= tau <= 1.0 + 1e-12:
        raise ValueError(f"tau 必须位于 [0,1]，当前为 {tau}")
    tau = float(np.clip(tau, 0.0, 1.0))
    mapping_type = str(time_mapping["type"])
    if mapping_type == "uniform":
        progress = tau
    elif mapping_type == "power_accelerate":
        progress = tau ** float(time_mapping["gamma"])
    elif mapping_type == "power_decelerate":
        gamma = float(time_mapping["gamma"])
        progress = 1.0 - (1.0 - tau) ** gamma
    elif mapping_type == "smoothstep":
        progress = tau * tau * (3.0 - 2.0 * tau)
    else:
        raise ValueError(f"Unsupported time mapping: {mapping_type}")
    return float(np.clip(progress, 0.0, 1.0))


def quadratic_bezier_points(
    points: np.ndarray, parameter: np.ndarray | float
) -> np.ndarray:
    """计算二次 Bézier 位置；``parameter`` 是原始曲线参数 u。"""
    points = np.asarray(points, dtype=np.float64)
    parameter_array = np.asarray(parameter, dtype=np.float64)
    one_minus = 1.0 - parameter_array
    return (
        one_minus[..., None] ** 2 * points[0]
        + 2.0
        * one_minus[..., None]
        * parameter_array[..., None]
        * points[1]
        + parameter_array[..., None] ** 2 * points[2]
    )


@lru_cache(maxsize=32)
def cached_quadratic_bezier_arc_lookup(
    points_key: tuple[tuple[float, float], ...],
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """建立并缓存 ``归一化弧长 s -> 原始 Bézier 参数 u`` 查找表。"""
    points = np.asarray(points_key, dtype=np.float64)
    u_dense = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    positions = quadratic_bezier_points(points, u_dense)
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-12:
        raise ValueError("Bézier 路径总长度必须大于0")
    normalized_arc = cumulative / total_length
    return normalized_arc, u_dense


def native_path_parameter(path: dict[str, Any], progress: float) -> float:
    """把归一化路径进度 s 转为路径自身参数 u。"""
    progress = float(np.clip(progress, 0.0, 1.0))
    if (
        str(path["type"]) == "quadratic_bezier"
        and bool(path.get("arc_length_parameterized", False))
    ):
        points_key = tuple(
            tuple(float(value) for value in point)
            for point in path["points"]
        )
        sample_count = int(
            path.get("arc_length_samples", ARC_LENGTH_LOOKUP_SAMPLES)
        )
        normalized_arc, u_dense = cached_quadratic_bezier_arc_lookup(
            points_key, sample_count
        )
        return float(np.interp(progress, normalized_arc, u_dense))
    return progress


def evaluate_spatial_path(
    path: dict[str, Any], progress: float
) -> tuple[float, float]:
    """在归一化路径进度 ``s`` 上计算平台中心位移 P(s)。"""
    path_type = str(path["type"])
    parameter = native_path_parameter(path, progress)
    if path_type == "line":
        points = np.asarray(path["points"], dtype=np.float64)
        position = (1.0 - parameter) * points[0] + parameter * points[1]
    elif path_type == "quadratic_bezier":
        position = quadratic_bezier_points(
            np.asarray(path["points"], dtype=np.float64), parameter
        )
    elif path_type == "harmonic":
        position = np.asarray(path["origin_xy"], dtype=np.float64).copy()
        position += (
            np.asarray(path["drift_xy"], dtype=np.float64) * parameter
        )
        for harmonic in path["harmonics"]:
            amplitude = np.asarray(
                harmonic["amplitude_xy"], dtype=np.float64
            )
            phase_rad = np.deg2rad(float(harmonic["phase_deg"]))
            position += amplitude * np.sin(
                2.0
                * np.pi
                * float(harmonic["frequency"])
                * parameter
                + phase_rad
            )
    else:
        raise ValueError(f"Unsupported path type: {path_type}")
    return float(position[0]), float(position[1])


def evaluate_path_tangent(
    path: dict[str, Any], progress: float
) -> tuple[float, float]:
    """计算路径切线，仅供 ``follow_path_tangent`` 旋转模式使用。"""
    path_type = str(path["type"])
    parameter = native_path_parameter(path, progress)
    if path_type == "line":
        points = np.asarray(path["points"], dtype=np.float64)
        tangent = points[1] - points[0]
    elif path_type == "quadratic_bezier":
        points = np.asarray(path["points"], dtype=np.float64)
        tangent = (
            2.0 * (1.0 - parameter) * (points[1] - points[0])
            + 2.0 * parameter * (points[2] - points[1])
        )
    elif path_type == "harmonic":
        tangent = np.asarray(path["drift_xy"], dtype=np.float64).copy()
        for harmonic in path["harmonics"]:
            amplitude = np.asarray(
                harmonic["amplitude_xy"], dtype=np.float64
            )
            frequency = float(harmonic["frequency"])
            phase_rad = np.deg2rad(float(harmonic["phase_deg"]))
            tangent += (
                amplitude
                * 2.0
                * np.pi
                * frequency
                * np.cos(2.0 * np.pi * frequency * parameter + phase_rad)
            )
    else:
        raise ValueError(f"Unsupported path type: {path_type}")
    if float(np.linalg.norm(tangent)) <= 1e-12:
        raise ValueError("当前路径位置的切线长度为0，不能跟随切线旋转")
    return float(tangent[0]), float(tangent[1])


def evaluate_rotation_path(
    rotation: dict[str, Any],
    path: dict[str, Any],
    progress: float,
) -> float:
    """计算作为空间姿态路径一部分的旋转角，单位度。"""
    rotation_type = str(rotation["type"])
    if rotation_type == "constant":
        theta = float(rotation["value_deg"])
    elif rotation_type == "linear":
        theta = (1.0 - progress) * float(rotation["start_deg"])
        theta += progress * float(rotation["end_deg"])
    elif rotation_type == "harmonic":
        phase_rad = np.deg2rad(float(rotation["phase_deg"]))
        theta = float(rotation["base_deg"])
        theta += float(rotation["drift_deg"]) * progress
        theta += float(rotation["amplitude_deg"]) * np.sin(
            2.0 * np.pi * float(rotation["frequency"]) * progress
            + phase_rad
        )
    elif rotation_type == "follow_path_tangent":
        tangent = evaluate_path_tangent(path, progress)
        tangent_start = evaluate_path_tangent(path, 0.0)
        theta = np.rad2deg(np.arctan2(tangent[1], tangent[0]))
        theta -= np.rad2deg(
            np.arctan2(tangent_start[1], tangent_start[0])
        )
        theta += float(rotation.get("offset_deg", 0.0))
    else:
        raise ValueError(f"Unsupported rotation type: {rotation_type}")
    return float(theta)


def evaluate_scale_path(scale: dict[str, Any], progress: float) -> float:
    """计算作为空间姿态路径一部分的缩放比例。"""
    scale_type = str(scale["type"])
    if scale_type == "constant":
        value = float(scale["value"])
    elif scale_type == "linear":
        value = (1.0 - progress) * float(scale["start"])
        value += progress * float(scale["end"])
    elif scale_type == "log_linear":
        start = float(scale["start"])
        end = float(scale["end"])
        if start <= 0.0 or end <= 0.0:
            raise ValueError("log_linear scale 的 start/end 必须大于0")
        value = float(
            np.exp((1.0 - progress) * np.log(start) + progress * np.log(end))
        )
    else:
        raise ValueError(f"Unsupported scale type: {scale_type}")
    return float(value)


def evaluate_trajectory_at_time(
    trajectory: dict[str, Any], tau: float
) -> tuple[float, float, float, float]:
    """计算相对于曝光中心GT的相似变换参数。

    顶部配置描述一条原始绝对姿态路径。这里先分别求当前时刻与GT参考时刻
    的姿态，再计算 ``H_relative(t)=H_absolute(t)@inv(H_reference)``。
    因此无论配置采用平移、旋转还是缩放，参考时刻都严格返回恒等变换。
    """
    if str(trajectory["projection_model"]) != "similarity_2d":
        raise ValueError(
            "三维姿态轨迹没有唯一dx/dy/theta/scale；"
            "请调用evaluate_forward_homography_at_time"
        )
    current = evaluate_absolute_trajectory_at_time(trajectory, tau)
    reference = evaluate_absolute_trajectory_at_time(
        trajectory, GT_REFERENCE_TAU
    )
    current_dx, current_dy, current_theta, current_scale = current
    reference_dx, reference_dy, reference_theta, reference_scale = reference
    relative_theta = current_theta - reference_theta
    relative_scale = current_scale / reference_scale
    theta_rad = np.deg2rad(relative_theta)
    relative_linear = relative_scale * np.asarray(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ],
        dtype=np.float64,
    )
    current_translation = np.asarray(
        [current_dx, current_dy], dtype=np.float64
    )
    reference_translation = np.asarray(
        [reference_dx, reference_dy], dtype=np.float64
    )
    relative_translation = (
        current_translation - relative_linear @ reference_translation
    )
    return (
        float(relative_translation[0]),
        float(relative_translation[1]),
        float(relative_theta),
        float(relative_scale),
    )


def evaluate_absolute_trajectory_at_time(
    trajectory: dict[str, Any], tau: float
) -> tuple[float, float, float, float]:
    """计算顶部配置定义的原始姿态路径，尚未相对GT参考时刻归一化。"""
    progress = evaluate_time_mapping(trajectory["time_mapping"], tau)
    dx, dy = evaluate_spatial_path(trajectory["path"], progress)
    theta = evaluate_rotation_path(
        trajectory["rotation"], trajectory["path"], progress
    )
    scale_value = evaluate_scale_path(trajectory["scale"], progress)
    return dx, dy, theta, scale_value


def build_virtual_camera_intrinsics(
    height: int,
    width: int,
    horizontal_fov_deg: float = VIRTUAL_CAMERA_HORIZONTAL_FOV_DEG,
) -> np.ndarray:
    """仅由图像尺寸和水平FOV建立虚拟K；有真实标定时应替换本函数。"""
    if height < 2 or width < 2:
        raise ValueError("输入图像高宽必须都不小于2")
    if not 1.0 < horizontal_fov_deg < 179.0:
        raise ValueError("horizontal_fov_deg必须位于(1,179)度")
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    fx = (width / 2.0) / np.tan(
        np.deg2rad(horizontal_fov_deg) / 2.0
    )
    fy = fx  # 假设方形像素；不是把整张图做统一缩放。
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def evaluate_absolute_attitude_rotation(
    trajectory: dict[str, Any],
    tau: float,
    *,
    angular_scale_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回绝对相机姿态R及便于记录的(x,y,z)等效角，单位度。"""
    attitude = trajectory["attitude"]
    attitude_type = str(attitude["type"])
    if angular_scale_deg is None:
        if attitude_type == "smooth_axis_tilt":
            angular_scale_deg = float(attitude["total_angle_span_deg"])
        else:
            angular_scale_deg = float(attitude["angular_amplitude_deg"])

    if attitude_type == "smooth_axis_tilt":
        progress = evaluate_time_mapping(attitude["time_mapping"], tau)
        # total_angle_span_deg表示曝光开始到结束的总角度覆盖。
        angle_deg = float(angular_scale_deg) * (progress - 0.5)
        axis_angle_rad = np.deg2rad(float(attitude["axis_angle_deg"]))
        axis = np.asarray(
            [np.cos(axis_angle_rad), np.sin(axis_angle_rad), 0.0],
            dtype=np.float64,
        )
        rotation_vector = axis * np.deg2rad(angle_deg)
        rotation, _ = cv2.Rodrigues(rotation_vector)
        euler_like_deg = axis * angle_deg
    elif attitude_type == "harmonic_euler":
        frequencies = np.asarray(
            attitude["frequencies"], dtype=np.float64
        )
        ratios = np.asarray(
            attitude["amplitude_ratios"], dtype=np.float64
        )
        phases_rad = np.deg2rad(
            np.asarray(attitude["phase_deg"], dtype=np.float64)
        )
        euler_like_deg = (
            float(angular_scale_deg)
            * ratios
            * np.sin(2.0 * np.pi * frequencies * float(tau) + phases_rad)
        )
        x_rad, y_rad, z_rad = np.deg2rad(euler_like_deg)
        rotation = (
            rotation_matrix_z(float(z_rad))
            @ rotation_matrix_y(float(y_rad))
            @ rotation_matrix_x(float(x_rad))
        )
    else:
        raise ValueError(f"Unsupported attitude type: {attitude_type}")
    return np.asarray(rotation, dtype=np.float64), np.asarray(
        euler_like_deg, dtype=np.float64
    )


def pure_rotation_homography_at_time(
    trajectory: dict[str, Any],
    tau: float,
    camera_intrinsics: np.ndarray,
    *,
    angular_scale_deg: float | None = None,
) -> np.ndarray:
    """计算相对曝光中心GT的纯旋转单应矩阵 H=K R_rel K^-1。"""
    current_rotation, _ = evaluate_absolute_attitude_rotation(
        trajectory, tau, angular_scale_deg=angular_scale_deg
    )
    reference_rotation, _ = evaluate_absolute_attitude_rotation(
        trajectory,
        GT_REFERENCE_TAU,
        angular_scale_deg=angular_scale_deg,
    )
    relative_rotation = current_rotation @ reference_rotation.T
    camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    homography = (
        camera_intrinsics
        @ relative_rotation
        @ np.linalg.inv(camera_intrinsics)
    )
    if abs(float(homography[2, 2])) <= 1e-12:
        raise ValueError("姿态单应矩阵的H[2,2]过小")
    return homography / homography[2, 2]


def similarity_forward_homography(
    dx: float,
    dy: float,
    theta_deg: float,
    scale_value: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """把原有2D相似变换写成统一的3×3前向单应矩阵。"""
    if scale_value <= 0.0:
        raise ValueError("scale_value must be positive")
    theta_rad = np.deg2rad(theta_deg)
    cosine, sine = np.cos(theta_rad), np.sin(theta_rad)
    linear = scale_value * np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    center = np.asarray([cx, cy], dtype=np.float64)
    translation = (
        center + np.asarray([dx, dy], dtype=np.float64) - linear @ center
    )
    return np.asarray(
        [
            [linear[0, 0], linear[0, 1], translation[0]],
            [linear[1, 0], linear[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def apply_homography_forward(
    x: np.ndarray,
    y: np.ndarray,
    homography: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """用前向单应矩阵映射任意同形状坐标数组。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    homography = np.asarray(homography, dtype=np.float64)
    denominator = (
        homography[2, 0] * x
        + homography[2, 1] * y
        + homography[2, 2]
    )
    if np.any(np.abs(denominator) <= 1e-10):
        raise ValueError("单应变换把采样点映射到无穷远")
    x_position = (
        homography[0, 0] * x
        + homography[0, 1] * y
        + homography[0, 2]
    ) / denominator
    y_position = (
        homography[1, 0] * x
        + homography[1, 1] * y
        + homography[1, 2]
    ) / denominator
    return x_position, y_position


def attitude_motion_span_px(
    trajectory: dict[str, Any],
    camera_intrinsics: np.ndarray,
    height: int,
    width: int,
    angular_scale_deg: float,
) -> float:
    """测量一条姿态轨迹在图像网格上的最大首末包络跨度。"""
    grid_x, grid_y = np.meshgrid(
        np.linspace(
            0.0, width - 1.0, ATTITUDE_CALIBRATION_GRID_SIZE
        ),
        np.linspace(
            0.0, height - 1.0, ATTITUDE_CALIBRATION_GRID_SIZE
        ),
    )
    x_min = np.full(grid_x.shape, np.inf, dtype=np.float64)
    x_max = np.full(grid_x.shape, -np.inf, dtype=np.float64)
    y_min = np.full(grid_y.shape, np.inf, dtype=np.float64)
    y_max = np.full(grid_y.shape, -np.inf, dtype=np.float64)
    for tau in np.linspace(
        0.0, 1.0, ATTITUDE_CALIBRATION_TIME_SAMPLES
    ):
        homography = pure_rotation_homography_at_time(
            trajectory,
            float(tau),
            camera_intrinsics,
            angular_scale_deg=angular_scale_deg,
        )
        x_now, y_now = apply_homography_forward(
            grid_x, grid_y, homography
        )
        x_min = np.minimum(x_min, x_now)
        x_max = np.maximum(x_max, x_now)
        y_min = np.minimum(y_min, y_now)
        y_max = np.maximum(y_max, y_now)
    return float(np.max(np.hypot(x_max - x_min, y_max - y_min)))


def prepare_trajectory_for_image(
    trajectory: dict[str, Any],
    height: int,
    width: int,
) -> None:
    """知道图像尺寸后，准备姿态分量并检查中心GT严格对应恒等变换。"""
    projection_model = str(trajectory["projection_model"])
    if projection_model not in {
        "pure_rotation_homography",
        "composite_homography",
    }:
        return
    if (
        int(trajectory.get("prepared_height", -1)) == height
        and int(trajectory.get("prepared_width", -1)) == width
    ):
        return

    if projection_model == "composite_homography":
        for component in trajectory["components"]:
            prepare_trajectory_for_image(component, height, width)
        attitude_components = [
            component
            for component in trajectory["components"]
            if str(component["projection_model"])
            == "pure_rotation_homography"
        ]
        if attitude_components:
            trajectory["camera_intrinsics_K"] = np.asarray(
                attitude_components[0]["camera_intrinsics_K"],
                dtype=np.float64,
            )
            trajectory["virtual_camera_horizontal_fov_deg"] = float(
                attitude_components[0][
                    "virtual_camera_horizontal_fov_deg"
                ]
            )
        trajectory["prepared_height"] = int(height)
        trajectory["prepared_width"] = int(width)
        center_homography = evaluate_forward_homography_at_time(
            trajectory, GT_REFERENCE_TAU, height, width
        )
        if not np.allclose(center_homography, np.eye(3), atol=1e-9):
            raise RuntimeError(
                "复合轨迹没有在曝光中心归一化为GT恒等变换"
            )
        return

    camera_intrinsics = build_virtual_camera_intrinsics(height, width)
    target_span = float(trajectory["attitude"]["target_span_px"])
    low_angle, high_angle = 0.0, 1.0
    while (
        attitude_motion_span_px(
            trajectory,
            camera_intrinsics,
            height,
            width,
            high_angle,
        )
        < target_span
    ):
        high_angle *= 2.0
        if high_angle > 45.0:
            raise ValueError(
                "目标像素跨度需要超过45度姿态角，请检查FOV和运动范围"
            )
    for _ in range(24):
        middle_angle = 0.5 * (low_angle + high_angle)
        if (
            attitude_motion_span_px(
                trajectory,
                camera_intrinsics,
                height,
                width,
                middle_angle,
            )
            < target_span
        ):
            low_angle = middle_angle
        else:
            high_angle = middle_angle
    resolved_angle = 0.5 * (low_angle + high_angle)
    achieved_span = attitude_motion_span_px(
        trajectory,
        camera_intrinsics,
        height,
        width,
        resolved_angle,
    )
    attitude = trajectory["attitude"]
    if str(attitude["type"]) == "smooth_axis_tilt":
        attitude["total_angle_span_deg"] = float(resolved_angle)
    else:
        attitude["angular_amplitude_deg"] = float(resolved_angle)
    attitude["achieved_span_px"] = float(achieved_span)
    trajectory["camera_intrinsics_K"] = camera_intrinsics
    trajectory["virtual_camera_horizontal_fov_deg"] = float(
        VIRTUAL_CAMERA_HORIZONTAL_FOV_DEG
    )
    trajectory["prepared_height"] = int(height)
    trajectory["prepared_width"] = int(width)

    center_homography = pure_rotation_homography_at_time(
        trajectory, GT_REFERENCE_TAU, camera_intrinsics
    )
    if not np.allclose(center_homography, np.eye(3), atol=1e-9):
        raise RuntimeError("姿态轨迹没有在曝光中心归一化为GT恒等变换")


def evaluate_forward_homography_at_time(
    trajectory: dict[str, Any],
    tau: float,
    height: int,
    width: int,
) -> np.ndarray:
    """为基础与复合轨迹统一返回源GT到当前子帧的3×3前向变换。"""
    projection_model = str(trajectory["projection_model"])
    if projection_model == "similarity_2d":
        cx, cy = (width - 1.0) / 2.0, (height - 1.0) / 2.0
        dx, dy, theta, scale_value = evaluate_trajectory_at_time(
            trajectory, tau
        )
        return similarity_forward_homography(
            dx, dy, theta, scale_value, cx, cy
        )
    if projection_model == "pure_rotation_homography":
        prepare_trajectory_for_image(trajectory, height, width)
        return pure_rotation_homography_at_time(
            trajectory,
            tau,
            np.asarray(trajectory["camera_intrinsics_K"]),
        )
    if projection_model == "composite_homography":
        prepare_trajectory_for_image(trajectory, height, width)
        homography = np.eye(3, dtype=np.float64)
        for component in trajectory["components"]:
            component_homography = evaluate_forward_homography_at_time(
                component, tau, height, width
            )
            # component_case_names描述源GT经历变换的先后次序。列向量约定下，
            # 后发生的变换左乘，因此主路径、姿态、振动对应 Hvib@Hatt@Hmain。
            homography = component_homography @ homography
        if abs(float(homography[2, 2])) <= 1e-12:
            raise ValueError("复合单应矩阵的H[2,2]过小")
        return homography / homography[2, 2]
    raise ValueError(f"Unsupported projection model: {projection_model}")


def warp_frame_homography(
    image: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    """按源到目标的H做一次透视warp；OpenCV内部执行逆向双线性采样。"""
    height, width = image.shape[:2]
    result = cv2.warpPerspective(
        np.asarray(image, dtype=np.float64),
        np.asarray(homography, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return clip01_uav(result)


def warp_frame_similarity(
    image: np.ndarray,
    dx: float,
    dy: float,
    theta_deg: float,
    scale_value: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """对应 MATLAB ``interp2`` 的逆向双线性采样，图外补 0。"""
    if scale_value <= 0:
        raise ValueError("scale_value must be positive")
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
    result = cv2.remap(
        np.asarray(image, dtype=np.float64),
        x_input.astype(np.float32),
        y_input.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return clip01_uav(result)


def apply_similarity_forward(
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
    theta_deg: float,
    scale_value: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(theta_deg)
    x_centered = x - cx
    y_centered = y - cy
    x_position = scale_value * (
        np.cos(theta) * x_centered - np.sin(theta) * y_centered
    ) + cx + dx
    y_position = scale_value * (
        np.sin(theta) * x_centered + np.cos(theta) * y_centered
    ) + cy + dy
    return x_position, y_position


def build_motion_summary_from_trajectory(
    height: int,
    width: int,
    tau_list: np.ndarray,
    trajectory: dict[str, Any],
    cx: float,
    cy: float,
) -> dict[str, np.ndarray]:
    """统计每个像素的起止位移、路径长度、跨度和最大偏移。"""
    prepare_trajectory_for_image(trajectory, height, width)
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    x_min = np.full((height, width), np.inf)
    x_max = np.full((height, width), -np.inf)
    y_min = np.full((height, width), np.inf)
    y_max = np.full((height, width), -np.inf)
    path_length = np.zeros((height, width), dtype=np.float64)
    max_abs_x = np.zeros((height, width), dtype=np.float64)
    max_abs_y = np.zeros((height, width), dtype=np.float64)

    x_previous = y_previous = x_start = y_start = None
    for index, tau in enumerate(tau_list):
        homography = evaluate_forward_homography_at_time(
            trajectory, float(tau), height, width
        )
        x_position, y_position = apply_homography_forward(
            x, y, homography
        )
        if index == 0:
            x_start, y_start = x_position.copy(), y_position.copy()
        else:
            path_length += np.sqrt(
                (x_position - x_previous) ** 2 + (y_position - y_previous) ** 2
            )
        x_min = np.minimum(x_min, x_position)
        x_max = np.maximum(x_max, x_position)
        y_min = np.minimum(y_min, y_position)
        y_max = np.maximum(y_max, y_position)
        max_abs_x = np.maximum(max_abs_x, np.abs(x_position - x))
        max_abs_y = np.maximum(max_abs_y, np.abs(y_position - y))
        x_previous, y_previous = x_position, y_position

    assert x_start is not None and y_start is not None
    assert x_previous is not None and y_previous is not None
    span_u = x_max - x_min
    span_v = y_max - y_min
    return {
        "end_mu": x_previous - x_start,
        "end_mv": y_previous - y_start,
        "span_u": span_u,
        "span_v": span_v,
        "span_mag": np.sqrt(span_u**2 + span_v**2),
        "path_len": path_length,
        "max_abs_x_from_ref": max_abs_x,
        "max_abs_y_from_ref": max_abs_y,
    }


def maximum_periodic_frequency(trajectory: dict[str, Any]) -> float:
    """返回平移/旋转谐波中的最高频率（一次曝光内的周期数）。"""
    if str(trajectory["projection_model"]) == "composite_homography":
        return max(
            (
                maximum_periodic_frequency(component)
                for component in trajectory["components"]
            ),
            default=0.0,
        )
    if (
        str(trajectory["projection_model"])
        == "pure_rotation_homography"
    ):
        attitude = trajectory["attitude"]
        if str(attitude["type"]) == "harmonic_euler":
            return float(
                np.max(
                    np.asarray(
                        attitude["frequencies"], dtype=np.float64
                    )
                )
            )
        return 0.0

    frequencies: list[float] = []
    path = trajectory["path"]
    if str(path["type"]) == "harmonic":
        frequencies.extend(
            float(harmonic["frequency"])
            for harmonic in path["harmonics"]
        )
    rotation = trajectory["rotation"]
    if str(rotation["type"]) == "harmonic":
        frequencies.append(float(rotation["frequency"]))
    return max(frequencies, default=0.0)


def temporal_trapezoid_weights(tau_list: np.ndarray) -> np.ndarray:
    """为非均匀曝光时刻建立梯形积分权重，权重和严格归一化为1。"""
    tau_list = np.asarray(tau_list, dtype=np.float64)
    if tau_list.ndim != 1 or tau_list.size < 2:
        raise ValueError("tau_list 至少需要两个严格递增的时间点")
    if not np.all(np.diff(tau_list) > 0.0):
        raise ValueError("tau_list 必须严格递增")
    weights = np.empty_like(tau_list)
    weights[0] = 0.5 * (tau_list[1] - tau_list[0])
    weights[-1] = 0.5 * (tau_list[-1] - tau_list[-2])
    if tau_list.size > 2:
        weights[1:-1] = 0.5 * (tau_list[2:] - tau_list[:-2])
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("曝光时间权重之和必须大于0")
    return weights / weight_sum


def build_adaptive_subframe_schedule(
    trajectory: dict[str, Any],
    height: int,
    width: int,
    *,
    distance_step_px: float = SUBFRAME_DISTANCE_STEP_PX,
    samples_per_cycle: int = RECIPROCAL_SAMPLES_PER_CYCLE,
    preview_samples: int = ADAPTIVE_PREVIEW_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """按统一前向变换的最大屏幕运动选择非均匀曝光时刻。

    使用覆盖全图的规则网格作为参考点。每个稠密预采样小区间的运动量定义为
    所有参考点位移中的最大值，再沿时间累计。这样能处理曲线、旋转、缩放
    和透视姿态，以及回到原位的往复轨迹，不会把首末净位移0误判成无运动。
    """
    if distance_step_px <= 0.0:
        raise ValueError("distance_step_px 必须大于0")
    if samples_per_cycle < 2:
        raise ValueError("samples_per_cycle 必须不小于2")
    if preview_samples < 101 or preview_samples % 2 == 0:
        raise ValueError("preview_samples 必须是不小于101的奇数")

    if ADAPTIVE_SAMPLING_GRID_SIZE < 3:
        raise ValueError("ADAPTIVE_SAMPLING_GRID_SIZE必须不小于3")
    prepare_trajectory_for_image(trajectory, height, width)
    anchor_x_grid, anchor_y_grid = np.meshgrid(
        np.linspace(0.0, width - 1.0, ADAPTIVE_SAMPLING_GRID_SIZE),
        np.linspace(0.0, height - 1.0, ADAPTIVE_SAMPLING_GRID_SIZE),
    )
    anchor_x = anchor_x_grid.ravel()
    anchor_y = anchor_y_grid.ravel()
    dense_tau = np.linspace(0.0, 1.0, preview_samples, dtype=np.float64)
    anchor_positions = np.empty(
        (preview_samples, anchor_x.size, 2), dtype=np.float64
    )
    for index, tau in enumerate(dense_tau):
        homography = evaluate_forward_homography_at_time(
            trajectory, float(tau), height, width
        )
        x_now, y_now = apply_homography_forward(
            anchor_x, anchor_y, homography
        )
        anchor_positions[index, :, 0] = x_now
        anchor_positions[index, :, 1] = y_now

    per_anchor_segment = np.linalg.norm(
        np.diff(anchor_positions, axis=0), axis=2
    )
    segment_screen_motion = np.max(per_anchor_segment, axis=1)
    cumulative_screen_motion = np.concatenate(
        ([0.0], np.cumsum(segment_screen_motion))
    )

    sampling_type = str(trajectory["sampling"]["type"])
    highest_frequency = maximum_periodic_frequency(trajectory)
    max_period_time_step = np.inf
    if sampling_type == "screen_distance_or_period":
        if highest_frequency <= 0.0:
            raise ValueError(
                "screen_distance_or_period 要求路径或旋转中至少有一个正频率谐波"
            )
        max_period_time_step = 1.0 / (
            float(samples_per_cycle) * highest_frequency
        )

    selected_indices = [0]
    last_index = 0
    for dense_index in range(1, preview_samples - 1):
        distance_due = (
            cumulative_screen_motion[dense_index]
            - cumulative_screen_motion[last_index]
            >= distance_step_px
        )
        period_due = (
            dense_tau[dense_index] - dense_tau[last_index]
            >= max_period_time_step
        )
        if distance_due or period_due:
            selected_indices.append(dense_index)
            last_index = dense_index
    selected_indices.append(preview_samples - 1)

    # 强制加入物理曝光中间时刻，使输入GT仍然是一个真实参与积分的子帧。
    gt_reference_index_dense = int(
        round(GT_REFERENCE_TAU * (preview_samples - 1))
    )
    selected_indices.append(gt_reference_index_dense)
    selected_indices_array = np.asarray(
        sorted(set(selected_indices)), dtype=np.int64
    )
    tau_list = dense_tau[selected_indices_array]
    gt_matches = np.flatnonzero(
        np.isclose(tau_list, GT_REFERENCE_TAU, atol=1e-12, rtol=0.0)
    )
    if gt_matches.size != 1:
        raise RuntimeError("自适应采样必须且只能包含一个GT参考时刻")

    temporal_weights = temporal_trapezoid_weights(tau_list)
    selected_cumulative = cumulative_screen_motion[
        selected_indices_array
    ]
    selected_motion_steps = np.diff(selected_cumulative)
    selected_time_steps = np.diff(tau_list)
    schedule_metadata: dict[str, Any] = {
        "adaptive_sampling": True,
        "sampling_type": sampling_type,
        "distance_step_target_px": float(distance_step_px),
        "reciprocal_samples_per_cycle": int(samples_per_cycle),
        "highest_periodic_frequency_cycles_per_exposure": float(
            highest_frequency
        ),
        "period_time_step_limit": (
            float(max_period_time_step)
            if np.isfinite(max_period_time_step)
            else np.nan
        ),
        "preview_samples": int(preview_samples),
        "screen_motion_reference_grid_size": int(
            ADAPTIVE_SAMPLING_GRID_SIZE
        ),
        "preview_total_screen_motion_px": float(
            cumulative_screen_motion[-1]
        ),
        "selected_max_accumulated_screen_step_px": float(
            selected_motion_steps.max(initial=0.0)
        ),
        "selected_max_time_step": float(
            selected_time_steps.max(initial=0.0)
        ),
        "gt_reference_subframe_index_zero_based": int(gt_matches[0]),
    }
    return tau_list, temporal_weights, schedule_metadata


def build_fixed_subframe_schedule(
    n_subframes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """兼容旧实验：使用固定奇数帧、均匀时间和等权离散平均。"""
    if n_subframes < 3 or n_subframes % 2 == 0:
        raise ValueError("固定中心GT采样要求 N_subframes 为不小于3的奇数")
    tau_list = np.linspace(0.0, 1.0, n_subframes, dtype=np.float64)
    temporal_weights = np.full(
        n_subframes, 1.0 / n_subframes, dtype=np.float64
    )
    return tau_list, temporal_weights, {
        "adaptive_sampling": False,
        "sampling_type": "fixed_uniform_time",
        "distance_step_target_px": np.nan,
        "reciprocal_samples_per_cycle": 0,
        "highest_periodic_frequency_cycles_per_exposure": np.nan,
        "period_time_step_limit": np.nan,
        "preview_samples": 0,
        "preview_total_screen_motion_px": np.nan,
        "selected_max_accumulated_screen_step_px": np.nan,
        "selected_max_time_step": float(1.0 / (n_subframes - 1)),
        "gt_reference_subframe_index_zero_based": int(n_subframes // 2),
    }


def synthesize_blur_by_multisubframe_v2(
    image: np.ndarray,
    trajectory: dict[str, Any],
    n_subframes: int,
    extra_crop_margin: int,
    *,
    use_adaptive_sampling: bool = adaptive_subframe_sampling,
    sensor_parameters: sensor_model.SensorSampleParameters | None = None,
) -> tuple[Any, ...]:
    """以中心GT在线性光强空间加权积分，并计算公共真实有效区域。"""
    image_srgb = np.asarray(image, dtype=np.float64)
    image_linear = srgb_to_linear_uav(image_srgb)
    height, width = image_linear.shape[:2]
    # MATLAB 的 (w+1)/2 使用 1-based 坐标，换成 0-based 后是 (w-1)/2。
    cx, cy = (width - 1.0) / 2.0, (height - 1.0) / 2.0
    prepare_trajectory_for_image(trajectory, height, width)
    projection_model = str(trajectory["projection_model"])
    if use_adaptive_sampling:
        tau_list, temporal_weights, schedule_metadata = (
            build_adaptive_subframe_schedule(
                trajectory,
                height,
                width,
            )
        )
    else:
        tau_list, temporal_weights, schedule_metadata = (
            build_fixed_subframe_schedule(n_subframes)
        )
    actual_n_subframes = int(tau_list.size)
    gt_reference_subframe_index = int(
        schedule_metadata["gt_reference_subframe_index_zero_based"]
    )

    blur_accumulator = np.zeros_like(image_linear, dtype=np.float64)
    sensor_accumulator = (
        np.zeros_like(image_linear, dtype=np.float64)
        if sensor_parameters is not None
        else None
    )
    valid_source = np.ones((height, width), dtype=np.float64)
    if image_linear.ndim == 2:
        packed_linear_and_valid = np.dstack((image_linear, valid_source))
        color_channel_count = 1
    else:
        color_channel_count = int(image_linear.shape[2])
        packed_linear_and_valid = np.concatenate(
            (image_linear, valid_source[..., None]), axis=2
        )
    common_valid_coverage = np.ones((height, width), dtype=np.float64)
    dx_list = np.full(actual_n_subframes, np.nan, dtype=np.float64)
    dy_list = np.full(actual_n_subframes, np.nan, dtype=np.float64)
    theta_list = np.full(actual_n_subframes, np.nan, dtype=np.float64)
    scale_list = np.full(actual_n_subframes, np.nan, dtype=np.float64)
    path_progress_list = np.full(
        actual_n_subframes, np.nan, dtype=np.float64
    )
    native_path_parameter_list = np.full(
        actual_n_subframes, np.nan, dtype=np.float64
    )
    camera_axis_angle_deg_list = np.full(
        (actual_n_subframes, 3), np.nan, dtype=np.float64
    )
    homography_list = np.empty(
        (actual_n_subframes, 3, 3), dtype=np.float64
    )
    reference_attitude_parameter = np.zeros(3, dtype=np.float64)
    if projection_model == "pure_rotation_homography":
        _, reference_attitude_parameter = (
            evaluate_absolute_attitude_rotation(
                trajectory, GT_REFERENCE_TAU
            )
        )

    for index, (tau, temporal_weight) in enumerate(
        strict_zip(tau_list, temporal_weights)
    ):
        if projection_model == "similarity_2d":
            path_progress = evaluate_time_mapping(
                trajectory["time_mapping"], float(tau)
            )
            path_progress_list[index] = path_progress
            native_path_parameter_list[index] = native_path_parameter(
                trajectory["path"], path_progress
            )
            dx, dy, theta, scale_value = evaluate_trajectory_at_time(
                trajectory, float(tau)
            )
            dx_list[index] = dx
            dy_list[index] = dy
            theta_list[index] = theta
            scale_list[index] = scale_value
        elif projection_model == "pure_rotation_homography":
            _, attitude_parameter = evaluate_absolute_attitude_rotation(
                trajectory, float(tau)
            )
            camera_axis_angle_deg_list[index] = (
                attitude_parameter - reference_attitude_parameter
            )
        homography = evaluate_forward_homography_at_time(
            trajectory, float(tau), height, width
        )
        homography_list[index] = homography
        warped_packed = warp_frame_homography(
            packed_linear_and_valid, homography
        )
        if image_linear.ndim == 2:
            frame_linear = warped_packed[..., 0]
        else:
            frame_linear = warped_packed[..., :color_channel_count]
        valid_now = warped_packed[..., -1]

        # GT参考时刻显式采用输入图，避免恒等remap产生任何亚像素数值误差。
        if index == gt_reference_subframe_index:
            frame_linear = image_linear.copy()
            valid_now = valid_source
        blur_accumulator += float(temporal_weight) * frame_linear
        if sensor_accumulator is not None:
            assert sensor_parameters is not None
            sensor_frame_linear = (
                sensor_model.restore_saturated_highlights_linear(
                    frame_linear, sensor_parameters
                )
            )
            sensor_accumulator += (
                float(temporal_weight) * sensor_frame_linear
            )
        common_valid_coverage = np.minimum(
            common_valid_coverage, valid_now
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
    motion_summary = build_motion_summary_from_trajectory(
        height, width, tau_list, trajectory, cx, cy
    )
    motion_field = {
        "mu": motion_summary["end_mu"],
        "mv": motion_summary["end_mv"],
    }

    common_valid_mask = (
        common_valid_coverage >= valid_coverage_threshold
    )
    crop_top, crop_bottom, crop_left, crop_right = (
        largest_valid_rectangle(
            common_valid_mask,
            safety_margin=int(extra_crop_margin),
        )
    )
    motion_summary["common_valid_mask"] = common_valid_mask.astype(
        np.uint8
    )
    motion_summary["common_valid_coverage"] = common_valid_coverage
    # 旧字段仅用于兼容查看器；新裁剪使用下面四个非对称矩形坐标。
    crop_bound_h = max(crop_top, height - crop_bottom)
    crop_bound_w = max(crop_left, width - crop_right)
    end_magnitude = np.sqrt(motion_field["mu"] ** 2 + motion_field["mv"] ** 2)
    parameters: dict[str, Any] = {
        "case_name": trajectory["case_name"],
        "case_seed": trajectory["case_seed"],
        "projection_model": projection_model,
        "component_case_names": tuple(
            trajectory.get("component_case_names", ())
        ),
        "component_seeds": np.asarray(
            trajectory.get("component_seeds", ()), dtype=np.int64
        ),
        "component_count": int(
            len(trajectory.get("components", ()))
        ),
        "composition_order": str(
            trajectory.get("composition_order", "single_component")
        ),
        "formation_model": (
            "B_srgb=g(sum_t(w_t*W(g_inverse(I_srgb),H_t)))"
        ),
        "gt_reference": "center",
        "gt_reference_tau": float(GT_REFERENCE_TAU),
        "N_subframes": actual_n_subframes,
        "N_subframes_actual": actual_n_subframes,
        "N_subframes_fixed_fallback": int(n_subframes),
        "temporal_weights": temporal_weights[:, None],
        "temporal_weight_sum": float(temporal_weights.sum()),
        **schedule_metadata,
        "crop_margin": int(extra_crop_margin),
        "crop_bound_h": crop_bound_h,
        "crop_bound_w": crop_bound_w,
        "crop_rectangle_top_bottom_left_right": np.asarray(
            [crop_top, crop_bottom, crop_left, crop_right],
            dtype=np.int32,
        ),
        "crop_rectangle_coordinate_convention": (
            "zero_based_half_open_[top,bottom,left,right]"
        ),
        "common_valid_threshold": float(valid_coverage_threshold),
        "common_valid_area_before_crop": int(common_valid_mask.sum()),
        "full_image_area": int(height * width),
        "cropped_height": int(crop_bottom - crop_top),
        "cropped_width": int(crop_right - crop_left),
        "boundary_policy": (
            "constant_zero_during_warp_then_crop_all_subframe_valid_intersection"
        ),
        "tau_list": tau_list,
        "dx_list": dx_list[:, None],
        "dy_list": dy_list[:, None],
        "theta_list": theta_list[:, None],
        "scale_list": scale_list[:, None],
        "homography_source_gt_to_subframe_list": homography_list,
        "camera_axis_angle_xyz_deg_relative_list": (
            camera_axis_angle_deg_list
        ),
        "camera_intrinsics_K": np.asarray(
            trajectory.get(
                "camera_intrinsics_K",
                np.full((3, 3), np.nan, dtype=np.float64),
            ),
            dtype=np.float64,
        ),
        "virtual_camera_horizontal_fov_deg": float(
            trajectory.get(
                "virtual_camera_horizontal_fov_deg", np.nan
            )
        ),
        "attitude_type": str(
            trajectory.get("attitude", {}).get("type", "none")
        ),
        "attitude_target_span_px": float(
            trajectory.get("attitude", {}).get(
                "target_span_px", np.nan
            )
        ),
        "attitude_achieved_span_px": float(
            trajectory.get("attitude", {}).get(
                "achieved_span_px", np.nan
            )
        ),
        "path_type": str(
            trajectory.get("path", {}).get(
                "type", "attitude_rotation"
            )
        ),
        "time_mapping_type": str(
            trajectory.get("time_mapping", {}).get(
                "type",
                trajectory.get("attitude", {})
                .get("time_mapping", {})
                .get("type", "harmonic_physical_time"),
            )
        ),
        "rotation_type": str(
            trajectory.get("rotation", {}).get(
                "type",
                trajectory.get("attitude", {}).get("type", "none"),
            )
        ),
        "scale_type": str(
            trajectory.get("scale", {}).get("type", "not_used")
        ),
        "arc_length_parameterized": bool(
            trajectory.get("path", {}).get(
                "arc_length_parameterized", False
            )
        ),
        "path_progress_s_list": path_progress_list[:, None],
        "native_path_parameter_u_list": (
            native_path_parameter_list[:, None]
        ),
        "radiometric_model": "linear_srgb_exposure_integration",
        "input_transfer_function": "IEC_61966_2_1_sRGB",
        "temporal_integration_space": "linear_rgb",
        "output_transfer_function": "IEC_61966_2_1_sRGB",
        "start_end_mean_mag": float(end_magnitude.mean()),
        "start_end_max_mag": float(end_magnitude.max()),
        "path_len_mean": float(motion_summary["path_len"].mean()),
        "path_len_max": float(motion_summary["path_len"].max()),
        "span_mag_mean": float(motion_summary["span_mag"].mean()),
        "span_mag_max": float(motion_summary["span_mag"].max()),
    }
    if sensor_blur_image is not None:
        return (
            blur_image,
            sensor_blur_image,
            motion_field,
            parameters,
            motion_summary,
        )
    return blur_image, motion_field, parameters, motion_summary


def build_mfmap_uav(
    motion_field: dict[str, np.ndarray], label_type: str
) -> np.ndarray:
    mu = np.asarray(motion_field["mu"], dtype=np.float64)
    mv = np.asarray(motion_field["mv"], dtype=np.float64)
    if label_type == "uv_basic":
        return np.stack([mu, mv], axis=-1)
    if label_type == "uv_half":
        output_u, output_v = mu.copy(), mv.copy()
        negative = output_u < 0
        output_u[negative] *= -1
        output_v[negative] *= -1
        return np.stack([output_u, output_v], axis=-1)
    raise ValueError(f"Unsupported q_type: {label_type}")


def largest_valid_rectangle(
    valid_mask: np.ndarray,
    *,
    safety_margin: int = 0,
) -> tuple[int, int, int, int]:
    """返回二值有效区域内面积最大的轴对齐矩形。

    返回值采用零基、半开区间 ``(top, bottom, left, right)``。若多个矩形
    面积相同，优先选择中心更接近输入图像中心的矩形。
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 2 or not valid_mask.any():
        raise ValueError("公共有效区域为空，无法生成无补边的训练样本")
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
                if rectangle_height <= 0:
                    continue
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
            "公共有效矩形在应用 crop_margin 后为空；请减小运动或crop_margin"
        )
    if not valid_mask[top:bottom, left:right].all():
        raise RuntimeError("内部错误：计算出的裁剪矩形包含无效边界像素")
    return top, bottom, left, right


def _crop_rectangle(
    array: np.ndarray,
    rectangle: tuple[int, int, int, int],
) -> np.ndarray:
    top, bottom, left, right = rectangle
    return array[top:bottom, left:right, ...]


def crop_valid_rectangle_uav(
    blur_image: np.ndarray,
    mfmap: np.ndarray,
    gt_image: np.ndarray,
    rectangle: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用完全相同的公共有效矩形裁剪模糊图、运动场和GT。"""
    return (
        _crop_rectangle(blur_image, rectangle),
        _crop_rectangle(mfmap, rectangle),
        _crop_rectangle(gt_image, rectangle),
    )


def crop_motion_summary_rectangle(
    motion_summary: dict[str, np.ndarray],
    rectangle: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    """裁剪所有与输入图同高宽的二维运动统计和有效性图。"""
    return {
        name: _crop_rectangle(value, rectangle)
        if isinstance(value, np.ndarray) and value.ndim == 2
        else value
        for name, value in motion_summary.items()
    }


def _crop_array(array: np.ndarray, bound_h: int, bound_w: int) -> np.ndarray:
    height, width = array.shape[:2]
    if bound_h * 2 >= height or bound_w * 2 >= width:
        return array
    row_slice = slice(bound_h, height - bound_h if bound_h else height)
    col_slice = slice(bound_w, width - bound_w if bound_w else width)
    return array[row_slice, col_slice, ...]


def crop_boundary_uav(
    blur_image: np.ndarray,
    mfmap: np.ndarray,
    gt_image: np.ndarray,
    bound_h: int,
    bound_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = blur_image.shape[:2]
    if bound_h * 2 >= height or bound_w * 2 >= width:
        print("Warning: boundary crop is too large; skip this sample.")
        return blur_image, mfmap, gt_image
    return (
        _crop_array(blur_image, bound_h, bound_w),
        _crop_array(mfmap, bound_h, bound_w),
        _crop_array(gt_image, bound_h, bound_w),
    )


def crop_motion_summary(
    motion_summary: dict[str, np.ndarray], bound_h: int, bound_w: int
) -> dict[str, np.ndarray]:
    return {
        name: _crop_array(value, bound_h, bound_w)
        if isinstance(value, np.ndarray) and value.ndim == 2
        else value
        for name, value in motion_summary.items()
    }


def clip01_uav(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0.0, 1.0)


def show_debug_result(
    gt: np.ndarray,
    blur: np.ndarray,
    mfmap: np.ndarray,
    motion_summary: dict[str, np.ndarray],
    case_name: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gt, cmap="gray" if gt.ndim == 2 else None)
    axes[0].set_title("GT")
    axes[1].imshow(blur, cmap="gray" if blur.ndim == 2 else None)
    axes[1].set_title(case_name)
    image1 = axes[2].imshow(np.linalg.norm(mfmap[..., :2], axis=-1))
    axes[2].set_title("start-end mag")
    figure.colorbar(image1, ax=axes[2])
    image2 = axes[3].imshow(motion_summary["path_len"])
    axes[3].set_title("path length")
    figure.colorbar(image2, ax=axes[3])
    for axis in axes:
        axis.axis("off")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=image_path)
    parser.add_argument("--output-root", type=Path, default=output_path)
    parser.add_argument("--splits", default=",".join(splits))
    parser.add_argument("--cases", default=",".join(blur_cases))
    parser.add_argument("--seed", type=int, default=base_seed)
    parser.add_argument(
        "--motion-strength",
        type=float,
        default=1.0,
        help=(
            "固定对比倍率：同时缩放所有运动幅度；1.0为中等基础轨迹，"
            "1.8为强模糊。在线训练仍使用各参数独立随机倍率"
        ),
    )
    parser.add_argument(
        "--n-subframes",
        type=int,
        default=N_subframes,
        help="仅在 --no-adaptive-subframe-sampling 时使用的固定奇数帧数",
    )
    parser.add_argument(
        "--adaptive-subframe-sampling",
        action=BooleanOptionalAction,
        default=adaptive_subframe_sampling,
        help="默认按1.5像素距离及往复周期约束自动选择子帧",
    )
    parser.add_argument("--crop-margin", type=int, default=crop_margin)
    parser.add_argument("--q-type", choices=("uv_basic", "uv_half"), default=q_type)
    parser.add_argument("--noise-sigma-255", type=float, default=noise_sigma_255)
    parser.add_argument(
        "--sensor-variant",
        action=BooleanOptionalAction,
        default=generate_sensor_variant,
        help="额外保存带高光饱和和曝光后传感器噪声的模糊图",
    )
    parser.add_argument(
        "--sensor-saturation-highlights",
        action=BooleanOptionalAction,
        default=sensor_saturation_highlights,
        help="在各线性RGB子帧中恢复一部分饱和高光能量",
    )
    parser.add_argument(
        "--sensor-poisson-gaussian-noise",
        action=BooleanOptionalAction,
        default=sensor_poisson_gaussian_noise,
        help="曝光积分完成后添加一次Poisson-Gaussian传感器噪声",
    )
    parser.add_argument(
        "--crop-boundary", action=BooleanOptionalAction, default=is_crop_boundary
    )
    parser.add_argument("--show", action=BooleanOptionalAction, default=is_show)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    run_generation(
        arguments.input_root,
        arguments.output_root,
        tuple(item.strip() for item in arguments.splits.split(",") if item.strip()),
        tuple(item.strip() for item in arguments.cases.split(",") if item.strip()),
        seed=arguments.seed,
        motion_strength=arguments.motion_strength,
        n_subframes=arguments.n_subframes,
        use_adaptive_sampling=arguments.adaptive_subframe_sampling,
        crop_boundary=arguments.crop_boundary,
        extra_crop_margin=arguments.crop_margin,
        label_type=arguments.q_type,
        noise_sigma=arguments.noise_sigma_255,
        sensor_variant=arguments.sensor_variant,
        saturation_highlights=arguments.sensor_saturation_highlights,
        poisson_gaussian_noise=arguments.sensor_poisson_gaussian_noise,
        show=arguments.show,
    )


if __name__ == "__main__":
    main()

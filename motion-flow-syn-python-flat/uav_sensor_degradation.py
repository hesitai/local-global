"""UAV 合成模糊共用的轻量传感器退化模型。

本模块只负责两件事：

1. 在每个完整线性 RGB 子帧中，用软饱和 mask 恢复 JPEG 截断掉的一部分
   高光能量，使高光能够随运动轨迹参与曝光积分；
2. 在所有子帧已经积分完成后，仅添加一次 Poisson shot noise 和 Gaussian
   read noise，再使用标准 sRGB 传递函数编码输出。

它不是经过真实相机标定的 RAW/ISP 模型，因此元数据明确使用
``linear_rgb_pseudo_sensor``；Bayer、白平衡和去马赛克不在本阶段模拟。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SensorDegradationConfig:
    """一组生成任务共用的传感器参数范围。"""

    saturation_enabled: bool = True
    saturation_threshold_low: float = 0.95
    saturation_threshold_high: float = 0.99
    saturation_gain_range: tuple[float, float] = (0.25, 1.25)
    poisson_gaussian_enabled: bool = True
    shot_noise_scale_range: tuple[float, float] = (1.0e-5, 1.0e-4)
    read_noise_sigma_range: tuple[float, float] = (3.0e-6, 5.0e-5)


@dataclass(frozen=True)
class SensorSampleParameters:
    """某一张传感器退化图实际采用的可复现参数。"""

    saturation_enabled: bool
    saturation_gain: float
    saturation_threshold_low: float
    saturation_threshold_high: float
    poisson_gaussian_enabled: bool
    shot_noise_scale: float
    read_noise_sigma: float
    sensor_seed: int
    noise_domain: str = "linear_rgb_pseudo_sensor"
    saturation_domain: str = "linear_rgb"
    noise_application_count: int = 1
    sensor_model_version: str = "uav_sat_poisson_gaussian_v1"


def validate_sensor_config(config: SensorDegradationConfig) -> None:
    """检查配置范围，避免生成无意义或数值不稳定的退化。"""

    if not (
        0.0
        <= config.saturation_threshold_low
        < config.saturation_threshold_high
        <= 1.0
    ):
        raise ValueError(
            "饱和阈值必须满足 "
            "0 <= saturation_threshold_low < saturation_threshold_high <= 1"
        )
    gain_min, gain_max = config.saturation_gain_range
    if gain_min < 0.0 or gain_max < gain_min:
        raise ValueError("saturation_gain_range 必须满足 0 <= min <= max")
    shot_min, shot_max = config.shot_noise_scale_range
    if shot_min <= 0.0 or shot_max < shot_min:
        raise ValueError("shot_noise_scale_range 必须满足 0 < min <= max")
    read_min, read_max = config.read_noise_sigma_range
    if read_min < 0.0 or read_max < read_min:
        raise ValueError("read_noise_sigma_range 必须满足 0 <= min <= max")


def _sample_log_uniform(
    rng: np.random.Generator, minimum: float, maximum: float
) -> float:
    if minimum == maximum:
        return float(minimum)
    return float(np.exp(rng.uniform(np.log(minimum), np.log(maximum))))


def sample_sensor_parameters(
    config: SensorDegradationConfig,
    sensor_seed: int,
) -> SensorSampleParameters:
    """从配置范围中为一个输出样本确定参数；同一 seed 完全可复现。"""

    validate_sensor_config(config)
    rng = np.random.default_rng(int(sensor_seed))
    gain_min, gain_max = config.saturation_gain_range
    saturation_gain = (
        float(rng.uniform(gain_min, gain_max))
        if config.saturation_enabled
        else 0.0
    )
    shot_min, shot_max = config.shot_noise_scale_range
    read_min, read_max = config.read_noise_sigma_range
    shot_noise_scale = (
        _sample_log_uniform(rng, shot_min, shot_max)
        if config.poisson_gaussian_enabled
        else 0.0
    )
    read_noise_sigma = (
        _sample_log_uniform(rng, max(read_min, np.finfo(float).tiny), read_max)
        if config.poisson_gaussian_enabled and read_max > 0.0
        else 0.0
    )
    return SensorSampleParameters(
        saturation_enabled=bool(config.saturation_enabled),
        saturation_gain=saturation_gain,
        saturation_threshold_low=float(config.saturation_threshold_low),
        saturation_threshold_high=float(config.saturation_threshold_high),
        poisson_gaussian_enabled=bool(config.poisson_gaussian_enabled),
        shot_noise_scale=shot_noise_scale,
        read_noise_sigma=read_noise_sigma,
        sensor_seed=int(sensor_seed),
    )


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def restore_saturated_highlights_linear(
    frame_linear: np.ndarray,
    parameters: SensorSampleParameters,
) -> np.ndarray:
    """恢复一部分饱和能量，但不裁剪大于 1 的临时线性光强。

    输入必须是一张已经完成背景和所有目标合成的瞬时清晰子帧。软 mask
    由各像素最大颜色通道构造，所有通道使用同一个增益，避免改变高光色相。
    """

    frame = np.asarray(frame_linear, dtype=np.float64)
    if not parameters.saturation_enabled or parameters.saturation_gain <= 0.0:
        return frame

    if frame.ndim == 2:
        peak = frame
    elif frame.ndim == 3:
        peak = np.max(frame, axis=2)
    else:
        raise ValueError("frame_linear 必须是 HxW 或 HxWxC 数组")

    threshold_span = (
        parameters.saturation_threshold_high
        - parameters.saturation_threshold_low
    )
    normalized = (
        peak - parameters.saturation_threshold_low
    ) / threshold_span
    highlight_alpha = _smoothstep01(normalized)
    if frame.ndim == 3:
        highlight_alpha = highlight_alpha[..., None]
    return frame * (1.0 + parameters.saturation_gain * highlight_alpha)


def add_poisson_gaussian_noise_after_exposure(
    exposure_linear: np.ndarray,
    parameters: SensorSampleParameters,
) -> np.ndarray:
    """在一次完整曝光积分后添加一次 shot/read noise。"""

    signal = np.maximum(np.asarray(exposure_linear, dtype=np.float64), 0.0)
    if not parameters.poisson_gaussian_enabled:
        return np.clip(signal, 0.0, 1.0)

    # 与参数抽样使用不同但可由 sensor_seed 唯一推导的随机序列。
    noise_seed = int(parameters.sensor_seed) ^ 0x5EED5EED
    rng = np.random.default_rng(noise_seed)
    shot_scale = float(parameters.shot_noise_scale)
    noisy = rng.poisson(signal / shot_scale).astype(np.float64) * shot_scale
    if parameters.read_noise_sigma > 0.0:
        noisy += rng.normal(
            0.0,
            float(parameters.read_noise_sigma),
            size=signal.shape,
        )
    return np.clip(noisy, 0.0, 1.0)


def linear_to_srgb(image_linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 标准 sRGB 正向传递函数。"""

    linear = np.clip(np.asarray(image_linear, dtype=np.float64), 0.0, 1.0)
    return np.clip(
        np.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * linear ** (1.0 / 2.4) - 0.055,
        ),
        0.0,
        1.0,
    )


def finalize_sensor_variant_srgb(
    sensor_exposure_linear: np.ndarray,
    parameters: SensorSampleParameters,
) -> np.ndarray:
    """将高光累加器变成最终带传感器退化的 sRGB 图。"""

    noisy_linear = add_poisson_gaussian_noise_after_exposure(
        sensor_exposure_linear, parameters
    )
    return linear_to_srgb(noisy_linear)


def sensor_metadata_dict(
    parameters: SensorSampleParameters,
) -> dict[str, object]:
    """返回既能写入 NPZ、也能嵌入 MAT 元数据的普通字典。"""

    return asdict(parameters)


def save_sensor_metadata_npz(
    path: str | Path,
    parameters: SensorSampleParameters,
) -> None:
    """单独保存一张 ``*_blurryimg_sensor.png`` 的退化参数。"""

    metadata = sensor_metadata_dict(parameters)
    np.savez_compressed(Path(path), **metadata)


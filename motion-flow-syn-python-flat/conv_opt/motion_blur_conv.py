"""逐式改写 ``motionBlurConv.cc`` 和 ``motionBlurConv_mirror.cc``。

原 ``.mexw64`` 是 Windows MATLAB 二进制，Python 不能直接调用。本文件保留
C++ 中的线核生成、归一化和边界夹取规则，并使用 Numba 加速。
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

EPS = 2.2204e-16
LINE_WIDTH = 1.0


@njit(cache=True, inline="always")
def _kernel_weight(k: int, l: int, half: float, cos_phi: float, sin_phi: float) -> float:
    """对应 C++ 中偏移 ``(k,l)`` 的线核权重计算。"""
    dist_to_line = l * cos_phi + k * sin_phi
    dist_to_center = math.sqrt(float(l * l + k * k))

    if abs(dist_to_line) <= LINE_WIDTH and dist_to_center >= half:
        # 原 C++ 在 cos(phi)==0 时会除零；这里只做数值保护。
        cos_safe = cos_phi
        if abs(cos_safe) < 1e-12:
            cos_safe = 1e-12 if cos_safe >= 0.0 else -1e-12
        x_to_last_pixel = half - abs((k + dist_to_line * sin_phi) / cos_safe)
        dist_to_line = math.sqrt(dist_to_line * dist_to_line + x_to_last_pixel * x_to_last_pixel)

    return max(LINE_WIDTH + EPS - abs(dist_to_line), 0.0)


@njit(cache=True, parallel=True)
def _motion_blur_conv_numba(image: np.ndarray, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    h, w, channels = image.shape
    output = np.zeros((h, w, channels), dtype=np.float64)

    # C++ 中 i 是 MATLAB 第一维（行），j 是第二维（列）。
    for row in prange(h):
        for col in range(w):
            mag = magnitude[row, col]
            ori = orientation[row, col]
            half = (mag - 1.0) / 2.0
            phi = (ori % 180.0) / 180.0 * math.pi
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            x_sign = -1.0 if cos_phi < 0.0 else 1.0

            sx = int(math.floor(abs(half * cos_phi + LINE_WIDTH * x_sign - mag * EPS)))
            sy = int(math.floor(abs(half * sin_phi + LINE_WIDTH - mag * EPS)))

            kernel_sum = 0.0
            accum = np.zeros(channels, dtype=np.float64)
            for l in range(-sy, sy + 1):
                for k in range(-sx, sx + 1):
                    weight = _kernel_weight(k, l, half, cos_phi, sin_phi)
                    if weight == 0.0:
                        continue

                    # 与 C++ 一致：越界坐标夹到最近边界像素。
                    source_row = min(max(row + k, 0), h - 1)
                    source_col = min(max(col + l, 0), w - 1)
                    for channel in range(channels):
                        accum[channel] += weight * image[source_row, source_col, channel]
                    kernel_sum += weight

            if kernel_sum > 0.0:
                for channel in range(channels):
                    output[row, col, channel] = accum[channel] / kernel_sum
    return output


@njit(cache=True)
def _motion_blur_conv_mirror_numba(image: np.ndarray, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """对应 mirror C++ 的 source-driven scatter 版本。"""
    h, w, channels = image.shape
    output = np.zeros((h, w, channels), dtype=np.float64)

    for row in range(h):
        for col in range(w):
            mag = magnitude[row, col]
            ori = orientation[row, col]
            half = (mag - 1.0) / 2.0
            phi = (ori % 180.0) / 180.0 * math.pi
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            x_sign = -1.0 if cos_phi < 0.0 else 1.0
            sx = int(math.floor(abs(half * cos_phi + LINE_WIDTH * x_sign - mag * EPS)))
            sy = int(math.floor(abs(half * sin_phi + LINE_WIDTH - mag * EPS)))

            kernel_sum = 0.0
            for l in range(-sy, sy + 1):
                for k in range(-sx, sx + 1):
                    kernel_sum += _kernel_weight(k, l, half, cos_phi, sin_phi)

            denominator = kernel_sum if kernel_sum > 0.0 else 1.0
            for l in range(-sy, sy + 1):
                for k in range(-sx, sx + 1):
                    weight = _kernel_weight(k, l, half, cos_phi, sin_phi)
                    if weight == 0.0:
                        continue
                    target_row = min(max(row + k, 0), h - 1)
                    target_col = min(max(col + l, 0), w - 1)
                    for channel in range(channels):
                        output[target_row, target_col, channel] += (
                            weight * image[row, col, channel] / denominator
                        )
    return output


def _validate_inputs(
    image: np.ndarray,
    magnitude: np.ndarray,
    orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    image_array = np.asarray(image, dtype=np.float64)
    was_grayscale = image_array.ndim == 2
    if was_grayscale:
        image_array = image_array[..., None]
    if image_array.ndim != 3:
        raise ValueError("image 必须是 HxW 或 HxWxC 数组。")

    magnitude_array = np.asarray(magnitude, dtype=np.float64)
    orientation_array = np.asarray(orientation, dtype=np.float64)
    if magnitude_array.shape != image_array.shape[:2] or orientation_array.shape != image_array.shape[:2]:
        raise ValueError("magnitude 和 orientation 必须与图像前两维一致。")
    if np.any(magnitude_array < 0.0):
        raise ValueError("magnitude 不能包含负数。")
    if not (
        np.isfinite(image_array).all()
        and np.isfinite(magnitude_array).all()
        and np.isfinite(orientation_array).all()
    ):
        raise ValueError("输入不能包含 NaN 或 Inf。")

    return (
        np.ascontiguousarray(image_array),
        np.ascontiguousarray(magnitude_array),
        np.ascontiguousarray(orientation_array),
        was_grayscale,
    )


def motion_blur_conv(image: np.ndarray, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """替代 ``motionBlurConv.mexw64``；首次调用会触发 Numba 编译。"""
    image_array, magnitude_array, orientation_array, was_grayscale = _validate_inputs(
        image, magnitude, orientation
    )
    result = _motion_blur_conv_numba(image_array, magnitude_array, orientation_array)
    return result[..., 0] if was_grayscale else result


def motion_blur_conv_mirror(image: np.ndarray, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """替代 ``motionBlurConv_mirror.mexw64``。"""
    image_array, magnitude_array, orientation_array, was_grayscale = _validate_inputs(
        image, magnitude, orientation
    )
    result = _motion_blur_conv_mirror_numba(image_array, magnitude_array, orientation_array)
    return result[..., 0] if was_grayscale else result


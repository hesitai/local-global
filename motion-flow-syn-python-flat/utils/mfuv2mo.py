"""对应原 ``utils/mfuv2mo.m``。"""

from __future__ import annotations

import numpy as np


def mfuv2mo(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把两个运动分量转换为线核长度和 [0,180) 度方向。"""
    u_array = np.asarray(u, dtype=np.float64)
    v_array = np.asarray(v, dtype=np.float64)
    if u_array.shape != v_array.shape:
        raise ValueError("u 和 v 的尺寸必须一致。")

    magnitude = np.sqrt(u_array * u_array + v_array * v_array)
    u_safe = u_array.copy()
    u_safe[u_safe == 0.0] = 1e-16
    orientation = np.degrees(np.arctan(v_array / u_safe))
    orientation[orientation < 0.0] += 180.0
    return magnitude, orientation


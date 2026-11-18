"""对应原 MATLAB ``gen_blurry_img.m``。"""

from __future__ import annotations

import numpy as np

from conv_opt import motion_blur_conv
from gen_mf import MotionField


def gen_blurry_img(image: np.ndarray, motion_field: MotionField) -> np.ndarray:
    """使用 motion field 的每像素线核生成单通道或多通道模糊图。"""
    if motion_field.mag is None or motion_field.ori is None:
        raise ValueError("motion_field.mag 和 motion_field.ori 必须先计算。")
    return motion_blur_conv(np.asarray(image, dtype=np.float64), motion_field.mag, motion_field.ori)


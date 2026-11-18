"""对应 ``flowToColor.m`` 与 ``computeColor.m`` 的 Middlebury 色轮显示。"""

from __future__ import annotations

import numpy as np


def _make_color_wheel() -> np.ndarray:
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    wheel = np.zeros((ry + yg + gc + cb + bm + mr, 3), dtype=np.float64)
    col = 0
    wheel[col : col + ry, 0] = 255
    wheel[col : col + ry, 1] = np.floor(255 * np.arange(ry) / ry)
    col += ry
    wheel[col : col + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    wheel[col : col + yg, 1] = 255
    col += yg
    wheel[col : col + gc, 1] = 255
    wheel[col : col + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    col += gc
    wheel[col : col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    wheel[col : col + cb, 2] = 255
    col += cb
    wheel[col : col + bm, 2] = 255
    wheel[col : col + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    col += bm
    wheel[col : col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    wheel[col : col + mr, 0] = 255
    return wheel


def flow_to_color(flow: np.ndarray, max_motion: float | None = None) -> np.ndarray:
    """将 HxWx2 的 ``(u,v)`` flow 转为 RGB uint8 图。"""
    flow_array = np.asarray(flow, dtype=np.float64)
    if flow_array.ndim != 3 or flow_array.shape[2] != 2:
        raise ValueError("flow 必须为 HxWx2。")

    u = flow_array[..., 0].copy()
    v = flow_array[..., 1].copy()
    invalid = ~np.isfinite(u) | ~np.isfinite(v)
    u[invalid] = 0.0
    v[invalid] = 0.0
    radius = np.sqrt(u * u + v * v)
    maximum = float(radius.max()) if max_motion is None else float(max_motion)
    maximum = max(maximum, np.finfo(np.float64).eps)
    u /= maximum
    v /= maximum

    wheel = _make_color_wheel()
    angle = np.arctan2(-v, -u) / np.pi
    fk = (angle + 1.0) / 2.0 * (wheel.shape[0] - 1)
    k0 = np.floor(fk).astype(np.int64)
    k1 = (k0 + 1) % wheel.shape[0]
    fraction = fk - k0
    normalized_radius = np.sqrt(u * u + v * v)
    output = np.empty((*u.shape, 3), dtype=np.uint8)

    for channel in range(3):
        color0 = wheel[k0, channel] / 255.0
        color1 = wheel[k1, channel] / 255.0
        color = (1.0 - fraction) * color0 + fraction * color1
        inside = normalized_radius <= 1.0
        color[inside] = 1.0 - normalized_radius[inside] * (1.0 - color[inside])
        color[~inside] *= 0.75
        color[invalid] = 0.0
        output[..., channel] = np.floor(255.0 * color).astype(np.uint8)
    return output


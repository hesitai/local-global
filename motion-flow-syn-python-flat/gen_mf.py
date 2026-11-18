"""对应原 MATLAB ``gen_mf.m``。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class MotionField:
    """像素级运动场；mu 向右为正，mv 向下为正。"""

    mu: np.ndarray
    mv: np.ndarray
    mag: np.ndarray | None = None
    ori: np.ndarray | None = None

    def to_mat_dict(self) -> dict[str, np.ndarray]:
        result = {"mu": self.mu, "mv": self.mv}
        if self.mag is not None:
            result["mag"] = self.mag
        if self.ori is not None:
            result["ori"] = self.ori
        return result


@dataclass
class MotionParameters:
    tx: float
    ty: float
    tx_acc: float
    ty_acc: float
    tz: float
    cen_z: tuple[float, float]
    rot_z: float

    def to_mat_dict(self) -> dict[str, Any]:
        return {
            "tx": self.tx,
            "ty": self.ty,
            "tx_acc": self.tx_acc,
            "ty_acc": self.ty_acc,
            "tz": self.tz,
            "cen_z": np.asarray(self.cen_z, dtype=np.float64),
            "rot_z": self.rot_z,
        }


def _parameter_value(parameters: MotionParameters | Mapping[str, Any], name: str) -> Any:
    if isinstance(parameters, Mapping):
        return parameters[name]
    return getattr(parameters, name)


def gen_mf(h: int, w: int, parameters: MotionParameters | Mapping[str, Any]) -> MotionField:
    """逐式复现原 ``gen_mf.m`` 的平移、旋转和沿光轴运动场。"""
    if h <= 0 or w <= 0:
        raise ValueError("h 和 w 必须为正整数。")

    tx = float(_parameter_value(parameters, "tx"))
    ty = float(_parameter_value(parameters, "ty"))
    tx_acc = float(_parameter_value(parameters, "tx_acc"))
    ty_acc = float(_parameter_value(parameters, "ty_acc"))
    tz = float(_parameter_value(parameters, "tz"))
    center_row, center_col = _parameter_value(parameters, "cen_z")
    rot_z = float(_parameter_value(parameters, "rot_z"))

    mu = np.zeros((h, w), dtype=np.float64)
    mv = np.zeros((h, w), dtype=np.float64)

    # x trans：tx_acc 控制的是空间梯度，不是时间加速度。
    left_tx = tx * (1.0 - tx_acc / 2.0)
    tx_line = np.arange(1, w + 1, dtype=np.float64) * (tx_acc * tx / w) + left_tx
    mu += np.broadcast_to(tx_line[None, :], (h, w))

    # y trans。
    bottom_ty = ty * (1.0 - ty_acc / 2.0)
    ty_line = np.arange(h, 0, -1, dtype=np.float64) * (ty_acc * ty / h) + bottom_ty
    mv += np.broadcast_to(ty_line[:, None], (h, w))

    # MATLAB 1-based 坐标。
    x, y = np.meshgrid(
        np.arange(1, w + 1, dtype=np.float64),
        np.arange(1, h + 1, dtype=np.float64),
    )
    radius = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

    # z rotation。
    rotation_length = np.tan(rot_z) * radius
    tmp_h = (1.0 + h - y) - center_row
    tmp_w = x - center_col
    alpha = np.arctan2(tmp_h, tmp_w)
    alpha_1 = alpha - np.pi / 2.0
    mu += np.sin(alpha_1) * rotation_length
    mv += np.cos(alpha_1) * rotation_length

    # z trans (forward)。
    radial_length = tz * radius**1.5
    mu += radial_length * np.sin(alpha)
    mv += radial_length * np.cos(alpha)
    return MotionField(mu=mu, mv=mv)


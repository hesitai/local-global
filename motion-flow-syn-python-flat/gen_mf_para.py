"""对应原 MATLAB ``gen_mf_para.m``。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gen_mf import MotionField, MotionParameters, gen_mf


@dataclass
class MotionOptions:
    tx_max: float = 27.0
    tx_acc_max: float = 0.3
    ty_max: float = 27.0
    ty_acc_max: float = 0.3
    tz_max: float = 5e-3
    cen_z_shift_max: float = 15.0
    rot_z_max: float = np.pi / 70.0
    isblur: bool = True


def gen_mf_para(
    h: int,
    w: int,
    options: MotionOptions,
    rng: np.random.Generator | None = None,
) -> tuple[MotionField, MotionParameters | None]:
    """随机生成运动参数。

    原 MATLAB 函数每次调用 ``rng('shuffle')``。Python 默认同样使用系统熵；若调用者
    传入 ``rng``，则可获得可复现实验，这是此处唯一有意增加的接口。
    """
    if not options.isblur:
        zeros = np.zeros((h, w), dtype=np.float64)
        return MotionField(zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()), None

    generator = np.random.default_rng() if rng is None else rng
    parameters = MotionParameters(
        tx=(2.0 * generator.random() - 1.0) * options.tx_max,
        ty=(2.0 * generator.random() - 1.0) * options.ty_max,
        tx_acc=(2.0 * generator.random() - 1.0) * options.tx_acc_max,
        ty_acc=(2.0 * generator.random() - 1.0) * options.ty_acc_max,
        tz=(2.0 * generator.random() - 1.0) * options.tz_max,
        cen_z=(
            h / 2.0 + (2.0 * generator.random() - 1.0) * options.cen_z_shift_max,
            w / 2.0 + (2.0 * generator.random() - 1.0) * options.cen_z_shift_max,
        ),
        rot_z=(2.0 * generator.random() - 1.0) * options.rot_z_max,
    )
    return gen_mf(h, w, parameters), parameters


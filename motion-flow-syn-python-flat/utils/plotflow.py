"""对应原 ``utils/plotflow.m`` 的常用显示模式。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plotflow(flow: np.ndarray, kind: str = "vector", axis=None):
    flow = np.asarray(flow, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow 必须是 HxWx2")
    axis = plt.gca() if axis is None else axis
    u, v = flow[..., 0], flow[..., 1]
    kind = kind.lower()
    if kind in {"quiver", "vector"}:
        step = max(1, int(np.ceil(max(flow.shape[:2]) / 40)))
        y, x = np.mgrid[0 : flow.shape[0] : step, 0 : flow.shape[1] : step]
        axis.quiver(x, y, u[::step, ::step], v[::step, ::step], color="k", angles="xy", scale_units="xy", scale=1)
        axis.invert_yaxis(); axis.set_aspect("equal")
    elif kind == "rgb":
        blue = normalize(u); green = normalize(v); red = np.zeros_like(blue)
        image = np.stack([red, green, blue], axis=-1)
        image[np.sqrt(u**2 + v**2) < 0.1] = 1.0
        axis.imshow(image); axis.axis("off")
    elif kind == "hsv":
        import matplotlib.colors as colors
        angle = (np.arctan2(v, u) + np.pi) / (2 * np.pi)
        magnitude = np.sqrt(u**2 + v**2)
        magnitude /= max(float(magnitude.max()), np.finfo(float).eps)
        axis.imshow(colors.hsv_to_rgb(np.stack([angle, np.ones_like(angle), magnitude], axis=-1))); axis.axis("off")
    elif kind in {"bw", "bwscale"}:
        image = np.concatenate([u, v], axis=1)
        maximum = max(float(np.max(np.abs(image))), np.finfo(float).eps)
        shown = axis.imshow(image, cmap="gray", vmin=-maximum, vmax=maximum); axis.axis("off")
        if kind == "bwscale": plt.colorbar(shown, ax=axis)
    elif kind in {"mag", "magscale"}:
        shown = axis.imshow(np.sqrt(u**2 + v**2), cmap="gray"); axis.axis("off")
        if kind == "magscale": plt.colorbar(shown, ax=axis)
    else:
        raise ValueError(f"Invalid plot type: {kind}")
    return axis


def normalize(array: np.ndarray) -> np.ndarray:
    output = array - np.nanmin(array)
    maximum = float(np.nanmax(output))
    return output / maximum if maximum > 0 else np.zeros_like(output)


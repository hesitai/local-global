"""对应原 ``utils/draw_mf_on_img.m`` 的稀疏运动线叠加显示。"""

from __future__ import annotations

import cv2
import numpy as np


def draw_mf_on_img(
    image: np.ndarray,
    magnitude: np.ndarray,
    orientation: np.ndarray,
    interval: int = 21,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:
        rgb = np.clip(image, 0.0, 1.0)
        gray = cv2.cvtColor(np.rint(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    else:
        gray = np.clip(image, 0.0, 1.0)
    output = np.repeat(np.clip(0.7 + 0.4 * gray, 0.0, 1.0)[..., None], 3, axis=2)
    height, width = gray.shape
    for row in range(interval - 1, height - interval, interval):
        for column in range(interval - 1, width - interval, interval):
            length = max(float(magnitude[row, column]), 1.0)
            angle = np.deg2rad(float(orientation[row, column]))
            delta_x = 0.5 * length * np.sin(angle)
            delta_y = 0.5 * length * np.cos(angle)
            start = (int(round(column - delta_x)), int(round(row - delta_y)))
            end = (int(round(column + delta_x)), int(round(row + delta_y)))
            cv2.line(output, start, end, (1.0, 0.0, 0.0), 1, cv2.LINE_AA)
            output[row, column] = (0.0, 0.5, 0.0)
    return np.clip(output, 0.0, 1.0)


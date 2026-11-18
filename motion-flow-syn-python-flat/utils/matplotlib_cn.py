"""为可视化脚本选择系统中已安装的中文字体。"""

from __future__ import annotations

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


def configure_chinese_font() -> None:
    candidates = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "SimHei",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


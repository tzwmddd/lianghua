"""绘图:策略净值曲线对比,遵循统一视觉规范。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import CHINESE_FONTS, RESULTS_DIR

_COLORS = {
    "LSTM": "#1769AA",
    "LightGBM": "#D62728",
    "沪深300": "#777777",
}


def plot_equity(curves, title, filename):
    """curves: {label: DataFrame(date, equity)},绘制多条净值曲线。"""
    plt.rcParams["font.sans-serif"] = CHINESE_FONTS
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, c in curves.items():
        ax.plot(c["date"], c["equity"], label=label,
                color=_COLORS.get(label, "#333333"), linewidth=1.6)

    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(frameon=False)
    ax.tick_params(direction="out", width=1.1)
    for s in ax.spines.values():
        s.set_linewidth(1.2)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

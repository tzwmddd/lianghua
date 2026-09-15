"""两融因子单因子 Rank IC 验证(embargo 口径)。

两融是独立于技术面的正交源(信用账户杠杆资金行为)。构造两个横截面可比的比率因子:
  margin_mcap_ratio = 融资余额 / 流通市值     (杠杆比例)
  margin_buy_ratio  = 融资买入额 / 成交额      (杠杆买入占比)

单因子 IC 无训练无标签重叠,天然干净;点-in-time 成分股过滤消幸存者偏差。
"""
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import CACHE_DIR, RESULTS_DIR
from cross_section import attach_fundamentals
import turnover_model_test as T

TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
MIN_CROSS = 30


def factor_ic(pool, col):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]
    ics = []
    for t in rebal_dates:
        cross = pool[pool["date"] == t].dropna(subset=[col, "excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        ics.append(cross[col].corr(cross["excess_fwd"], method="spearman"))
    ic = np.array(ics)
    icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")
    return ic.mean(), icir, (ic > 0).mean(), len(ic)


def main():
    pool = T.load_pool()
    pool = T.apply_constituents(pool)
    pool = attach_fundamentals(pool, CACHE_DIR)

    margin = pd.read_csv(os.path.join(CACHE_DIR, "_margin_panel.csv"),
                         parse_dates=["date"], dtype={"code": str})
    pool = pool.merge(margin, on=["code", "date"], how="left")
    # mktcap 单位亿元, margin_balance 单位元 -> 换算
    pool["margin_mcap_ratio"] = pool["margin_balance"] / (pool["mktcap"] * 1e8)
    pool["margin_buy_ratio"] = pool["margin_buy"] / pool["amount"]

    print(f"样本 {len(pool)}, 两融覆盖 {pool['margin_balance'].notna().mean():.0%}", flush=True)

    lines = []
    header = f"{'因子':<20}{'Rank IC':>10}{'ICIR':>9}{'IC>0':>8}{'期数':>6}"
    print(header, flush=True)
    lines.append(header)
    for col in ["margin_balance", "margin_buy", "margin_mcap_ratio", "margin_buy_ratio"]:
        m, ir, pos, n = factor_ic(pool, col)
        row = f"{col:<20}{m:>+10.4f}{ir:>+9.3f}{pos:>7.0%}{n:>6}"
        print(row, flush=True)
        lines.append(row)

    print("\n(对照: 换手率原始 -0.077 是 size 别名; 技术面模型 +0.030)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "margin_factor_ic_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("两融因子单因子 Rank IC (embargo 口径, 点-in-time 成分股)\n")
        f.write("====================================================\n\n")
        f.write(f"OOS {TEST_START} 起, 每 {REBALANCE_EVERY} 交易日调仓\n\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)


if __name__ == "__main__":
    main()

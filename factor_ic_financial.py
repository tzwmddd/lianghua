"""财务因子单因子 Rank IC(10年窗口) + size 拆解。

定位 0.47 里"财务+size"贡献(~+0.33)的具体来源:
  - 9 个财务因子各自中性化后 Rank IC(定位哪个字段泄露/有效)
  - size 因子单因子 IC(验证"幸存者偏差载体"假设)
  - 估值因子 + 原始 log_mcap 对照

与 stress_test_ablate.py 同窗口: TEST_START=2017, 每20交易日调仓。
不训练模型,只算横截面 Spearman 相关,几秒出结果。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import CACHE_DIR, RESULTS_DIR, FORECAST_HORIZON
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import attach_fundamentals, neutralize, VALUATION_COLS
from fundamental import attach_financial, FINANCIAL_COLS

TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]

# 中性化后(模型实际输入)重点因子: 财务9 + size + 估值3(对照)
NEU_FACTORS = FINANCIAL_COLS + ["size"] + VAL_LOG_COLS
# 原始值(未中性化)对照: 财务9 + log_mcap
RAW_FACTORS = FINANCIAL_COLS + ["log_mcap"]


def load_index():
    idx = pd.read_csv(os.path.join(CACHE_DIR, "sh000300.csv"), parse_dates=["date"])
    return idx.set_index("date")["close"].sort_index()


def load_pool():
    idx = load_index()
    idx_ret = idx.pct_change()
    idx_fwd = idx.shift(-FORECAST_HORIZON) / idx - 1

    codes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(CACHE_DIR, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    parts = []
    for code in codes:
        df = load_cached(code, CACHE_DIR)
        if df is None or len(df) < 120:
            continue
        feat = build_features(df, FORECAST_HORIZON)
        feat["idx_ret"] = feat["date"].map(idx_ret)
        feat["idx_fwd"] = feat["date"].map(idx_fwd)
        feat["ret"] = feat["close"].pct_change()
        for n in EXCESS_WINDOWS:
            feat[f"excess_{n}"] = feat["ret"].rolling(n).sum() - feat["idx_ret"].rolling(n).sum()
        feat["stock_fwd"] = feat["close"].shift(-FORECAST_HORIZON) / feat["close"] - 1
        feat["excess_fwd"] = feat["stock_fwd"] - feat["idx_fwd"]
        feat["target_rel"] = (feat["excess_fwd"] > 0).astype(int)
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool.dropna(subset=["target_rel", "excess_fwd"] + EXCESS_COLS)
    return pool.sort_values("date").reset_index(drop=True)


def factor_ic(pool, factor_col):
    """单因子横截面 Rank IC:每个调仓日因子值 vs excess_fwd 的 Spearman 相关。"""
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]
    ics = []
    for t in rebal_dates:
        cross = pool[pool["date"] == t].dropna(subset=[factor_col, "excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        ics.append(cross[factor_col].corr(cross["excess_fwd"], method="spearman"))
    ic = np.array(ics)
    if len(ic) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")
    return float(ic.mean()), float(icir), float((ic > 0).mean()), len(ic)


def main():
    t0 = time.time()
    pool = load_pool()
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本 "
          f"({pool['date'].min().date()} ~ {pool['date'].max().date()})", flush=True)

    raw = attach_fundamentals(pool.copy(), CACHE_DIR)
    raw = attach_financial(raw, CACHE_DIR)
    all_style = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS
    neu = neutralize(raw.copy(), all_style)

    lines = []
    lines.append(f"财务因子单因子 Rank IC(10年窗口, TEST_START={TEST_START}, 每{REBALANCE_EVERY}日调仓)")
    lines.append("对照: 模型整体 full +0.471 / tech_only +0.126 / 财务贡献 ~+0.15 / size ~+0.18")
    lines.append("")

    # 中性化后
    neu_rows = sorted(
        ((f, *factor_ic(neu, f)) for f in NEU_FACTORS),
        key=lambda r: abs(r[1]), reverse=True,
    )
    print(f"\n{'因子':<22}{'Rank IC':>10}{'ICIR':>9}{'IC>0':>8}{'期数':>6}", flush=True)
    lines.append(f"{'因子':<22}{'Rank IC':>10}{'ICIR':>9}{'IC>0':>8}{'期数':>6}")
    for f, mic, icir, pos, n in neu_rows:
        print(f"{f:<22}{mic:>+10.4f}{icir:>+9.3f}{pos:>7.0%}{n:>6}", flush=True)
        lines.append(f"{f:<22}{mic:>+10.4f}{icir:>+9.3f}{pos:>7.0%}{n:>6}")

    # 原始值
    raw_rows = sorted(
        ((f, *factor_ic(raw, f)) for f in RAW_FACTORS),
        key=lambda r: abs(r[1]), reverse=True,
    )
    print(f"\n---- 原始值(未中性化) ----", flush=True)
    lines.append("")
    lines.append("---- 原始值(未中性化) ----")
    for f, mic, icir, pos, n in raw_rows:
        print(f"{f:<22}{mic:>+10.4f}{icir:>+9.3f}{pos:>7.0%}{n:>6}", flush=True)
        lines.append(f"{f:<22}{mic:>+10.4f}{icir:>+9.3f}{pos:>7.0%}{n:>6}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "factor_ic_financial_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

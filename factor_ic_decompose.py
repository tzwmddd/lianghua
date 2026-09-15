"""单因子 Rank IC 拆解:定位"模型 0.44 的 IC 是不是一个因子撑起来的"。

对每个因子单独算横截面 Rank IC(因子值 vs 未来30日超额收益 excess_fwd),
不训练任何模型。分两组:
  原始值(raw)  : 因子在行业+市值中性化之前的原始值,经济含义直观
  中性化后(neu) : 模型实际看到的中性化残差(含 size 因子)

若某个单因子的 |IC| 已接近模型整体 IC,说明模型主要靠这一个因子;
若所有单因子 |IC| 都很低(<0.1),说明 0.44 来自组合(但也可能是过拟合)。
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

TEST_START = "2024-02-01"    # 与 evaluate_financial2.py 基线(0.442)同窗口,直接可比
REBALANCE_EVERY = 20
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
RAW_EXTRA = ["log_mcap"] + VAL_LOG_COLS
RAW_FACTORS = FEATURE_COLS + EXCESS_COLS + RAW_EXTRA + FINANCIAL_COLS
NEU_FACTORS = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS + ["size"]


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
        ic = cross[factor_col].corr(cross["excess_fwd"], method="spearman")
        ics.append(ic)
    ic = np.array(ics)
    if len(ic) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")
    return float(ic.mean()), float(icir), float((ic > 0).mean()), len(ic)


def table(name, pool, factors):
    print(f"\n=== {name} ===", flush=True)
    rows = []
    for f in factors:
        mic, icir, pos, n = factor_ic(pool, f)
        rows.append((f, mic, icir, pos, n))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    print(f"{'因子':<20}{'Rank IC':>10}{'ICIR':>9}{'IC>0':>8}{'期数':>6}", flush=True)
    for f, mic, icir, pos, n in rows:
        print(f"{f:<20}{mic:>+10.4f}{icir:>+9.3f}{pos:>7.0%}{n:>6}", flush=True)
    return rows


def main():
    t0 = time.time()
    pool = load_pool()
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本 "
          f"({pool['date'].min().date()} ~ {pool['date'].max().date()})", flush=True)

    # 原始因子(中性化前)
    raw = pool.copy()
    raw = attach_fundamentals(raw, CACHE_DIR)
    raw = attach_financial(raw, CACHE_DIR)

    # 中性化(模型实际输入)
    neu = raw.copy()
    all_style = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS
    neu = neutralize(neu, all_style)

    raw_rows = table("原始因子 Rank IC (未中性化)", raw, RAW_FACTORS)
    neu_rows = table("中性化后因子 Rank IC (模型实际输入)", neu, NEU_FACTORS)

    # 关键结论
    top_raw = max(raw_rows, key=lambda r: abs(r[1]))
    top_neu = max(neu_rows, key=lambda r: abs(r[1]))
    print("\n===== 结论 =====", flush=True)
    print(f"最强原始因子: {top_raw[0]}  |IC|={abs(top_raw[1]):.4f}", flush=True)
    print(f"最强中性化因子: {top_neu[0]}  |IC|={abs(top_neu[1]):.4f}", flush=True)
    print(f"(对照: 模型整体 Rank IC ≈ +0.442, ICIR +3.18)", flush=True)
    strong = [r for r in neu_rows if abs(r[1]) >= 0.20]
    if strong:
        print(f"中性化后 |IC|>=0.20 的因子 {len(strong)} 个: "
              f"{', '.join(r[0] for r in strong)}", flush=True)
    else:
        print("没有单因子 |IC|>=0.20 —— 0.44 主要来自因子组合(需警惕过拟合)。", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "factor_ic_decompose_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("==============================================================\n")
        f.write("单因子 Rank IC 拆解报告 (因子值 vs 未来30日超额收益)\n")
        f.write("==============================================================\n\n")
        f.write(f"OOS 起点 : {TEST_START},每 {REBALANCE_EVERY} 交易日调仓\n")
        f.write(f"对照基准 : 模型整体 Rank IC +0.442, ICIR +3.18\n\n")
        f.write("---- 原始因子(未中性化) ----\n")
        for f, mic, icir, pos, n in raw_rows:
            f.write(f"{f:<20}{mic:>+10.4f}{icir:>+9.3f}{pos:>7.0%}{n:>6}\n")
        f.write("\n---- 中性化后(模型输入) ----\n")
        for f, mic, icir, pos, n in neu_rows:
            f.write(f"{f:<20}{mic:>+10.4f}{icir:>+9.3f}{pos:>7.0%}{n:>6}\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""线性模型对照:逻辑回归 vs LightGBM,区分"真实非线性 alpha"与"过拟合"。

同一批 38 个中性化特征、同一扩张窗口、同一 Rank IC 口径。
若逻辑回归(只能学线性组合)IC 接近单因子(约0.1),而 LightGBM 到 0.44,
说明 0.44 里的非线性部分大概率是过拟合;若逻辑回归也能到 0.3+,则因子线性能量确实强。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import CACHE_DIR, RESULTS_DIR, SEED, FORECAST_HORIZON
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import attach_fundamentals, neutralize, VALUATION_COLS
from fundamental import attach_financial, FINANCIAL_COLS

TEST_START = "2024-02-01"
REBALANCE_EVERY = 20
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]
EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
ALL_STYLE = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS
FEATURES = ALL_STYLE + ["size"]


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


def main():
    t0 = time.time()
    pool = load_pool()
    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    pool = neutralize(pool, ALL_STYLE)

    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    ic_list = []
    for t in rebal_dates:
        train = pool[pool["date"] < t].dropna(subset=FEATURES)
        if len(train) < MIN_TRAIN:
            continue
        Xtr = train[FEATURES].to_numpy()
        ytr = train["target_rel"].to_numpy()
        m = Xtr.mean(axis=0)
        s = Xtr.std(axis=0)
        s[s < 1e-12] = 1.0
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((Xtr - m) / s, ytr)
        cross = pool[pool["date"] == t].dropna(subset=FEATURES + ["excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        prob = clf.predict_proba((cross[FEATURES].to_numpy() - m) / s)[:, 1]
        df = pd.DataFrame({"prob": prob, "excess_fwd": cross["excess_fwd"].to_numpy()})
        ic = df["prob"].corr(df["excess_fwd"], method="spearman")
        ic_list.append(ic)

    ic = np.array(ic_list)
    icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")
    print(f"\n=== 逻辑回归(线性) Rank IC ===", flush=True)
    print(f"  Rank IC: {ic.mean():+.4f}   ICIR {icir:+.3f}   IC>0 {(ic > 0).mean():.1%}   ({len(ic)}期)", flush=True)
    print(f"\n=== 对照 ===", flush=True)
    print(f"  最强单因子 |IC|: ~0.074 (rev_growth)", flush=True)
    print(f"  LightGBM 非线性 : +0.442 / ICIR +3.18", flush=True)
    print(f"  逻辑回归(线性) : {ic.mean():+.4f} / ICIR {icir:+.3f}", flush=True)
    print(f"\n结论: 线性模型 IC = {ic.mean():+.4f}。", flush=True)
    print(f"  - 若接近 0.1(单因子水平) => LightGBM 的 0.44 主要是非线性过拟合", flush=True)
    print(f"  - 若接近 0.3+          => 因子线性能量强,0.44 含真实非线性 alpha", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "linear_ic_check_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"逻辑回归(线性) Rank IC: {ic.mean():+.4f}  ICIR {icir:+.3f}  IC>0 {(ic>0).mean():.1%}  ({len(ic)}期)\n")
        f.write(f"对照: LightGBM +0.442 / ICIR +3.18 ; 最强单因子 |IC| 0.074\n")
    print(f"\n报告已保存: results/linear_ic_check_report.txt", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

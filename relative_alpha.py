"""横截面相对强弱实验:相对收益标签 + 超额收益特征,看能否提升选股 alpha。

思路:选股本质是"跑赢市场",而非"预测涨跌"。把标签从绝对涨跌方向改成
"是否跑赢沪深300",并加入个股相对市场的超额收益特征,直接对齐选股目标。
对比:绝对标签(22特征) vs 相对标签(22+3超额特征)。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import CACHE_DIR, SEED, FORECAST_HORIZON
from features import build_features, FEATURE_COLS
from data import load_cached

TEST_START = "2024-02-01"
REBALANCE_EVERY = 20
N_GROUPS = 5
TOP_FRAC = 0.2
EXCESS_WINDOWS = [5, 20, 60]


def load_index():
    idx = pd.read_csv(os.path.join(CACHE_DIR, "sh000300.csv"), parse_dates=["date"])
    return idx.set_index("date")["close"].sort_index()


def load_pool():
    """加载个股面板,额外算超额收益特征与相对收益标签。"""
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
        feat = build_features(df, FORECAST_HORIZON)   # 含 target(绝对方向)
        feat["idx_ret"] = feat["date"].map(idx_ret)
        feat["idx_fwd"] = feat["date"].map(idx_fwd)
        feat["ret"] = feat["close"].pct_change()
        for n in EXCESS_WINDOWS:
            feat[f"excess_{n}"] = feat["ret"].rolling(n).sum() - feat["idx_ret"].rolling(n).sum()
        feat["stock_fwd"] = feat["close"].shift(-FORECAST_HORIZON) / feat["close"] - 1
        feat["excess_fwd"] = feat["stock_fwd"] - feat["idx_fwd"]      # 未来超额收益
        feat["target_rel"] = (feat["excess_fwd"] > 0).astype(int)     # 是否跑赢市场
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool.dropna(subset=["target_rel", "excess_fwd"] + [f"excess_{n}" for n in EXCESS_WINDOWS])
    return pool.sort_values("date").reset_index(drop=True)


def run_cross(pool, feature_cols, target_col):
    """pooled 扩张训练 + 横截面评估,返回 Rank IC(相对超额收益)与分层超额收益。"""
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    ic_list, group_excess = [], {g: [] for g in range(N_GROUPS)}
    for t in rebal_dates:
        train = pool[pool["date"] < t]
        if len(train) < 5000:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(train[feature_cols].to_numpy(), train[target_col].to_numpy())
        cross = pool[pool["date"] == t].dropna(subset=["excess_fwd"])
        if len(cross) < 30:
            continue
        prob = model.predict_proba(cross[feature_cols].to_numpy())[:, 1]
        df = cross[["excess_fwd"]].copy()
        df["prob"] = prob
        ic = df["prob"].corr(df["excess_fwd"], method="spearman")
        ic_list.append(ic)
        df["rank"] = df["prob"].rank(method="first", pct=True)
        df["group"] = np.ceil(df["rank"] * N_GROUPS).astype(int) - 1
        for g in range(N_GROUPS):
            group_excess[g].append(df.loc[df["group"] == g, "excess_fwd"].mean())
    return np.array(ic_list), group_excess


def summarize(name, ic, group_excess):
    icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")
    print(f"\n=== {name} ===", flush=True)
    print(f"  相对 Rank IC: {ic.mean():+.4f}  ICIR {icir:+.3f}  IC>0占比 {(ic > 0).mean():.1%}", flush=True)
    print(f"  分层超额收益(个股收益-沪深300):", flush=True)
    for g in range(N_GROUPS - 1, -1, -1):
        r = np.array(group_excess[g])
        print(f"    Q{g + 1}: {r.mean():+.3%}/期", flush=True)
    top = np.array(group_excess[N_GROUPS - 1])
    bot = np.array(group_excess[0])
    print(f"  多空 Q5-Q1 超额: {(top - bot).mean():+.3%}/期", flush=True)


def main():
    t0 = time.time()
    pool = load_pool()
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本", flush=True)

    base_cols = FEATURE_COLS
    rel_cols = FEATURE_COLS + [f"excess_{n}" for n in EXCESS_WINDOWS]

    # 基线:绝对方向标签 + 22特征(用超额收益做评估,对齐口径)
    ic_abs, ge_abs = run_cross(pool, base_cols, "target")
    summarize("基线(绝对涨跌标签, 22特征)", ic_abs, ge_abs)

    # 新:相对标签 + 25特征
    ic_rel, ge_rel = run_cross(pool, rel_cols, "target_rel")
    summarize("相对标签(跑赢沪深300) + 超额收益特征(25特征)", ic_rel, ge_rel)

    print(f"\n总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""幸存者偏差检验:按上市日期过滤次新股,看纯多头 Q5 真实收益还剩多少。

幸存者偏差来源:用 2026 年当前的沪深300成分股回测 2024 年,系统高估收益
(表现好才留得下/进得来,表现差被踢出的没纳入)。akshare 无历史成分股接口,
这里用 cache 数据起点近似上市日期,做三个版本的对比。
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

from config import (
    CACHE_DIR, SEED, FORECAST_HORIZON, COMMISSION, STAMP_TAX, SLIPPAGE,
)
from features import build_features, FEATURE_COLS
from data import load_cached

TRAIN_START = "2021-01-01"
TEST_START = "2024-02-01"
REBALANCE_EVERY = FORECAST_HORIZON
TOP_FRAC = 0.2
ROUND_TRIP = COMMISSION * 2 + STAMP_TAX + SLIPPAGE * 2


def load_pool(min_list_date=None):
    codes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(CACHE_DIR, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    parts = []
    skipped = 0
    for code in codes:
        df = load_cached(code, CACHE_DIR)
        if df is None or len(df) < 120:
            continue
        if min_list_date is not None and df["date"].min() > pd.Timestamp(min_list_date):
            skipped += 1
            continue
        feat = build_features(df, FORECAST_HORIZON)
        raw = df.set_index("date")["close"]
        fwd = (raw.shift(-FORECAST_HORIZON) / raw - 1).rename("fwd_ret")
        feat = feat.merge(fwd.reset_index(), on="date", how="left")
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)
    return pool.sort_values("date").reset_index(drop=True), skipped


def run_q5_long(pool):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]
    rets = []
    for t in rebal_dates:
        train = pool[pool["date"] < t]
        if len(train) < 5000:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(train[FEATURE_COLS].to_numpy(), train["target"].to_numpy())
        cross = pool[pool["date"] == t].dropna(subset=["fwd_ret"])
        if len(cross) < 30:
            continue
        prob = model.predict_proba(cross[FEATURE_COLS].to_numpy())[:, 1]
        df = cross[["fwd_ret"]].copy()
        df["prob"] = prob
        k = max(1, int(len(df) * TOP_FRAC))
        q5_ret = df.nlargest(k, "prob")["fwd_ret"].mean()
        rets.append(q5_ret - ROUND_TRIP)
    return np.array(rets)


def summarize(name, rets):
    nav = float(np.prod(1 + rets))
    total_days = len(rets) * FORECAST_HORIZON
    ann = nav ** (252 / total_days) - 1
    sr = rets.mean() / rets.std(ddof=1) * np.sqrt(252 / FORECAST_HORIZON) if rets.std(ddof=1) > 0 else float("nan")
    print(f"{name:<22} {len(rets)}期  平均{rets.mean():+.3%}/期  年化{ann:+.2%}  "
          f"夏普{sr:.3f}  净值{nav:.4f}  胜率{(rets > 0).mean():.1%}", flush=True)
    return {"nav": nav, "ann": ann, "sr": sr, "mean": rets.mean(), "win": (rets > 0).mean()}


def main():
    t0 = time.time()
    print("纯多头 Q5(扣单边成本) 按上市日期过滤对比:\n", flush=True)

    versions = [
        ("全部300只(有偏差)", None),
        ("剔除2024后上市", "2024-02-01"),
        ("只留2021前上市", "2022-01-01"),
    ]
    results = {}
    for name, min_date in versions:
        pool, skipped = load_pool(min_date)
        n_stocks = pool["code"].nunique()
        rets = run_q5_long(pool)
        print(f"[{name}] 纳入 {n_stocks} 只 (剔除 {skipped} 只次新)", flush=True)
        results[name] = summarize(name, rets)

    print(f"\n=== 结论 ===", flush=True)
    base = results["全部300只(有偏差)"]["ann"]
    for name, _ in versions[1:]:
        ann = results[name]["ann"]
        print(f"  {name}: 年化 {ann:+.2%} (vs 全部300只 {base:+.2%}, 缩水 {base - ann:.2%})", flush=True)
    print(f"\n总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""连续打分评估:每笔得分 = 预测方向(+1涨/-1跌) × 实际收益(%)。

比二元准确率更合理:准确率只看对错(正负号),这里把涨跌幅度也纳入,
每笔得分就是"这笔交易赚/亏了百分之几",汇总后即策略实际盈亏。
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

from config import CACHE_DIR, RESULTS_DIR, SEED, FORECAST_HORIZON
from features import build_features, FEATURE_COLS
from data import load_cached

TRAIN_START = "2021-01-01"
TEST_START = "2024-02-01"
REBALANCE_EVERY = 20
MIN_TRAIN = 5000
MIN_CROSS = 30


def load_panel():
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
        raw = df.set_index("date")["close"]
        fwd = (raw.shift(-FORECAST_HORIZON) / raw - 1).rename("fwd_ret")
        feat = feat.merge(fwd.reset_index(), on="date", how="left")
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)
    return pool.sort_values("date").reset_index(drop=True)


def train_pooled(train):
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(train[FEATURE_COLS].to_numpy(), train["target"].to_numpy())
    return model


def main():
    t0 = time.time()
    pool = load_panel()
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]
    print(f"加载 {pool['code'].nunique()} 只股票,调仓日 {len(rebal_dates)} 个,"
          f"预测目标未来 {FORECAST_HORIZON} 日", flush=True)

    scores, probs = [], []
    per_rebal_long, per_rebal_short = [], []

    for t in rebal_dates:
        train = pool[pool["date"] < t]
        if len(train) < MIN_TRAIN:
            continue
        model = train_pooled(train)
        cross = pool[pool["date"] == t].dropna(subset=["fwd_ret"])
        if len(cross) < MIN_CROSS:
            continue
        prob = model.predict_proba(cross[FEATURE_COLS].to_numpy())[:, 1]
        fwd = cross["fwd_ret"].to_numpy()
        d = np.where(prob > 0.5, 1.0, -1.0)
        scores.append(d * fwd)
        probs.append(prob)
        per_rebal_long.append(fwd[d > 0].mean())
        per_rebal_short.append(fwd[d < 0].mean())

    score = np.concatenate(scores)
    prob = np.concatenate(probs)
    long = np.array(per_rebal_long)
    short = np.array(per_rebal_short)
    ls = long - short
    cum_nav = float(np.prod(1.0 + ls))

    print("\n========== 连续打分评估 ==========", flush=True)
    print(f"打分规则: 每笔得分 = 预测方向(+1涨/-1跌) × 实际收益", flush=True)
    print(f"预测目标: 未来 {FORECAST_HORIZON} 日涨跌方向\n", flush=True)

    print(f"总笔数           : {len(score)}", flush=True)
    print(f"平均每笔得分     : {score.mean():+.4%}   <-- 核心:平均每笔盈亏", flush=True)
    print(f"得分中位数       : {np.median(score):+.4%}", flush=True)
    print(f"盈利笔数占比     : {(score > 0).mean():.1%}  (= 二元方向准确率)", flush=True)
    print(f"得分标准差       : {score.std(ddof=1):.4%}", flush=True)

    print("\n=== 得分分布(按幅度分档) ===", flush=True)
    bins = [
        ("大赢  (>+5%)", (score > 0.05).mean()),
        ("小赢  (0~+5%)", ((score > 0) & (score <= 0.05)).mean()),
        ("小亏  (-5%~0)", ((score < 0) & (score >= -0.05)).mean()),
        ("大亏  (<-5%)", (score < -0.05).mean()),
    ]
    for name, pct in bins:
        print(f"  {name:<16} {pct:6.1%}", flush=True)

    print("\n=== 多空组合(每期做多预测涨 / 做空预测跌, 等权重) ===", flush=True)
    print(f"  预测涨组平均收益 : {long.mean():+.4%}", flush=True)
    print(f"  预测跌组平均收益 : {short.mean():+.4%}", flush=True)
    print(f"  多空价差(每期)   : {ls.mean():+.4%}", flush=True)
    print(f"  多空胜率         : {(ls > 0).mean():.1%}", flush=True)
    print(f"  累计净值({len(ls)}期): {cum_nav:.4f}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "score_pnl_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("==============================================================\n")
        f.write("连续打分评估报告 (得分 = 预测方向 × 实际收益)\n")
        f.write("==============================================================\n\n")
        f.write(f"预测目标: 未来 {FORECAST_HORIZON} 日涨跌方向\n")
        f.write(f"总笔数   : {len(score)}\n\n")
        f.write(f"平均每笔得分   : {score.mean():+.4%}\n")
        f.write(f"得分中位数     : {np.median(score):+.4%}\n")
        f.write(f"盈利笔数占比   : {(score > 0).mean():.1%}  (= 二元方向准确率)\n")
        f.write(f"得分标准差     : {score.std(ddof=1):.4%}\n\n")
        f.write("得分分布:\n")
        for name, pct in bins:
            f.write(f"  {name:<16} {pct:6.1%}\n")
        f.write(f"\n多空组合:\n")
        f.write(f"  预测涨组平均收益 : {long.mean():+.4%}\n")
        f.write(f"  预测跌组平均收益 : {short.mean():+.4%}\n")
        f.write(f"  多空价差(每期)   : {ls.mean():+.4%}\n")
        f.write(f"  多空胜率         : {(ls > 0).mean():.1%}\n")
        f.write(f"  累计净值({len(ls)}期): {cum_nav:.4f}\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""泄露测试:判断 0.43 IC 是"真学会预测未来"还是"特征/标签里藏着泄露"。

单次 train/test split(省时),三组对照:
  1. 真标签: 正常训练,测 Rank IC(应 ~0.43)
  2. 随机标签: 把训练标签打乱,再训练,测 IC(应 ~0,若仍高 => 特征含泄露)
  3. 预测过去: 模型预测 vs 过去30日收益的相关(看是否在"预测已发生的事")
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
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import attach_fundamentals, neutralize, VALUATION_COLS
from fundamental import attach_financial, FINANCIAL_COLS

SPLIT = "2023-01-01"
EXCESS_WINDOWS = [5, 20, 60]
EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
STYLE_COLS = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS
FEATURES = STYLE_COLS + ["size"]


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
        feat["past_fwd"] = feat["close"] / feat["close"].shift(-FORECAST_HORIZON) - 1  # 过去30日收益(符号取反)
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool.dropna(subset=["target_rel", "excess_fwd"] + EXCESS_COLS)
    return pool.sort_values("date").reset_index(drop=True)


def spearman(a, b):
    return pd.Series(a).corr(pd.Series(b), method="spearman")


def train_predict(train, test, ycol):
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.10, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(train[FEATURES].to_numpy(), train[ycol].to_numpy())
    return model.predict_proba(test[FEATURES].to_numpy())[:, 1]


def main():
    t0 = time.time()
    pool = load_pool()
    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    pool = neutralize(pool, STYLE_COLS)

    train = pool[pool["date"] < SPLIT].dropna(subset=FEATURES)
    test = pool[pool["date"] >= SPLIT].dropna(subset=FEATURES + ["excess_fwd", "past_fwd"])
    print(f"训练 {len(train)} 条, 测试 {len(test)} 条 ({test['date'].min().date()} ~ {test['date'].max().date()})", flush=True)

    # 1. 真标签
    prob = train_predict(train, test, "target_rel")
    ic_real = spearman(prob, test["excess_fwd"].to_numpy())
    print(f"\n1. 真标签 Rank IC: {ic_real:+.4f}", flush=True)

    # 2. 随机标签
    train_r = train.copy()
    train_r["target_rel"] = np.random.RandomState(SEED).permutation(train_r["target_rel"].to_numpy())
    prob_r = train_predict(train_r, test, "target_rel")
    ic_rand = spearman(prob_r, test["excess_fwd"].to_numpy())
    print(f"2. 随机标签 Rank IC: {ic_rand:+.4f}  (应≈0;若仍高=>特征含泄露)", flush=True)

    # 3. 预测过去(模型 prob vs 过去30日收益)
    ic_past = spearman(prob, test["past_fwd"].to_numpy())
    print(f"3. 模型prob vs 过去30日收益: {ic_past:+.4f}  (若≈真标签=>在预测已发生的事/纯动量)", flush=True)

    # 4. 测试期内按月的 IC 稳定性(看是否集中在某段)
    test2 = test.copy()
    test2["prob"] = prob
    test2["ym"] = test2["date"].dt.to_period("M")
    monthly = test2.groupby("ym").apply(lambda g: spearman(g["prob"], g["excess_fwd"]))
    print(f"\n4. 测试期月度 IC 分布: mean {monthly.mean():+.3f} / 中位 {monthly.median():+.3f} / IC>0 {(monthly>0).mean():.1%}", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

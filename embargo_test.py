"""隔离期(embargo)测试:验证 0.43 是"标签重叠泄露"还是真实预测力。

机理怀疑:FORECAST_HORIZON=30天标签,但训练与测试只隔1个交易日。
  - 训练样本 s 的标签 = return[s, s+30]
  - 测试日期 t 的标签 = return[t, t+30]
  - 当 s > t-30 时,两者标签重叠 30-(t-s) 天
  - 模型记住最近的样本(t-1,t-2...),其标签与测试标签 97% 相同
  - features[t] ≈ features[t-1](平滑),于是"预测"≈"昨天的标签"≈"今天的标签"

若加上 embargo(测试前 30+ 交易日不参与训练),IC 应崩到 ~0 => 坐实泄露。
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

TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
MIN_TRAIN = 5000
MIN_CROSS = 30
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
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool.dropna(subset=["target_rel", "excess_fwd"] + EXCESS_COLS)
    return pool.sort_values("date").reset_index(drop=True)


def run(pool, embargo_days):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]
    ic_list = []
    for t in rebal_dates:
        cutoff = t - pd.Timedelta(days=embargo_days)
        train = pool[pool["date"] < cutoff].dropna(subset=FEATURES)
        if len(train) < MIN_TRAIN:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.10, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(train[FEATURES].to_numpy(), train["target_rel"].to_numpy())
        cross = pool[pool["date"] == t].dropna(subset=FEATURES + ["excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        prob = model.predict_proba(cross[FEATURES].to_numpy())[:, 1]
        ic_list.append(pd.Series(prob).corr(pd.Series(cross["excess_fwd"].to_numpy()), method="spearman"))
    ic = np.array(ic_list)
    icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else np.nan
    return ic.mean(), icir, (ic > 0).mean(), len(ic)


def main():
    t0 = time.time()
    pool = load_pool()
    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    pool = neutralize(pool, STYLE_COLS)
    print(f"加载 {pool['code'].nunique()} 只, {len(pool)} 条", flush=True)

    print(f"\n{'embargo':<10}{'Rank IC':>10}{'ICIR':>8}{'IC>0':>8}{'期数':>6}", flush=True)
    for emb in [0, 10, 20, 30, 45]:
        m, icir, pos, n = run(pool, emb)
        print(f"{emb:<10}{m:>+10.4f}{icir:>+8.2f}{pos:>8.1%}{n:>6}", flush=True)

    print(f"\n解读:", flush=True)
    print(f"  embargo=0   => 训练和测试隔1天, 标签重叠29/30天 => 若 IC~0.43 是泄露", flush=True)
    print(f"  embargo=45  => 训练排除测试前45天, 标签无重叠   => 若 IC 崩到~0 坐实泄露", flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

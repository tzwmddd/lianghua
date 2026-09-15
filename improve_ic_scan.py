"""提高 IC 的扫描实验(embargo 口径):标签形式 x 预测周期。

真实信号只有 +0.03~0.04(技术面),本脚本系统扫两个高杠杆维度:
  1. 标签形式:class(二分类"跑赢") / reg(L2 回归 excess_fwd) / rank(lambdarank 排序)
  2. 预测周期 horizon:{10,20,30,60} 交易日,embargo 随 horizon 调整(≈1.5x 日历天)

Rank IC 是排序指标,分类强制二值化丢失幅度信息,回归/排序理论上更贴合。
每个组合在点-in-time 成分股 + embargo 下算 Rank IC/ICIR/IC>0/SE。
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

from config import CACHE_DIR, RESULTS_DIR, SEED
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import attach_fundamentals, neutralize

TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]
EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
STYLE_COLS = FEATURE_COLS + EXCESS_COLS          # tech_only:技术22+超额3
FEATURES = STYLE_COLS
HORIZONS = [10, 20, 30, 60]
LABEL_MODES = ["class", "reg", "rank"]


def load_index():
    idx = pd.read_csv(os.path.join(CACHE_DIR, "sh000300.csv"), parse_dates=["date"])
    return idx.set_index("date")["close"].sort_index()


def load_pool(horizon):
    idx = load_index()
    idx_ret = idx.pct_change()
    idx_fwd = idx.shift(-horizon) / idx - 1

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
        feat = build_features(df, horizon)
        feat["idx_ret"] = feat["date"].map(idx_ret)
        feat["idx_fwd"] = feat["date"].map(idx_fwd)
        feat["ret"] = feat["close"].pct_change()
        for n in EXCESS_WINDOWS:
            feat[f"excess_{n}"] = feat["ret"].rolling(n).sum() - feat["idx_ret"].rolling(n).sum()
        feat["stock_fwd"] = feat["close"].shift(-horizon) / feat["close"] - 1
        feat["excess_fwd"] = feat["stock_fwd"] - feat["idx_fwd"]
        feat["target_rel"] = (feat["excess_fwd"] > 0).astype(int)
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool.dropna(subset=["excess_fwd"] + EXCESS_COLS)
    return pool.sort_values("date").reset_index(drop=True)


def run_cross(pool, horizon, label_mode):
    embargo_days = int(np.ceil(horizon * 1.5))
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    ic_list = []
    for t in rebal_dates:
        cutoff = t - pd.Timedelta(days=embargo_days)
        train = pool[pool["date"] < cutoff].dropna(subset=FEATURES)
        if len(train) < MIN_TRAIN:
            continue

        if label_mode == "class":
            y = train["target_rel"].to_numpy()
            model = lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.10, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, verbosity=-1, n_jobs=-1,
            )
        elif label_mode == "reg":
            y = train["excess_fwd"].to_numpy()
            model = lgb.LGBMRegressor(
                n_estimators=150, learning_rate=0.10, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, verbosity=-1, n_jobs=-1,
            )
        else:  # rank
            tr = train.sort_values("date").copy()
            # lambdarank 要求整数 relevance;把连续 excess_fwd 按每期横截面分位离散成 5 档(0-4)
            tr["label_rank"] = tr.groupby("date")["excess_fwd"].transform(
                lambda s: (s.rank(method="first", pct=True) * 5).astype(int).clip(0, 4)
            )
            y = tr["label_rank"].to_numpy()
            groups = tr.groupby("date").size().to_numpy()
            model = lgb.LGBMRanker(
                n_estimators=150, learning_rate=0.10, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, verbosity=-1, n_jobs=-1,
            )

        if label_mode == "rank":
            model.fit(tr[FEATURES].to_numpy(), y, group=groups)
        else:
            model.fit(train[FEATURES].to_numpy(), y)

        cross = pool[pool["date"] == t].dropna(subset=FEATURES + ["excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        if label_mode == "class":
            pred = model.predict_proba(cross[FEATURES].to_numpy())[:, 1]
        else:
            pred = model.predict(cross[FEATURES].to_numpy())
        ic_list.append(pd.Series(pred).corr(pd.Series(cross["excess_fwd"].to_numpy()),
                                           method="spearman"))
    return np.array(ic_list)


def icir(ic):
    return ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = []
    header = (f"{'horizon':<8}{'label':<8}{'Rank IC':>10}{'ICIR':>8}{'IC>0':>8}"
              f"{'SE':>8}{'期数':>6}")
    print(header, flush=True)
    lines.append(header)

    # 点-in-time 成分股过滤(与 stress_test_pit 一致,消幸存者偏差)
    cons = pd.read_csv(os.path.join(CACHE_DIR, "_constituents.csv"),
                       parse_dates=["snapshot_date"], dtype={"code": str})
    snap_dates = pd.to_datetime(sorted(cons["snapshot_date"].unique()))

    for horizon in HORIZONS:
        pool = load_pool(horizon)
        pool = pool.sort_values("date").copy()
        sidx = np.searchsorted(snap_dates.values, pool["date"].to_numpy(), side="right") - 1
        sidx = np.clip(sidx, 0, len(snap_dates) - 1)
        pool["snap_date"] = snap_dates.values[sidx]
        pool = pool.merge(cons, left_on=["snap_date", "code"],
                          right_on=["snapshot_date", "code"], how="inner")
        pool = pool.drop(columns=["snap_date", "snapshot_date"]).sort_values("date")

        pool = attach_fundamentals(pool, CACHE_DIR)
        pool = neutralize(pool, STYLE_COLS)

        for mode in LABEL_MODES:
            ic = run_cross(pool, horizon, mode)
            se = ic.std(ddof=1) / np.sqrt(len(ic)) if len(ic) > 1 else float("nan")
            row = (f"{horizon:<8}{mode:<8}{ic.mean():>+10.4f}{icir(ic):>+8.3f}"
                   f"{(ic > 0).mean():>8.1%}{se:>8.4f}{len(ic):>6}")
            print(row, flush=True)
            lines.append(row)

    path = os.path.join(RESULTS_DIR, "improve_ic_scan_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("提高 IC 扫描 (embargo 口径, tech_only 技术22+超额3)\n")
        f.write("=====================================================\n\n")
        f.write("标签形式 x 预测周期;embargo = ceil(horizon*1.5) 日历天\n")
        f.write(f"OOS {TEST_START} 起, 每 {REBALANCE_EVERY} 交易日调仓, 点-in-time 成分股\n\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

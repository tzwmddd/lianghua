"""换手率因子单因子 Rank IC 验证(embargo 口径)。

换手率是 A股横截面强因子(文献:低换手->高收益),当前特征缺失。先算单因子
横截面 Rank IC(因子值 vs 未来30日超额收益 excess_fwd),确认方向与强度,再决定
是否接入模型。

换手率特征:
  log_turnover   = log(turnover)             水平(横截面)
  log_turnover_ma20 = log(20日均换手率)       平滑水平
  turnover_ratio = turnover / turnover_ma20  异动(相对自身均值)

单因子 IC 无训练/无标签重叠,天然干净,但仍按点-in-time 成分股过滤消幸存者偏差。
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
from features import build_features

TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

TURNOVER_COLS = ["log_turnover", "log_turnover_ma20", "turnover_ratio"]


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
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)

    # 附加换手率特征
    to = pd.read_csv(os.path.join(CACHE_DIR, "_turnover_panel.csv"),
                     parse_dates=["date"], dtype={"code": str})
    pool = pool.merge(to, on=["code", "date"], how="left")
    pool = pool.sort_values(["code", "date"])
    pool["log_turnover"] = np.log(pool["turnover"].clip(lower=1e-8))
    pool["turnover_ma20"] = pool.groupby("code")["turnover"].transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    pool["log_turnover_ma20"] = np.log(pool["turnover_ma20"].clip(lower=1e-8))
    pool["turnover_ratio"] = pool["turnover"] / pool["turnover_ma20"]

    pool = pool.dropna(subset=["excess_fwd"])
    return pool.sort_values("date").reset_index(drop=True)


def apply_constituents(pool):
    cons = pd.read_csv(os.path.join(CACHE_DIR, "_constituents.csv"),
                       parse_dates=["snapshot_date"], dtype={"code": str})
    snap_dates = pd.to_datetime(sorted(cons["snapshot_date"].unique()))
    pool = pool.sort_values("date").copy()
    sidx = np.searchsorted(snap_dates.values, pool["date"].to_numpy(), side="right") - 1
    sidx = np.clip(sidx, 0, len(snap_dates) - 1)
    pool["snap_date"] = snap_dates.values[sidx]
    pool = pool.merge(cons, left_on=["snap_date", "code"],
                      right_on=["snapshot_date", "code"], how="inner")
    return pool.drop(columns=["snap_date", "snapshot_date"]).sort_values("date")


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
    t0 = time.time()
    pool = load_pool()
    pool = apply_constituents(pool)
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本", flush=True)

    lines = []
    header = f"{'因子':<22}{'Rank IC':>10}{'ICIR':>9}{'IC>0':>8}{'期数':>6}"
    print(header, flush=True)
    lines.append(header)
    for col in ["turnover"] + TURNOVER_COLS:
        m, ir, pos, n = factor_ic(pool, col)
        row = f"{col:<22}{m:>+10.4f}{ir:>+9.3f}{pos:>7.0%}{n:>6}"
        print(row, flush=True)
        lines.append(row)

    print("\n解读: 换手率单因子 |IC| 若 >0.03 且方向稳定,即有接入价值", flush=True)
    print("(对照: tech_only 模型整体 IC +0.030,单因子普遍 <0.06)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "turnover_factor_ic_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("换手率因子单因子 Rank IC (embargo 口径, 点-in-time 成分股)\n")
        f.write("======================================================\n\n")
        f.write(f"OOS {TEST_START} 起, 每 {REBALANCE_EVERY} 交易日调仓, horizon {FORECAST_HORIZON}\n\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

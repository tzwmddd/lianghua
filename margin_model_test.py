"""两融因子接入模型验证(embargo 口径)。

margin_buy_ratio(融资买入额/成交额)是正交因子:与市值相关仅 -0.085,中性化后
单因子 IC -0.031(方向稳定,69%期数为负)。本脚本验证接入模型能否提升整体 IC:
  tech_only   : 技术22+超额3
  tech_margin : + margin_buy_ratio(前向填充,调仓日快照)
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

from config import CACHE_DIR, RESULTS_DIR, SEED, FORECAST_HORIZON, EMBARGO_DAYS
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import attach_fundamentals, neutralize

TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]
EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
BASE_STYLE = FEATURE_COLS + EXCESS_COLS

CONFIGS = {
    "tech_only": {"style": BASE_STYLE, "features": BASE_STYLE},
    "tech_margin": {"style": BASE_STYLE + ["margin_buy_ratio"],
                    "features": BASE_STYLE + ["margin_buy_ratio"]},
}


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

    to = pd.read_csv(os.path.join(CACHE_DIR, "_turnover_panel.csv"),
                     parse_dates=["date"], dtype={"code": str})
    pool = pool.merge(to, on=["code", "date"], how="left")
    pool = pool.dropna(subset=["excess_fwd"] + EXCESS_COLS)
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


def attach_margin(pool):
    margin = pd.read_csv(os.path.join(CACHE_DIR, "_margin_panel.csv"),
                         parse_dates=["date"], dtype={"code": str})
    pool = pool.merge(margin, on=["code", "date"], how="left")
    pool = pool.sort_values(["code", "date"])
    pool["margin_buy_ratio"] = pool["margin_buy"] / pool["amount"]
    pool["margin_buy_ratio"] = pool.groupby("code")["margin_buy_ratio"].ffill()
    return pool.sort_values("date").reset_index(drop=True)


def run_cross(pool, feature_cols):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]
    ic_list = []
    for t in rebal_dates:
        cutoff = t - pd.Timedelta(days=EMBARGO_DAYS)
        train = pool[pool["date"] < cutoff].dropna(subset=feature_cols)
        if len(train) < MIN_TRAIN:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.10, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(train[feature_cols].to_numpy(), train["target_rel"].to_numpy())
        cross = pool[pool["date"] == t].dropna(subset=feature_cols + ["excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        prob = model.predict_proba(cross[feature_cols].to_numpy())[:, 1]
        ic_list.append(pd.Series(prob).corr(pd.Series(cross["excess_fwd"].to_numpy()),
                                            method="spearman"))
    return np.array(ic_list)


def icir(ic):
    return ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")


def main():
    t0 = time.time()
    pool = load_pool()
    pool = apply_constituents(pool)
    pool = attach_margin(pool)
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本, "
          f"margin 覆盖 {pool['margin_buy_ratio'].notna().mean():.0%}", flush=True)

    lines = []
    header = f"{'配置':<16}{'Rank IC':>10}{'ICIR':>8}{'IC>0':>8}{'SE':>8}{'期数':>6}"
    print(header, flush=True)
    lines.append(header)
    for name, cfg in CONFIGS.items():
        p = attach_fundamentals(pool.copy(), CACHE_DIR)
        p = neutralize(p, cfg["style"])
        ic = run_cross(p, cfg["features"])
        se = ic.std(ddof=1) / np.sqrt(len(ic)) if len(ic) > 1 else float("nan")
        row = (f"{name:<16}{ic.mean():>+10.4f}{icir(ic):>+8.3f}"
               f"{(ic > 0).mean():>8.1%}{se:>8.4f}{len(ic):>6}")
        print(row, flush=True)
        lines.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "margin_model_test_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("两融因子接入模型验证 (embargo 口径)\n")
        f.write("================================\n\n")
        f.write(f"horizon {FORECAST_HORIZON}, embargo {EMBARGO_DAYS}, 点-in-time 成分股\n")
        f.write("margin_buy_ratio 前向填充(调仓日快照), 模型 num_leaves=31\n\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

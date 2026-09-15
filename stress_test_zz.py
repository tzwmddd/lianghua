"""中证500/1000 点-in-time 成分股回测(消除幸存者偏差 + embargo 防标签泄露)。

复刻 stress_test_pit.py 的严谨评估逻辑,数据源换成 tushare(cache_zz500/1000),
基准换成"跑赢自身指数"。输出真实 alpha 量级(embargo 后 Rank IC 不应虚高)。

用法: python stress_test_zz.py [500|1000] [tech_only|tech_size|no_fundamental|no_valuation|full ...]
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

from config import RESULTS_DIR, SEED, FORECAST_HORIZON, EMBARGO_DAYS
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import neutralize
from fundamental_ts import (
    attach_fundamentals_ts, attach_financial_ts,
    VALUATION_COLS_TS, FINANCIAL_COLS_TS,
)

UNIVERSES = {
    "500": ("000905.SH", "cache_zz500"),
    "1000": ("000852.SH", "cache_zz1000"),
}
TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
N_GROUPS = 5
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS_TS]
BASE_STYLE = FEATURE_COLS + EXCESS_COLS

CONFIGS = {
    "tech_only": {
        "style": BASE_STYLE, "features": BASE_STYLE,
        "needs_financial": False, "desc": "技术22+超额3",
    },
    "tech_size": {
        "style": BASE_STYLE, "features": BASE_STYLE + ["size"],
        "needs_financial": False, "desc": "技术+超额+size",
    },
    "no_valuation": {
        "style": BASE_STYLE + FINANCIAL_COLS_TS, "features": BASE_STYLE + FINANCIAL_COLS_TS + ["size"],
        "needs_financial": True, "desc": "技术+超额+财务+size",
    },
    "no_fundamental": {
        "style": BASE_STYLE + VAL_LOG_COLS, "features": BASE_STYLE + VAL_LOG_COLS + ["size"],
        "needs_financial": False, "desc": "技术+超额+估值+size",
    },
    "full": {
        "style": BASE_STYLE + VAL_LOG_COLS + FINANCIAL_COLS_TS,
        "features": BASE_STYLE + VAL_LOG_COLS + FINANCIAL_COLS_TS + ["size"],
        "needs_financial": True, "desc": "全部(技术+超额+估值+财务+size)",
    },
}


def load_index(cache_dir):
    idx = pd.read_csv(os.path.join(cache_dir, "index.csv"), parse_dates=["date"])
    return idx.set_index("date")["close"].sort_index()


def load_pool(cache_dir):
    idx = load_index(cache_dir)
    idx_ret = idx.pct_change()
    idx_fwd = idx.shift(-FORECAST_HORIZON) / idx - 1

    codes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(cache_dir, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    parts = []
    for code in codes:
        df = load_cached(code, cache_dir)
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


def apply_constituents(pool, cache_dir):
    """点-in-time 过滤:每个日期只保留"当时在成分里"的股票。"""
    cons = pd.read_csv(os.path.join(cache_dir, "_constituents.csv"),
                       parse_dates=["snapshot_date"], dtype={"code": str})
    snap_dates = pd.to_datetime(sorted(cons["snapshot_date"].unique()))
    pool = pool.sort_values("date").copy()
    idx = np.searchsorted(snap_dates.values, pool["date"].to_numpy(), side="right") - 1
    idx = np.clip(idx, 0, len(snap_dates) - 1)
    pool["snap_date"] = snap_dates.values[idx]
    before = len(pool)
    pool = pool.merge(cons, left_on=["snap_date", "code"],
                      right_on=["snapshot_date", "code"], how="inner")
    pool = pool.drop(columns=["snap_date", "snapshot_date"])
    print(f"  点-in-time 过滤: {before} -> {len(pool)} 条 ({len(pool) / before:.0%} 保留)", flush=True)
    return pool.sort_values("date").reset_index(drop=True)


def run_cross(pool, feature_cols):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    ic_list, group_excess, n_cross = [], {g: [] for g in range(N_GROUPS)}, []
    for t in rebal_dates:
        cutoff = t - pd.Timedelta(days=EMBARGO_DAYS)
        train = pool[pool["date"] < cutoff].dropna(subset=feature_cols)
        if len(train) < MIN_TRAIN:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.10, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(train[feature_cols].to_numpy(), train["target_rel"].to_numpy())
        cross = pool[pool["date"] == t].dropna(subset=feature_cols + ["excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        prob = model.predict_proba(cross[feature_cols].to_numpy())[:, 1]
        df = pd.DataFrame({"prob": prob, "excess_fwd": cross["excess_fwd"].to_numpy()})
        ic_list.append(df["prob"].corr(df["excess_fwd"], method="spearman"))
        df["rank"] = df["prob"].rank(method="first", pct=True)
        df["group"] = np.ceil(df["rank"] * N_GROUPS).astype(int) - 1
        for g in range(N_GROUPS):
            group_excess[g].append(df.loc[df["group"] == g, "excess_fwd"].mean())
        n_cross.append(len(cross))
    return np.array(ic_list), group_excess, n_cross


def icir(ic):
    return ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit("用法: python stress_test_zz.py [500|1000] [config...]")
    index_code, cache_dir = UNIVERSES[universe]
    configs = sys.argv[2:] or list(CONFIGS.keys())

    t0 = time.time()
    pool = load_pool(cache_dir)
    print(f"{index_code}: 加载 {pool['code'].nunique()} 只, {len(pool)} 条样本", flush=True)
    pool = apply_constituents(pool, cache_dir)

    header = f"{'配置':<16}{'Rank IC':>10}{'ICIR':>8}{'IC>0':>8}{'期数':>6}{'横截面均数':>10}"
    print(f"\n{header}", flush=True)
    lines = [header]
    for name in configs:
        cfg = CONFIGS[name]
        p = attach_fundamentals_ts(pool.copy(), cache_dir)
        if cfg["needs_financial"]:
            p = attach_financial_ts(p, cache_dir)
        p = neutralize(p, cfg["style"])
        ic, ge, nc = run_cross(p, cfg["features"])
        q5 = np.mean([ge[N_GROUPS - 1][i] for i in range(len(ic))])
        q1 = np.mean([ge[0][i] for i in range(len(ic))])
        row = (f"{name:<16}{ic.mean():>+10.4f}{icir(ic):>+8.3f}{(ic > 0).mean():>8.1%}"
               f"{len(ic):>6}{np.mean(nc):>10.0f}")
        print(row, flush=True)
        qstr = f"    Q5超额 {q5:+.2%}  Q1超额 {q1:+.2%}  多空 {q5 - q1:+.2%}"
        print(qstr, flush=True)
        lines.append(row)
        lines.append(qstr)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"stress_test_zz{universe}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"中证{universe} 点-in-time 成分股回测 (tushare 源, embargo 修正)\n")
        f.write("================================================\n\n")
        f.write(f"指数 {index_code}, embargo {EMBARGO_DAYS} 天, FORECAST_HORIZON {FORECAST_HORIZON}\n")
        f.write(f"OOS 起点 {TEST_START}, 每 {REBALANCE_EVERY} 交易日调仓, num_leaves=63/lr=0.10\n\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

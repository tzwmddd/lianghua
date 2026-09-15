"""点-in-time 成分股回测:消除幸存者偏差。

每个调仓日用"当时真实在沪深300里"的股票(baostock 历史成分股),而非固定 300 只。
对比 stress_test_ablate.py 的幸存者版,看 Rank IC 是否回落——若从 +0.47 大幅回落,
坐实幸存者偏差是主因。

支持命令行传特征配置(默认 tech_only,因被调出股估值/财务数据可能尚未下载)。
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
from cross_section import attach_fundamentals, neutralize, VALUATION_COLS
from fundamental import attach_financial, FINANCIAL_COLS

TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
N_GROUPS = 5
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
BASE_STYLE = FEATURE_COLS + EXCESS_COLS

CONFIGS = {
    "tech_only": {
        "style": BASE_STYLE,
        "features": BASE_STYLE,
        "needs_financial": False,
        "desc": "技术22+超额3",
    },
    "tech_size": {
        "style": BASE_STYLE,
        "features": BASE_STYLE + ["size"],
        "needs_financial": False,
        "desc": "技术+超额+size",
    },
    "no_valuation": {
        "style": BASE_STYLE + FINANCIAL_COLS,
        "features": BASE_STYLE + FINANCIAL_COLS + ["size"],
        "needs_financial": True,
        "desc": "技术+超额+财务+size",
    },
    "no_fundamental": {
        "style": BASE_STYLE + VAL_LOG_COLS,
        "features": BASE_STYLE + VAL_LOG_COLS + ["size"],
        "needs_financial": False,
        "desc": "技术+超额+估值+size",
    },
    "full": {
        "style": BASE_STYLE + VAL_LOG_COLS + FINANCIAL_COLS,
        "features": BASE_STYLE + VAL_LOG_COLS + FINANCIAL_COLS + ["size"],
        "needs_financial": True,
        "desc": "全部38列(技术+超额+估值+财务+size)",
    },
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
    pool = pool.dropna(subset=["target_rel", "excess_fwd"] + EXCESS_COLS)
    return pool.sort_values("date").reset_index(drop=True)


def apply_constituents(pool):
    """点-in-time 过滤:每个日期只保留"当时在沪深300成分里"的股票。"""
    cons = pd.read_csv(os.path.join(CACHE_DIR, "_constituents.csv"),
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
    print(f"  点-in-time 过滤: {before} -> {len(pool)} 条样本 "
          f"({len(pool)/before:.0%} 保留)", flush=True)
    return pool.sort_values("date").reset_index(drop=True)


def run_cross(pool, feature_cols):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    ic_list, group_excess, period_years, n_cross = [], {g: [] for g in range(N_GROUPS)}, [], []
    for t in rebal_dates:
        # embargo:排除测试日之前 EMBARGO_DAYS 日历天内的训练样本,消除标签重叠泄露
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
        period_years.append(t.year)
        n_cross.append(len(cross))
    return np.array(ic_list), group_excess, period_years, n_cross


def icir(ic):
    return ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")


def main():
    t0 = time.time()
    configs = sys.argv[1:] or list(CONFIGS.keys())
    pool = load_pool()
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本", flush=True)
    pool = apply_constituents(pool)

    header = f"{'配置':<16}{'Rank IC':>10}{'ICIR':>8}{'IC>0':>8}{'期数':>6}{'横截面均数':>10}"
    print(f"\n{header}", flush=True)
    lines = [header]
    for name in configs:
        cfg = CONFIGS[name]
        p = attach_fundamentals(pool.copy(), CACHE_DIR)
        if cfg["needs_financial"]:
            p = attach_financial(p, CACHE_DIR)
        p = neutralize(p, cfg["style"])
        ic, ge, years, nc = run_cross(p, cfg["features"])
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
    path = os.path.join(RESULTS_DIR, "stress_test_pit_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("点-in-time 成分股回测 (embargo 修正标签重叠泄露)\n")
        f.write("================================================\n\n")
        f.write(f"embargo {EMBARGO_DAYS} 日历天(≈30交易日), FORECAST_HORIZON {FORECAST_HORIZON}\n")
        f.write(f"OOS 起点 {TEST_START}, 每 {REBALANCE_EVERY} 交易日调仓\n")
        f.write("模型 num_leaves=63 / lr=0.10 / n_estimators=150\n\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""扫描持有期(标签 horizon)对纯多头净超额的影响,量化降换手省多少成本。

horizon 30→45→60 天:调仓频率成比例下降,年化成本减半,看净超额能否提升。
标签 = 未来 horizon 日是否跑赢自身指数;持有期 = 调仓期 = horizon(持仓不重叠)。

用法: python backtest_zz_scan.py [500|1000]
"""
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import RESULTS_DIR, SEED, EMBARGO_DAYS
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import neutralize
from fundamental_ts import attach_fundamentals_ts
from backtest_zz import load_index, apply_constituents, max_drawdown

warnings.filterwarnings("ignore")

UNIVERSES = {
    "500": ("000905.SH", "cache_zz500", 0.004),
    "1000": ("000852.SH", "cache_zz1000", 0.005),
}
TEST_START = "2017-01-01"
TOP_FRAC = 0.2
LIQUIDITY_DROP = 0.2
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]
EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
FEATURES = FEATURE_COLS + EXCESS_COLS
HORIZONS = [30, 45, 60]


def load_pool(cache_dir, horizon):
    idx = load_index(cache_dir)
    idx_ret = idx.pct_change()
    idx_fwd = idx.shift(-horizon) / idx - 1

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
    pool = pool.dropna(subset=["target_rel", "excess_fwd", "stock_fwd", "idx_fwd"] + EXCESS_COLS)

    db_path = os.path.join(cache_dir, "_daily_basic.csv")
    if os.path.exists(db_path):
        db = pd.read_csv(db_path, parse_dates=["date"], dtype={"code": str})
        db = db.sort_values(["code", "date"])
        db["turnover"] = db.groupby("code")["turnover_rate"].transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
        pool = pool.merge(db[["code", "date", "turnover"]], on=["code", "date"], how="left")

    pool = attach_fundamentals_ts(pool, cache_dir)
    pool = neutralize(pool, FEATURES)
    return pool.sort_values("date").reset_index(drop=True)


def annualize(nav, n_periods, horizon):
    return float(nav ** (252 / (n_periods * horizon)) - 1)


def sharpe(rets, horizon):
    if rets.std(ddof=1) <= 1e-12:
        return float("nan")
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(252 / horizon))


def run_long_only(pool, cost_per_turn, horizon):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::horizon]

    gross_list, net_list, idx_list, excess_list = [], [], [], []
    turn_list = []
    prev_hold = set()
    for t in rebal_dates:
        cutoff = t - pd.Timedelta(days=EMBARGO_DAYS)
        train = pool[pool["date"] < cutoff].dropna(subset=FEATURES)
        if len(train) < MIN_TRAIN:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.10, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(train[FEATURES].to_numpy(), train["target_rel"].to_numpy())
        cross = pool[pool["date"] == t].dropna(subset=FEATURES + ["stock_fwd", "idx_fwd", "excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        if "turnover" in cross.columns and cross["turnover"].notna().sum() > 10:
            thresh = cross["turnover"].quantile(LIQUIDITY_DROP)
            cross = cross[cross["turnover"] >= thresh]
        prob = model.predict_proba(cross[FEATURES].to_numpy())[:, 1]
        cross = cross.copy()
        cross["prob"] = prob
        k = max(1, int(len(cross) * TOP_FRAC))
        q5 = cross.nlargest(k, "prob")

        hold = set(q5["code"])
        turnover = len(hold - prev_hold) / len(hold) if prev_hold else 1.0
        prev_hold = hold

        gross = q5["stock_fwd"].mean()
        net = gross - turnover * cost_per_turn
        gross_list.append(gross)
        net_list.append(net)
        idx_list.append(q5["idx_fwd"].mean())
        excess_list.append(q5["excess_fwd"].mean())
        turn_list.append(turnover)

    return {
        "gross": np.array(gross_list), "net": np.array(net_list),
        "idx": np.array(idx_list), "excess": np.array(excess_list),
        "turnover": np.array(turn_list),
    }


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit("用法: python backtest_zz_scan.py [500|1000]")
    index_code, cache_dir, cost = UNIVERSES[universe]
    t0 = time.time()

    print(f"\n{'持有期':>8}{'期数':>6}{'换手率':>8}{'毛收益/期':>10}{'净收益/期':>10}"
          f"{'年化超额':>10}{'绝对年化':>10}{'回撤':>8}{'夏普':>8}", flush=True)
    lines = []
    for horizon in HORIZONS:
        pool = load_pool(cache_dir, horizon)
        pool = apply_constituents(pool, cache_dir)
        r = run_long_only(pool, cost, horizon)
        net, idx = r["net"], r["idx"]
        nav = float(np.prod(1 + net))
        idx_nav = float(np.prod(1 + idx))
        n = len(net)
        ann_ex = (nav / idx_nav) ** (252 / (n * horizon)) - 1
        ann_abs = annualize(nav, n, horizon)
        dd = max_drawdown(np.cumprod(1 + net))
        sh = sharpe(net, horizon)
        row = (f"{horizon:>6}天{n:>6}{r['turnover'].mean():>8.1%}{r['gross'].mean():>+10.3%}"
               f"{net.mean():>+10.3%}{ann_ex:>+10.2%}{ann_abs:>+10.2%}{dd:>8.2%}{sh:>8.3f}")
        print(row, flush=True)
        lines.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"backtest_zz{universe}_scan_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"中证{universe} 持有期扫描(降换手实验)\n")
        f.write(f"指数 {index_code}, 成本 {cost:.2%}/次换手, 点-in-time + embargo {EMBARGO_DAYS}天\n\n")
        f.write("持有期   期数  换手率  毛收益/期  净收益/期  年化超额  绝对年化  回撤   夏普\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

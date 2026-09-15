"""中证500/1000 纯多头 Q5 实盘回测(只买入,不做空),严谨版。

相比旧 backtest_long_only.py 修正两处关键偏差:
  1. 点-in-time 成分过滤(_constituents.csv,消除幸存者偏差)
  2. embargo(训练集排除测试日前 EMBARGO_DAYS 天,消除标签重叠泄露)

另加: 真实换手率算成本(相邻两期 Q5 组合重叠度),不做"每次 100% 换手"的乐观假设;
流动性过滤(剔除近20日均换手率最低 20%)。

持有期 = 调仓期 = FORECAST_HORIZON(30交易日),持仓不重叠,等权。

用法: python backtest_zz.py [500|1000]
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

from config import RESULTS_DIR, SEED, FORECAST_HORIZON, EMBARGO_DAYS
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import neutralize
from fundamental_ts import attach_fundamentals_ts

warnings.filterwarnings("ignore")

UNIVERSES = {
    "500": ("000905.SH", "cache_zz500", 0.004),   # 单次完整换手成本(含小盘股滑点)
    "1000": ("000852.SH", "cache_zz1000", 0.005),
}
TEST_START = "2017-01-01"
REBALANCE_EVERY = FORECAST_HORIZON
TOP_FRAC = 0.2
LIQUIDITY_DROP = 0.2
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
FEATURES = FEATURE_COLS + EXCESS_COLS  # 纯技术面(评估证实最优)


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
    pool = pool.dropna(subset=["target_rel", "excess_fwd", "stock_fwd", "idx_fwd"] + EXCESS_COLS)

    # 合并近20日均换手率(点-in-time 流动性)
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


def apply_constituents(pool, cache_dir):
    """点-in-time 过滤:每个日期只保留当时在成分里的股票。"""
    cons = pd.read_csv(os.path.join(cache_dir, "_constituents.csv"),
                       parse_dates=["snapshot_date"], dtype={"code": str})
    snap_dates = pd.to_datetime(sorted(cons["snapshot_date"].unique()))
    pool = pool.sort_values("date").copy()
    i = np.searchsorted(snap_dates.values, pool["date"].to_numpy(), side="right") - 1
    i = np.clip(i, 0, len(snap_dates) - 1)
    pool["snap_date"] = snap_dates.values[i]
    pool = pool.merge(cons, left_on=["snap_date", "code"],
                      right_on=["snapshot_date", "code"], how="inner")
    return pool.drop(columns=["snap_date", "snapshot_date"]).sort_values("date").reset_index(drop=True)


def max_drawdown(nav):
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1).min())


def annualize(nav, n_periods):
    return float(nav ** (252 / (n_periods * FORECAST_HORIZON)) - 1)


def sharpe(rets):
    if rets.std(ddof=1) <= 1e-12:
        return float("nan")
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(252 / FORECAST_HORIZON))


def run_long_only(pool, cost_per_turn):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    gross_list, net_list, idx_list, excess_list = [], [], [], []
    turn_list, n_hold = [], []
    rows = []
    prev_hold = set()
    for t in rebal_dates:
        cutoff = t - pd.Timedelta(days=EMBARGO_DAYS)  # embargo 防标签重叠
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
        # 流动性过滤:剔除换手率最低 20%
        if "turnover" in cross.columns and cross["turnover"].notna().sum() > 10:
            thresh = cross["turnover"].quantile(LIQUIDITY_DROP)
            cross = cross[cross["turnover"] >= thresh]
        prob = model.predict_proba(cross[FEATURES].to_numpy())[:, 1]
        cross = cross.copy()
        cross["prob"] = prob
        k = max(1, int(len(cross) * TOP_FRAC))
        q5 = cross.nlargest(k, "prob")

        hold = set(q5["code"])
        if prev_hold:
            turnover = len(hold - prev_hold) / len(hold)  # 新进入比例(换手率)
        else:
            turnover = 1.0
        prev_hold = hold

        gross = q5["stock_fwd"].mean()
        cost = turnover * cost_per_turn
        net = gross - cost
        idx_r = q5["idx_fwd"].mean()
        excess = q5["excess_fwd"].mean()

        gross_list.append(gross)
        net_list.append(net)
        idx_list.append(idx_r)
        excess_list.append(excess)
        turn_list.append(turnover)
        n_hold.append(len(q5))
        rows.append((t.strftime("%Y-%m-%d"), turnover, gross, net, idx_r, excess))

    return {
        "gross": np.array(gross_list), "net": np.array(net_list),
        "idx": np.array(idx_list), "excess": np.array(excess_list),
        "turnover": np.array(turn_list), "n_hold": n_hold, "rows": rows,
    }


def report(universe, index_code, r, cost_per_turn):
    net, idx = r["net"], r["idx"]
    nav = float(np.prod(1 + net))
    idx_nav = float(np.prod(1 + idx))
    n = len(net)
    ann = annualize(nav, n)
    ann_idx = annualize(idx_nav, n)
    ann_ex = (nav / idx_nav) ** (252 / (n * FORECAST_HORIZON)) - 1
    print(f"\n=== 中证{universe} 纯多头 Q5 回测(严谨版: 点-in-time + embargo + 真实换手成本) ===", flush=True)
    print(f"  指数 {index_code}, 调仓/持有 {FORECAST_HORIZON} 交易日, 期数 {n}", flush=True)
    print(f"  平均持仓 {np.mean(r['n_hold']):.0f} 只, 平均换手率 {r['turnover'].mean():.1%}/期", flush=True)
    print(f"  单次换手成本 {cost_per_turn:.2%}", flush=True)
    print(f"  Q5 毛收益(每期) : {r['gross'].mean():+.3%}", flush=True)
    print(f"  Q5 净收益(每期) : {net.mean():+.3%}  (扣换手成本 {np.mean(r['turnover'] * cost_per_turn):+.3%})", flush=True)
    print(f"  累计净值        : {nav:.4f}  年化 {ann:+.2%}", flush=True)
    print(f"  同期指数        : {idx_nav:.4f}  年化 {ann_idx:+.2%}", flush=True)
    print(f"  年化超额        : {ann_ex:+.2%}  (超额均值 {r['excess'].mean():+.3%}/期, 胜率 {(r['excess'] > 0).mean():.1%})", flush=True)
    print(f"  夏普(净)        : {sharpe(net):.3f}", flush=True)
    print(f"  最大回撤(净)    : {max_drawdown(np.cumprod(1 + net)):.2%}", flush=True)
    print(f"  盈利期占比      : {(net > 0).mean():.1%}", flush=True)
    return nav, idx_nav, ann, ann_idx, ann_ex


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit("用法: python backtest_zz.py [500|1000]")
    index_code, cache_dir, cost = UNIVERSES[universe]
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pool = load_pool(cache_dir)
    print(f"加载 {pool['code'].nunique()} 只, {len(pool)} 条样本", flush=True)
    pool = apply_constituents(pool, cache_dir)
    print(f"点-in-time 过滤后 {len(pool)} 条", flush=True)

    r = run_long_only(pool, cost)
    nav, idx_nav, ann, ann_idx, ann_ex = report(universe, index_code, r, cost)

    path = os.path.join(RESULTS_DIR, f"backtest_zz{universe}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"中证{universe} 纯多头 Q5 回测(严谨版)\n")
        f.write("========================================\n\n")
        f.write(f"指数 {index_code}, 调仓/持有 {FORECAST_HORIZON} 交易日, embargo {EMBARGO_DAYS} 天\n")
        f.write(f"点-in-time 成分过滤 + 流动性过滤(换手率最低 {LIQUIDITY_DROP:.0%})\n")
        f.write(f"单次换手成本 {cost:.2%}, 平均换手率 {r['turnover'].mean():.1%}/期\n\n")
        f.write(f"Q5 毛收益 {r['gross'].mean():+.3%}/期, 净收益 {r['net'].mean():+.3%}/期\n")
        f.write(f"累计净值 {nav:.4f} 年化 {ann:+.2%}\n")
        f.write(f"同期指数 {idx_nav:.4f} 年化 {ann_idx:+.2%}\n")
        f.write(f"年化超额 {ann_ex:+.2%} (胜率 {(r['excess'] > 0).mean():.1%})\n")
        f.write(f"夏普 {sharpe(r['net']):.3f}, 最大回撤 {max_drawdown(np.cumprod(1+r['net'])):.2%}\n\n")
        f.write("逐期(日期, 换手率, 毛收益, 净收益, 指数, 超额):\n")
        for date, turn, g, net, i, e in r["rows"]:
            f.write(f"  {date}  换手 {turn:.0%}  Q5毛 {g:+.3%}  Q5净 {net:+.3%}  指数 {i:+.3%}  超额 {e:+.3%}\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

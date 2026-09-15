"""Q5 做多 / Q1 做空 实盘回测(含交易成本)。

只挑概率最高 20%(Q5) 做多、最低 20%(Q1) 做空,中间 60% 没把握的跳过。
调仓间隔 = 持有期 = FORECAST_HORIZON(30 交易日),每期重训、换仓、扣成本。
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

from config import (
    CACHE_DIR, RESULTS_DIR, SEED, FORECAST_HORIZON,
    COMMISSION, STAMP_TAX, SLIPPAGE,
)
from features import build_features, FEATURE_COLS
from data import load_cached

TRAIN_START = "2021-01-01"
TEST_START = "2024-02-01"
REBALANCE_EVERY = FORECAST_HORIZON   # 调仓间隔 = 持有期,持仓不重叠
TOP_FRAC = 0.2                        # Q5/Q1 各占 20%
MIN_TRAIN = 5000
MIN_CROSS = 30

# 单边一次完整换手(卖旧+买新)成本
ROUND_TRIP = COMMISSION * 2 + STAMP_TAX + SLIPPAGE * 2


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


def max_drawdown(nav):
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1).min())


def annualize(nav, n_periods):
    """每期 FORECAST_HORIZON 交易日,年化 = 净值^(252/总交易日)-1。"""
    total_days = n_periods * FORECAST_HORIZON
    return float(nav ** (252 / total_days) - 1)


def sharpe(rets):
    """每期收益 -> 年化夏普。"""
    if rets.std(ddof=1) == 0:
        return float("nan")
    periods_per_year = 252 / FORECAST_HORIZON
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year))


def main():
    t0 = time.time()
    pool = load_panel()
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]
    print(f"加载 {pool['code'].nunique()} 只股票,调仓日 {len(rebal_dates)} 个,"
          f"持有期 {FORECAST_HORIZON} 交易日,成本 {ROUND_TRIP:.2%}/次换手", flush=True)

    ls_raw, ls_net, long_only_raw, long_only_net = [], [], [], []
    n_q5, n_q1 = [], []
    rows = []

    for t in rebal_dates:
        train = pool[pool["date"] < t]
        if len(train) < MIN_TRAIN:
            continue
        model = train_pooled(train)
        cross = pool[pool["date"] == t].dropna(subset=["fwd_ret"])
        if len(cross) < MIN_CROSS:
            continue
        prob = model.predict_proba(cross[FEATURE_COLS].to_numpy())[:, 1]
        df = cross[["code", "fwd_ret"]].copy()
        df["prob"] = prob
        k = max(1, int(len(df) * TOP_FRAC))
        q5 = df.nlargest(k, "prob")
        q1 = df.nsmallest(k, "prob")

        long_ret = q5["fwd_ret"].mean()
        short_ret = q1["fwd_ret"].mean()
        ls_raw.append(long_ret - short_ret)          # 未扣成本多空
        ls_net.append(long_ret - short_ret - 2 * ROUND_TRIP)  # 多空各换一次仓
        long_only_raw.append(long_ret)
        long_only_net.append(long_ret - ROUND_TRIP)  # 纯多头换一次仓
        n_q5.append(len(q5))
        n_q1.append(len(q1))
        rows.append((t.strftime("%Y-%m-%d"), long_ret, short_ret, long_ret - short_ret))

    ls_raw = np.array(ls_raw)
    ls_net = np.array(ls_net)
    long_only_raw = np.array(long_only_raw)
    long_only_net = np.array(long_only_net)
    n_periods = len(ls_net)

    def report(name, rets, nav):
        print(f"\n=== {name} (扣成本后) ===", flush=True)
        print(f"  调仓期数   : {len(rets)}", flush=True)
        print(f"  平均每期收益: {rets.mean():+.3%}", flush=True)
        print(f"  累计净值   : {nav:.4f}", flush=True)
        print(f"  年化收益   : {annualize(nav, len(rets)):+.2%}", flush=True)
        print(f"  夏普比率   : {sharpe(rets):.3f}", flush=True)
        print(f"  最大回撤   : {max_drawdown(np.cumprod(1 + rets)):.2%}", flush=True)
        print(f"  胜率       : {(rets > 0).mean():.1%}", flush=True)
        win = rets[rets > 0]
        loss = rets[rets <= 0]
        pl = win.mean() / abs(loss.mean()) if len(loss) and loss.mean() != 0 else float("nan")
        print(f"  盈亏比     : {pl:.2f}", flush=True)
        return nav

    print(f"\n每期平均持仓: 多头 Q5 {int(np.mean(n_q5))} 只 / 空头 Q1 {int(np.mean(n_q1))} 只", flush=True)

    print(f"\n========== 未扣成本基准 ==========", flush=True)
    print(f"  多空价差平均 : {ls_raw.mean():+.3%}/期  (胜率 {(ls_raw > 0).mean():.1%})", flush=True)
    print(f"  纯多头 Q5    : {long_only_raw.mean():+.3%}/期", flush=True)

    nav_ls = report("多空组合(Q5多 / Q1空)", ls_net, float(np.prod(1 + ls_net)))
    report("纯多头(只做多 Q5)", long_only_net, float(np.prod(1 + long_only_net)))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "backtest_ls_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("==============================================================\n")
        f.write("Q5 做多 / Q1 做空 实盘回测报告\n")
        f.write("==============================================================\n\n")
        f.write(f"预测目标: 未来 {FORECAST_HORIZON} 交易日涨跌方向\n")
        f.write(f"调仓间隔: {REBALANCE_EVERY} 交易日(持仓不重叠)\n")
        f.write(f"分组比例: Q5/Q1 各 {TOP_FRAC:.0%}\n")
        f.write(f"单次换手成本: {ROUND_TRIP:.3%}\n\n")
        f.write(f"未扣成本多空价差: {ls_raw.mean():+.3%}/期 (胜率 {(ls_raw > 0).mean():.1%})\n")
        f.write(f"扣成本后多空    : {ls_net.mean():+.3%}/期\n")
        f.write(f"多空累计净值    : {nav_ls:.4f}\n")
        f.write(f"多空年化        : {annualize(nav_ls, n_periods):+.2%}\n")
        f.write(f"多空夏普        : {sharpe(ls_net):.3f}\n")
        f.write(f"多空最大回撤    : {max_drawdown(np.cumprod(1 + ls_net)):.2%}\n\n")
        f.write("逐期明细(日期, Q5收益, Q1收益, 多空价差):\n")
        for date, lr, sr, diff in rows:
            f.write(f"  {date}  Q5 {lr:+.3%}  Q1 {sr:+.3%}  多空 {diff:+.3%}\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

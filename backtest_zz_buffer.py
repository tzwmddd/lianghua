"""换手缓冲(buffer)实验:持有期固定 60 天,扫描缓冲带宽度对换手率/净超额的影响。

缓冲逻辑:调仓日不强制卖光旧持仓,只卖出"掉出 top BUFFER_FRAC"的旧股,
买入新进入 top 20% 的新股填满仓位。BUFFER_FRAC 越大,保留越多,换手越低。

BUFFER_FRAC = 0.2 时等价于无缓冲(每期强制换到新 Q5)。

用法: python backtest_zz_buffer.py [500|1000]
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import RESULTS_DIR, SEED, EMBARGO_DAYS
from backtest_zz_scan import load_pool, apply_constituents, max_drawdown, annualize, sharpe, FEATURES

warnings.filterwarnings("ignore")

UNIVERSES = {
    "500": ("000905.SH", "cache_zz500", 0.004),
    "1000": ("000852.SH", "cache_zz1000", 0.005),
}
TEST_START = "2017-01-01"
HORIZON = 60
TOP_FRAC = 0.2
LIQUIDITY_DROP = 0.2
MIN_TRAIN = 5000
MIN_CROSS = 30
BUFFER_FRACS = [0.2, 0.3, 0.4, 0.5]


def run_buffered(pool, cost_per_turn, buffer_frac):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::HORIZON]

    net_list, excess_list, turn_list = [], [], []
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
        cross = cross.sort_values("prob", ascending=False)

        k = max(1, int(len(cross) * TOP_FRAC))
        buffer_k = max(1, int(len(cross) * buffer_frac))
        buffer_codes = set(cross.head(buffer_k)["code"])
        top_codes = list(cross.head(k)["code"])

        # 保留旧持仓中仍在缓冲带内的;新买入 top20% 填满 k 只
        keep = [c for c in prev_hold if c in buffer_codes]
        final = keep[:]
        for c in top_codes:
            if c not in final and len(final) < k:
                final.append(c)
        final = final[:k]
        hold = set(final)

        turnover = len(hold - prev_hold) / len(hold) if prev_hold else 1.0
        prev_hold = hold

        held = cross[cross["code"].isin(hold)]
        gross = held["stock_fwd"].mean()
        net = gross - turnover * cost_per_turn
        net_list.append(net)
        excess_list.append(held["excess_fwd"].mean())
        turn_list.append(turnover)

    return np.array(net_list), np.array(excess_list), np.array(turn_list)


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit("用法: python backtest_zz_buffer.py [500|1000]")
    index_code, cache_dir, cost = UNIVERSES[universe]
    t0 = time.time()

    pool = load_pool(cache_dir, HORIZON)
    pool = apply_constituents(pool, cache_dir)
    print(f"中证{universe} 持有期 {HORIZON} 天, 加载 {len(pool)} 条", flush=True)

    print(f"\n{'缓冲带':>8}{'换手率':>8}{'净收益/期':>10}{'年化超额':>10}"
          f"{'绝对年化':>10}{'回撤':>8}{'夏普':>8}", flush=True)
    lines = []
    for bf in BUFFER_FRACS:
        net, excess, turn = run_buffered(pool, cost, bf)
        nav = float(np.prod(1 + net))
        n = len(net)
        ann_abs = annualize(nav, n, HORIZON)
        # 指数年化近似:用净收益减去超额还原,但这里只报超额均值对应的年化
        ann_ex = (excess.mean()) * (252 / HORIZON)  # 超额均值年化(近似)
        dd = max_drawdown(np.cumprod(1 + net))
        sh = sharpe(net, HORIZON)
        row = (f"{bf:>7.0%}{turn.mean():>8.1%}{net.mean():>+10.3%}"
               f"{ann_ex:>+10.2%}{ann_abs:>+10.2%}{dd:>8.2%}{sh:>8.3f}")
        print(row, flush=True)
        lines.append(row)

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"backtest_zz{universe}_buffer_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"中证{universe} 换手缓冲实验(持有期 {HORIZON} 天)\n")
        f.write(f"指数 {index_code}, 成本 {cost:.2%}/次换手, embargo {EMBARGO_DAYS}天\n\n")
        f.write("缓冲带   换手率  净收益/期  年化超额(近似)  绝对年化  回撤   夏普\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""纯多头 Q5 实盘回测(只买入,不做空),对比默认参数 vs 欠拟合优化参数。

口径与推荐工具一致:相对标签(未来30日是否跑赢沪深300)+ 完整特征(技术+超额+
估值+财务+size)+ 横截面中性化。每 FORECAST_HORIZON(30交易日)调仓,持仓不重叠,
只买"跑赢概率"最高的20%(Q5),等权持有,扣换手成本。

输出纯多头 Q5 的绝对收益、相对沪深300超额、年化/夏普/回撤/胜率,
并对比两组模型参数:
  A 默认   : num_leaves=31, learning_rate=0.05
  B 优化   : num_leaves=63, learning_rate=0.10  (敏感性扫描提示欠拟合,加大容量)
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

from config import CACHE_DIR, RESULTS_DIR, SEED, FORECAST_HORIZON, ROUND_TRIP_COST
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import attach_fundamentals, neutralize, VALUATION_COLS
from fundamental import attach_financial, FINANCIAL_COLS

warnings.filterwarnings("ignore")

TEST_START = "2024-02-01"
REBALANCE_EVERY = FORECAST_HORIZON   # 调仓间隔 = 持有期,持仓不重叠
TOP_FRAC = 0.2
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
STYLE_COLS = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS
FEATURES = STYLE_COLS + ["size"]

# 对比的两组参数
PARAM_GRID = [
    ("A 默认(num_leaves=31, lr=0.05)", 31, 0.05),
    ("B 优化(num_leaves=63, lr=0.10)", 63, 0.10),
]


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
    pool = pool.dropna(subset=["target_rel", "excess_fwd", "stock_fwd", "idx_fwd"] + EXCESS_COLS)
    pool = pool.sort_values("date").reset_index(drop=True)

    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    pool = neutralize(pool, STYLE_COLS)
    return pool


def max_drawdown(nav):
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1).min())


def annualize(nav, n_periods):
    total_days = n_periods * FORECAST_HORIZON
    return float(nav ** (252 / total_days) - 1)


def sharpe(rets):
    if rets.std(ddof=1) <= 1e-12:
        return float("nan")
    periods_per_year = 252 / FORECAST_HORIZON
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year))


def run_long_only(pool, num_leaves, learning_rate):
    """扩张窗口 pooled 训练,每期选 Q5 等权买入,返回逐期收益序列与净值。"""
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    q5_gross, q5_net, idx_ret_list, excess_list = [], [], [], []
    n_hold = []
    rows = []
    for t in rebal_dates:
        train = pool[pool["date"] < t].dropna(subset=FEATURES)
        if len(train) < MIN_TRAIN:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=learning_rate, num_leaves=num_leaves,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(train[FEATURES].to_numpy(), train["target_rel"].to_numpy())
        cross = pool[pool["date"] == t].dropna(subset=FEATURES + ["stock_fwd", "idx_fwd", "excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        prob = model.predict_proba(cross[FEATURES].to_numpy())[:, 1]
        df = cross[["code", "stock_fwd", "idx_fwd", "excess_fwd"]].copy()
        df["prob"] = prob
        k = max(1, int(len(df) * TOP_FRAC))
        q5 = df.nlargest(k, "prob")

        gross = q5["stock_fwd"].mean()              # Q5 绝对收益(未扣成本)
        net = gross - ROUND_TRIP_COST               # 扣一次完整换手成本
        idx_r = q5["idx_fwd"].mean()                # 同期沪深300
        excess = q5["excess_fwd"].mean()            # 超额收益

        q5_gross.append(gross)
        q5_net.append(net)
        idx_ret_list.append(idx_r)
        excess_list.append(excess)
        n_hold.append(len(q5))
        rows.append((t.strftime("%Y-%m-%d"), gross, net, idx_r, excess))

    return {
        "gross": np.array(q5_gross),
        "net": np.array(q5_net),
        "idx": np.array(idx_ret_list),
        "excess": np.array(excess_list),
        "n_hold": n_hold,
        "rows": rows,
    }


def report(name, r):
    net = r["net"]
    nav = float(np.prod(1 + net))
    idx_nav = float(np.prod(1 + r["idx"]))
    print(f"\n=== {name} (纯多头 Q5,扣成本后) ===", flush=True)
    print(f"  持仓数(每期) : {int(np.mean(r['n_hold']))} 只", flush=True)
    print(f"  调仓期数     : {len(net)}", flush=True)
    print(f"  平均每期收益 : {net.mean():+.3%}", flush=True)
    print(f"  累计净值     : {nav:.4f}   (年化 {annualize(nav, len(net)):+.2%})", flush=True)
    print(f"  同期沪深300  : {idx_nav:.4f}   (年化 {annualize(idx_nav, len(net)):+.2%})", flush=True)
    print(f"  累计超额     : {nav - idx_nav:+.2%}  (年化超额 {(nav/idx_nav)**(252/(len(net)*FORECAST_HORIZON))-1:+.2%})", flush=True)
    print(f"  超额均值     : {r['excess'].mean():+.3%}/期  (胜率 {(r['excess'] > 0).mean():.1%})", flush=True)
    print(f"  夏普比率     : {sharpe(net):.3f}", flush=True)
    print(f"  最大回撤     : {max_drawdown(np.cumprod(1 + net)):.2%}", flush=True)
    print(f"  盈利期占比   : {(net > 0).mean():.1%}", flush=True)
    win = net[net > 0]
    loss = net[net <= 0]
    pl = win.mean() / abs(loss.mean()) if len(loss) and loss.mean() != 0 else float("nan")
    print(f"  盈亏比       : {pl:.2f}", flush=True)
    return nav, idx_nav


def main():
    t0 = time.time()
    pool = load_pool()
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本;"
          f"持有期 {FORECAST_HORIZON} 交易日,调仓同周期(持仓不重叠),成本 {ROUND_TRIP_COST:.2%}/次",
          flush=True)

    results = {}
    for name, nl, lr in PARAM_GRID:
        print(f"\n>>>>>> {name} <<<<<<", flush=True)
        r = run_long_only(pool, nl, lr)
        results[name] = r
        report(name, r)

    # 汇总对比
    print("\n========== 两组对比(纯多头 Q5,扣成本) ==========", flush=True)
    for name, r in results.items():
        nav = float(np.prod(1 + r["net"]))
        idx_nav = float(np.prod(1 + r["idx"]))
        ann = annualize(nav, len(r["net"]))
        ann_idx = annualize(idx_nav, len(r["net"]))
        ann_ex = (nav / idx_nav) ** (252 / (len(r["net"]) * FORECAST_HORIZON)) - 1
        print(f"  {name:<32} 年化 {ann:+6.2%}  沪深300 {ann_idx:+6.2%}  "
              f"年化超额 {ann_ex:+6.2%}  回撤 {max_drawdown(np.cumprod(1+r['net'])):6.2%}",
              flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "backtest_long_only_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("==============================================================\n")
        f.write("纯多头 Q5 实盘回测报告(只买入,不做空)\n")
        f.write("==============================================================\n\n")
        f.write(f"预测目标: 未来 {FORECAST_HORIZON} 交易日是否跑赢沪深300\n")
        f.write(f"调仓间隔: {REBALANCE_EVERY} 交易日(持仓不重叠)\n")
        f.write(f"分组比例: Q5 前 {TOP_FRAC:.0%}\n")
        f.write(f"单次换手成本: {ROUND_TRIP_COST:.3%}\n")
        f.write(f"特征数: {len(FEATURES)}(技术+超额+估值+财务+size)\n\n")
        for name, r in results.items():
            nav = float(np.prod(1 + r["net"]))
            idx_nav = float(np.prod(1 + r["idx"]))
            ann = annualize(nav, len(r["net"]))
            ann_ex = (nav / idx_nav) ** (252 / (len(r["net"]) * FORECAST_HORIZON)) - 1
            f.write(f"[{name}]\n")
            f.write(f"  累计净值 {nav:.4f}  年化 {ann:+.2%}\n")
            f.write(f"  同期沪深300净值 {idx_nav:.4f}  年化 {annualize(idx_nav, len(r['net'])):+.2%}\n")
            f.write(f"  年化超额 {ann_ex:+.2%}  超额均值 {r['excess'].mean():+.3%}/期 "
                    f"(胜率 {(r['excess'] > 0).mean():.1%})\n")
            f.write(f"  夏普 {sharpe(r['net']):.3f}  最大回撤 {max_drawdown(np.cumprod(1+r['net'])):.2%} "
                    f"  盈利期占比 {(r['net'] > 0).mean():.1%}\n\n")
        f.write("逐期明细(日期, Q5毛收益, Q5净收益, 沪深300, 超额):\n")
        for date, g, n, i, e in results[list(results.keys())[0]]["rows"]:
            f.write(f"  {date}  Q5毛 {g:+.3%}  Q5净 {n:+.3%}  沪深300 {i:+.3%}  超额 {e:+.3%}\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""市场状态压测:把 OOS 窗口拉到 2023 下跌年,看信号是否"牛市幻觉"。

与生产口径 evaluate_financial2.py 的 B 组完全一致:
相对标签(未来30日是否跑赢沪深300) + 相对 Rank IC + 扩张窗口 + 中性化,
特征 = 22技术 + 3超额 + 3估值(log) + 9财务 + size,模型 num_leaves=63 / lr=0.10。

区别:TEST_START 提前到 2023-02(原 2024-02),让 2023 下跌年进入样本外;
结果按年拆分 Rank IC / 多空,并附沪深300年度收益作为市场状态标签。
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

from config import CACHE_DIR, RESULTS_DIR, SEED, FORECAST_HORIZON
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import attach_fundamentals, neutralize, VALUATION_COLS
from fundamental import attach_financial, FINANCIAL_COLS

TEST_START = "2017-01-01"    # 拉长历史后提前到 2017,覆盖 2018 熊市/2016 熔断
REBALANCE_EVERY = 20
N_GROUPS = 5
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
BASE_STYLE = FEATURE_COLS + EXCESS_COLS
ALL_STYLE = BASE_STYLE + VAL_LOG_COLS + FINANCIAL_COLS
FEATURES = ALL_STYLE + ["size"]


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


def index_annual_returns():
    idx = load_index()
    year_end = idx.groupby(idx.index.year).last()
    ann = year_end / year_end.shift(1) - 1
    return {int(y): float(r) for y, r in ann.items()}


def run_cross(pool, feature_cols):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    ic_list, group_excess, period_years, train_years = [], {g: [] for g in range(N_GROUPS)}, [], []
    for t in rebal_dates:
        train = pool[pool["date"] < t].dropna(subset=feature_cols)
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
        ic = df["prob"].corr(df["excess_fwd"], method="spearman")
        ic_list.append(ic)
        df["rank"] = df["prob"].rank(method="first", pct=True)
        df["group"] = np.ceil(df["rank"] * N_GROUPS).astype(int) - 1
        for g in range(N_GROUPS):
            group_excess[g].append(df.loc[df["group"] == g, "excess_fwd"].mean())
        period_years.append(t.year)
        # 训练窗口长度(年),用于标记"短训练 vs 长训练"的混淆
        train_years.append((train["date"].max() - train["date"].min()).days / 365.25)
    return np.array(ic_list), group_excess, period_years, train_years


def icir(ic):
    return ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")


def main():
    t0 = time.time()
    pool = load_pool()
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本 "
          f"({pool['date'].min().date()} ~ {pool['date'].max().date()})", flush=True)

    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    pool = neutralize(pool, ALL_STYLE)
    print(f"中性化完成(风格因子 {len(ALL_STYLE)} 个 + size),特征 {len(FEATURES)} 列", flush=True)

    ic, group_excess, period_years, train_years = run_cross(pool, FEATURES)
    ann = index_annual_returns()

    years = sorted(set(period_years))
    print(f"\nOOS 调仓期 {len(ic)} 个: {TEST_START} 起", flush=True)
    print(f"\n{'年份':<6}{'市场(沪深300)':>14}{'Rank IC':>10}{'IC>0占比':>10}"
          f"{'Q5超额':>10}{'Q1超额':>10}{'多空Q5-Q1':>12}{'训练窗(年)':>10}", flush=True)
    for yr in years:
        m = np.array([period_years[i] == yr for i in range(len(ic))])
        ic_y = ic[m]
        ge = {g: np.array([group_excess[g][i] for i in range(len(ic)) if period_years[i] == yr])
              for g in range(N_GROUPS)}
        q5 = ge[N_GROUPS - 1].mean()
        q1 = ge[0].mean()
        tr = np.array([train_years[i] for i in range(len(ic)) if period_years[i] == yr]).mean()
        mkt = ann.get(yr, float("nan"))
        print(f"{yr:<6}{mkt:>13.1%}{ic_y.mean():>+10.4f}{(ic_y > 0).mean():>9.1%}"
              f"{q5:>+9.2%}{q1:>+9.2%}{(q5 - q1):>+11.2%}{tr:>9.1f}", flush=True)

    print(f"\n{'合计':<6}{'':>14}{ic.mean():>+10.4f}{(ic > 0).mean():>9.1%}"
          f"{'':>10}{'':>10}{'':>12}", flush=True)
    print(f"  整体 ICIR {icir(ic):+.3f}", flush=True)

    # 明确对比:下跌年 vs 上涨年
    down = [2022, 2023]
    up = [2024, 2025, 2026]
    for label, yrs in [("下跌年", down), ("上涨年", up)]:
        m = np.array([y in yrs for y in period_years])
        if m.sum() == 0:
            continue
        ic_y = ic[m]
        ge = {g: np.array([group_excess[g][i] for i in range(len(ic)) if period_years[i] in yrs])
              for g in range(N_GROUPS)}
        ls = (ge[N_GROUPS - 1] - ge[0]).mean()
        print(f"  {label} {yrs}: Rank IC {ic_y.mean():+.4f}  ICIR {icir(ic_y):+.3f}  "
              f"多空 {ls:+.3%}/期  ({int(m.sum())}期)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "stress_test_regime_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("==============================================================\n")
        f.write("市场状态压测报告 (相对标签 + 相对 Rank IC, 生产口径)\n")
        f.write("==============================================================\n\n")
        f.write(f"预测目标 : 未来 {FORECAST_HORIZON} 日是否跑赢沪深300\n")
        f.write(f"OOS 起点 : {TEST_START} (提前到 2023,让下跌年进样本外)\n")
        f.write(f"调仓间隔 : 每 {REBALANCE_EVERY} 交易日\n")
        f.write(f"特征     : {len(FEATURES)} 列 (技术+超额+估值+财务+size)\n")
        f.write(f"模型     : num_leaves=63, lr=0.10\n\n")
        f.write(f"整体 Rank IC {ic.mean():+.4f}  ICIR {icir(ic):+.3f}  "
                f"IC>0 {(ic > 0).mean():.1%}\n\n")
        f.write(f"{'年份':<6}{'市场':>10}{'RankIC':>10}{'IC>0':>8}{'Q5超额':>10}{'Q1超额':>10}"
                f"{'多空':>10}{'训练年':>8}\n")
        for yr in years:
            m = np.array([period_years[i] == yr for i in range(len(ic))])
            ic_y = ic[m]
            ge = {g: np.array([group_excess[g][i] for i in range(len(ic)) if period_years[i] == yr])
                  for g in range(N_GROUPS)}
            q5 = ge[N_GROUPS - 1].mean()
            q1 = ge[0].mean()
            tr = np.array([train_years[i] for i in range(len(ic)) if period_years[i] == yr]).mean()
            f.write(f"{yr:<6}{ann.get(yr, 0):>9.1%}{ic_y.mean():>+10.4f}{(ic_y > 0).mean():>7.1%}"
                    f"{q5:>+9.2%}{q1:>+9.2%}{(q5 - q1):>+9.2%}{tr:>7.1f}\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

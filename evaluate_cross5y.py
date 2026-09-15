"""5年扩张窗口横截面评估:pooled 训练 + Rank IC + 分层回测。

对比旧的"单标的1年训练"(Rank IC=-0.069 反向指标),验证 5 年 pooled 扩张训练
能否把横截面信号转正、多空组合能否赚钱。
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
from features import build_features, FEATURE_COLS
from data import load_cached
from cross_section import attach_fundamentals, neutralize

TRAIN_START = "2021-01-01"
TEST_START = "2024-02-01"
REBALANCE_EVERY = 20       # 调仓间隔(交易日),约每月一次
N_GROUPS = 5
MIN_TRAIN = 5000
MIN_CROSS = 30

FEATURES = FEATURE_COLS + ["size"]


def load_panel():
    """全池特征面板,含未来5日连续收益 fwd_ret(用于 Rank IC)。"""
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
    model.fit(train[FEATURES].to_numpy(), train["target"].to_numpy())
    return model


def summarize(ic_list, group_returns, n_cross, n_rebal):
    ic = np.array(ic_list)
    avg_n = float(np.mean(n_cross))
    print(f"\n有效调仓期 {len(ic)} / {n_rebal},每期平均 {avg_n:.0f} 只股票")
    print("\n=== Rank IC (预测概率 vs 未来收益 秩相关) ===")
    print(f"  平均 Rank IC : {ic.mean():+.4f}")
    print(f"  IC 标准差    : {ic.std(ddof=1):.4f}")
    icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")
    print(f"  ICIR (信息比率): {icir:+.3f}")
    print(f"  IC>0 占比    : {(ic > 0).mean():.1%}")

    print("\n=== 分层回测 (按概率分5组, Q5=概率最高) ===")
    print(f"{'分组':<6}{'平均单期收益':>14}{'累计净值':>12}")
    for g in range(N_GROUPS - 1, -1, -1):
        r = np.array(group_returns[g])
        cum = float(np.prod(1.0 + r))
        print(f"  Q{g + 1}   {r.mean():>+12.2%}   {cum:>10.3f}")

    top = np.array(group_returns[N_GROUPS - 1])
    bot = np.array(group_returns[0])
    ls = top - bot
    print(f"\n  多空(Q{N_GROUPS}-Q1) 平均单期收益: {ls.mean():+.2%}")
    print(f"  多空胜率(>0 占比): {(ls > 0).mean():.1%}")

    return {
        "avg_rank_ic": ic.mean(),
        "ic_std": ic.std(ddof=1),
        "icir": icir,
        "ic_positive_ratio": (ic > 0).mean(),
        "group_returns": {f"Q{g + 1}": float(np.array(group_returns[g]).mean()) for g in range(N_GROUPS)},
        "long_short": float(ls.mean()),
        "ls_win_ratio": float((ls > 0).mean()),
        "n_periods": len(ic),
    }


def main():
    t0 = time.time()
    pool = load_panel()
    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = neutralize(pool, FEATURE_COLS)
    n_stocks = pool["code"].nunique()
    print(f"加载 {n_stocks} 只股票,{len(pool)} 条样本", flush=True)

    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]
    print(f"调仓日 {len(rebal_dates)} 个: {rebal_dates[0].date()} ~ {rebal_dates[-1].date()}", flush=True)

    ic_list, group_returns, n_cross = [], {g: [] for g in range(N_GROUPS)}, []
    for t in rebal_dates:
        train = pool[pool["date"] < t]
        if len(train) < MIN_TRAIN:
            continue
        model = train_pooled(train)
        cross = pool[pool["date"] == t].dropna(subset=["fwd_ret"])
        if len(cross) < MIN_CROSS:
            continue
        prob = model.predict_proba(cross[FEATURES].to_numpy())[:, 1]
        df = pd.DataFrame({"prob": prob, "fwd_ret": cross["fwd_ret"].to_numpy()})
        ic = df["prob"].corr(df["fwd_ret"], method="spearman")
        ic_list.append(ic)
        df["rank"] = df["prob"].rank(method="first", pct=True)
        df["group"] = np.ceil(df["rank"] * N_GROUPS).astype(int) - 1
        for g in range(N_GROUPS):
            group_returns[g].append(df.loc[df["group"] == g, "fwd_ret"].mean())
        n_cross.append(len(df))

    result = summarize(ic_list, group_returns, n_cross, len(rebal_dates))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "evaluate_cross5y_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("==============================================================\n")
        f.write("5年扩张窗口横截面评估 (Rank IC + 分层回测)\n")
        f.write("==============================================================\n\n")
        f.write(f"训练方式 : pooled 全池扩张窗口,{TRAIN_START} 起\n")
        f.write(f"测试区间 : {rebal_dates[0].date()} ~ {rebal_dates[-1].date()} "
                f"({result['n_periods']} 期)\n")
        f.write(f"预测目标 : 未来 {FORECAST_HORIZON} 日涨跌方向\n\n")
        f.write(f"平均 Rank IC     : {result['avg_rank_ic']:+.4f}\n")
        f.write(f"IC 标准差        : {result['ic_std']:.4f}\n")
        f.write(f"ICIR (信息比率)  : {result['icir']:+.3f}\n")
        f.write(f"IC>0 占比        : {result['ic_positive_ratio']:.1%}\n\n")
        f.write("分层平均单期收益:\n")
        for g in range(N_GROUPS - 1, -1, -1):
            f.write(f"  Q{g + 1}  {result['group_returns'][f'Q{g + 1}']:+.2%}\n")
        f.write(f"\n多空(Q{N_GROUPS}-Q1): {result['long_short']:+.2%}  "
                f"(胜率 {result['ls_win_ratio']:.1%})\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

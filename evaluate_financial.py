"""财务因子 A/B 评估:对比 加/不加 ROE等季度财务因子 的横截面选股信号质量。

口径与 evaluate_fundamental.py 对齐:相对标签(未来30日是否跑赢沪深300)+ 相对 Rank IC。
A 组 = 22技术 + 3超额 + size + 3估值(现状,无财务因子)
B 组 = A 组 + roe/np_growth/rev_growth/debt_ratio/gross_margin(财务质量/成长/杠杆)

财务因子为季度报告期数据,按"报告期 + 公布滞后"点-in-time 对齐(见 attach_financial),
避免用未来财报预测过去(未来数据泄露)。
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

TEST_START = "2024-02-01"
REBALANCE_EVERY = 20
N_GROUPS = 5
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
BASE_STYLE = FEATURE_COLS + EXCESS_COLS


def load_index():
    idx = pd.read_csv(os.path.join(CACHE_DIR, "sh000300.csv"), parse_dates=["date"])
    return idx.set_index("date")["close"].sort_index()


def load_pool():
    """个股面板:22技术 + 3超额特征 + 相对标签(是否跑赢沪深300)。"""
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


def run_cross(pool, feature_cols):
    """扩张窗口 pooled 训练 + 横截面评估。返回 Rank IC 序列与分层超额收益。"""
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    ic_list, group_excess = [], {g: [] for g in range(N_GROUPS)}
    for t in rebal_dates:
        train = pool[pool["date"] < t].dropna(subset=feature_cols)
        if len(train) < MIN_TRAIN:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
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
    return np.array(ic_list), group_excess


def summarize(name, ic, group_excess):
    icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")
    print(f"\n=== {name} ===", flush=True)
    print(f"  相对 Rank IC: {ic.mean():+.4f}  ICIR {icir:+.3f}  "
          f"IC>0占比 {(ic > 0).mean():.1%}", flush=True)
    print(f"  分层超额收益(个股-沪深300):", flush=True)
    for g in range(N_GROUPS - 1, -1, -1):
        r = np.array(group_excess[g])
        print(f"    Q{g + 1}: {r.mean():+.3%}/期", flush=True)
    top = np.array(group_excess[N_GROUPS - 1])
    bot = np.array(group_excess[0])
    print(f"  多空 Q5-Q1 超额: {(top - bot).mean():+.3%}/期", flush=True)
    return {"ic": float(ic.mean()), "icir": float(icir),
            "ic_pos": float((ic > 0).mean()), "ls": float((top - bot).mean())}


def main():
    t0 = time.time()
    pool = load_pool()
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本", flush=True)

    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    all_style = BASE_STYLE + VAL_LOG_COLS + FINANCIAL_COLS
    pool = neutralize(pool, all_style)
    print(f"中性化完成(风格因子 {len(all_style)} 个 + size)", flush=True)

    feats_a = BASE_STYLE + VAL_LOG_COLS + ["size"]
    feats_b = BASE_STYLE + VAL_LOG_COLS + FINANCIAL_COLS + ["size"]

    print(f"\n>>>>>> A 组: 基线 {len(feats_a)} 特征(技术+超额+估值+size)", flush=True)
    ic_a, ge_a = run_cross(pool, feats_a)
    ra = summarize("A 组(无财务因子)", ic_a, ge_a)

    print(f"\n>>>>>> B 组: +财务因子 {len(feats_b)} 特征", flush=True)
    ic_b, ge_b = run_cross(pool, feats_b)
    rb = summarize("B 组(+roe/np_growth/rev_growth/debt_ratio/gross_margin)", ic_b, ge_b)

    print(f"\n===== 结论 =====", flush=True)
    print(f"  Rank IC : {ra['ic']:+.4f} -> {rb['ic']:+.4f}", flush=True)
    print(f"  ICIR    : {ra['icir']:+.3f} -> {rb['icir']:+.3f}", flush=True)
    print(f"  IC>0占比: {ra['ic_pos']:.1%} -> {rb['ic_pos']:.1%}", flush=True)
    print(f"  多空超额: {ra['ls']:+.3%} -> {rb['ls']:+.3%}/期", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "evaluate_financial_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("==============================================================\n")
        f.write("财务因子 A/B 评估 (相对标签 + 相对 Rank IC)\n")
        f.write("==============================================================\n\n")
        f.write(f"预测目标 : 未来 {FORECAST_HORIZON} 日是否跑赢沪深300\n")
        f.write(f"测试区间 : {TEST_START} 起,每 {REBALANCE_EVERY} 交易日调仓\n")
        f.write(f"财务因子 : {', '.join(FINANCIAL_COLS)} (季度,点-in-time 对齐)\n\n")
        f.write(f"A 组(基线 {len(feats_a)} 特征): "
                f"Rank IC {ra['ic']:+.4f}  ICIR {ra['icir']:+.3f}  "
                f"IC>0 {ra['ic_pos']:.1%}  多空 {ra['ls']:+.3%}\n")
        f.write(f"B 组(+财务 {len(feats_b)} 特征): "
                f"Rank IC {rb['ic']:+.4f}  ICIR {rb['icir']:+.3f}  "
                f"IC>0 {rb['ic_pos']:.1%}  多空 {rb['ls']:+.3%}\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

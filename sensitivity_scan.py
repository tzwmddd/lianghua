"""参数敏感性扫描(坑5:过拟合防护)。

对关键超参做单变量网格扫描,输出 Rank IC/ICIR 在参数范围内的分布,
证明选股信号是"在合理参数范围内都稳健",而非"只在一个甜点才有效"的单点过拟合。

扫描维度(固定其他为默认值,单变量变化):
  - 预测周期 horizon:  [20, 30, 60]
  - 树深度 num_leaves: [15, 31, 63]
  - 学习率 lr:         [0.03, 0.05, 0.10]

口径与 evaluate_financial.py 对齐:相对标签 + 相对 Rank IC + 扩张窗口。
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

from config import CACHE_DIR, RESULTS_DIR, SEED
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

DEFAULTS = {"horizon": 30, "num_leaves": 31, "learning_rate": 0.05}

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
STYLE_COLS = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS
FEATURE_COLS_FULL = STYLE_COLS + ["size"]


def load_index():
    idx = pd.read_csv(os.path.join(CACHE_DIR, "sh000300.csv"), parse_dates=["date"])
    return idx.set_index("date")["close"].sort_index()


def build_pool(horizon):
    """构建并中性化面板:技术+超额+估值+财务+size,标签为未来 horizon 日是否跑赢沪深300。"""
    idx = load_index()
    idx_ret = idx.pct_change()
    idx_fwd = idx.shift(-horizon) / idx - 1

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
    pool = pool.dropna(subset=["target_rel", "excess_fwd"] + EXCESS_COLS)
    pool = pool.sort_values("date").reset_index(drop=True)

    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    pool = neutralize(pool, STYLE_COLS)
    return pool


def run_cross(pool, num_leaves, learning_rate):
    """扩张窗口 pooled 训练 + 横截面 Rank IC。返回 (ic_mean, icir, ic_pos)。"""
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    ic_list = []
    for t in rebal_dates:
        train = pool[pool["date"] < t].dropna(subset=FEATURE_COLS_FULL)
        if len(train) < MIN_TRAIN:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=learning_rate, num_leaves=num_leaves,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(train[FEATURE_COLS_FULL].to_numpy(), train["target_rel"].to_numpy())
        cross = pool[pool["date"] == t].dropna(subset=FEATURE_COLS_FULL + ["excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        prob = model.predict_proba(cross[FEATURE_COLS_FULL].to_numpy())[:, 1]
        df = pd.DataFrame({"prob": prob, "excess_fwd": cross["excess_fwd"].to_numpy()})
        ic_list.append(df["prob"].corr(df["excess_fwd"], method="spearman"))

    ic = np.array(ic_list)
    icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")
    return float(ic.mean()), float(icir), float((ic > 0).mean())


def scan_dim(dim, values, pools):
    """对单个维度扫描,复用对应 pool(horizon 需独立 pool,其他共用默认 horizon 的 pool)。"""
    print(f"\n--- 扫描 {dim} ---", flush=True)
    rows = []
    for v in values:
        t0 = time.time()
        if dim == "horizon":
            pool = build_pool(v)
            nl, lr = DEFAULTS["num_leaves"], DEFAULTS["learning_rate"]
        else:
            pool = pools["default"]
            nl = v if dim == "num_leaves" else DEFAULTS["num_leaves"]
            lr = v if dim == "learning_rate" else DEFAULTS["learning_rate"]
        ic, icir, ic_pos = run_cross(pool, nl, lr)
        mark = "  <- 默认" if v == DEFAULTS[dim] else ""
        print(f"  {dim}={v}: Rank IC {ic:+.4f}  ICIR {icir:+.3f}  IC>0 {ic_pos:.1%}  "
              f"({time.time()-t0:.0f}s){mark}", flush=True)
        rows.append((v, ic, icir, ic_pos))
    return rows


def main():
    t0 = time.time()
    print(f"特征集: {len(FEATURE_COLS_FULL)} 个(技术+超额+估值+财务+size)", flush=True)

    # 默认 horizon 的 pool 只构建一次,供 num_leaves/lr 扫描复用
    pools = {"default": build_pool(DEFAULTS["horizon"])}
    print(f"默认 pool 就绪(horizon={DEFAULTS['horizon']}),耗时 {time.time()-t0:.0f}s", flush=True)

    results = {}
    results["horizon"] = scan_dim("horizon", [20, 30, 60], pools)
    results["num_leaves"] = scan_dim("num_leaves", [15, 31, 63], pools)
    results["learning_rate"] = scan_dim("learning_rate", [0.03, 0.05, 0.10], pools)

    # 稳健性判定:该维度所有取值下 Rank IC 是否都 > 0
    print("\n===== 稳健性结论 =====", flush=True)
    for dim, rows in results.items():
        ics = [r[1] for r in rows]
        all_pos = all(ic > 0 for ic in ics)
        spread = max(ics) - min(ics)
        print(f"  {dim}: IC 范围 [{min(ics):+.4f}, {max(ics):+.4f}], "
              f"极差 {spread:.4f}, {'全部为正(稳健)' if all_pos else '有负值(不稳健)'}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "sensitivity_scan.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("参数敏感性扫描 (相对 Rank IC,扩张窗口)\n")
        f.write("======================================\n\n")
        for dim, rows in results.items():
            f.write(f"[{dim}]\n")
            for v, ic, icir, ic_pos in rows:
                f.write(f"  {dim}={v}: IC {ic:+.4f}  ICIR {icir:+.3f}  IC>0 {ic_pos:.1%}\n")
            f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""非线性交互拆解:定位驱动 LightGBM 0.43 IC 的因子组合。

Part 1 树结构:训练生产口径 LightGBM,dump 分裂,统计根节点/浅层(深度0-2)用什么因子,
        展示顶部因子的主要分裂阈值与方向(是否"低负债+高ROE+高增长")。
Part 2 双因素 double-sort:对顶部因子两两 3x3 分组,算每格实际未来超额收益,
        用"实际 - 可加预测"残差检测交互效应(角落格残差大 = 有真实交互)。

只读冻结 cache,不触发网络。
"""
import glob
import os
import sys
import time
from collections import Counter

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

EXCESS_WINDOWS = [5, 20, 60]
EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
STYLE_COLS = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS
FEATURES = STYLE_COLS + ["size"]

# 顶部因子(据历史特征重要性 debt_ratio/np_growth/rev_growth/size/roe + 估值 log_pe_ttm)
TOP_FACTORS = ["debt_ratio", "roe", "np_growth", "rev_growth", "size", "log_pe_ttm"]


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


# ---------------------------------------------------------------------------
# Part 1: 树结构分析
# ---------------------------------------------------------------------------
def part1_tree(pool):
    data = pool.dropna(subset=FEATURES + ["target_rel"])
    X, y = data[FEATURES].to_numpy(), data["target_rel"].to_numpy()
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.10, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(X, y)
    dump = model.booster_.dump_model()

    depth_feat = {d: Counter() for d in range(5)}
    gain_feat = Counter()
    root_feat = Counter()
    for tree in dump["tree_info"]:
        st = tree["tree_structure"]
        stack = [(st, 0)]
        while stack:
            node, d = stack.pop()
            if "split_feature" not in node:
                continue
            f = FEATURES[node["split_feature"]]
            if d < 5:
                depth_feat[d][f] += 1
            gain_feat[f] += node.get("split_gain", 0)
            if d == 0:
                root_feat[f] += 1
            stack.append((node["left_child"], d + 1))
            stack.append((node["right_child"], d + 1))

    print("\n===== Part 1: 树结构分析 =====", flush=True)
    print(f"树数 {len(dump['tree_info'])}, 样本 {len(data)}", flush=True)
    print(f"\n根节点(深度0)分裂特征 Top10:", flush=True)
    for f, c in root_feat.most_common(10):
        print(f"  {f:<18}{c:>4}", flush=True)
    print(f"\n各深度分裂特征 Top5 (深度越深=交互越深):", flush=True)
    for d in range(4):
        top = depth_feat[d].most_common(5)
        s = "  ".join(f"{f}({c})" for f, c in top)
        print(f"  深度{d}: {s}", flush=True)
    print(f"\n累计 split_gain Top12 特征:", flush=True)
    for f, g in gain_feat.most_common(12):
        print(f"  {f:<18}{g:>12.0f}", flush=True)

    # 顶部特征的分裂阈值分布(看方向:阈值是低值还是高值)
    print(f"\n顶部因子的分裂阈值分布(min/中位/max 阈值 + 分裂次数):", flush=True)
    thr = {f: [] for f in TOP_FACTORS}
    for tree in dump["tree_info"]:
        st = tree["tree_structure"]
        stack = [st]
        while stack:
            node = stack.pop()
            if "split_feature" not in node:
                continue
            f = FEATURES[node["split_feature"]]
            if f in thr:
                thr[f].append(node["threshold"])
            stack.append(node["left_child"])
            stack.append(node["right_child"])
    for f in TOP_FACTORS:
        if thr[f]:
            a = np.array(thr[f])
            print(f"  {f:<18} n={len(a):>3} 阈值 min/中位/max = {a.min():+.2f}/{np.median(a):+.2f}/{a.max():+.2f}",
                  flush=True)
    return model


# ---------------------------------------------------------------------------
# Part 2: 双因素 double-sort(检测交互效应)
# ---------------------------------------------------------------------------
def double_sort(pool, fa, fb, n=3):
    """对 fa/fb 两两分 n 组,算每格实际未来超额收益均值,返回 (R, I, 样本数)。

    R[a][b] = 实际超额收益均值; I[a][b] = 实际 - 可加预测(交互残差)。
    """
    d = pool.dropna(subset=[fa, fb, "excess_fwd"]).copy()
    d["qa"] = d.groupby("date")[fa].transform(
        lambda x: pd.qcut(x, n, labels=False, duplicates="drop"))
    d["qb"] = d.groupby("date")[fb].transform(
        lambda x: pd.qcut(x, n, labels=False, duplicates="drop"))
    d = d.dropna(subset=["qa", "qb"])
    R = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            m = d[(d["qa"] == a) & (d["qb"] == b)]["excess_fwd"].mean()
            R[a][b] = m
    G = np.nanmean(R)
    A = np.nanmean(R, axis=1)  # fa 边际
    B = np.nanmean(R, axis=0)  # fb 边际
    P = G + (A - G)[:, None] + (B - G)[None, :]
    I = R - P
    return R, I


def part2_double_sort(pool):
    print("\n===== Part 2: 双因素 double-sort (交互效应检测) =====", flush=True)
    print("每格 = 实际未来30日超额收益均值(%), I = 实际 - 可加预测(残差,>0=正交互)", flush=True)
    pairs = [
        ("debt_ratio", "roe"), ("debt_ratio", "np_growth"), ("debt_ratio", "size"),
        ("roe", "np_growth"), ("roe", "size"), ("np_growth", "size"),
        ("log_pe_ttm", "debt_ratio"), ("log_pe_ttm", "roe"),
    ]
    for fa, fb in pairs:
        R, I = double_sort(pool, fa, fb)
        print(f"\n--- {fa} x {fb} ---", flush=True)
        print("实际超额收益 R (行=fa低→高, 列=fb低→高):", flush=True)
        for a in range(3):
            print("  " + "  ".join(f"{R[a][b]:+.2f}" for b in range(3)), flush=True)
        print("交互残差 I (角落格大 = 有交互):", flush=True)
        for a in range(3):
            print("  " + "  ".join(f"{I[a][b]:+.2f}" for b in range(3)), flush=True)
        # 交互强度 = 角落对比(高-高 + 低-低) - (高-低 + 低-高)
        strength = (R[0][0] + R[2][2]) - (R[0][2] + R[2][0])
        print(f"  交互强度(对角-反对角) = {strength:+.2f}%", flush=True)


def main():
    t0 = time.time()
    pool = load_pool()
    print(f"加载 {pool['code'].nunique()} 只, {len(pool)} 条", flush=True)
    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    pool = neutralize(pool, STYLE_COLS)

    part1_tree(pool)
    part2_double_sort(pool)

    print(f"\n总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

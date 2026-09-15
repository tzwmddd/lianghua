"""特征消融压测:定位 0.47 Rank IC 的污染源(基本面数据 vs 幸存者偏差 vs 技术面)。

对比特征子集的 10 年相对 Rank IC,基线 full(38列) +0.471 见 stress_test_regime_report.txt:

  tech_only      25列 = 技术22 + 超额3                 (无估值/财务/市值/行业因子)
  no_valuation   35列 = 技术22 + 超额3 + 财务9 + size   (有财务,无估值)
  no_fundamental 29列 = 技术22 + 超额3 + 估值3 + size   (有估值,无财务)

判读:
  tech_only 崩到 ~0.1     -> 污染在基本面数据(估值/财务/市值/行业)
  tech_only 仍 ~0.4       -> 污染在别处(幸存者偏差/技术面本身)
  no_valuation 崩         -> 污染在估值因子(PE/PB/PCF)
  no_fundamental 崩       -> 污染在财务因子(ROE/成长/杠杆)

所有配置对各自风格因子做相同的中性化(行业+log_mcap 回归残差),保证唯一变量是特征子集。
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

TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
N_GROUPS = 5
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]

EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
BASE_STYLE = FEATURE_COLS + EXCESS_COLS

CONFIGS = {
    "tech_only": {
        "style": BASE_STYLE,
        "features": BASE_STYLE,
        "needs_financial": False,
        "desc": "技术22+超额3",
    },
    "tech_size": {
        "style": BASE_STYLE,
        "features": BASE_STYLE + ["size"],
        "needs_financial": False,
        "desc": "技术+超额+size",
    },
    "no_valuation": {
        "style": BASE_STYLE + FINANCIAL_COLS,
        "features": BASE_STYLE + FINANCIAL_COLS + ["size"],
        "needs_financial": True,
        "desc": "技术+超额+财务+size(无估值)",
    },
    "no_fundamental": {
        "style": BASE_STYLE + VAL_LOG_COLS,
        "features": BASE_STYLE + VAL_LOG_COLS + ["size"],
        "needs_financial": False,
        "desc": "技术+超额+估值+size(无财务)",
    },
}


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


def run_cross(pool, feature_cols):
    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    ic_list = []
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
        ic_list.append(df["prob"].corr(df["excess_fwd"], method="spearman"))
    return np.array(ic_list)


def icir(ic):
    return ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan")


def main():
    t0 = time.time()
    configs = sys.argv[1:] or list(CONFIGS.keys())
    pool = load_pool()
    print(f"加载 {pool['code'].nunique()} 只,{len(pool)} 条样本 "
          f"({pool['date'].min().date()} ~ {pool['date'].max().date()})", flush=True)

    lines = []
    header = f"{'配置':<16}{'特征数':>6}{'Rank IC':>10}{'ICIR':>8}{'IC>0':>8}{'期数':>6}"
    print("\n" + header, flush=True)
    lines.append(header)

    for name in configs:
        cfg = CONFIGS[name]
        p = attach_fundamentals(pool.copy(), CACHE_DIR)
        if cfg["needs_financial"]:
            p = attach_financial(p, CACHE_DIR)
        p = neutralize(p, cfg["style"])
        ic = run_cross(p, cfg["features"])
        row = (f"{name:<16}{len(cfg['features']):>6}{ic.mean():>+10.4f}"
               f"{icir(ic):>+8.3f}{(ic > 0).mean():>8.1%}{len(ic):>6}")
        print(row, flush=True)
        lines.append(row)

    lines.append("")
    lines.append("baseline full(38列) : Rank IC +0.471  ICIR +3.526  IC>0 100%  (stress_test_regime)")
    print("\nbaseline full(38列) : +0.471 / ICIR 3.53 / IC>0 100%", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "stress_test_ablate_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("特征消融压测 (相对标签 + 相对 Rank IC, 生产口径)\n")
        f.write("===============================================\n\n")
        f.write(f"OOS 起点 {TEST_START}, 每 {REBALANCE_EVERY} 交易日调仓\n")
        f.write("模型 num_leaves=63 / lr=0.10 / n_estimators=150\n\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

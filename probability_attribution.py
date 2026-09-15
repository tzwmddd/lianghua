"""概率分桶溯源:把 Q5-Q1 的 12.77% 利差拆到具体股票和时期。

回答:0.43 IC(12.77% 利差)是集中在少数股票/少数时期,还是均匀分布?
- 集中 => 过拟合/特定异常(可定位)
- 均匀 => 某种更系统性的假象(更难解释)

方法:完整 walk-forward(full 配置),逐期记录 Q5/Q1 成分与超额收益,然后:
  A. 单期利差分布(是否少数期爆拉)
  B. Q5 vs Q1 的贡献分解(是 Q5 太好还是 Q1 太差)
  C. 收益稳健性(mean vs median vs 截尾,是否离群值驱动)
  D. 股票集中度(是否少数股票长期霸榜 Q5)
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

TEST_START = "2017-01-01"
REBALANCE_EVERY = 20
N_GROUPS = 5
MIN_TRAIN = 5000
MIN_CROSS = 30
EXCESS_WINDOWS = [5, 20, 60]
EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
STYLE_COLS = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS
FEATURES = STYLE_COLS + ["size"]


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


def apply_constituents(pool):
    cons = pd.read_csv(os.path.join(CACHE_DIR, "_constituents.csv"),
                       parse_dates=["snapshot_date"], dtype={"code": str})
    snap_dates = pd.to_datetime(sorted(cons["snapshot_date"].unique()))
    pool = pool.sort_values("date").copy()
    idx = np.searchsorted(snap_dates.values, pool["date"].to_numpy(), side="right") - 1
    idx = np.clip(idx, 0, len(snap_dates) - 1)
    pool["snap_date"] = snap_dates.values[idx]
    pool = pool.merge(cons, left_on=["snap_date", "code"],
                      right_on=["snapshot_date", "code"], how="inner")
    pool = pool.drop(columns=["snap_date", "snapshot_date"])
    return pool.sort_values("date").reset_index(drop=True)


def main():
    t0 = time.time()
    pool = load_pool()
    pool = apply_constituents(pool)
    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    pool = neutralize(pool, STYLE_COLS)

    dates = sorted(pool["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START))
    rebal_dates = dates[start_idx::REBALANCE_EVERY]

    period_rows = []       # 每期: date, Q5均值, Q1均值, 利差, 期数n
    q5_membership = Counter()   # 股票出现在Q5的次数
    q5_ret_list = []       # 所有Q5样本的 excess_fwd
    q1_ret_list = []       # 所有Q1样本的 excess_fwd
    q5_code_excess = []    # (code, excess_fwd) 对,用于按股票聚合
    ic_list = []

    for t in rebal_dates:
        train = pool[pool["date"] < t].dropna(subset=FEATURES)
        if len(train) < MIN_TRAIN:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.10, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(train[FEATURES].to_numpy(), train["target_rel"].to_numpy())
        cross = pool[pool["date"] == t].dropna(subset=FEATURES + ["excess_fwd"])
        if len(cross) < MIN_CROSS:
            continue
        prob = model.predict_proba(cross[FEATURES].to_numpy())[:, 1]
        df = pd.DataFrame({
            "code": cross["code"].to_numpy(),
            "prob": prob,
            "excess_fwd": cross["excess_fwd"].to_numpy(),
        })
        ic_list.append(df["prob"].corr(df["excess_fwd"], method="spearman"))
        df["rank"] = df["prob"].rank(method="first", pct=True)
        df["group"] = np.ceil(df["rank"] * N_GROUPS).astype(int) - 1
        q5 = df[df["group"] == N_GROUPS - 1]
        q1 = df[df["group"] == 0]
        q5m, q1m = q5["excess_fwd"].mean(), q1["excess_fwd"].mean()
        period_rows.append((t, q5m, q1m, q5m - q1m, len(df)))
        for c in q5["code"]:
            q5_membership[c] += 1
        q5_ret_list.extend(q5["excess_fwd"].tolist())
        q1_ret_list.extend(q1["excess_fwd"].tolist())
        q5_code_excess.extend(zip(q5["code"].tolist(), q5["excess_fwd"].tolist()))

    ic = np.array(ic_list)
    pr = pd.DataFrame(period_rows, columns=["date", "q5", "q1", "spread", "n"])
    q5r = np.array(q5_ret_list)
    q1r = np.array(q1_ret_list)

    print(f"\n===== 总览 =====", flush=True)
    print(f"期数 {len(pr)}, Rank IC {ic.mean():+.4f} / ICIR {ic.mean()/ic.std(ddof=1):+.2f} / IC>0 {(ic>0).mean():.1%}", flush=True)
    print(f"Q5 平均超额 {q5r.mean():+.2%}, Q1 平均超额 {q1r.mean():+.2%}, 利差 {(q5r.mean()-q1r.mean()):+.2%}", flush=True)

    print(f"\n===== A. 单期利差分布(是否少数期爆拉) =====", flush=True)
    s = pr["spread"]
    print(f"  利差 mean {s.mean():+.2%} / median {s.median():+.2%} / std {s.std():+.2%}", flush=True)
    print(f"  利差>0 占比 {(s>0).mean():.1%}  (若少数期贡献大部分,这里应显示长尾)", flush=True)
    top = pr.nlargest(10, "spread")[["date", "q5", "q1", "spread"]]
    bot = pr.nsmallest(10, "spread")[["date", "q5", "q1", "spread"]]
    print(f"  利差最大10期:", flush=True)
    for _, r in top.iterrows():
        print(f"    {r['date'].date()}  Q5 {r['q5']:+.2%}  Q1 {r['q1']:+.2%}  利差 {r['spread']:+.2%}", flush=True)
    print(f"  利差最小10期:", flush=True)
    for _, r in bot.iterrows():
        print(f"    {r['date'].date()}  Q5 {r['q5']:+.2%}  Q1 {r['q1']:+.2%}  利差 {r['spread']:+.2%}", flush=True)
    # 前10期贡献
    top10_contrib = top["spread"].sum() / s.sum() if s.sum() != 0 else np.nan
    print(f"  前10期利差合计占总利差: {top10_contrib:.1%}", flush=True)

    print(f"\n===== B. 收益稳健性(是否离群值驱动) =====", flush=True)
    def robust(x):
        return x.mean(), np.median(x), np.mean(np.sort(x)[int(len(x)*0.05):int(len(x)*0.95)])
    for name, arr in [("Q5", q5r), ("Q1", q1r)]:
        m, med, tr = robust(arr)
        print(f"  {name}: mean {m:+.2%} / median {med:+.2%} / 截尾5% {tr:+.2%} / 正收益占比 {(arr>0).mean():.1%}", flush=True)

    print(f"\n===== C. 股票集中度(是否少数股票长期霸榜 Q5) =====", flush=True)
    total_q5_slots = len(q5_ret_list)
    top_stocks = q5_membership.most_common(20)
    print(f"  Q5 总样本槽位 {total_q5_slots}, 不同股票 {len(q5_membership)} 只", flush=True)
    top20_slots = sum(c for _, c in top_stocks)
    print(f"  Top20 股票占 Q5 槽位 {top20_slots}/{total_q5_slots} = {top20_slots/total_q5_slots:.1%}", flush=True)
    print(f"  Top10 股票(代码, 出现在Q5次数):", flush=True)
    for c, n in top_stocks[:10]:
        print(f"    {c}: {n} 次", flush=True)

    # D. Q5 收益的股票来源: 按股票聚合平均超额,看头部股票贡献
    print(f"\n===== D. Q5 收益按股票聚合(哪几只贡献了 Q5 的高收益) =====", flush=True)
    d = pd.DataFrame(q5_code_excess, columns=["code", "excess"])
    grp = d.groupby("code")["excess"].agg(["count", "mean"]).sort_values("mean", ascending=False)
    print(f"  Q5 内平均超额收益 Top15 股票(出现次数>=3):", flush=True)
    grp_f = grp[grp["count"] >= 3].head(15)
    for code, row in grp_f.iterrows():
        print(f"    {code}: 出现 {int(row['count'])} 次, 平均超额 {row['mean']:+.2%}", flush=True)
    print(f"  Q5 内平均超额收益 Bottom10 股票:", flush=True)
    for code, row in grp[grp["count"] >= 3].tail(10).iterrows():
        print(f"    {code}: 出现 {int(row['count'])} 次, 平均超额 {row['mean']:+.2%}", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

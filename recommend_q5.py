"""Q5 纯多头选股推荐(相对收益版):预测"是否跑赢沪深300",输出买入/回避榜单。

标签 = 个股未来30日收益 是否 > 沪深300未来30日收益(跑赢大盘);
特征 = 22个技术指标 + 3个超额收益特征(个股N日收益 - 指数N日收益)。
pooled 5年训练,预测最新交易日横截面,做多"跑赢概率"最高的 20%(Q5)。
"""
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import CACHE_DIR, RESULTS_DIR, SEED, FORECAST_HORIZON
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import attach_fundamentals, neutralize, VALUATION_COLS

warnings.filterwarnings("ignore")

Q_FRAC = 0.2          # Q5/Q1 各占 20%
TOP_SHOW = 15         # 高亮显示前 15 只
EXCESS_WINDOWS = [5, 20, 60]
BASE_STYLE = FEATURE_COLS + [f"excess_{n}" for n in EXCESS_WINDOWS]
STYLE_COLS = BASE_STYLE + [f"log_{c}" for c in VALUATION_COLS]


def get_cons():
    """沪深300成分股 -> DataFrame(code, name)。失败返回空名称。"""
    try:
        import akshare as ak
        cons = ak.index_stock_cons_csindex(symbol="000300")
        cons = cons[["成分券代码", "成分券名称"]].drop_duplicates()
        cons.columns = ["code", "name"]
        cons["code"] = cons["code"].astype(str).str.zfill(6)
        return cons.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["code", "name"])


def load_index():
    idx = pd.read_csv(os.path.join(CACHE_DIR, "sh000300.csv"), parse_dates=["date"])
    return idx.set_index("date")["close"].sort_index()


def load_pool():
    """加载个股面板,含超额收益特征与"是否跑赢大盘"标签。"""
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
        feat = build_features(df, FORECAST_HORIZON, keep_tail=True)
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
    return pool.sort_values("date").reset_index(drop=True)


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pool = load_pool()
    pool = pool.dropna(subset=BASE_STYLE)
    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = neutralize(pool, STYLE_COLS)
    features = STYLE_COLS + ["size"]
    train = pool[pool["target_rel"].notna()].dropna(subset=features)
    latest = pool["date"].max()
    today = pool[pool["date"] == latest].dropna(subset=features)
    print(f"训练样本 {len(train)} 条({train['code'].nunique()} 只),"
          f"区间 {train['date'].min().date()} ~ {train['date'].max().date()}", flush=True)
    print(f"预测截面: {latest.date()},共 {len(today)} 只,标签=未来 {FORECAST_HORIZON} 日是否跑赢沪深300", flush=True)

    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(train[features].to_numpy(), train["target_rel"].to_numpy())

    prob = model.predict_proba(today[features].to_numpy())[:, 1]
    res = today[["code", "date", "close"]].copy()
    res["prob"] = prob

    cons = get_cons()
    name_map = dict(zip(cons["code"], cons["name"])) if len(cons) else {}
    res["name"] = res["code"].map(name_map).fillna("")

    res = res.sort_values("prob", ascending=False).reset_index(drop=True)
    k = max(1, int(len(res) * Q_FRAC))

    print(f"\n===== Q5 买入推荐(跑赢概率最高 {Q_FRAC:.0%},共 {k} 只) =====", flush=True)
    print(f"{'代码':<8}{'名称':<10}{'收盘价':>10}{'跑赢概率':>10}", flush=True)
    for _, r in res.head(TOP_SHOW).iterrows():
        print(f"{r['code']:<8}{r['name']:<10}{r['close']:>10.2f}{r['prob']:>9.1%}", flush=True)
    print(f"  ... (共 {k} 只,完整见 CSV)", flush=True)

    print(f"\n===== Q1 回避(跑赢概率最低 {Q_FRAC:.0%},共 {k} 只) =====", flush=True)
    for _, r in res.tail(TOP_SHOW).iloc[::-1].iterrows():
        print(f"{r['code']:<8}{r['name']:<10}{r['close']:>10.2f}{r['prob']:>9.1%}", flush=True)

    res["group"] = "hold"
    res.loc[res.index < k, "group"] = "Q5_buy"
    res.loc[res.index >= len(res) - k, "group"] = "Q1_avoid"

    out = os.path.join(RESULTS_DIR, "recommend_q5.csv")
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n完整结果已保存: {out}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

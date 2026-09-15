"""中证500/1000 纯多头选股推荐:预测"是否跑赢自身指数",输出 Q5 买入榜。

只做多不做空。标签 = 个股未来30日收益是否跑赢中证500/1000 指数;pooled 全历史
训练(forward-30 标签天然留出 30 交易日 embargo,训练集与预测日零重叠),预测最新
交易日横截面,做多"跑赢概率"最高的 20%(Q5)。特征 = 技术22 + 超额3 + 估值2 + 财务6 + size。

用法: python recommend_zz.py [500|1000]
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

from config import RESULTS_DIR, SEED, FORECAST_HORIZON
from data import load_cached
from features import build_features, FEATURE_COLS
from cross_section import neutralize
from fundamental_ts import attach_fundamentals_ts

warnings.filterwarnings("ignore")

UNIVERSES = {
    "500": ("000905.SH", "cache_zz500"),
    "1000": ("000852.SH", "cache_zz1000"),
}
Q_FRAC = 0.2
TOP_SHOW = 20
LIQUIDITY_DROP = 0.2       # 剔除近20日均换手率最低 20%(流动性过滤,小盘股滑点大)
TURNOVER_WINDOW = 20       # 换手率均值窗口(交易日)
EXCESS_WINDOWS = [5, 20, 60]
BASE_STYLE = FEATURE_COLS + [f"excess_{n}" for n in EXCESS_WINDOWS]
# 评估证实:技术面是唯一有效信号,财务/估值/size 都是噪声(会稀释 IC)。
# 故推荐特征只用纯技术面(tech_only 最优配置),不做 size/财务/估值。
STYLE_COLS = BASE_STYLE


def get_names(universe):
    """tushare stock_basic 拿代码->名称(当前上市)。失败返回空。"""
    try:
        import tushare_common as tc
        pro = tc.get_pro()
        df = pro.stock_basic(list_status="L")
        return dict(zip(df["ts_code"].map(tc.to_code), df["name"]))
    except Exception:
        return {}


def load_index(cache_dir):
    idx = pd.read_csv(os.path.join(cache_dir, "index.csv"), parse_dates=["date"])
    return idx.set_index("date")["close"].sort_index()


def load_pool(cache_dir):
    idx = load_index(cache_dir)
    idx_ret = idx.pct_change()
    idx_fwd = idx.shift(-FORECAST_HORIZON) / idx - 1

    codes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(cache_dir, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    parts = []
    for code in codes:
        df = load_cached(code, cache_dir)
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


def load_turnover(cache_dir, window=TURNOVER_WINDOW):
    """读 daily_basic,算每只股票近 window 日平均换手率,返回最新一期 code->turnover。

    换手率是流动性代理:小盘股换手率过低会导致买入滑点大、涨跌停买不进。
    """
    path = os.path.join(cache_dir, "_daily_basic.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["code", "turnover"])
    db = pd.read_csv(path, parse_dates=["date"], dtype={"code": str})
    db = db.sort_values(["code", "date"])
    db["turnover"] = db.groupby("code")["turnover_rate"].transform(
        lambda s: s.rolling(window, min_periods=10).mean()
    )
    return db.drop_duplicates("code", keep="last")[["code", "turnover"]]


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit("用法: python recommend_zz.py [500|1000]")
    index_code, cache_dir = UNIVERSES[universe]
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pool = load_pool(cache_dir)
    pool = pool.dropna(subset=BASE_STYLE)
    pool = attach_fundamentals_ts(pool, cache_dir)
    pool = neutralize(pool, STYLE_COLS)
    features = STYLE_COLS
    train = pool[pool["target_rel"].notna()].dropna(subset=features)
    latest = pool["date"].max()
    today = pool[pool["date"] == latest].dropna(subset=features)
    # 预测截面只保留当前成分(最新 snapshot),不推荐已调出成分
    cons = pd.read_csv(os.path.join(cache_dir, "_constituents.csv"), dtype={"code": str})
    latest_snap = cons["snapshot_date"].max()
    current = set(cons.loc[cons["snapshot_date"] == latest_snap, "code"])
    today = today[today["code"].isin(current)]
    # 流动性过滤:剔除近20日均换手率最低 LIQUIDITY_DROP 的股票(小盘股滑点大、涨跌停买不进)
    turnover = load_turnover(cache_dir)
    today = today.merge(turnover, on="code", how="left")
    n_before = len(today)
    thresh = today["turnover"].quantile(LIQUIDITY_DROP)
    today = today[today["turnover"] >= thresh]
    print(f"流动性过滤: 剔除换手率最低 {LIQUIDITY_DROP:.0%} 的 {n_before - len(today)} 只"
          f"(阈值 {thresh:.2f}%),剩余 {len(today)} 只", flush=True)
    print(f"训练样本 {len(train)} 条({train['code'].nunique()} 只),"
          f"区间 {train['date'].min().date()} ~ {train['date'].max().date()}", flush=True)
    print(f"预测截面: {latest.date()},共 {len(today)} 只,标签=未来 {FORECAST_HORIZON} 日是否跑赢{index_code}", flush=True)

    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.10, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(train[features].to_numpy(), train["target_rel"].to_numpy())

    prob = model.predict_proba(today[features].to_numpy())[:, 1]
    res = today[["code", "date", "close", "turnover"]].copy()
    res["prob"] = prob

    name_map = get_names(universe)
    res["name"] = res["code"].map(name_map).fillna("")

    res = res.sort_values("prob", ascending=False).reset_index(drop=True)
    k = max(1, int(len(res) * Q_FRAC))

    print(f"\n===== 中证{universe} Q5 买入推荐(跑赢概率最高 {Q_FRAC:.0%},共 {k} 只) =====", flush=True)
    print(f"{'代码':<8}{'名称':<10}{'收盘价':>10}{'换手率%':>8}{'跑赢概率':>10}", flush=True)
    for _, r in res.head(TOP_SHOW).iterrows():
        print(f"{r['code']:<8}{r['name']:<10}{r['close']:>10.2f}{r['turnover']:>8.2f}{r['prob']:>9.1%}", flush=True)
    print(f"  ... (共 {k} 只,完整见 CSV)", flush=True)

    print(f"\n===== 回避参考(跑赢概率最低 {Q_FRAC:.0%},仅供参考不做空) =====", flush=True)
    for _, r in res.tail(10).iloc[::-1].iterrows():
        print(f"{r['code']:<8}{r['name']:<10}{r['close']:>10.2f}{r['turnover']:>8.2f}{r['prob']:>9.1%}", flush=True)

    res["group"] = "hold"
    res.loc[res.index < k, "group"] = "Q5_buy"
    res.loc[res.index >= len(res) - k, "group"] = "Q1_avoid"

    out = os.path.join(RESULTS_DIR, f"recommend_zz{universe}.csv")
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n完整结果已保存: {out}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""增强版选股推荐(坑6:打开黑盒 + 降低单模型方差)。

在 recommend_q5.py 基础上增加三件事:
1. 多 seed 集成:跑 K 个随机种子平均概率,降低单模型随机方差,输出概率稳定性(std)。
2. 特征重要性:LightGBM gain 重要性,展示模型主要靠哪些因子。
3. 线性对照:逻辑回归,系数可直接解读(哪些因子正向/负向贡献),验证信号可解释、非纯黑盒。

输出 Q5 买入榜(用集成平均概率)+ Q1 回避榜,完整结果 results/recommend_robust.csv。
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
from fundamental import attach_financial, FINANCIAL_COLS

warnings.filterwarnings("ignore")

Q_FRAC = 0.2
TOP_SHOW = 15
N_SEEDS = 7
EXCESS_WINDOWS = [5, 20, 60]
EXCESS_COLS = [f"excess_{n}" for n in EXCESS_WINDOWS]
VAL_LOG_COLS = [f"log_{c}" for c in VALUATION_COLS]
STYLE_COLS = FEATURE_COLS + EXCESS_COLS + VAL_LOG_COLS + FINANCIAL_COLS
FEATURES = STYLE_COLS + ["size"]


def get_cons():
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
    pool = pool.dropna(subset=FEATURE_COLS + EXCESS_COLS)
    pool = attach_fundamentals(pool, CACHE_DIR)
    pool = attach_financial(pool, CACHE_DIR)
    pool = neutralize(pool, STYLE_COLS)

    train = pool[pool["target_rel"].notna()].dropna(subset=FEATURES)
    latest = pool["date"].max()
    today = pool[pool["date"] == latest].dropna(subset=FEATURES)
    print(f"训练样本 {len(train)} 条({train['code'].nunique()} 只),", flush=True)
    print(f"预测截面: {latest.date()},共 {len(today)} 只,特征 {len(FEATURES)} 个", flush=True)

    X_tr = train[FEATURES].to_numpy()
    y_tr = train["target_rel"].to_numpy()
    X_te = today[FEATURES].to_numpy()

    # ---- 1) 多 seed 集成(降方差 + 概率稳定性) ----
    import lightgbm as lgb
    prob_mat = np.zeros((len(today), N_SEEDS))
    for i in range(N_SEEDS):
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.10, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED + i, verbosity=-1, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        prob_mat[:, i] = model.predict_proba(X_te)[:, 1]
        if i == 0:
            first_model = model
    prob = prob_mat.mean(axis=1)
    prob_std = prob_mat.std(axis=1)

    # ---- 2) 特征重要性(LightGBM gain) ----
    imp = pd.DataFrame({
        "feature": FEATURES,
        "gain": first_model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    print(f"\n===== 特征重要性 Top 10(gain) =====", flush=True)
    for _, r in imp.head(10).iterrows():
        print(f"  {r['feature']:<18} {r['gain']:>12.0f}", flush=True)

    # ---- 3) 线性对照(逻辑回归,可解释系数) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(scaler.transform(X_tr), y_tr)
    coef = pd.DataFrame({"feature": FEATURES, "coef": lr.coef_[0]})
    coef = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index)
    print(f"\n===== 逻辑回归系数(标准化后,可解读) =====", flush=True)
    print("  正向贡献(越大越容易跑赢):", flush=True)
    for _, r in coef.head(8).iterrows():
        sign = "+" if r["coef"] > 0 else "-"
        print(f"    {r['feature']:<18} {sign}{abs(r['coef']):.3f}", flush=True)

    # ---- 输出榜单(用集成平均概率) ----
    res = today[["code", "date", "close"]].copy()
    res["prob"] = prob
    res["prob_std"] = prob_std
    cons = get_cons()
    name_map = dict(zip(cons["code"], cons["name"])) if len(cons) else {}
    res["name"] = res["code"].map(name_map).fillna("")
    res = res.sort_values("prob", ascending=False).reset_index(drop=True)
    k = max(1, int(len(res) * Q_FRAC))

    print(f"\n===== Q5 买入推荐(集成概率最高 {Q_FRAC:.0%},共 {k} 只) =====", flush=True)
    print(f"{'代码':<8}{'名称':<10}{'收盘价':>10}{'概率':>9}{'std':>7}", flush=True)
    for _, r in res.head(TOP_SHOW).iterrows():
        print(f"{r['code']:<8}{r['name']:<10}{r['close']:>10.2f}{r['prob']:>9.1%}{r['prob_std']:>7.1%}", flush=True)

    print(f"\n===== Q1 回避(集成概率最低 {Q_FRAC:.0%}) =====", flush=True)
    for _, r in res.tail(TOP_SHOW).iloc[::-1].iterrows():
        print(f"{r['code']:<8}{r['name']:<10}{r['close']:>10.2f}{r['prob']:>9.1%}{r['prob_std']:>7.1%}", flush=True)

    res["group"] = "hold"
    res.loc[res.index < k, "group"] = "Q5_buy"
    res.loc[res.index >= len(res) - k, "group"] = "Q1_avoid"
    out = os.path.join(RESULTS_DIR, "recommend_robust.csv")
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n完整结果已保存: {out}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

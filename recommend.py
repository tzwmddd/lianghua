"""个股推荐:对沪深300成分股批量预测未来方向,输出买入/回避推荐与持仓建议。"""
import datetime
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from config import CACHE_DIR, RESULTS_DIR, FORECAST_HORIZON, SEED
from data import get_data
from features import build_features, FEATURE_COLS

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

TOP_N = 15

# 信号方向:历史横截面评估(Rank IC=-0.069, 见 evaluate.py)显示模型概率与未来
# 收益负相关——"预测上涨"的股票短期反而回落(均值回归)。故做多得分 = 1 - 上涨概率。
# 置 False 可切回"概率即做多得分"的原始口径。
SIGN_FLIP = True


def get_cons():
    """沪深300成分股 -> DataFrame(code, name)。"""
    import akshare as ak
    cons = ak.index_stock_cons_csindex(symbol="000300")
    cons = cons[["成分券代码", "成分券名称"]].drop_duplicates()
    cons.columns = ["code", "name"]
    cons["code"] = cons["code"].astype(str).str.zfill(6)
    return cons.reset_index(drop=True)


def predict_one(code, start, end):
    """训练 LightGBM,预测"最新一天"未来 horizon 天上涨概率。数据不足返回 None。"""
    import lightgbm as lgb
    df = get_data(code, start, end, CACHE_DIR)
    feat = build_features(df, FORECAST_HORIZON, keep_tail=True)
    if len(feat) < 120:
        return None
    train = feat[feat["target"].notna()]
    if len(train) < 60:
        return None

    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1,
    )
    model.fit(train[FEATURE_COLS].to_numpy(), train["target"].to_numpy())

    today = feat.iloc[-1:]
    prob = float(model.predict_proba(today[FEATURE_COLS].to_numpy())[:, 1][0])
    score = 1.0 - prob if SIGN_FLIP else prob
    return {
        "code": code,
        "prob": prob,
        "score": score,
        "close": float(today["close"].iloc[0]),
        "date": today["date"].iloc[0],
    }


def advise_holdings(res, holdings):
    """holdings: {code: name};按预测概率给出减仓/持有建议。"""
    print("\n=== 持仓建议 ===")
    m = res.set_index("code")
    for code, name in holdings.items():
        if code in m.index:
            r = m.loc[code]
            score = r["score"]
            if score >= 0.60:
                advice = "看涨,建议持有/可加仓"
            elif score >= 0.50:
                advice = "中性偏多,继续持有"
            elif score >= 0.40:
                advice = "中性偏空,考虑减仓"
            else:
                advice = "看跌,建议卖出/回避"
            print(f"  {code} {name:<8} 做多得分 {score:.1%}  -> {advice}")
        else:
            print(f"  {code} {name:<8} 不在沪深300成分股中,无法预测")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    today = datetime.date.today()
    end = today.strftime("%Y%m%d")
    start = (today - datetime.timedelta(days=365)).strftime("%Y%m%d")
    print(f"预测日期: {today}  训练区间: {start}-{end}")

    cons = get_cons()
    print(f"沪深300成分股: {len(cons)} 只,开始批量预测...\n")

    rows = []
    for i, (code, name) in enumerate(zip(cons["code"], cons["name"]), 1):
        try:
            r = predict_one(code, start, end)
            if r:
                r["name"] = name
                rows.append(r)
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  进度 {i}/{len(cons)}", flush=True)
        time.sleep(0.1)

    res = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"\n成功预测 {len(res)} 只\n")
    if SIGN_FLIP:
        print("【信号方向】已按反向信号使用(历史 Rank IC=-0.069):做多得分 = 1 - 模型上涨概率\n")

    print(f"===== 买入推荐(做多得分最高 TOP {TOP_N}) =====")
    for _, r in res.head(TOP_N).iterrows():
        print(f"  {r['code']} {r['name']:<8} 收盘 {r['close']:>10.2f}  做多得分 {r['score']:.1%} (模型上涨概率 {r['prob']:.1%})")

    print(f"\n===== 回避/卖出(做多得分最低 TOP {TOP_N}) =====")
    for _, r in res.tail(TOP_N).iloc[::-1].iterrows():
        print(f"  {r['code']} {r['name']:<8} 收盘 {r['close']:>10.2f}  做多得分 {r['score']:.1%} (模型上涨概率 {r['prob']:.1%})")

    out = os.path.join(RESULTS_DIR, "recommend.csv")
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n完整结果已保存: {out}")


if __name__ == "__main__":
    main()

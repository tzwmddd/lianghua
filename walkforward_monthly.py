"""月度滚动 walk-forward:扩张窗口(2021-01 起) vs 1年滚动窗口,输出方向准确率提升效果。

流程:用 [2021-01, 2024-01] 训练预测 2024-02,然后训练窗末端逐月前移(扩张),
依次预测 2024-03、2024-04 ... 直到数据末尾。每个测试月同时用"1年滚动窗口"
作为基线(近似此前的单标的1年训练史),量化 5 年数据 + 扩张重训带来的提升。
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

TRAIN_START = "2021-01-01"   # 扩张窗口起点
TEST_START = "2024-02-01"    # 第一个测试月
WINDOW_1Y = pd.Timedelta(days=365)
MIN_TRAIN_ROWS = 5000        # 训练样本下限,不足则跳过该月
MIN_TEST_ROWS = 100          # 测试样本下限


def load_pooled():
    """加载全部个股特征面板(已含 target),过滤指数与样本过少的标的。"""
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
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)
    return pool.sort_values("date").reset_index(drop=True)


def train_predict(train, test):
    """在 train 上训练 LightGBM,返回 test 方向准确率与预测概率。"""
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(train[FEATURE_COLS].to_numpy(), train["target"].to_numpy())
    prob = model.predict_proba(test[FEATURE_COLS].to_numpy())[:, 1]
    pred = (prob > 0.5).astype(int)
    acc = float((pred == test["target"].to_numpy()).mean())
    return acc, prob


def month_key(ts):
    return ts.strftime("%Y-%m")


def main():
    t0 = time.time()
    pool = load_pooled()
    n_stocks = pool["code"].nunique()
    print(f"加载 {n_stocks} 只股票,共 {len(pool)} 条样本,"
          f"区间 {pool['date'].min().date()} ~ {pool['date'].max().date()}", flush=True)

    train_start = pd.Timestamp(TRAIN_START)
    months = pd.date_range(pd.Timestamp(TEST_START), pool["date"].max(), freq="MS")

    rows = []
    for m in months:
        month_end = m + pd.offsets.MonthEnd(1)
        test = pool[(pool["date"] >= m) & (pool["date"] <= month_end)]
        if len(test) < MIN_TEST_ROWS:
            continue

        train_exp = pool[(pool["date"] >= train_start) & (pool["date"] < m)]
        train_1y = pool[(pool["date"] >= m - WINDOW_1Y) & (pool["date"] < m)]
        if len(train_exp) < MIN_TRAIN_ROWS or len(train_1y) < MIN_TRAIN_ROWS:
            continue

        acc_exp, _ = train_predict(train_exp, test)
        acc_1y, _ = train_predict(train_1y, test)
        rows.append({
            "month": month_key(m),
            "train_exp": len(train_exp),
            "train_1y": len(train_1y),
            "test": len(test),
            "acc_exp": acc_exp,
            "acc_1y": acc_1y,
        })
        print(f"  {month_key(m)}: 扩张窗 {acc_exp:6.2%}  1年窗 {acc_1y:6.2%}  "
              f"(训练 {len(train_exp)}/{len(train_1y)}, 测试 {len(test)})", flush=True)

    res = pd.DataFrame(rows)
    if res.empty:
        print("无有效测试月,退出。", flush=True)
        return

    exp = res["acc_exp"].to_numpy()
    one = res["acc_1y"].to_numpy()
    diff = exp - one

    print("\n========== 汇总 ==========", flush=True)
    print(f"有效测试月 {len(res)} 个: {res['month'].iloc[0]} ~ {res['month'].iloc[-1]}", flush=True)
    print(f"{'指标':<24}{'扩张窗口(2021起)':>16}{'1年滚动窗口':>14}{'提升':>10}", flush=True)
    print(f"{'平均方向准确率':<22}{exp.mean():>15.2%}{one.mean():>13.2%}{diff.mean():>+9.2%}", flush=True)
    print(f"{'准确率中位数':<22}{np.median(exp):>15.2%}{np.median(one):>13.2%}"
          f"{np.median(exp) - np.median(one):>+9.2%}", flush=True)
    print(f"{'>50% 月份占比':<22}{(exp > 0.5).mean():>15.1%}{(one > 0.5).mean():>13.1%}", flush=True)
    print(f"{'扩张窗优于1年窗月份占比':<24}{(diff > 0).mean():>13.1%}", flush=True)

    # 反向信号提示(此前发现模型为稳定反向指标)
    rev_exp = 1.0 - exp
    print(f"\n反向信号提示: 扩张窗平均准确率 {exp.mean():.2%} -> "
          f"反向(预测取反)后 {rev_exp.mean():.2%}", flush=True)

    # 按年分组的平均准确率
    res["year"] = res["month"].str[:4]
    print("\n=== 按年统计(扩张窗口方向准确率) ===", flush=True)
    for yr, g in res.groupby("year"):
        print(f"  {yr}: 平均 {g['acc_exp'].mean():.2%}  (1年窗 {g['acc_1y'].mean():.2%})", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "walkforward_monthly_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("==============================================================\n")
        f.write("月度滚动 walk-forward 报告 (LightGBM 方向预测)\n")
        f.write("==============================================================\n\n")
        f.write(f"样本池   : {n_stocks} 只股票,{len(pool)} 条样本\n")
        f.write(f"预测目标 : 未来 {FORECAST_HORIZON} 日涨跌方向\n")
        f.write(f"扩张窗口 : {TRAIN_START} 起\n")
        f.write(f"测试区间 : {res['month'].iloc[0]} ~ {res['month'].iloc[-1]} "
                f"({len(res)} 个月)\n\n")
        f.write(f"平均方向准确率(扩张窗口) : {exp.mean():.2%}\n")
        f.write(f"平均方向准确率(1年窗口)  : {one.mean():.2%}\n")
        f.write(f"提升幅度                 : {diff.mean():+.2%}\n")
        f.write(f"扩张窗优于1年窗月份占比  : {(diff > 0).mean():.1%}\n")
        f.write(f"反向信号后准确率         : {rev_exp.mean():.2%}\n\n")
        f.write("逐月明细:\n")
        f.write(f"{'月份':<10}{'扩张窗':>10}{'1年窗':>10}{'提升':>10}{'测试数':>8}\n")
        for _, r in res.iterrows():
            f.write(f"{r['month']:<10}{r['acc_exp']:>9.2%}{r['acc_1y']:>9.2%}"
                    f"{r['acc_exp'] - r['acc_1y']:>+9.2%}{int(r['test']):>8}\n")
    print(f"\n报告已保存: {path}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

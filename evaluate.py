"""横截面评估:Rank IC + 分层回测,替代单一方向准确率。

衡量模型的"排序能力"而非"猜对比例":选股只要求预测概率高的股票
确实比低的涨得好,即概率与未来收益的秩相关(IC)为正即可。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import CACHE_DIR, RESULTS_DIR, FORECAST_HORIZON, SEED
from data import load_cached
from features import build_features, FEATURE_COLS

MIN_TRAIN = 120          # 每个调仓日最少需要的历史交易日
REBALANCE_EVERY = 10     # 调仓间隔(交易日),约每两周一次
N_GROUPS = 5             # 分层组数


def load_panel(code):
    """单只股票 -> 含特征与连续未来收益的面板,每行日期对齐。"""
    df = load_cached(code, CACHE_DIR)
    if df is None:
        return None
    feat = build_features(df, FORECAST_HORIZON)
    raw = df.set_index("date")["close"]
    fwd = (raw.shift(-FORECAST_HORIZON) / raw - 1).rename("fwd_ret")
    return feat.merge(fwd.reset_index(), on="date", how="left")


def _train_predict(train, test_row):
    if len(train) < 60:
        return None
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1,
    )
    model.fit(train[FEATURE_COLS].to_numpy(), train["target"].to_numpy())
    return float(model.predict_proba(test_row[FEATURE_COLS].to_numpy())[:, 1][0])


def run_cross_sectional(min_train=MIN_TRAIN, rebalance_every=REBALANCE_EVERY):
    codes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(CACHE_DIR, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()  # 6位纯数字个股,排除指数与旧格式
    )
    panels = {}
    for c in codes:
        p = load_panel(c)
        if p is not None and len(p) >= min_train + FORECAST_HORIZON:
            panels[c] = p
    print(f"加载 {len(panels)} 只股票面板")

    calendar = sorted(set().union(*[set(p["date"]) for p in panels.values()]))
    rebal_dates = [
        calendar[i]
        for i in range(min_train, len(calendar) - FORECAST_HORIZON, rebalance_every)
    ]
    print(f"调仓日 {len(rebal_dates)} 个: {rebal_dates[0].date()} ... {rebal_dates[-1].date()}")

    ic_list = []
    group_returns = {g: [] for g in range(N_GROUPS)}
    n_stocks_per_rebal = []

    for t in rebal_dates:
        cross = []
        for c, p in panels.items():
            mask = p["date"] == t
            if not mask.any():
                continue
            i = int(np.flatnonzero(mask.to_numpy())[0])
            # 训练集只用标签已在 t 日收盘前兑现的样本,杜绝未来信息
            train = p.iloc[: i - FORECAST_HORIZON + 1]
            if len(train) < 60:
                continue
            prob = _train_predict(train, p.iloc[[i]])
            if prob is None:
                continue
            fwd_ret = p["fwd_ret"].iloc[i]
            if pd.isna(fwd_ret):
                continue
            cross.append((c, prob, fwd_ret))

        if len(cross) < 30:
            continue
        df = pd.DataFrame(cross, columns=["code", "prob", "fwd_ret"])
        ic = df["prob"].corr(df["fwd_ret"], method="spearman")
        ic_list.append(ic)

        df["rank"] = df["prob"].rank(method="first", pct=True)
        df["group"] = np.ceil(df["rank"] * N_GROUPS).astype(int) - 1  # 0=最低, N-1=最高
        for g in range(N_GROUPS):
            group_returns[g].append(df.loc[df["group"] == g, "fwd_ret"].mean())
        n_stocks_per_rebal.append(len(df))

    return summarize(ic_list, group_returns, n_stocks_per_rebal, len(rebal_dates))


def summarize(ic_list, group_returns, n_stocks, n_rebal):
    ic = np.array(ic_list)
    avg_n = float(np.mean(n_stocks))
    print(f"\n有效调仓期 {len(ic)} / {n_rebal},每期平均 {avg_n:.0f} 只股票")
    print("\n=== Rank IC (预测概率 vs 未来收益 秩相关) ===")
    print(f"  平均 Rank IC : {ic.mean():+.4f}")
    print(f"  IC 标准差    : {ic.std(ddof=1):.4f}")
    print(f"  ICIR (信息比率): {ic.mean() / ic.std(ddof=1):.3f}" if ic.std(ddof=1) > 0 else "  ICIR: N/A")
    print(f"  IC>0 占比    : {(ic > 0).mean():.1%}")

    print("\n=== 分层回测 (按概率分5组, 组5=概率最高) ===")
    print(f"{'分组':<6}{'平均单期收益':>14}{'累计净值':>12}")
    lines = []
    for g in range(N_GROUPS - 1, -1, -1):
        r = np.array(group_returns[g])
        cum = float(np.prod(1.0 + r))
        lines.append(f"  Q{g + 1}   {r.mean():>+12.2%}   {cum:>10.3f}")
        print(f"  Q{g + 1}   {r.mean():>+12.2%}   {cum:>10.3f}")

    top = np.array(group_returns[N_GROUPS - 1])
    bot = np.array(group_returns[0])
    ls = top - bot
    print(f"\n  多空(Q{N_GROUPS}-Q1) 平均单期收益: {ls.mean():+.2%}")
    print(f"  多空胜率(>0 占比): {(ls > 0).mean():.1%}")

    out = {
        "avg_rank_ic": ic.mean(),
        "ic_std": ic.std(ddof=1),
        "icir": ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else float("nan"),
        "ic_positive_ratio": (ic > 0).mean(),
        "group_returns": {f"Q{g + 1}": float(np.array(group_returns[g]).mean()) for g in range(N_GROUPS)},
        "long_short": float(ls.mean()),
        "ls_win_ratio": float((ls > 0).mean()),
    }
    return out


def save_report(result):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("==============================================================\n")
        f.write("横截面评估报告 (Rank IC + 分层回测)\n")
        f.write("==============================================================\n\n")
        f.write(f"平均 Rank IC     : {result['avg_rank_ic']:+.4f}\n")
        f.write(f"IC 标准差        : {result['ic_std']:.4f}\n")
        f.write(f"ICIR (信息比率)  : {result['icir']:+.3f}\n")
        f.write(f"IC>0 占比        : {result['ic_positive_ratio']:.1%}\n\n")
        f.write("分层平均单期收益:\n")
        for g in range(N_GROUPS - 1, -1, -1):
            f.write(f"  Q{g + 1}  {result['group_returns'][f'Q{g + 1}']:+.2%}\n")
        f.write(f"\n多空(Q{N_GROUPS}-Q1): {result['long_short']:+.2%}  "
                f"(胜率 {result['ls_win_ratio']:.1%})\n")
    print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    res = run_cross_sectional()
    save_report(res)

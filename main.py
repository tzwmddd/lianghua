"""主入口:数据 -> 特征 -> walk-forward -> 回测 -> 指标与绘图。"""
import os

import numpy as np
import pandas as pd

from config import (
    SYMBOL, START_DATE, END_DATE, FORECAST_HORIZON,
    TRAIN_RATIO, STEP, ROUND_TRIP_COST, CACHE_DIR, RESULTS_DIR,
)
from data import get_data
from features import build_features
from models import fit_predict_lstm, fit_predict_lgb
from backtest import run_backtest
from plot import plot_equity


def walk_forward(feat):
    """滚动训练 + 预测,返回 (lstm_pred, lgb_pred),均为 date/prob。"""
    n = len(feat)
    train_end = int(n * TRAIN_RATIO)
    lstm_parts, lgb_parts = [], []
    while train_end < n:
        test_end = min(train_end + STEP, n)
        train_df = feat.iloc[:train_end]
        test_df = feat.iloc[train_end:test_end]
        lstm_parts.append(fit_predict_lstm(train_df, test_df))
        lgb_parts.append(fit_predict_lgb(train_df, test_df))
        train_end = test_end
    return pd.concat(lstm_parts, ignore_index=True), pd.concat(lgb_parts, ignore_index=True)


def to_signal(pred):
    pred = pred.copy()
    pred["signal"] = np.where(pred["prob"].to_numpy() > 0.5, 1.0, -1.0)
    return pred


def accuracy(pred, feat):
    tgt = feat.set_index("date").loc[pred["date"], "target"].to_numpy()
    return float((np.where(pred["prob"].to_numpy() > 0.5, 1, 0) == tgt).mean())


def benchmark_curve(feat, start_date):
    b = feat[feat["date"] >= start_date][["date", "close"]].copy()
    b["equity"] = b["close"] / b["close"].iloc[0]
    return b[["date", "equity"]]


def print_metrics(name, acc, metrics):
    print(f"\n=== {name} ===")
    print(f"  方向准确率: {acc:.2%}")
    print(f"  累计收益:   {metrics['total_return']:.2%}")
    print(f"  年化收益:   {metrics['annual_return']:.2%}")
    print(f"  年化波动:   {metrics['annual_vol']:.2%}")
    print(f"  夏普比率:   {metrics['sharpe']:.3f}")
    print(f"  最大回撤:   {metrics['max_drawdown']:.2%}")
    print(f"  调仓次数:   {metrics['n_trades']}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"标的 {SYMBOL}  区间 {START_DATE}-{END_DATE}")

    df = get_data(SYMBOL, START_DATE, END_DATE, CACHE_DIR)
    feat = build_features(df, FORECAST_HORIZON)
    print(f"有效样本: {len(feat)} 个交易日")

    lstm_pred, lgb_pred = walk_forward(feat)
    lstm_pred = to_signal(lstm_pred)
    lgb_pred = to_signal(lgb_pred)
    print(f"样本外预测: {len(lstm_pred)} 天")

    lstm_acc = accuracy(lstm_pred, feat)
    lgb_acc = accuracy(lgb_pred, feat)

    lstm_curve, lstm_metrics = run_backtest(feat, lstm_pred, FORECAST_HORIZON, ROUND_TRIP_COST)
    lgb_curve, lgb_metrics = run_backtest(feat, lgb_pred, FORECAST_HORIZON, ROUND_TRIP_COST)

    print_metrics("LSTM", lstm_acc, lstm_metrics)
    print_metrics("LightGBM", lgb_acc, lgb_metrics)

    start_date = lstm_curve["date"].iloc[0]
    curves = {
        "LSTM": lstm_curve,
        "LightGBM": lgb_curve,
        "沪深300": benchmark_curve(feat, start_date),
    }
    out = plot_equity(curves, f"{SYMBOL} 多空策略净值对比", "equity_curve.png")
    print(f"\n净值曲线已保存: {out}")


if __name__ == "__main__":
    main()

"""回测引擎:多空双向、信号次日生效、每 horizon 天调仓,含 A股交易成本。"""
import numpy as np
import pandas as pd


def run_backtest(feat_df, pred_df, horizon, round_trip_cost):
    """按方向信号回测,返回净值曲线与指标。

    feat_df: build_features 输出(含 date/close)
    pred_df: 含 date 与 signal(+1 做多 / -1 做空),覆盖样本外测试期
    horizon: 调仓周期(交易日)
    round_trip_cost: 一次完整换手(平旧仓 + 开新仓)的总成本

    持仓逻辑:每隔 horizon 天在收盘后根据当日信号调仓,次日生效;
    方向不变则继续持有不产生成本,首次建仓收半个 round-trip,
    多空翻转收一个完整 round-trip。
    """
    df = feat_df.set_index("date")
    ret = df["close"].pct_change().fillna(0.0)
    signal = pred_df.set_index("date")["signal"]

    dates = list(signal.index)
    pos = 0.0
    equity = 1.0
    n_trades = 0
    curve = []

    for k, d in enumerate(dates):
        equity *= (1.0 + pos * float(ret.get(d, 0.0)))   # 用当前持仓结算当日收益
        if k % horizon == 0:                              # 收盘后调仓,次日生效
            new_pos = float(signal.loc[d])
            if new_pos != pos:
                cost = round_trip_cost / 2.0 if pos == 0.0 else round_trip_cost
                equity *= (1.0 - cost)
                pos = new_pos
                n_trades += 1
        curve.append((d, equity))

    curve_df = pd.DataFrame(curve, columns=["date", "equity"])
    return curve_df, _metrics(curve_df, n_trades)


def _metrics(curve_df, n_trades):
    eq = curve_df["equity"].to_numpy()
    n = len(eq)
    years = n / 252.0

    total_return = eq[-1] - 1.0
    annual_return = eq[-1] ** (1.0 / years) - 1.0 if years > 0 else 0.0

    daily = np.diff(eq) / eq[:-1]
    annual_vol = daily.std(ddof=1) * np.sqrt(252.0)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1.0).min())

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
    }

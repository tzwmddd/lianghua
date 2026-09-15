"""特征工程:技术指标 + 目标变量,滞后对齐,严禁未来数据。"""
import numpy as np
import pandas as pd


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _macd(close, fast=12, slow=26, signal=9):
    dif = _ema(close, fast) - _ema(close, slow)
    dea = _ema(dif, signal)
    hist = dif - dea
    return dif, dea, hist


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def _rsrs(df, n=18, m=250):
    """RSRS 阻力支撑相对强弱(光大证券):high=alpha+beta*low 的 beta 斜率,再做 z-score。

    捕捉支撑强度,是横截面选股的有效因子,只用最高/最低价。
    """
    high = df["high"]
    low = df["low"]
    cov = low.rolling(n).cov(high)
    var = low.rolling(n).var()
    beta = cov / var
    mean = beta.rolling(m).mean()
    std = beta.rolling(m).std()
    return (beta - mean) / (std + 1e-8)


def build_features(df, horizon, keep_tail=False):
    """由 OHLCV 日线构造特征与标签。

    返回 DataFrame:date / open / close / 各特征列 / target。
    所有特征只用 t 及以前的数据;target 为未来 horizon 天涨跌方向。
    keep_tail=True 时保留最后 horizon-1 天(target 未知),用于预测"今天"方向。
    """
    df = df.copy()
    close = df["close"]
    volume = df["volume"]

    ret = close.pct_change()
    feat = pd.DataFrame(index=df.index)

    # 滞后收益率(过去 1..5 日)
    for k in range(1, 6):
        feat[f"ret_lag{k}"] = ret.shift(k)

    # 价格相对均线位置
    feat["ma_ratio_5"] = close / close.rolling(5).mean() - 1
    feat["ma_ratio_20"] = close / close.rolling(20).mean() - 1
    feat["ma_ratio_60"] = close / close.rolling(60).mean() - 1

    # MACD
    dif, dea, hist = _macd(close)
    feat["macd_dif"] = dif
    feat["macd_dea"] = dea
    feat["macd_hist"] = hist

    # RSI
    feat["rsi"] = _rsi(close)

    # 布林带位置(偏离均线几个标准差)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["boll"] = (close - ma20) / (2 * std20)

    # 历史波动率(年化)
    feat["vol_20"] = ret.rolling(20).std() * np.sqrt(252)

    # 量能比(当日成交量 / 20日均量)
    feat["volume_ratio"] = volume / volume.rolling(20).mean()

    # ---- 新增特征(2026-08-27) ----
    # RSRS 阻力支撑相对强弱(横截面因子)
    feat["rsrs"] = _rsrs(df, m=60)

    # 日内振幅(最高最低价差 / 收盘价)
    feat["amplitude"] = (df["high"] - df["low"]) / close

    # 多周期波动率
    feat["vol_5"] = ret.rolling(5).std() * np.sqrt(252)
    feat["vol_60"] = ret.rolling(60).std() * np.sqrt(252)

    # 中期动量
    feat["ret_lag10"] = ret.shift(10)
    feat["ret_lag20"] = ret.shift(20)

    # 量能结构(5日均量 / 20日均量)
    feat["volume_ma_ratio"] = volume.rolling(5).mean() / volume.rolling(20).mean()

    # 目标变量:未来 horizon 天涨跌方向(仅作标签,非特征)
    future_ret = close.shift(-horizon) / close - 1
    feat["target"] = (future_ret > 0).astype(int).where(future_ret.notna())

    out = pd.concat([df[["date", "open", "close"]], feat], axis=1)
    if keep_tail:
        # 只按特征列去 NaN(前 60 天),保留 target 为 NaN 的最后几天
        feat_cols = [c for c in feat.columns if c != "target"]
        out = out.dropna(subset=feat_cols).reset_index(drop=True)
    else:
        out = out.dropna().reset_index(drop=True)
    return out


FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
    "ma_ratio_5", "ma_ratio_20", "ma_ratio_60",
    "macd_dif", "macd_dea", "macd_hist",
    "rsi", "boll", "vol_20", "volume_ratio",
    "rsrs", "amplitude", "vol_5", "vol_60",
    "ret_lag10", "ret_lag20", "volume_ma_ratio",
]

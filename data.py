"""数据层:akshare 拉取 + 增量缓存 + 列名统一。"""
import glob
import os
import random
import time

import pandas as pd

# 个股接口返回的中文列名 -> 统一英文列名
CN_TO_EN = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}


def _is_index(symbol):
    """指数代码带 sh/sz/bj 前缀;个股为 6 位纯数字。"""
    return symbol.lower().startswith(("sh", "sz", "bj"))


def _to_sina_symbol(symbol):
    """6 位数字代码 -> 新浪带交易所前缀代码。"""
    if symbol.startswith(("6", "5", "9")):
        return "sh" + symbol
    if symbol.startswith(("0", "3", "2")):
        return "sz" + symbol
    if symbol.startswith(("4", "8")):
        return "bj" + symbol
    return symbol


def _with_retry(fn, retries=3, delay=(3, 5)):
    """随机延时 + 指数退避,缓解东财源断连(RemoteDisconnected)。"""
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - 捕获网络/上游所有异常并重试
            last = e
            if i < retries - 1:
                time.sleep(random.uniform(*delay) * (i + 1))
    raise last


def _require_not_frozen(cache_dir):
    """数据冻结保护:cache/_FROZEN 存在时禁止网络重拉(保证结果可复现)。"""
    if os.path.exists(os.path.join(cache_dir, "_FROZEN")):
        raise RuntimeError(
            "数据已冻结(见 cache/_FROZEN): 禁止网络重拉,请使用冻结缓存"
        )


def _fetch(symbol, start, end):
    import akshare as ak  # 延迟导入,避免未安装时阻断模块加载

    if _is_index(symbol):
        # 新浪源,返回全历史,列名已为英文(date/open/high/low/close/volume)
        df = ak.stock_zh_index_daily(symbol=symbol)
    else:
        # 新浪源前复权日线,列名已为英文
        df = ak.stock_zh_a_daily(
            symbol=_to_sina_symbol(symbol),
            start_date=start, end_date=end, adjust="qfq",
        )
    return df


def _postprocess(df):
    df = df.rename(columns=CN_TO_EN).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    keep = ["date", "open", "high", "low", "close", "volume"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def _cache_path(symbol, cache_dir):
    return os.path.join(cache_dir, f"{symbol}.csv")


def _read_cache(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)


def load_cached(symbol, cache_dir):
    """读完整缓存(不做网络更新),无缓存返回 None。供评估/回测直接复用。"""
    return _read_cache(_cache_path(symbol, cache_dir))


def _migrate_legacy(symbol, cache_dir):
    """旧格式 {symbol}_{start}_{end}.csv 合并成新格式的初始内容(仅首次)。"""
    files = glob.glob(os.path.join(cache_dir, f"{symbol}_*.csv"))
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, parse_dates=["date"]))
        except Exception:
            pass
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)


def _is_consistent(cached, new):
    """增量与缓存衔接处是否连续。

    前复权以"最新交易日"为基准,期间除权会导致历史价格整体重估,
    增量首日与缓存末日出现跳变。此时须回退全量重拉。
    """
    if len(cached) == 0 or len(new) == 0:
        return True
    gap = float(new["close"].iloc[0] / cached["close"].iloc[-1] - 1.0)
    return abs(gap) < 0.20


def get_data(symbol, start, end, cache_dir, retain_days=730):
    """增量缓存:复用已有数据,只拉新增区间,丢弃窗口外的过期数据。

    返回 [start, end] 的标准化日线;缓存文件 {symbol}.csv 持续滚动更新。
    """
    os.makedirs(cache_dir, exist_ok=True)
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)

    # 指数源返回全历史且仅单标的,直接按需拉取并缓存,无需增量
    if _is_index(symbol):
        path = _cache_path(symbol, cache_dir)
        cached = _read_cache(path)
        if cached is None or cached["date"].min() > start_ts or cached["date"].max() < end_ts:
            _require_not_frozen(cache_dir)
            cached = _postprocess(_with_retry(lambda: _fetch(symbol, start, end)))
            cached.to_csv(path, index=False)
        return cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)].reset_index(drop=True)

    path = _cache_path(symbol, cache_dir)
    cached = _read_cache(path)
    if cached is None or len(cached) == 0:
        cached = _migrate_legacy(symbol, cache_dir)

    need_full = cached is None or len(cached) == 0
    if not need_full:
        cache_min, cache_max = cached["date"].min(), cached["date"].max()
        if cache_min > start_ts:
            need_full = True  # 请求比缓存更早的数据,重拉
        elif cache_max < end_ts:
            _require_not_frozen(cache_dir)
            inc_start = (cache_max + pd.Timedelta(days=1)).strftime("%Y%m%d")
            try:
                new = _postprocess(_with_retry(
                    lambda: _fetch(symbol, inc_start, end_ts.strftime("%Y%m%d"))
                ))
            except Exception:
                new = None
            if new is not None and len(new) > 0:
                if not _is_consistent(cached, new):
                    need_full = True  # 除权导致基准跳变,重拉全量
                else:
                    cached = pd.concat([cached, new], ignore_index=True)
                    cached = cached.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if need_full:
        _require_not_frozen(cache_dir)
        cached = _postprocess(_with_retry(
            lambda: _fetch(symbol, start_ts.strftime("%Y%m%d"), end_ts.strftime("%Y%m%d"))
        ))

    # 丢弃窗口外的过期数据,仅保留 start 之前 retain_days 用于特征预热
    keep_from = start_ts - pd.Timedelta(days=retain_days)
    cached = cached[cached["date"] >= keep_from].reset_index(drop=True)
    cached.to_csv(path, index=False)

    return cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)].reset_index(drop=True)

"""下载两融(融资融券)横截面面板:调仓日的融资余额/融资买入额。

两融是独立于技术面的正交数据源(信用交易账户的杠杆资金行为)。akshare 免费源
stock_margin_detail_szse(深市,纯个股) / stock_margin_detail_sse(沪市,含ETF) 逐日
横截面,点-in-time 无泄露,历史可拉到 2017。

只下载调仓日(2017-01 起每20交易日)的两融快照,存 cache/_margin_panel.csv,
后续前向填充对齐日频 pool。断点续传。
"""
import glob
import os
import random
import time

import pandas as pd


def _retry(fn, retries=3, delay=(2.0, 4.0)):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(random.uniform(*delay) * (i + 1))
    raise last


def rebal_dates(cache_dir, test_start="2015-01-01", every=20):
    idx = pd.read_csv(os.path.join(cache_dir, "sh000300.csv"), parse_dates=["date"])
    dates = sorted(idx["date"].unique())
    start_idx = next(i for i, d in enumerate(dates) if d >= pd.Timestamp(test_start))
    return dates[start_idx::every]


def fetch_day(ak, date_str):
    """拉某日深市+沪市两融明细,统一字段。返回 DataFrame(code/margin_balance/margin_buy)。"""
    parts = []
    sz = _retry(lambda: ak.stock_margin_detail_szse(date=date_str), retries=2)
    sz = sz[["证券代码", "融资余额", "融资买入额"]].copy()
    sz.columns = ["code", "margin_balance", "margin_buy"]
    sz["code"] = sz["code"].astype(str).str.zfill(6)
    parts.append(sz)

    sh = _retry(lambda: ak.stock_margin_detail_sse(date=date_str), retries=2)
    sh = sh[["标的证券代码", "融资余额", "融资买入额"]].copy()
    sh.columns = ["code", "margin_balance", "margin_buy"]
    sh["code"] = sh["code"].astype(str).str.zfill(6)
    parts.append(sh)

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.Timestamp(date_str)
    return df[["date", "code", "margin_balance", "margin_buy"]]


def main():
    import akshare as ak
    base = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base, "cache")
    path = os.path.join(cache_dir, "_margin_panel.csv")

    dates = rebal_dates(cache_dir)
    dates_str = [d.strftime("%Y%m%d") for d in dates]
    print(f"共 {len(dates_str)} 个调仓日: {dates_str[0]} ~ {dates_str[-1]}", flush=True)

    done = set()
    if os.path.exists(path):
        done = set(pd.read_csv(path, parse_dates=["date"])["date"].dt.strftime("%Y%m%d").unique())
    todo = [d for d in dates_str if d not in done]
    print(f"已下载 {len(done)} 日, 待下载 {len(todo)} 日", flush=True)

    parts = []
    if os.path.exists(path):
        parts.append(pd.read_csv(path, parse_dates=["date"], dtype={"code": str}))

    for i, d in enumerate(todo):
        ok = False
        for _ in range(3):
            try:
                df = fetch_day(ak, d)
                parts.append(df)
                ok = True
                break
            except Exception:
                time.sleep(random.uniform(2.0, 4.0))
        if not ok:
            print(f"  {d} 失败,跳过", flush=True)
        if (i + 1) % 10 == 0:
            pd.concat(parts, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  进度 {i + 1}/{len(todo)}, 已保存", flush=True)
            time.sleep(random.uniform(0.5, 1.0))

    pd.concat(parts, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"完成, 保存到 {path}", flush=True)


if __name__ == "__main__":
    main()

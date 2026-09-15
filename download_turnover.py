"""下载换手率/成交额面板:新浪 stock_zh_a_daily 的 turnover/amount 列。

现有日线缓存(cache/*.csv)只保留 OHLCV,turnover/amount/outstanding_share 被
data.py 的 _postprocess 丢弃。换手率是 A股横截面强因子(低换手->未来收益更高),
日频、点-in-time、无未来泄露。

本脚本只【新增】cache/_turnover_panel.csv,不修改/不删除现有冻结日线缓存,
直接调新浪接口(不经过 data.py 的冻结保护),支持断点续传。
"""
import glob
import os
import random
import time

import pandas as pd


def _to_sina_symbol(symbol):
    if symbol.startswith(("6", "5", "9")):
        return "sh" + symbol
    if symbol.startswith(("0", "3", "2")):
        return "sz" + symbol
    if symbol.startswith(("4", "8")):
        return "bj" + symbol
    return symbol


def main():
    import akshare as ak
    base = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base, "cache")
    path = os.path.join(cache_dir, "_turnover_panel.csv")

    codes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(cache_dir, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    print(f"共 {len(codes)} 只股票", flush=True)

    done = set()
    if os.path.exists(path):
        done = set(pd.read_csv(path, dtype={"code": str})["code"].unique())
    todo = [c for c in codes if c not in done]
    print(f"已下载 {len(done)} 只, 待下载 {len(todo)} 只", flush=True)

    parts = []
    if os.path.exists(path):
        parts.append(pd.read_csv(path, parse_dates=["date"], dtype={"code": str}))

    for i, code in enumerate(todo):
        ok = False
        for attempt in range(3):
            try:
                df = ak.stock_zh_a_daily(
                    symbol=_to_sina_symbol(code),
                    start_date="20150101", end_date="20261231", adjust="qfq",
                )
                sub = df[["date", "turnover", "amount"]].copy()
                sub["date"] = pd.to_datetime(sub["date"])
                sub["code"] = code
                parts.append(sub)
                ok = True
                break
            except Exception:
                time.sleep(random.uniform(2.0, 4.0) * (attempt + 1))
        if not ok:
            print(f"  {code} 失败,跳过", flush=True)
        if (i + 1) % 20 == 0:
            pd.concat(parts, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  进度 {i + 1}/{len(todo)}, 已保存", flush=True)
            time.sleep(random.uniform(0.5, 1.0))

    pd.concat(parts, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"完成, 保存到 {path}", flush=True)


if __name__ == "__main__":
    main()

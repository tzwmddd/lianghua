"""下载中证500/1000 每日指标(市值/PE/PB/换手率), 用 tushare daily_basic。

逐股拉历史(单股 11 年约 2750 行 < 单次 6000 行上限)。total_mv 单位万元,
在 fundamental_ts.py 里转亿(÷1e4)后取 log。缓存 {cache}/_daily_basic.csv
(code/date/total_mv/pe_ttm/pb/turnover_rate)。
"""
import os
import sys
import time

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import tushare_common as tc

UNIVERSES = {
    "500": ("000905.SH", "cache_zz500"),
    "1000": ("000852.SH", "cache_zz1000"),
}
START, END = "20150101", "20261231"
COLS = ["total_mv", "pe_ttm", "pb", "turnover_rate"]


def fetch(pro, ts_code):
    df = pro.daily_basic(ts_code=ts_code, start_date=START, end_date=END)
    if df is None or df.empty:
        return None
    keep = ["trade_date"] + [c for c in COLS if c in df.columns]
    sub = df[keep].copy()
    sub["date"] = pd.to_datetime(sub["trade_date"])
    sub["code"] = tc.to_code(ts_code)
    return sub[["code", "date"] + [c for c in COLS if c in keep]].sort_values("date").reset_index(drop=True)


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit(f"用法: python download_zz_daily_basic.py [500|1000]")
    _, cache_dir = UNIVERSES[universe]

    cons = pd.read_csv(os.path.join(cache_dir, "_constituents.csv"), dtype={"code": str})
    codes = sorted(cons["code"].unique())
    path = os.path.join(cache_dir, "_daily_basic.csv")
    done = set()
    if os.path.exists(path):
        done = set(pd.read_csv(path, usecols=["code"], dtype={"code": str})["code"])
    todo = [c for c in codes if c not in done]
    print(f"{cache_dir}: 成分股 {len(codes)} 只, 已有 {len(done)} 只, 待下载 {len(todo)} 只", flush=True)

    pro = tc.get_pro()
    parts = []
    t0 = time.time()
    for i, code in enumerate(todo):
        try:
            df = tc._retry(lambda: fetch(pro, tc.to_ts_code(code)))
            if df is not None and len(df):
                parts.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"  {code} 失败: {repr(e)[:50]}", flush=True)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  进度 {i + 1}/{len(todo)} 用时 {el:.0f}s", flush=True)
        tc.sleep(0.35)

    if parts:
        new = pd.concat(parts, ignore_index=True)
        if os.path.exists(path):
            old = pd.read_csv(path, parse_dates=["date"], dtype={"code": str})
            new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"])
        new.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n保存 {path}: {len(new)} 行, {new['code'].nunique()} 只", flush=True)


if __name__ == "__main__":
    main()

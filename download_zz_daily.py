"""下载中证500/1000 成分股并集 + 指数日线(前复权), 用 tushare daily + adj_factor。

前复权 qfq = price × adj_factor / adj_factor[最新],对 OHLC 统一用收盘复权因子
(tushare 官方做法,月度选股误差可忽略)。缓存 {code}.csv 列名与现有 data.py 一致
(date/open/high/low/close/volume),可直接被 load_cached 读。

退市/停牌拉不到的股票记录失败并跳过(存 {cache}/_daily_failed.txt)。
"""
import glob
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


def fetch_qfq_daily(pro, ts_code):
    """拉不复权日线 + 复权因子,算前复权 OHLCV。失败返回 None。"""
    daily = pro.daily(ts_code=ts_code, start_date=START, end_date=END)
    adj = pro.adj_factor(ts_code=ts_code, start_date=START, end_date=END)
    if daily is None or daily.empty or adj is None or adj.empty:
        return None
    daily = daily.rename(columns={"vol": "volume"})
    merged = daily.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
    merged = merged.sort_values("trade_date")
    merged["adj_factor"] = merged["adj_factor"].ffill().bfill()
    latest_adj = merged["adj_factor"].iloc[-1]
    factor = merged["adj_factor"] / latest_adj
    for col in ["open", "high", "low", "close"]:
        merged[col] = merged[col] * factor
    merged["date"] = pd.to_datetime(merged["trade_date"])
    out = merged[["date", "open", "high", "low", "close", "volume"]].copy()
    return out.sort_values("date").reset_index(drop=True)


def fetch_index_daily(pro, index_code):
    df = pro.index_daily(ts_code=index_code, start_date=START, end_date=END)
    if df is None or df.empty:
        return None
    df = df.rename(columns={"vol": "volume"})
    df["date"] = pd.to_datetime(df["trade_date"])
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit(f"用法: python download_zz_daily.py [500|1000]")
    index_code, cache_dir = UNIVERSES[universe]
    os.makedirs(cache_dir, exist_ok=True)

    cons = pd.read_csv(os.path.join(cache_dir, "_constituents.csv"), dtype={"code": str})
    all_codes = sorted(cons["code"].unique())
    have = set(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(cache_dir, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    todo = [c for c in all_codes if c not in have]
    print(f"{index_code}: 历史成分股 {len(all_codes)} 只, 已下载 {len(have)} 只, "
          f"待下载 {len(todo)} 只", flush=True)

    pro = tc.get_pro()
    ok = fail = 0
    failed = []
    t0 = time.time()
    for i, code in enumerate(todo):
        try:
            df = tc._retry(lambda: fetch_qfq_daily(pro, tc.to_ts_code(code)))
            if df is None or len(df) == 0:
                raise ValueError("empty")
            df.to_csv(os.path.join(cache_dir, f"{code}.csv"), index=False)
            ok += 1
        except Exception as e:  # noqa: BLE001 - 退市/停牌拉不到,记录后跳过
            fail += 1
            failed.append((code, repr(e)[:50]))
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  进度 {i + 1}/{len(todo)} (成功 {ok} 失败 {fail}) "
                  f"用时 {el:.0f}s, 预计剩余 {el / max(i + 1, 1) * (len(todo) - i - 1):.0f}s", flush=True)
        tc.sleep(0.5)  # daily/adj_factor 各自限流,每循环各 1 次调用,0.5s→约 120次/分钟 安全

    # 指数日线
    idx = tc._retry(lambda: fetch_index_daily(pro, index_code))
    if idx is not None:
        idx.to_csv(os.path.join(cache_dir, "index.csv"), index=False)
        print(f"指数日线 {index_code}: {len(idx)} 行 -> index.csv", flush=True)

    if failed:
        with open(os.path.join(cache_dir, "_daily_failed.txt"), "w", encoding="utf-8") as f:
            for c, e in failed:
                f.write(f"{c}: {e}\n")
    print(f"\n完成: 成功 {ok} 失败 {fail} 总用时 {time.time() - t0:.0f}s", flush=True)
    if failed:
        print(f"失败 {len(failed)} 只(退市/停牌),见 _daily_failed.txt", flush=True)


if __name__ == "__main__":
    main()

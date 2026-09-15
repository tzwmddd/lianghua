"""下载中证500/1000 历史成分股名单(点-in-time), 用 tushare index_weight。

中证500=000905.SH / 中证1000=000852.SH,成分每半年(6月/12月)调整,但 index_weight
提供月度权重快照,比半年快照更精确。**按月分段拉**,避免单次行数上限静默截断。

输出 cache_zz{universe}/_constituents.csv (snapshot_date, code),格式与现有
cache/_constituents.csv 一致,供点-in-time 回测复用 apply_constituents 逻辑。
"""
import os
import sys

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import tushare_common as tc

UNIVERSES = {
    "500": ("000905.SH", "cache_zz500"),
    "1000": ("000852.SH", "cache_zz1000"),
}
START, END = "2015-01", "2026-12"


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit(f"用法: python download_zz_constituents.py [500|1000]")
    index_code, cache_dir = UNIVERSES[universe]
    os.makedirs(cache_dir, exist_ok=True)

    pro = tc.get_pro()
    months = pd.period_range(START, END, freq="M")
    print(f"{index_code} 拉取 {len(months)} 个月成分 ({START}~{END})", flush=True)

    rows = []
    for m in months:
        start = m.start_time.strftime("%Y%m%d")
        end = m.end_time.strftime("%Y%m%d")
        df = tc._retry(lambda: pro.index_weight(
            index_code=index_code, start_date=start, end_date=end
        ))
        if df is None or df.empty:
            print(f"  {m}: 无数据", flush=True)
            continue
        trade_date = str(df["trade_date"].iloc[0])
        for con in df["con_code"].astype(str):
            rows.append((trade_date, tc.to_code(con)))
        if len(months) > 200 or int(m.strftime("%m")) == 12:
            print(f"  {m}: {len(df)} 只 (trade_date={trade_date})", flush=True)
        tc.sleep(0.5)  # index_weight 限流 200次/分钟,144次×0.5s≈72s 安全

    out = pd.DataFrame(rows, columns=["snapshot_date", "code"]).drop_duplicates()
    out = out.sort_values(["snapshot_date", "code"]).reset_index(drop=True)
    path = os.path.join(cache_dir, "_constituents.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n保存 {path}: {len(out)} 行, {out['code'].nunique()} 只历史成分股(并集), "
          f"{out['snapshot_date'].nunique()} 个月快照", flush=True)


if __name__ == "__main__":
    main()

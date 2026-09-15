"""下载沪深300历史成分股名单(点-in-time), 用 baostock 免费源。

沪深300 每半年(6月/12月)调整一次成分股。对每个调整期查询当时成分股,
输出 cache/_constituents.csv (snapshot_date, code),供回测做点-in-time 过滤。

已验证(2026-08-30): baostock query_hs300_stocks 返回不同时点的真实成分,
2018 vs 2026 有 129 只被调出/调入,非固定一套。
"""
import os
import sys

import baostock as bs
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CACHE = "cache"


def snapshot_dates():
    d = []
    for y in range(2015, 2027):
        d.append(f"{y}-06-30")
        d.append(f"{y}-12-31")
    return d


def main():
    lg = bs.login()
    if lg.error_code != "0":
        print("login 失败:", lg.error_msg, flush=True)
        return
    rows = []
    for day in snapshot_dates():
        rs = bs.query_hs300_stocks(day)
        df = rs.get_data()
        if df.empty or "code" not in df.columns:
            print(f"{day}: 无数据 ({rs.error_code} {rs.error_msg})", flush=True)
            continue
        codes = (df["code"].str.replace("sh.", "").str.replace("sz.", "")
                 .str.replace("bj.", "").str.zfill(6))
        rows.extend((day, c) for c in codes)
        print(f"{day}: {len(codes)} 只", flush=True)
    bs.logout()

    out = pd.DataFrame(rows, columns=["snapshot_date", "code"])
    out.to_csv(os.path.join(CACHE, "_constituents.csv"), index=False, encoding="utf-8-sig")
    print(f"\n保存 cache/_constituents.csv: {len(out)} 行, "
          f"{out['code'].nunique()} 只历史成分股(并集)", flush=True)


if __name__ == "__main__":
    main()

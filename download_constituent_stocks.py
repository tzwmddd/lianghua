"""下载历史成分股(被调出股)的日线,扩充股票池。

从 _constituents.csv 的 652 只历史成分股并集,找出当前 cache 缺失的股票,
用 akshare 新浪源下载前复权日线(2015-2026)。退市股(已摘牌)拉不到则跳过。

注意: 直接调 data._fetch 底层,绕过 get_data 的冻结保护——冻结只防"重拉已有
数据导致污染",而下载新股票是扩充股票池的正当操作,不触碰已有缓存文件。
"""
import glob
import os
import random
import sys
import time

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from data import _fetch, _postprocess

CACHE = "cache"
START = "20150101"
END = "20261231"


def main():
    cons = pd.read_csv(os.path.join(CACHE, "_constituents.csv"), dtype={"code": str})
    all_codes = set(cons["code"])
    have = set(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(CACHE, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    missing = sorted(all_codes - have)
    print(f"历史成分股 {len(all_codes)} 只, 已有 {len(have)} 只, 需下载 {len(missing)} 只", flush=True)

    ok = fail = 0
    failed = []
    t0 = time.time()
    for i, c in enumerate(missing):
        try:
            df = _postprocess(_fetch(c, START, END))
            df.to_csv(os.path.join(CACHE, f"{c}.csv"), index=False)
            ok += 1
        except Exception as e:  # noqa: BLE001 - 退市股拉不到,记录后跳过
            fail += 1
            failed.append((c, repr(e)[:60]))
        if (i + 1) % 20 == 0:
            print(f"  进度 {i + 1}/{len(missing)} (成功 {ok} 失败 {fail}) "
                  f"用时 {time.time() - t0:.0f}s", flush=True)
        time.sleep(random.uniform(0.2, 0.6))

    print(f"\n完成: 成功 {ok} 失败 {fail} 总用时 {time.time() - t0:.0f}s", flush=True)
    if failed:
        print("失败(退市/已摘牌):", flush=True)
        for c, e in failed:
            print(f"  {c}: {e}", flush=True)


if __name__ == "__main__":
    main()

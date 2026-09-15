"""把沪深300成分股缓存从1年扩展到5年(2021-01-01 ~ 2026-now)。"""
import glob
import os
import sys
import time

from config import CACHE_DIR
from data import get_data

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

START = "20210101"
END = "20261231"


def main():
    codes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(CACHE_DIR, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    print(f"待更新 {len(codes)} 只,区间 {START}-{END}", flush=True)

    ok = fail = 0
    for i, code in enumerate(codes, 1):
        try:
            df = get_data(code, START, END, CACHE_DIR)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [失败] {code}: {type(e).__name__}", flush=True)
        if i % 25 == 0:
            print(f"  进度 {i}/{len(codes)} (成功{ok} 失败{fail})", flush=True)
        time.sleep(0.1)

    print(f"\n完成: 成功 {ok} / 失败 {fail}", flush=True)


if __name__ == "__main__":
    main()

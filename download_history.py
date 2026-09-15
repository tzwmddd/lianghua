"""把个股日线缓存扩展到 2015 年,覆盖 2018 熊市/2016 熔断等多市场状态。

用途:验证过拟合——当前模型 IC 0.44 建立在 2024-2026 约 28 个牛市调仓期上,
拉长到 2015 后样本外期数翻几倍、且含熊市,LightGBM 的 IC 应会显著回落。

注意:仍是"当前沪深300成分股"回溯,幸存者偏差未消除(免费源无历史成分股名单)。
"""
import glob
import os
import random
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import CACHE_DIR
from data import get_data

START = "20150101"
END = "20261231"


def main():
    codes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(CACHE_DIR, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    print(f"共 {len(codes)} 只股票,扩展到 {START} ~ {END}", flush=True)
    t0 = time.time()
    ok = fail = 0
    for i, code in enumerate(codes):
        try:
            get_data(code, START, END, CACHE_DIR)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  {code} 失败: {repr(e)[:100]}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  进度 {i + 1}/{len(codes)} (成功 {ok} 失败 {fail}) "
                  f"用时 {time.time() - t0:.0f}s", flush=True)
        time.sleep(random.uniform(0.2, 0.6))
    print(f"\n完成: 成功 {ok} 失败 {fail} 总用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

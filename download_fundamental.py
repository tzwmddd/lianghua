"""下载并缓存估值因子面板:pe_ttm / pb / pcf(百度估值点-in-time 历史序列)。

单独脚本,便于失败后重跑;fetch_valuation_panel 命中缓存则直接跳过。
"""
import sys
import time
import warnings

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from config import CACHE_DIR
from fundamental import fetch_valuation_panel, VALUATION_INDICATORS

if __name__ == "__main__":
    for col in VALUATION_INDICATORS:
        t0 = time.time()
        df = fetch_valuation_panel(CACHE_DIR, col)
        print(f"{col}: {len(df)} 行, {df['code'].nunique()} 只, 耗时 {time.time() - t0:.0f}s", flush=True)
    print("估值面板下载完成", flush=True)

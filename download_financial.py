"""下载沪深300成分股季度财务指标,缓存到 cache/_financial.csv。"""
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import CACHE_DIR
from fundamental import fetch_financial_panel

if __name__ == "__main__":
    panel = fetch_financial_panel(CACHE_DIR)
    print(f"财务因子面板: {panel['code'].nunique()} 只,{len(panel)} 条")
    print(f"字段: {list(panel.columns)}")
    print(panel.head(3).to_string())

"""为历史成分股(被调出股)补充估值+市值+财务数据。

现有 _val_pe_ttm/_val_pb/_val_pcf/_mktcap_panel 只覆盖 300 只当前成分股,
_financial.csv 覆盖 267 只。被调出的历史成分股缺失,需补充(否则 full 点-in-time
回测会因缺基本面因子而大量 dropna)。

直接调 akshare 下载并追加到现有缓存(保留已有数据,不重拉)。
"""
import glob
import os
import random
import sys
import time

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from fundamental import _retry, VALUATION_INDICATORS, FINANCIAL_FIELD_MAP, FINANCIAL_COLS

CACHE = "cache"


def existing_codes(csv):
    p = os.path.join(CACHE, csv)
    if not os.path.exists(p):
        return set()
    df = pd.read_csv(p, dtype={"code": str})
    return set(df["code"])


def missing_codes():
    cons = pd.read_csv(os.path.join(CACHE, "_constituents.csv"), dtype={"code": str})
    all_codes = set(cons["code"])
    have_daily = set(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(CACHE, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    # 只需为"有日线但缺基本面"的股票补充
    targets = all_codes & have_daily
    miss_val = targets - existing_codes("_val_pe_ttm.csv")
    miss_fin = targets - existing_codes("_financial.csv")
    return sorted(miss_val), sorted(miss_fin)


def download_valuation(miss_val):
    """下载缺失股票的 4 个估值指标(总市值+pe_ttm+pb+pcf),追加到对应缓存。"""
    import akshare as ak
    if not miss_val:
        print("估值无需补充", flush=True)
        return
    print(f"补充估值: {len(miss_val)} 只", flush=True)
    panels = {"_mktcap_panel.csv": ("总市值", "mktcap")}
    for col, ind in VALUATION_INDICATORS.items():
        panels[f"_val_{col}.csv"] = (ind, col)

    t0 = time.time()
    for name, (indicator, out_col) in panels.items():
        path = os.path.join(CACHE, name)
        base = pd.read_csv(path, parse_dates=["date"], dtype={"code": str}) if os.path.exists(path) else pd.DataFrame()
        base_codes = set(base["code"]) if len(base) else set()
        todo = [c for c in miss_val if c not in base_codes]
        parts = []
        for i, c in enumerate(todo):
            try:
                s = _retry(lambda: ak.stock_zh_valuation_baidu(
                    symbol=c, indicator=indicator, period="全部"), retries=2, delay=(1.0, 2.0))
                s = s.rename(columns={"value": out_col})
                s["code"] = c
                s["date"] = pd.to_datetime(s["date"])
                parts.append(s[["code", "date", out_col]])
            except Exception:
                pass
            if (i + 1) % 20 == 0:
                print(f"  {name} 进度 {i + 1}/{len(todo)} 用时 {time.time() - t0:.0f}s", flush=True)
            time.sleep(random.uniform(0.2, 0.5))
        if parts:
            new = pd.concat(parts, ignore_index=True)
            merged = pd.concat([base, new], ignore_index=True).drop_duplicates(subset=["code", "date"])
            merged.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  {name}: 新增 {len(new)} 行, 现 {merged['code'].nunique()} 只", flush=True)


def download_financial(miss_fin):
    """下载缺失股票的季度财务指标,追加到 _financial.csv。"""
    import akshare as ak
    if not miss_fin:
        print("财务无需补充", flush=True)
        return
    print(f"补充财务: {len(miss_fin)} 只", flush=True)
    path = os.path.join(CACHE, "_financial.csv")
    base = pd.read_csv(path, parse_dates=["report_date"], dtype={"code": str})
    fields = list(FINANCIAL_FIELD_MAP.values())
    t0 = time.time()
    parts = []
    for i, c in enumerate(miss_fin):
        try:
            df = _retry(lambda: ak.stock_financial_analysis_indicator(symbol=c, start_year="2015"),
                        retries=2, delay=(1.0, 2.0))
            sub = df[["日期"] + fields].copy()
            sub.columns = ["report_date"] + FINANCIAL_COLS
            sub["report_date"] = pd.to_datetime(sub["report_date"])
            sub["code"] = c
            parts.append(sub[["code", "report_date"] + FINANCIAL_COLS])
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  财务进度 {i + 1}/{len(miss_fin)} 用时 {time.time() - t0:.0f}s", flush=True)
        time.sleep(random.uniform(0.3, 0.6))
    if parts:
        new = pd.concat(parts, ignore_index=True)
        merged = pd.concat([base, new], ignore_index=True).drop_duplicates(subset=["code", "report_date"])
        merged.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  财务: 新增 {len(new)} 行, 现 {merged['code'].nunique()} 只", flush=True)


def main():
    miss_val, miss_fin = missing_codes()
    print(f"历史成分股需补: 估值 {len(miss_val)} 只, 财务 {len(miss_fin)} 只", flush=True)
    download_valuation(miss_val)
    download_financial(miss_fin)
    print("\n完成", flush=True)


if __name__ == "__main__":
    main()

"""数据源盘点:确定每个因子的来源 + 稳定性(覆盖/空值/离群/格式)。

产出一份"因子 -> 数据源 -> 缓存文件 -> 稳定性"清单,是固定数据、保证可复现的基础。
只读 cache/,不触发任何网络请求。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CACHE = "cache"


def codes():
    return sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(CACHE, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )


def audit_daily():
    cs = codes()
    rows, starts, ends = [], [], []
    for c in cs:
        df = pd.read_csv(os.path.join(CACHE, f"{c}.csv"), parse_dates=["date"])
        rows.append(len(df))
        starts.append(df["date"].min())
        ends.append(df["date"].max())
    print("=== 1. 个股日线 (cache/{code}.csv, 新浪 stock_zh_a_daily 前复权) ===")
    print(f"  股票数 {len(cs)}, 总行数 {sum(rows)}, "
          f"单股行数 min/中位/max = {min(rows)}/{int(np.median(rows))}/{max(rows)}")
    print(f"  日期范围 {min(starts).date()} ~ {max(ends).date()}")
    idx = pd.read_csv(os.path.join(CACHE, "sh000300.csv"), parse_dates=["date"])
    print(f"  指数 sh000300: {len(idx)} 行, {idx['date'].min().date()} ~ {idx['date'].max().date()}")


def audit_industry():
    p = os.path.join(CACHE, "_industry_map.csv")
    cs = set(codes())
    m = pd.read_csv(p, dtype={"code": str})
    covered = set(m["code"])
    missing = cs - covered
    print("\n=== 2. 行业映射 (cache/_industry_map.csv, 新浪 stock_sector_spot/stock_sector_detail) ===")
    print(f"  映射 {len(m)} 只, 覆盖 {len(cs & covered)}/{len(cs)} ({len(cs & covered)/len(cs):.0%})")
    print(f"  板块数 {m['industry'].nunique()}")
    if missing:
        print(f"  缺失 {len(missing)} 只: {sorted(missing)[:10]}")


def audit_valuation(name, col):
    p = os.path.join(CACHE, f"_val_{name}.csv")
    if not os.path.exists(p):
        print(f"  {name}: 缺失")
        return
    df = pd.read_csv(p, parse_dates=["date"], dtype={"code": str})
    v = df[col]
    pos = (v > 0).mean()
    print(f"  {name}: {len(df)} 行, {df['code'].nunique()} 只, "
          f"正值占比 {pos:.1%}, 中位 {v.median():.2f}")


def audit_mktcap():
    p = os.path.join(CACHE, "_mktcap_panel.csv")
    df = pd.read_csv(p, parse_dates=["date"], dtype={"code": str})
    v = df["mktcap"]
    print(f"  总市值: {len(df)} 行, {df['code'].nunique()} 只, "
          f"正值占比 {(v > 0).mean():.1%}, 中位 {v.median():.1f} 亿")


def audit_financial():
    p = os.path.join(CACHE, "_financial.csv")
    df = pd.read_csv(p, parse_dates=["report_date"], dtype={"code": str})
    cs = set(codes())
    print("\n=== 4. 财务因子 (cache/_financial.csv, 新浪 stock_financial_analysis_indicator) ===")
    print(f"  {len(df)} 行, {df['code'].nunique()} 只 / {len(cs)} 成分股 ({df['code'].nunique()/len(cs):.0%})")
    print(f"  报告期 {df['report_date'].min().date()} ~ {df['report_date'].max().date()}")
    print(f"\n  {'字段':<22}{'非空率':>8}{'中位':>10}{'min':>12}{'max':>12}")
    for c in [x for x in df.columns if x not in ("code", "report_date")]:
        v = df[c]
        nn = v.notna().mean()
        fin = v[np.isfinite(v)]
        if len(fin) == 0:
            print(f"  {c:<22}{nn:>8.0%}{'NaN':>10}")
        else:
            print(f"  {c:<22}{nn:>8.0%}{fin.median():>10.2f}{fin.min():>12.2f}{fin.max():>12.2f}")


def main():
    audit_daily()
    print("\n=== 3. 估值+市值 (cache/_val_*.csv + _mktcap_panel.csv, 百度 stock_zh_valuation_baidu) ===")
    audit_mktcap()
    audit_valuation("pe_ttm", "pe_ttm")
    audit_valuation("pb", "pb")
    audit_valuation("pcf", "pcf")
    audit_financial()


if __name__ == "__main__":
    main()

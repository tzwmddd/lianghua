"""审计 tushare 数据完整性:日线/市值/财务覆盖率 + ann_date 点-in-time 合理性。

用法: python audit_ts_data.py [500|1000]
"""
import glob
import os
import sys

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from fundamental_ts import FINANCIAL_COLS_TS

UNIVERSES = {"500": "cache_zz500", "1000": "cache_zz1000"}


def audit(cache_dir):
    cons = pd.read_csv(os.path.join(cache_dir, "_constituents.csv"), dtype={"code": str})
    all_codes = set(cons["code"])
    have_daily = set(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(cache_dir, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    print(f"历史成分股并集: {len(all_codes)} 只")
    print(f"日线覆盖: {len(have_daily)}/{len(all_codes)} ({len(have_daily) / len(all_codes):.0%})")
    print(f"  缺失(退市/拉不到): {len(all_codes - have_daily)} 只")

    db_path = os.path.join(cache_dir, "_daily_basic.csv")
    if os.path.exists(db_path):
        db = pd.read_csv(db_path, dtype={"code": str})
        print(f"市值/估值覆盖: {db['code'].nunique()}/{len(all_codes)} "
              f"({db['code'].nunique() / len(all_codes):.0%}), 共 {len(db)} 行")

    fin_path = os.path.join(cache_dir, "_financial.csv")
    if os.path.exists(fin_path):
        fin = pd.read_csv(fin_path, parse_dates=["ann_date", "end_date"], dtype={"code": str})
        print(f"财务覆盖: {fin['code'].nunique()}/{len(all_codes)} "
              f"({fin['code'].nunique() / len(all_codes):.0%})")
        print("财务因子非空率:")
        for c in FINANCIAL_COLS_TS:
            print(f"  {c:<18} {fin[c].notna().mean():.1%}")
        lag = (fin["ann_date"] - fin["end_date"]).dt.days
        print(f"ann_date 滞后(公布-报告期): 中位 {lag.median():.0f} 天, "
              f"P5 {lag.quantile(0.05):.0f} 天, P95 {lag.quantile(0.95):.0f} 天")
        neg = (lag < 0).mean()
        print(f"ann_date < end_date(异常)占比: {neg:.2%}")


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit("用法: python audit_ts_data.py [500|1000]")
    audit(UNIVERSES[universe])


if __name__ == "__main__":
    main()

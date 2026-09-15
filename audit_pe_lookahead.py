"""审计百度估值 PE(TTM) 是否 look-ahead 泄露。

方法: 反推 TTM 净利润 E(t) = 总市值(t) / PE(t)。
- 若百度 PE 用"点-in-time"盈利,E(t) 应在财报【公布日】(avail_date)才跳变;
- 若用"事后盈利",E(t) 会在报告期(report_date)甚至更早就反映新财报 → look-ahead。

对比新浪财务 report_date(报告期) + 估算公布滞后,观察 E(t) 台阶跳变时点。
"""
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CACHE = "cache"


def _report_lag(report_date):
    m = report_date.month
    if m == 12:
        return pd.Timedelta(days=120)
    if m == 6:
        return pd.Timedelta(days=60)
    return pd.Timedelta(days=45)


def load():
    pe = pd.read_csv(os.path.join(CACHE, "_val_pe_ttm.csv"),
                     parse_dates=["date"], dtype={"code": str})
    mcap = pd.read_csv(os.path.join(CACHE, "_mktcap_panel.csv"),
                       parse_dates=["date"], dtype={"code": str})
    fin = pd.read_csv(os.path.join(CACHE, "_financial.csv"),
                      parse_dates=["report_date"], dtype={"code": str})
    m = pe.merge(mcap, on=["code", "date"], how="inner")
    m["e_ttm"] = m["mktcap"] / m["pe_ttm"]  # 反推 TTM 净利润(亿)
    m = m.dropna(subset=["e_ttm"])
    return m, fin


def audit_one(m, fin, code):
    sub = m[m["code"] == code].sort_values("date")
    f = fin[fin["code"] == code].sort_values("report_date")
    if sub.empty or f.empty:
        return
    f = f.assign(avail_date=f["report_date"] + f["report_date"].apply(_report_lag))

    print(f"\n===== {code} 反推TTM净利润 E(t)=市值/PE (亿) =====", flush=True)
    print(f"百度估值序列 {sub['date'].min().date()} ~ {sub['date'].max().date()}, "
          f"{len(sub)} 个交易日", flush=True)

    # 对每个财报报告期,打印 E(t) 在 report_date 前5日 / report_date 后5日 / avail_date 后5日 的值
    print(f"\n{'报告期':<12}{'公布日(估)':<12}{'E@报告期前5日':>14}{'E@报告期后5日':>14}{'E@公布日后5日':>14}", flush=True)
    for _, r in f.iterrows():
        rd = r["report_date"]
        ad = r["avail_date"]
        before = sub[sub["date"] <= rd]["e_ttm"].tail(1)
        after_rd = sub[(sub["date"] > rd) & (sub["date"] <= rd + pd.Timedelta(days=5))]["e_ttm"].tail(1)
        after_ad = sub[(sub["date"] > ad) & (sub["date"] <= ad + pd.Timedelta(days=5))]["e_ttm"].tail(1)
        def fmt(s):
            return f"{s.iloc[0]:>12.1f}" if len(s) else f"{'n/a':>12}"
        print(f"{rd.date()!s:<12}{ad.date()!s:<12}{fmt(before):>14}{fmt(after_rd):>14}{fmt(after_ad):>14}", flush=True)


def main():
    m, fin = load()
    print(f"百度 PE+市值合并 {len(m)} 条, 财务 {fin['code'].nunique()} 只", flush=True)
    for code in ["600519", "600036", "000002", "601318"]:
        audit_one(m, fin, code)


if __name__ == "__main__":
    main()

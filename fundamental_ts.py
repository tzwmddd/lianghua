"""tushare 版横截面基本面数据层:行业 + 市值 + 估值 + 财务。

产出与现有 cross_section.py 相同的 pool schema(industry/log_mcap/pe_ttm/pb/
财务因子),使 cross_section.neutralize 可直接复用(逐日行业+市值中性化)。

数据源(tushare 2000 积分):
- 行业: 申万 SW2021 一级行业(_industry_map.csv)
- 市值/估值: daily_basic(total_mv/pe_ttm/pb,点-in-time)
- 财务: fina_indicator(ann_date 精确公布日,点-in-time)

与旧版差异: 无 PCF(市现率,需 cashflow 接口另算,第一阶段弃用);财务点-in-time
用 ann_date 替代"报告期+固定滞后"。
"""
import os

import numpy as np
import pandas as pd

from fundamental import FINANCIAL_BOUNDS  # 复用财务经济范围截断

VALUATION_COLS_TS = ["pe_ttm", "pb"]
FINANCIAL_COLS_TS = ["roe", "np_growth", "rev_growth", "debt_ratio",
                     "net_asset_growth", "total_asset_growth"]


def fetch_industry_map_ts(cache_dir):
    path = os.path.join(cache_dir, "_industry_map.csv")
    if os.path.exists(path):
        m = pd.read_csv(path, dtype={"code": str})
        return dict(zip(m["code"], m["industry"]))
    return {}


def fetch_daily_basic_panel_ts(cache_dir):
    path = os.path.join(cache_dir, "_daily_basic.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"], dtype={"code": str})
    return pd.DataFrame(columns=["code", "date"])


def fetch_financial_panel_ts(cache_dir):
    path = os.path.join(cache_dir, "_financial.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["ann_date", "end_date"], dtype={"code": str})
    return pd.DataFrame(columns=["code", "ann_date", "end_date"])


def attach_fundamentals_ts(pool, cache_dir):
    """合并 log_mcap / industry / pe_ttm / pb(点-in-time),产出与 cross_section 相同列。"""
    mcap = fetch_daily_basic_panel_ts(cache_dir)
    imap = fetch_industry_map_ts(cache_dir)

    pool = pool.copy()
    if "total_mv" in mcap.columns:
        mv = mcap[["code", "date", "total_mv"]].copy()
        mv["mktcap"] = mv["total_mv"] / 1e4  # 万元 -> 亿元
        pool = pool.merge(mv[["code", "date", "mktcap"]], on=["code", "date"], how="left")
        pool = pool.sort_values(["code", "date"])
        pool["mktcap"] = pool.groupby("code")["mktcap"].ffill()
        pool["log_mcap"] = np.log(pool["mktcap"].where(pool["mktcap"] > 0))
    else:
        pool["log_mcap"] = np.nan

    pool["industry"] = pool["code"].map(imap).fillna("未知").astype(str)

    for col in VALUATION_COLS_TS:
        if col in mcap.columns:
            v = mcap[["code", "date", col]].copy()
            pool = pool.merge(v, on=["code", "date"], how="left")
            pool = pool.sort_values(["code", "date"])
            pool[col] = pool.groupby("code")[col].ffill()
            pool[f"log_{col}"] = np.log(pool[col].where(pool[col] > 0))
        else:
            pool[col] = np.nan
            pool[f"log_{col}"] = np.nan
    return pool


def attach_financial_ts(pool, cache_dir):
    """季度财务因子按 ann_date(精确公布日)点-in-time 对齐,前向填充。

    与旧版 attach_financial 逻辑一致,仅把 avail_date(报告期+固定滞后)换成
    ann_date(真实公布日)。merge_asof 要求按 on 键全局排序(见 fundamental.py 坑5)。
    """
    fin = fetch_financial_panel_ts(cache_dir)
    if fin.empty:
        pool = pool.copy()
        for c in FINANCIAL_COLS_TS:
            pool[c] = np.nan
        return pool

    fin = fin.dropna(subset=["ann_date"]).sort_values(["code", "ann_date"])
    fin_asof = fin[["code", "ann_date"] + FINANCIAL_COLS_TS].sort_values("ann_date")
    pool = pool.copy().sort_values("date")
    merged = pd.merge_asof(
        pool, fin_asof,
        left_on="date", right_on="ann_date", by="code", direction="backward",
    )
    return merged.sort_values("date").reset_index(drop=True)

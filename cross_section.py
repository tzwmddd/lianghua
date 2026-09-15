"""横截面中性化:去极值 -> z-score -> 对[行业哑变量 + 对数市值]回归取残差。

标准 Barra 式处理,逐交易日进行,只用到当日的横截面信息,无未来数据泄露。
"""
import numpy as np
import pandas as pd

from fundamental import (
    fetch_industry_map,
    fetch_mktcap_panel,
    fetch_valuation_panel,
    FINANCIAL_BOUNDS,
)


def _winsorize_zscore(col, bounds=None):
    """去极值(1%/99% 分位截断) + z-score。全 NaN 时返回 0。

    bounds 为财务因子的经济范围截断(如 roe ±100%),在分位截断之前先裁掉
    极端离群值(新浪财务有 roe -5762% 等),否则会污染分位统计。
    """
    col = np.asarray(col, dtype=float)
    if bounds is not None:
        lo, hi = bounds
        col = np.clip(col, lo, hi)
    if not np.isfinite(col).any():
        return np.zeros_like(col)
    med = np.nanmedian(col)
    col = np.where(np.isfinite(col), col, med)
    lo, hi = np.nanpercentile(col, [1, 99])
    col = np.clip(col, lo, hi)
    mu, sd = col.mean(), col.std()
    if sd < 1e-12:
        return np.zeros_like(col)
    return (col - mu) / sd


# 估值因子(价值类):pe_ttm/pb/pcf,百度点-in-time 序列,负值(亏损)置 NaN 后取对数
VALUATION_COLS = ["pe_ttm", "pb", "pcf"]


def attach_fundamentals(pool, cache_dir):
    """给 pooled 面板附加 log_mcap、industry 与估值因子列。

    mcap 为百度点-in-time 总市值(按 code+date 合并);行业缺失记为"未知"。
    估值因子 pe_ttm/pb/pcf 同源点-in-time,负值(亏损/负现金流)置 NaN 后取 log。
    """
    mcap = fetch_mktcap_panel(cache_dir)
    imap = fetch_industry_map(cache_dir)

    pool = pool.copy()
    pool = pool.merge(mcap, on=["code", "date"], how="left")
    pool = pool.sort_values(["code", "date"])
    pool["mktcap"] = pool.groupby("code")["mktcap"].ffill()
    pool["log_mcap"] = np.log(pool["mktcap"].where(pool["mktcap"] > 0))
    pool["industry"] = pool["code"].map(imap).fillna("未知").astype(str)

    for col in VALUATION_COLS:
        v = fetch_valuation_panel(cache_dir, col)
        pool = pool.merge(v, on=["code", "date"], how="left")
        pool = pool.sort_values(["code", "date"])
        pool[col] = pool.groupby("code")[col].ffill()
        pool[f"log_{col}"] = np.log(pool[col].where(pool[col] > 0))
    return pool


def neutralize(pool, style_cols, industry_col="industry", mcap_col="log_mcap"):
    """对 style_cols 逐日做行业+市值中性化;另产出 size 因子(行业中性化的 log_mcap)。

    返回原 DataFrame 副本:style_cols 被中性化残差覆盖,新增 'size' 列。
    """
    out = pool.copy()
    for c in style_cols:
        out[c] = np.nan
    out["size"] = np.nan

    for _, g in pool.groupby("date"):
        idx = g.index
        mc = _winsorize_zscore(g[mcap_col].to_numpy())
        dummies = pd.get_dummies(g[industry_col].to_numpy(), dtype=float).to_numpy()

        # size 因子:log_mcap 对行业哑变量回归取残差(只行业中性,不做市值中性)
        d_ind = np.hstack([np.ones((len(g), 1)), dummies])
        size_resid = mc.reshape(-1, 1) - d_ind @ np.linalg.lstsq(
            d_ind, mc.reshape(-1, 1), rcond=None
        )[0]
        out.loc[idx, "size"] = size_resid[:, 0]

        # 风格因子:去极值+zscore 后,对[行业哑变量 + log_mcap]回归取残差
        xz = np.column_stack([
            _winsorize_zscore(g[c].to_numpy(), FINANCIAL_BOUNDS.get(c)) for c in style_cols
        ])
        d_full = np.hstack([d_ind, mc.reshape(-1, 1)])
        resid = xz - d_full @ np.linalg.lstsq(d_full, xz, rcond=None)[0]
        out.loc[idx, style_cols] = resid

    return out

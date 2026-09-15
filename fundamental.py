"""横截面基本面数据:行业分类 + 市值,带重试与磁盘缓存。

数据源:
- 行业:新浪"行业"板块(84 个,约 88% 沪深300覆盖),东财/申万被限流时的可用替代。
- 市值:百度估值(逐股点-in-time 总市值,全覆盖),单位亿元。

均为一次性静态映射/快照,缓存到 cache/。
"""
import glob
import os
import random
import time

import pandas as pd


def _retry(fn, retries=3, delay=(2.0, 4.0)):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries - 1:
                time.sleep(random.uniform(*delay) * (i + 1))
    raise last


def _require_not_frozen(cache_dir, path):
    """数据冻结保护:cache/_FROZEN 存在且目标缓存缺失时,拒绝网络重拉。

    免费数据源(新浪财务等)重拉会变(字段空值/离群/格式),导致结果不可复现。
    """
    if os.path.exists(os.path.join(cache_dir, "_FROZEN")):
        raise RuntimeError(
            f"数据已冻结(见 cache/_FROZEN): {os.path.basename(path)} 缺失,"
            f"拒绝网络重拉以保证结果可复现"
        )


def _cached_codes(cache_dir):
    return sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(cache_dir, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )


def fetch_industry_map(cache_dir):
    """新浪"行业"板块成分 -> {code: 行业中文名}。缓存到 _industry_map.csv。"""
    import akshare as ak
    path = os.path.join(cache_dir, "_industry_map.csv")
    if os.path.exists(path):
        m = pd.read_csv(path, dtype={"code": str})
        return dict(zip(m["code"], m["industry"]))

    _require_not_frozen(cache_dir, path)
    sectors = _retry(lambda: ak.stock_sector_spot(indicator="行业"))
    name_map = dict(zip(sectors["label"], sectors["板块"]))
    mapping = {}
    for i, sec_code in enumerate(sectors["label"].tolist()):
        try:
            cons = _retry(
                lambda: ak.stock_sector_detail(sector=sec_code),
                retries=2, delay=(1.0, 2.0),
            )
            ind = name_map.get(sec_code, sec_code)
            for c in cons["code"].astype(str).str.zfill(6):
                mapping[c] = ind
        except Exception:
            continue
        if i % 15 == 14:
            time.sleep(random.uniform(0.5, 1.0))

    m = pd.DataFrame(sorted(mapping.items()), columns=["code", "industry"])
    m.to_csv(path, index=False, encoding="utf-8-sig")
    return mapping


def fetch_mktcap_panel(cache_dir):
    """百度估值逐股总市值(亿元) -> DataFrame(code/date/mktcap)。缓存到 _mktcap_panel.csv。"""
    import akshare as ak
    path = os.path.join(cache_dir, "_mktcap_panel.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"], dtype={"code": str})
        return df

    _require_not_frozen(cache_dir, path)
    codes = _cached_codes(cache_dir)
    parts = []
    for i, code in enumerate(codes):
        try:
            s = _retry(
                lambda: ak.stock_zh_valuation_baidu(
                    symbol=code, indicator="总市值", period="全部"
                ),
                retries=2, delay=(1.0, 2.0),
            )
            s = s.rename(columns={"value": "mktcap"})
            s["code"] = code
            s["date"] = pd.to_datetime(s["date"])
            parts.append(s[["code", "date", "mktcap"]])
        except Exception:
            continue
        if i % 20 == 19:
            time.sleep(random.uniform(0.5, 1.0))

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


# 百度估值接口实际可用的估值指标(实测 2026-08-28):总市值/市盈率(TTM)/市盈率(静)/市净率/市现率
# 股息率/ROE/EPS 等接口返回空,需另找数据源。
VALUATION_INDICATORS = {
    "pe_ttm": "市盈率(TTM)",
    "pb": "市净率",
    "pcf": "市现率",
}


def fetch_valuation_panel(cache_dir, out_col):
    """百度估值逐股历史序列 -> DataFrame(code/date/{out_col})。缓存到 _val_{out_col}.csv。

    与 fetch_mktcap_panel 同源同构,但按 out_col 缓存,避免重复拉总市值。
    """
    import akshare as ak
    indicator = VALUATION_INDICATORS[out_col]
    path = os.path.join(cache_dir, f"_val_{out_col}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"], dtype={"code": str})

    _require_not_frozen(cache_dir, path)
    codes = _cached_codes(cache_dir)
    parts = []
    for i, code in enumerate(codes):
        try:
            s = _retry(
                lambda: ak.stock_zh_valuation_baidu(
                    symbol=code, indicator=indicator, period="全部"
                ),
                retries=2, delay=(1.0, 2.0),
            )
            s = s.rename(columns={"value": out_col})
            s["code"] = code
            s["date"] = pd.to_datetime(s["date"])
            parts.append(s[["code", "date", out_col]])
        except Exception:
            continue
        if i % 20 == 19:
            time.sleep(random.uniform(0.5, 1.0))

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------------------
# 财务因子(新浪 stock_financial_analysis_indicator,季度报告期数据)
# 覆盖:ROE/净利润增长率/营收增长率/资产负债率/现金流/成长。银行券商无营收增长率,为 NaN。
# 注:gross_margin(销售毛利率)已移除——新浪自 2020 起对该字段 100% 返回空值,是死因子。
# ---------------------------------------------------------------------------

FINANCIAL_FIELD_MAP = {
    "roe": "净资产收益率(%)",            # 质量:盈利水平
    "np_growth": "净利润增长率(%)",       # 成长:利润增速
    "rev_growth": "主营业务收入增长率(%)",  # 成长:营收增速
    "debt_ratio": "资产负债率(%)",        # 杠杆:财务风险
    "ocf_to_np": "经营现金净流量与净利润的比率(%)",  # 现金流质量:利润含金量
    "ocf_roa": "资产的经营现金流量回报率(%)",         # 现金流质量
    "net_asset_growth": "净资产增长率(%)",             # 成长:净资产增速
    "total_asset_growth": "总资产增长率(%)",           # 成长:总资产增速
}
FINANCIAL_COLS = list(FINANCIAL_FIELD_MAP.keys())

# 财务因子经济范围截断(离群值清洗):新浪财务有极端离群(roe 最低 -5762%、
# np_growth ±408万%、rev_growth 达 1079万%),在 winsorize(1/99 分位)之前先按
# 合理经济范围截断,避免极端值污染分位统计。
FINANCIAL_BOUNDS = {
    "roe": (-100, 100),          # 净资产收益率:±100%(亏损超净资产即异常)
    "np_growth": (-1000, 1000),  # 净利润增长率:±1000%(扭亏/暴增可达十倍)
    "rev_growth": (-1000, 1000), # 营收增长率:±1000%
    "debt_ratio": (0, 100),      # 资产负债率:0~100%(超100%即资不抵债异常)
    "ocf_to_np": (-1000, 1000),  # 经营现金流/净利润:净利润接近0时比值可很大
    "ocf_roa": (-100, 100),      # 经营现金流回报率:±100%
    "net_asset_growth": (-1000, 1000),  # 净资产增长率:±1000%
    "total_asset_growth": (-1000, 1000),  # 总资产增长率:±1000%
}


def fetch_financial_panel(cache_dir):
    """逐股拉季度财务指标,缓存到 _financial.csv。

    返回 DataFrame(code/report_date/各财务因子列)。
    报告期(而非公布日)为索引;公布滞后由 attach_financial 处理。
    """
    import akshare as ak
    path = os.path.join(cache_dir, "_financial.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["report_date"], dtype={"code": str})

    _require_not_frozen(cache_dir, path)
    codes = _cached_codes(cache_dir)
    fields = list(FINANCIAL_FIELD_MAP.values())
    parts = []
    for i, code in enumerate(codes):
        try:
            df = _retry(
                lambda: ak.stock_financial_analysis_indicator(symbol=code, start_year="2015"),
                retries=2, delay=(1.0, 2.0),
            )
            sub = df[["日期"] + fields].copy()
            sub.columns = ["report_date"] + FINANCIAL_COLS
            sub["report_date"] = pd.to_datetime(sub["report_date"])
            sub["code"] = code
            parts.append(sub[["code", "report_date"] + FINANCIAL_COLS])
        except Exception:
            continue
        if i % 15 == 14:
            time.sleep(random.uniform(0.5, 1.0))

    panel = pd.concat(parts, ignore_index=True)
    panel.to_csv(path, index=False, encoding="utf-8-sig")
    return panel


def _report_lag(report_date):
    """财报公布滞后(保守):年报 120 天,半年报 60 天,一/三季报 45 天。

    报告期数据须等公布后才可用,否则是未来数据泄露。
    """
    m = report_date.month
    if m == 12:
        return pd.Timedelta(days=120)   # 年报:次年 4 月底
    if m == 6:
        return pd.Timedelta(days=60)    # 半年报:8 月底
    return pd.Timedelta(days=45)        # 一季报(3月)/三季报(9月):4月底/10月底


def attach_financial(pool, cache_dir):
    """季度财务因子按"报告期 + 公布滞后"点-in-time 对齐到日频面板,前向填充。

    用 merge_asof 对每只股票取 avail_date <= date 的最新一期财报,早期无财报为 NaN。
    """
    fin = fetch_financial_panel(cache_dir)
    fin = fin.dropna(subset=["report_date"]).sort_values(["code", "report_date"])
    fin["avail_date"] = fin["report_date"] + fin["report_date"].apply(_report_lag)
    # merge_asof 要求按 on 键(avail_date)全局排序,而非按 by+on 排序
    fin_asof = fin[["code", "avail_date"] + FINANCIAL_COLS].sort_values("avail_date")

    pool = pool.copy().sort_values("date")
    merged = pd.merge_asof(
        pool, fin_asof,
        left_on="date", right_on="avail_date", by="code", direction="backward",
    )
    return merged.sort_values("date").reset_index(drop=True)

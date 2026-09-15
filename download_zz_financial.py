"""下载中证500/1000 财务指标, 用 tushare fina_indicator(逐股)。

字段映射(tushare -> 内部名,与现有 FINANCIAL_COLS 对齐):
  roe_waa -> roe, netprofit_yoy -> np_growth, or_yoy -> rev_growth,
  debt_to_assets -> debt_ratio, eqt_yoy -> net_asset_growth, assets_yoy -> total_asset_growth

现金流两因子(ocf_to_np/ocf_roa)第一阶段放弃(需 cashflow 接口另算)。

**点-in-time 关键**: 存 ann_date(精确财报公布日),替代现有"报告期+固定滞后",
fundamental_ts.py 用 ann_date 做 merge_asof 对齐。同 end_date 重复记录(财报修正)
按 ann_date 去重取最新。

缓存 {cache}/_financial.csv (code/ann_date/end_date + 6 财务因子)。
"""
import os
import sys
import time

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import tushare_common as tc

UNIVERSES = {
    "500": ("000905.SH", "cache_zz500"),
    "1000": ("000852.SH", "cache_zz1000"),
}
START, END = "20150101", "20261231"

FIN_MAP = {
    "roe_waa": "roe",
    "netprofit_yoy": "np_growth",
    "or_yoy": "rev_growth",
    "debt_to_assets": "debt_ratio",
    "eqt_yoy": "net_asset_growth",
    "assets_yoy": "total_asset_growth",
}
FIN_COLS = list(FIN_MAP.values())


def fetch(pro, ts_code):
    df = pro.fina_indicator(ts_code=ts_code, start_date=START, end_date=END)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["code"] = tc.to_code(ts_code)
    df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
    df = df.rename(columns=FIN_MAP)
    keep = ["code", "ann_date", "end_date"] + FIN_COLS
    sub = df[[c for c in keep if c in df.columns]].copy()
    # 同报告期重复记录(财报修正)按 ann_date 去重取最新
    sub = sub.dropna(subset=["end_date"])
    sub = sub.sort_values(["code", "end_date", "ann_date"]).drop_duplicates(
        subset=["code", "end_date"], keep="last"
    )
    return sub


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit(f"用法: python download_zz_financial.py [500|1000]")
    _, cache_dir = UNIVERSES[universe]

    cons = pd.read_csv(os.path.join(cache_dir, "_constituents.csv"), dtype={"code": str})
    codes = sorted(cons["code"].unique())
    path = os.path.join(cache_dir, "_financial.csv")
    done = set()
    if os.path.exists(path):
        done = set(pd.read_csv(path, usecols=["code"], dtype={"code": str})["code"])
    todo = [c for c in codes if c not in done]
    print(f"{cache_dir}: 成分股 {len(codes)} 只, 已有 {len(done)} 只, 待下载 {len(todo)} 只", flush=True)

    pro = tc.get_pro()
    parts = []
    t0 = time.time()
    for i, code in enumerate(todo):
        try:
            df = tc._retry(lambda: fetch(pro, tc.to_ts_code(code)))
            if df is not None and len(df):
                parts.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"  {code} 失败: {repr(e)[:50]}", flush=True)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  进度 {i + 1}/{len(todo)} 用时 {el:.0f}s, 预计剩余 "
                  f"{el / (i + 1) * (len(todo) - i - 1):.0f}s", flush=True)
        tc.sleep(0.5)  # fina_indicator 限流约 200次/分钟

    if parts:
        new = pd.concat(parts, ignore_index=True)
        if os.path.exists(path):
            old = pd.read_csv(path, parse_dates=["ann_date", "end_date"], dtype={"code": str})
            new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=["code", "end_date"]).sort_values(["code", "end_date"])
        new.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n保存 {path}: {len(new)} 行, {new['code'].nunique()} 只", flush=True)


if __name__ == "__main__":
    main()

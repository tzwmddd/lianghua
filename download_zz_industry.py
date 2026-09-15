"""下载中证500/1000 申万行业映射, 用 tushare index_classify + index_member_all。

遍历 31 个申万一级行业(index_member_all 按 l1_code 拉最新成分),建立全市场
ts_code -> 一级行业中文名映射;再对成分股并集生成 {cache}/_industry_map.csv
(code/industry),退市/查不到的记为"未知"。静态映射(最新分类),与现有
fundamental.py 的 fetch_industry_map 用法一致。
"""
import os
import sys

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import tushare_common as tc

UNIVERSES = {
    "500": ("000905.SH", "cache_zz500"),
    "1000": ("000852.SH", "cache_zz1000"),
}


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "500"
    if universe not in UNIVERSES:
        sys.exit(f"用法: python download_zz_industry.py [500|1000]")
    _, cache_dir = UNIVERSES[universe]

    pro = tc.get_pro()
    l1 = pro.index_classify(level="L1", src="SW2021")
    print(f"申万一级行业 {len(l1)} 个", flush=True)

    mapping = {}
    for _, row in l1.iterrows():
        l1_code, l1_name = row["index_code"], row["industry_name"]
        try:
            members = tc._retry(lambda: pro.index_member_all(l1_code=l1_code))
        except Exception as e:  # noqa: BLE001
            print(f"  {l1_name} 失败: {repr(e)[:50]}", flush=True)
            continue
        for ts in members["ts_code"].astype(str):
            mapping[tc.to_code(ts)] = l1_name
        tc.sleep(0.3)

    cons = pd.read_csv(os.path.join(cache_dir, "_constituents.csv"), dtype={"code": str})
    codes = sorted(cons["code"].unique())
    rows = [(c, mapping.get(c, "未知")) for c in codes]
    out = pd.DataFrame(rows, columns=["code", "industry"])
    path = os.path.join(cache_dir, "_industry_map.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    covered = sum(1 for _, ind in rows if ind != "未知")
    print(f"保存 {path}: {len(out)} 只, 行业覆盖 {covered}/{len(out)} "
          f"({covered / len(out):.0%})", flush=True)


if __name__ == "__main__":
    main()

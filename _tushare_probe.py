"""tushare 接口探测:验证 2000 积分能否拉到中证500/1000 迁移所需的数据。

token 从环境变量 TUSHARE_TOKEN 读取,不硬编码。
"""
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import tushare as ts

token = os.environ.get("TUSHARE_TOKEN")
if not token:
    sys.exit("缺少环境变量 TUSHARE_TOKEN")

ts.set_token(token)
pro = ts.pro_api()


def probe(name, fn, show_head=3):
    try:
        df = fn()
        n = len(df)
        print(f"\n[OK] {name}: {n} 行")
        if n and show_head:
            print(df.head(show_head).to_string())
        return df
    except Exception as e:  # noqa: BLE001
        print(f"\n[FAIL] {name}: {type(e).__name__}: {e}")
        return None


# 1. 基础:交易日历(验证 token 有效 + 基础权限)
probe("trade_cal", lambda: pro.trade_cal(exchange="SSE", start_date="20240101", end_date="20240110"))

# 2. 核心:中证500 / 中证1000 历史成分权重
for icode in ["000905.SH", "000852.SH"]:
    df = probe(
        f"index_weight {icode} 2020 全年",
        lambda ic=icode: pro.index_weight(index_code=ic, start_date="20200101", end_date="20201231"),
    )
    if df is not None and len(df):
        print(f"    trade_date 唯一值数: {df['trade_date'].nunique()}, 成分股数: {df['con_code'].nunique()}")

# 3. 申万行业分类
probe("index_classify SW2021 L1", lambda: pro.index_classify(level="L1", src="SW2021"))

# 4. 申万行业成分(反查一只股票)
probe("index_member_all 000001.SZ", lambda: pro.index_member_all(ts_code="000001.SZ"))

# 5. 指数日线(中证500)
probe("index_daily 000905.SH", lambda: pro.index_daily(ts_code="000905.SH", start_date="20240101", end_date="20240115"))

# 6. 个股日线 + 复权因子(取中证500权重第一只做样本)
probe("daily 600000.SH", lambda: pro.daily(ts_code="600000.SH", start_date="20240101", end_date="20240115"))
probe("adj_factor 600000.SH", lambda: pro.adj_factor(ts_code="600000.SH", start_date="20240101", end_date="20240115"))

# 7. 每日指标(全市场某交易日:PE/PB/市值/换手率)
probe("daily_basic 20240102 全市场", lambda: pro.daily_basic(trade_date="20240102"))

# 8. 财务指标(逐只,单次上限100条)
probe("fina_indicator 600000.SH", lambda: pro.fina_indicator(ts_code="600000.SH", start_date="20150101"))

print("\n=== 探测完成 ===")

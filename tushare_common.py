"""tushare 基础设施:token 管理、代码转换、限流重试。

token 读取顺序:环境变量 TUSHARE_TOKEN -> 本目录 tushare_token.txt。
不硬编码、不进 git(建议把 tushare_token.txt 加入 .gitignore)。
"""
import os
import random
import time


def _load_token():
    tok = os.environ.get("TUSHARE_TOKEN")
    if tok:
        return tok.strip()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tushare_token.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    raise RuntimeError(
        "未找到 token: 请设置环境变量 TUSHARE_TOKEN, 或在本目录建 tushare_token.txt"
    )


def get_pro():
    import tushare as ts
    ts.set_token(_load_token())
    return ts.pro_api()


def to_ts_code(code):
    """6 位数字代码 -> tushare ts_code(600000.SH / 000001.SZ / 8xxxxx.BJ)。"""
    code = str(code).zfill(6)
    if code.startswith("920"):
        return code + ".BJ"
    if code.startswith("6"):
        return code + ".SH"
    if code.startswith(("0", "3")):
        return code + ".SZ"
    if code.startswith(("4", "8")):
        return code + ".BJ"
    return code + ".SH"


def to_code(ts_code):
    """ts_code -> 6 位数字代码。"""
    return str(ts_code).split(".")[0].zfill(6)


def _retry(fn, retries=3, delay=(1.0, 2.0)):
    """随机延时 + 指数退避,缓解 tushare 限流。"""
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries - 1:
                time.sleep(random.uniform(*delay) * (i + 1))
    raise last


def sleep(seconds):
    time.sleep(seconds)

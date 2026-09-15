"""全局配置:标的、日期、预测周期、交易成本、路径。"""
import os

# ---- 项目路径 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# ---- 数据 ----
SYMBOL = "sh000300"          # 默认沪深300指数;个股如 "600519"
START_DATE = "20210101"      # YYYYMMDD
END_DATE = "20261231"

# ---- 预测目标 ----
FORECAST_HORIZON = 30        # 未来 N 天涨跌方向

# ---- 标签重叠泄露防护 ----
# walk-forward 评估中,训练集与测试集只隔 1 个交易日时,FORECAST_HORIZON=30 天标签
# 会与训练样本标签重叠 29/30 天,导致 IC 被虚高 ~10 倍(见 embargo_test.py)。
# 训练集须排除测试日之前 EMBARGO_DAYS 日历天(≈30 交易日),使标签窗口不再重叠。
EMBARGO_DAYS = 45
LSTM_SEQ_LEN = 30            # LSTM 输入序列窗口(交易日)
LSTM_HIDDEN = 64
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 64
WALK_FORWARD_EPOCHS = 5      # walk-forward 滚动重训时 LSTM 的 epochs(降低以控制耗时)

# ---- walk-forward 切分 ----
TRAIN_RATIO = 0.7            # 初始训练窗占比
STEP = 21                    # 测试窗每次前进的交易日数

# ---- 回测成本(A股) ----
COMMISSION = 0.00025         # 佣金,双向,万2.5
STAMP_TAX = 0.0005           # 印花税,仅卖出,0.05%
SLIPPAGE = 0.001             # 滑点 0.1%
# 单次完整换手(卖旧+买新)的总成本
ROUND_TRIP_COST = COMMISSION * 2 + STAMP_TAX + SLIPPAGE * 2

# ---- 随机种子 ----
SEED = 42

# ---- 绘图 ----
CHINESE_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]

"""个股池化 walk-forward:整合沪深300成分股,训练 LSTM + LightGBM 方向分类器。

对比"1年训练史" vs "5年训练史"在同一测试窗口上的方向准确率,量化更多数据带来的提升。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import CACHE_DIR, SEED, LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_BATCH_SIZE, FORECAST_HORIZON
from features import build_features, FEATURE_COLS
from data import load_cached

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

TEST_FRAC = 0.2          # 最后20%日历作为测试窗口
LSTM_EPOCHS = 5


def set_seed():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_pooled():
    codes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(CACHE_DIR, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    )
    parts = []
    for code in codes:
        df = load_cached(code, CACHE_DIR)
        if df is None or len(df) < 120:
            continue
        feat = build_features(df, FORECAST_HORIZON)
        feat["code"] = code
        parts.append(feat)
    pool = pd.concat(parts, ignore_index=True)
    return pool


def make_sequences(pool, seq_len):
    """按个股建序列后跨股票池化,避免把不同股票的窗口拼在一起。"""
    Xs, ys = [], []
    for _, g in pool.groupby("code", sort=False):
        g = g.sort_values("date")
        X = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = g["target"].to_numpy()
        if len(g) < seq_len:
            continue
        seq = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
        Xs.append(np.ascontiguousarray(seq.transpose(0, 2, 1)))
        ys.append(y[seq_len - 1:])
    return np.concatenate(Xs), np.concatenate(ys)


def run_lgb(train, test):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(train[FEATURE_COLS].to_numpy(), train["target"].to_numpy())
    prob = model.predict_proba(test[FEATURE_COLS].to_numpy())[:, 1]
    acc = float(((prob > 0.5).astype(int) == test["target"].to_numpy()).mean())
    return acc


class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def run_lstm(train, test):
    set_seed()
    n_feat = len(FEATURE_COLS)
    X_tr, y_tr = make_sequences(train, LSTM_SEQ_LEN)
    X_te, y_te = make_sequences(test, LSTM_SEQ_LEN)

    mu = X_tr.mean(axis=(0, 1), keepdims=True)
    sd = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(n_feat, LSTM_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device).long()
    n = len(Xt)
    bs = LSTM_BATCH_SIZE

    model.train()
    for _ in range(LSTM_EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = []
        Xte = torch.from_numpy(X_te).to(device)
        for i in range(0, len(Xte), 4096):
            probs.append(torch.softmax(model(Xte[i:i + 4096]), dim=1)[:, 1].cpu().numpy())
        prob = np.concatenate(probs)
    acc = float(((prob > 0.5).astype(int) == y_te).mean())
    return acc


def main():
    t0 = time.time()
    pool = load_pooled()
    n_stocks = pool["code"].nunique()
    print(f"加载 {n_stocks} 只股票,共 {len(pool)} 条样本,"
          f"区间 {pool['date'].min().date()} ~ {pool['date'].max().date()}", flush=True)

    dates = sorted(pool["date"].unique())
    test_start = dates[int(len(dates) * (1 - TEST_FRAC))]
    test = pool[pool["date"] >= test_start]
    train_5y = pool[pool["date"] < test_start]
    train_1y_start = test_start - pd.Timedelta(days=365)
    train_1y = pool[(pool["date"] >= train_1y_start) & (pool["date"] < test_start)]

    print(f"\n测试窗口: {test_start.date()} ~ {test['date'].max().date()} ({len(test)} 条)")
    print(f"1年训练: {train_1y_start.date()} ~ {test_start.date()} ({len(train_1y)} 条)")
    print(f"5年训练: {pool['date'].min().date()} ~ {test_start.date()} ({len(train_5y)} 条)", flush=True)

    results = {}
    for name, train in [("1年", train_1y), ("5年", train_5y)]:
        print(f"\n=== {name}训练史 ===", flush=True)
        lgb_acc = run_lgb(train, test)
        print(f"  LightGBM 方向准确率: {lgb_acc:.2%}", flush=True)
        lstm_acc = run_lstm(train, test)
        print(f"  LSTM     方向准确率: {lstm_acc:.2%}", flush=True)
        results[name] = {"lgb": lgb_acc, "lstm": lstm_acc}

    print("\n========== 汇总 ==========", flush=True)
    print(f"{'模型':<10}{'1年':>10}{'5年':>10}{'提升幅度':>12}", flush=True)
    for model, key in [("LightGBM", "lgb"), ("LSTM", "lstm")]:
        a = results["1年"][key]
        b = results["5年"][key]
        print(f"{model:<10}{a:>9.2%}{b:>9.2%}{b - a:>+11.2%}", flush=True)
    print(f"\n总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

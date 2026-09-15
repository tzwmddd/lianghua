"""神经网络 vs LightGBM 方向预测对比(5年 pooled 数据, 2026 年样本外测试)。

回答"换神经网络能否提高准确率":在同一 train/val/test 切分下对比
LightGBM / MLP / GRU / LSTM 的 5 日涨跌方向准确率。
train=[2021-01, 2025-12], 其中最后 10% 交易日做验证(早停), test=2026 全年。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import CACHE_DIR, SEED, FORECAST_HORIZON, LSTM_SEQ_LEN, LSTM_HIDDEN
from features import build_features, FEATURE_COLS
from data import load_cached

TEST_START = "2026-01-01"
TRAIN_START = "2021-01-01"
VAL_FRAC = 0.1
MLP_HIDDEN = [256, 128, 64]
RNN_HIDDEN = LSTM_HIDDEN
MAX_EPOCHS_MLP = 60
MAX_EPOCHS_RNN = 30
PATIENCE = 8


def set_seed():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


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
    return pool.sort_values("date").reset_index(drop=True)


def split_by_date(pool):
    train = pool[(pool["date"] >= pd.Timestamp(TRAIN_START)) & (pool["date"] < pd.Timestamp(TEST_START))]
    test = pool[pool["date"] >= pd.Timestamp(TEST_START)]
    # 按日期切验证集:训练末尾 10% 交易日
    dates = sorted(train["date"].unique())
    val_start = dates[int(len(dates) * (1 - VAL_FRAC))]
    val = train[train["date"] >= val_start]
    train = train[train["date"] < val_start]
    return train, val, test


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


# ---------------- LightGBM ----------------
def run_lgb(train, val, test):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(
        train[FEATURE_COLS].to_numpy(), train["target"].to_numpy(),
        eval_set=[(val[FEATURE_COLS].to_numpy(), val["target"].to_numpy())],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    prob = model.predict_proba(test[FEATURE_COLS].to_numpy())[:, 1]
    acc = float(((prob > 0.5).astype(int) == test["target"].to_numpy()).mean())
    return acc


# ---------------- MLP ----------------
class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        layers = []
        prev = n_features
        for h in MLP_HIDDEN:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------- RNN (GRU / LSTM) ----------------
class RNNModel(nn.Module):
    def __init__(self, n_features, hidden, rnn_type="gru"):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


def _train_nn(model, Xtr, ytr, Xval, yval, max_epochs, batch_size, device):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device).long()
    Xv = torch.from_numpy(Xval).to(device)
    yv = torch.from_numpy(yval).to(device).long()
    train_ds = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_ds:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv))
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_nn(model, Xte, batch_size, device):
    Xt = torch.from_numpy(Xte).to(device)
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            probs.append(torch.softmax(model(Xt[i:i + batch_size]), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def _std_fit(Xtr, Xval, Xte, seq_mode):
    """标准化:MLP 用逐列均值/方差,序列用全量均值/方差。"""
    if seq_mode:
        mu = Xtr.mean(axis=(0, 1), keepdims=True)
        sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
    else:
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xval - mu) / sd, (Xte - mu) / sd


def run_nn(train, val, test, seq_mode=False, rnn_type="gru"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seq_mode:
        Xtr, ytr = make_sequences(train, LSTM_SEQ_LEN)
        Xval, yval = make_sequences(val, LSTM_SEQ_LEN)
        Xte, yte = make_sequences(test, LSTM_SEQ_LEN)
        Xtr, Xval, Xte = _std_fit(Xtr, Xval, Xte, seq_mode=True)
        model = RNNModel(len(FEATURE_COLS), RNN_HIDDEN, rnn_type)
        model = _train_nn(model, Xtr, ytr, Xval, yval, MAX_EPOCHS_RNN, 512, device)
        prob = _predict_nn(model, Xte, 4096, device)
    else:
        Xtr, ytr = train[FEATURE_COLS].to_numpy(dtype=np.float32), train["target"].to_numpy()
        Xval, yval = val[FEATURE_COLS].to_numpy(dtype=np.float32), val["target"].to_numpy()
        Xte, yte = test[FEATURE_COLS].to_numpy(dtype=np.float32), test["target"].to_numpy()
        Xtr, Xval, Xte = _std_fit(Xtr, Xval, Xte, seq_mode=False)
        model = MLP(len(FEATURE_COLS))
        model = _train_nn(model, Xtr, ytr, Xval, yval, MAX_EPOCHS_MLP, 2048, device)
        prob = _predict_nn(model, Xte, 8192, device)
    acc = float(((prob > 0.5).astype(int) == yte).mean())
    return acc


def main():
    t0 = time.time()
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}", flush=True)

    pool = load_pooled()
    train, val, test = split_by_date(pool)
    print(f"样本: train {len(train)} / val {len(val)} / test {len(test)}", flush=True)
    print(f"train 区间 {train['date'].min().date()} ~ {train['date'].max().date()}", flush=True)
    print(f"test  区间 {test['date'].min().date()} ~ {test['date'].max().date()} "
          f"({test['code'].nunique()} 只, {len(test)} 条)", flush=True)

    results = {}
    print("\n=== 训练与评估 ===", flush=True)

    results["LightGBM"] = run_lgb(train, val, test)
    print(f"  LightGBM 方向准确率: {results['LightGBM']:.2%}", flush=True)

    results["MLP"] = run_nn(train, val, test, seq_mode=False)
    print(f"  MLP      方向准确率: {results['MLP']:.2%}", flush=True)

    results["GRU"] = run_nn(train, val, test, seq_mode=True, rnn_type="gru")
    print(f"  GRU      方向准确率: {results['GRU']:.2%}", flush=True)

    results["LSTM"] = run_nn(train, val, test, seq_mode=True, rnn_type="lstm")
    print(f"  LSTM     方向准确率: {results['LSTM']:.2%}", flush=True)

    print("\n========== 汇总(2026 年样本外方向准确率) ==========", flush=True)
    for name, acc in results.items():
        print(f"  {name:<10} {acc:6.2%}   (反向 {1 - acc:6.2%})", flush=True)

    best_name = max(results, key=results.get)
    print(f"\n最优模型: {best_name} ({results[best_name]:.2%})", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

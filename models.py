"""模型层:PyTorch LSTM 与 LightGBM 方向分类器,统一 fit_predict 接口。"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from config import (
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_BATCH_SIZE,
    WALK_FORWARD_EPOCHS, SEED,
)
from features import FEATURE_COLS


def _set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class LSTMModel(nn.Module):
    """单层 LSTM + 全连接,输出 2 分类 logits(跌/涨)。"""

    def __init__(self, n_features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, seq_len, hidden)
        return self.head(out[:, -1, :])


def make_sequences(df, feature_cols, seq_len):
    """逐日特征表 -> 滑动窗口序列 (X, y)。

    X: (n - seq_len + 1, seq_len, n_features)
    y: (n - seq_len + 1,)  每个序列终点对应日期的 target
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy()
    # sliding_window_view(axis=0) 会把窗口维放到末尾,转置成 (w, seq_len, n_features)
    X = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
    return np.ascontiguousarray(X.transpose(0, 2, 1)), y[seq_len - 1:]


def _scale(X_tr, X_te):
    """在训练集上 fit 标准化器,应用到训练/测试集(逐特征维度)。"""
    n_feat = X_tr.shape[-1]
    scaler = StandardScaler().fit(X_tr.reshape(-1, n_feat))
    X_tr = scaler.transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_te = scaler.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape)
    return X_tr.astype(np.float32), X_te.astype(np.float32)


def fit_predict_lstm(train_df, test_df):
    """在一个 walk-forward fold 上训练 LSTM,预测 test 窗口上涨概率。

    返回 DataFrame: date / prob。
    """
    _set_seed()
    n_feat = len(FEATURE_COLS)

    X_tr, y_tr = make_sequences(train_df, FEATURE_COLS, LSTM_SEQ_LEN)
    # test 窗口起点需要 train 末尾 seq_len-1 天作前缀,保证序列完整
    prefix = train_df.iloc[-(LSTM_SEQ_LEN - 1):]
    test_ext = pd.concat([prefix, test_df], ignore_index=False)
    X_te, _ = make_sequences(test_ext, FEATURE_COLS, LSTM_SEQ_LEN)

    X_tr, X_te = _scale(X_tr, X_te)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(n_feat, LSTM_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device).long()
    n = len(Xt)
    bs = LSTM_BATCH_SIZE

    model.train()
    for _ in range(WALK_FORWARD_EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X_te).to(device)
        prob = torch.softmax(model(Xte), dim=1)[:, 1].cpu().numpy()

    return pd.DataFrame({"date": test_df["date"].to_numpy(), "prob": prob})


def fit_predict_lgb(train_df, test_df):
    """在一个 fold 上训练 LightGBM,预测 test 窗口上涨概率。"""
    import lightgbm as lgb

    X_tr = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_tr = train_df["target"].to_numpy()
    X_te = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)

    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1,
    )
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    return pd.DataFrame({"date": test_df["date"].to_numpy(), "prob": prob})

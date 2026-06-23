"""
心跳信号分类预测 — 数据集加载与预处理
使用 per-signal z-score 标准化，按 8:2 分层划分训练/测试集
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def load_and_parse(data_path):
    df = pd.read_csv(data_path)
    signals = df["heartbeat_signals"].apply(
        lambda x: np.array([float(v) for v in x.split(",")], dtype=np.float32)
    )
    X = np.stack(signals.values)
    y = df["label"].values.astype(np.int64)
    return X, y, df


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


class ECGDataset(Dataset):
    """心电信号数据集，per-signal z-score 标准化"""

    def __init__(self, X, y):
        self.X = self._normalize(X)
        self.y = y

    @staticmethod
    def _normalize(X):
        """向量化 per-signal z-score 标准化"""
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return ((X - mean) / std).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # (1, L)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def create_dataloaders(data_path, batch_size=64, test_size=0.2, random_state=42,
                       num_workers=0):
    """完整数据流水线：加载 → 分层划分 → Dataset → DataLoader"""
    X, y, _ = load_and_parse(data_path)

    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=test_size, random_state=random_state
    )

    signal_length = X.shape[1]

    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    for cls in range(4):
        print(f"  类别 {cls}: 训练 {sum(y_train == cls)}, 测试 {sum(y_test == cls)}")

    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin_memory)

    # 类别权重（用于加权交叉熵）
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return train_loader, test_loader, signal_length, class_weights

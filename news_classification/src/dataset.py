"""
新闻文本分类 — 数据集加载与预处理
将 train_set.csv 按 8:2 分层划分为训练集和测试集

BERT 风格: 每条序列前插入 [CLS] token，取首个 token 隐向量做分类
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter


class NewsDataset(Dataset):
    """新闻文本 PyTorch Dataset — 自动在序列开头插入 [CLS] token"""

    def __init__(self, texts, labels, max_len=1024, cls_token_id=None):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_len = max_len

        n = len(texts)
        data_arr = np.zeros((n, max_len), dtype=np.int64)
        mask_arr = np.zeros((n, max_len), dtype=np.float32)

        text_capacity = max_len - 1  # 预留给 [CLS] 一个位置

        for i, text in enumerate(texts):
            parts = text.split()
            seq_len = min(len(parts), text_capacity)

            # [CLS] token 放在位置 0
            data_arr[i, 0] = cls_token_id
            data_arr[i, 1:1 + seq_len] = [int(t) for t in parts[:seq_len]]

            mask_arr[i, 0] = 1.0  # [CLS] 永远不是 padding
            mask_arr[i, 1:1 + seq_len] = 1.0

        self.data = torch.tensor(data_arr, dtype=torch.long)
        self.masks = torch.tensor(mask_arr, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx], self.labels[idx]


def load_data(data_path):
    """加载 CSV 并返回 texts, labels, df"""
    df = pd.read_csv(data_path, sep="\t")
    texts = df["text"].values
    labels = df["label"].values.astype(np.int64)
    return texts, labels, df


def _fast_vocab_size(texts, sample_size=10000):
    """快速估算 vocab_size：随机采样 sample_size 条，扫描所有 token 取最大值"""
    n = len(texts)
    idx = np.random.RandomState(42).choice(n, min(sample_size, n), replace=False)
    max_tok = 0
    for i in idx:
        text = texts[i]
        for t in text.split():
            tid = int(t)
            if tid > max_tok:
                max_tok = tid
    return max_tok + 200  # buffer 防止采样遗漏稀有 token


def create_dataloaders(data_path, batch_size=128, max_len=1024,
                       test_size=0.2, random_state=42, num_workers=0):
    """完整数据流水线：加载 -> 划分 -> Dataset -> DataLoader"""
    texts, labels, df = load_data(data_path)
    num_classes = df["label"].nunique()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    for cls in range(num_classes):
        print(f"  类别 {cls}: 训练 {sum(y_train == cls)}, 测试 {sum(y_test == cls)}")

    # [CLS] token ID = vocab_size (embedding 需要 vocab_size + 1 行)
    vocab_size = _fast_vocab_size(texts)
    cls_token_id = vocab_size
    embedding_vocab_size = vocab_size + 1

    print(f"词汇表大小 (采样估算): {vocab_size}")
    print(f"[CLS] token id: {cls_token_id}, Embedding 词表: {embedding_vocab_size}")

    train_dataset = NewsDataset(X_train, y_train, max_len=max_len, cls_token_id=cls_token_id)
    test_dataset = NewsDataset(X_test, y_test, max_len=max_len, cls_token_id=cls_token_id)

    use_pin_memory = torch.cuda.is_available()
    if num_workers > 0 and not torch.cuda.is_available():
        num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin_memory,
                              persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin_memory,
                             persistent_workers=(num_workers > 0))

    # 类别权重
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return train_loader, test_loader, embedding_vocab_size, num_classes, class_weights

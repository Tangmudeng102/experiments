"""
心跳信号分类预测 — 1D-CNN 模型

架构 (4层卷积):
  Conv1d(1, 32, k=7) → BN → ReLU → MaxPool1d(2)
  Conv1d(32, 64, k=5) → BN → ReLU → MaxPool1d(2)
  Conv1d(64, 128, k=5) → BN → ReLU → MaxPool1d(2)
  Conv1d(128, 256, k=3) → BN → ReLU → AdaptiveAvgPool1d(1)
  Flatten → Linear(256, 128) → ReLU → Dropout(0.5) → Linear(128, 4)
"""

import torch
import torch.nn as nn


class ECG1DCNN(nn.Module):
    """1D-CNN 用于心电信号 4 分类"""

    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 205 -> 102
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 102 -> 51
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 51 -> 25
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # 25 -> 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


def create_model(input_length=None, num_classes=4, **kwargs):
    """工厂函数，保持与旧代码兼容"""
    return ECG1DCNN(num_classes=num_classes, **kwargs)

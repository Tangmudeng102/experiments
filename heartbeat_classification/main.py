"""
心跳信号分类预测 — 主入口脚本
实验一：使用 1D-CNN 对心电图信号进行 4 分类

模型架构:
  Conv1d(1, 32, k=7) → BN → ReLU → MaxPool1d(2)
  Conv1d(32, 64, k=5) → BN → ReLU → MaxPool1d(2)
  Conv1d(64, 128, k=5) → BN → ReLU → MaxPool1d(2)
  Conv1d(128, 256, k=3) → BN → ReLU → AdaptiveAvgPool1d(1)
  Flatten → Linear(256, 128) → ReLU → Dropout(0.5) → Linear(128, 4)

使用说明:
  python main.py                          # 默认 30 epochs
  python main.py --epochs 50              # 自定义训练轮数
  python main.py --batch_size 128         # 自定义批次大小
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train


def main():
    import argparse
    parser = argparse.ArgumentParser(description="心跳信号分类预测")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--threads", type=int, default=4, help="CPU 线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "train.csv")

    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        print("请将 train.csv 放入 data/ 目录后重试")
        sys.exit(1)

    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    print("\n" + "=" * 60)
    print("  心跳信号分类预测")
    print("  架构: 1D-CNN (4层卷积)")
    print("  损失: 加权交叉熵 (CrossEntropyLoss)")
    print("  优化: Adam + StepLR")
    print("=" * 60)

    train(
        data_path, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_threads=args.threads, seed=args.seed
    )


if __name__ == "__main__":
    main()

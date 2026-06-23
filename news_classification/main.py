"""
新闻文本分类 — 主入口脚本
实验二：使用 BERT 风格 Transformer 对新闻文本进行 14 分类

模型结构 (BERT 风格):
  - [CLS] token + Token/Position/Segment Embedding
  - Post-LN TransformerEncoder 堆叠
  - [CLS] Pooling → Linear 分类

使用说明:
  1. 确保 train_set.csv 位于上级目录
  2. 运行: python main.py
  3. 可选: python main.py --device cpu --epochs 50 --batch_size 128 --max_len 1024
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_analysis import run_analysis
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="新闻文本分类")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="设备选择 (default: auto)")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--max_len", type=int, default=None, help="最大序列长度")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "train_set.csv")

    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        print("请将 train_set.csv 放入上级目录后重试")
        sys.exit(1)

    # 阶段 1: 数据分析
    print("\n" + "=" * 60)
    print("  阶段 1: 数据分析与可视化")
    print("=" * 60)
    run_analysis(data_path)

    # 阶段 2: 模型训练与评估
    print("\n" + "=" * 60)
    print("  阶段 2: 模型训练与评估")
    print("=" * 60)

    # 设备选择
    if args.device == "cpu":
        use_gpu = False
    elif args.device == "cuda":
        use_gpu = True
    else:
        use_gpu = torch.cuda.is_available()

    # 用命令行参数覆盖默认值
    epochs = args.epochs if args.epochs else (30 if use_gpu else 20)
    batch_size = args.batch_size if args.batch_size else 64
    max_len = args.max_len if args.max_len else (512 if use_gpu else 128)
    d_model = 256 if use_gpu else 128
    nhead = 8 if use_gpu else 4
    num_layers = 6 if use_gpu else 4
    dim_feedforward = 1024 if use_gpu else 512

    device_label = "GPU" if use_gpu else "CPU"
    print(f"[配置] {device_label} 模式")
    train(data_path, epochs=epochs, batch_size=batch_size, max_len=max_len,
          d_model=d_model, nhead=nhead, num_layers=num_layers,
          dim_feedforward=dim_feedforward)


if __name__ == "__main__":
    main()

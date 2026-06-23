"""
心跳信号分类预测 — 数据分析与可视化
使用 NumPy、Pandas、Matplotlib 对训练数据进行探索性分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(data_path=None):
    """加载原始数据"""
    if data_path is None:
        data_path = os.path.join(BASE_DIR, "data", "train.csv")
    df = pd.read_csv(data_path)
    return df


def parse_signals(df):
    """将 heartbeat_signals 字符串解析为 numpy 数组"""
    signals = df["heartbeat_signals"].apply(
        lambda x: np.array([float(v) for v in x.split(",")], dtype=np.float32)
    )
    signal_matrix = np.stack(signals.values)
    return signal_matrix


def data_overview(df, signal_matrix):
    """打印数据集基本信息"""
    print("=" * 60)
    print("数据集基本概况")
    print("=" * 60)
    print(f"样本总数:       {df.shape[0]}")
    print(f"特征数(列):     {df.shape[1]}")
    print(f"列名:           {list(df.columns)}")
    print(f"信号序列长度:   {signal_matrix.shape[1]}")
    print(f"标签类别数:     {df['label'].nunique()}")
    print(f"标签分布:")
    for label, count in sorted(Counter(df["label"]).items()):
        print(f"  类别 {label}: {count} ({count/len(df)*100:.2f}%)")
    print(f"缺失值数量:")
    print(df.isnull().sum())
    print(f"信号值范围:     [{signal_matrix.min():.4f}, {signal_matrix.max():.4f}]")
    print(f"信号均值:       {signal_matrix.mean():.4f}")
    print(f"信号标准差:     {signal_matrix.std():.4f}")


def plot_label_distribution(df, save_path=None):
    """绘制标签分布柱状图"""
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "label_distribution.png")
    plt.figure(figsize=(8, 5))
    label_counts = df["label"].value_counts().sort_index()
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
    bars = plt.bar(label_counts.index, label_counts.values, color=colors)
    for bar, count in zip(bars, label_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{count}\n({count/len(df)*100:.1f}%)",
                 ha="center", fontsize=10)
    plt.xlabel("Label")
    plt.ylabel("Sample Count")
    plt.title("Label Distribution in Training Set")
    plt.xticks(label_counts.index, [f"Class {i}" for i in label_counts.index])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[保存] 标签分布图 -> {save_path}")


def plot_sample_signals(df, signal_matrix, samples_per_class=3, save_path=None):
    """绘制每个类别的样本信号曲线"""
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "sample_signals.png")
    fig, axes = plt.subplots(4, samples_per_class, figsize=(14, 10))
    for label in range(4):
        idxs = df[df["label"] == label].index[:samples_per_class]
        for j, idx in enumerate(idxs):
            ax = axes[label, j]
            ax.plot(signal_matrix[idx], color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"][label], linewidth=0.5)
            ax.set_title(f"Class {label} - Sample {idx}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Signal Value")
    plt.suptitle("Sample Heartbeat Signals by Class", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[保存] 样本信号图 -> {save_path}")


def plot_signal_statistics(signal_matrix, df, save_path=None):
    """绘制信号统计特征（均值、方差）按类别分布"""
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "signal_stats.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    mean_vals = signal_matrix.mean(axis=1)
    std_vals = signal_matrix.std(axis=1)
    max_vals = signal_matrix.max(axis=1)
    min_vals = signal_matrix.min(axis=1)

    for label in range(4):
        mask = df["label"] == label
        axes[0, 0].hist(mean_vals[mask], bins=40, alpha=0.5, label=f"Class {label}")
        axes[0, 1].hist(std_vals[mask], bins=40, alpha=0.5, label=f"Class {label}")
        axes[1, 0].hist(max_vals[mask], bins=40, alpha=0.5, label=f"Class {label}")
        axes[1, 1].hist(min_vals[mask], bins=40, alpha=0.5, label=f"Class {label}")

    axes[0, 0].set_title("Mean Value Distribution")
    axes[0, 1].set_title("Std Value Distribution")
    axes[1, 0].set_title("Max Value Distribution")
    axes[1, 1].set_title("Min Value Distribution")
    for ax in axes.flat:
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[保存] 信号统计图 -> {save_path}")


def plot_correlation_heatmap(signal_matrix, df, save_path=None):
    """绘制类别均值信号的相关性热力图"""
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "correlation.png")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for label in range(4):
        mask = df["label"] == label
        class_signals = signal_matrix[mask]
        # 每100个时间步采样一个点做相关性分析
        sampled = class_signals[:, ::max(1, class_signals.shape[1] // 30)]
        corr = np.corrcoef(sampled[:min(100, len(sampled))])
        im = axes[label].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        axes[label].set_title(f"Class {label} Correlation")
    plt.colorbar(im, ax=axes, shrink=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[保存] 相关性热力图 -> {save_path}")


def run_analysis(data_path=None):
    """执行完整的数据分析流程"""
    if data_path is None:
        data_path = os.path.join(BASE_DIR, "data", "train.csv")
    print("加载数据...")
    df = load_data(data_path)
    signal_matrix = parse_signals(df)

    data_overview(df, signal_matrix)
    plot_label_distribution(df)
    plot_sample_signals(df, signal_matrix)
    plot_signal_statistics(signal_matrix, df)
    plot_correlation_heatmap(signal_matrix, df)

    print("\n数据分析完成！所有图表已保存到 ../results/ 目录")


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../data/train.csv"
    run_analysis(data_path)

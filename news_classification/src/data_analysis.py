"""
新闻文本分类 — 数据分析与可视化
使用 NumPy、Pandas、Matplotlib 对训练数据进行探索性分析
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# --- 无头服务器兼容配置（必须在 import pyplot 之前）---
import matplotlib
matplotlib.use('Agg')

# 中文字体兜底：尝试匹配，失败则用默认字体
try:
    from matplotlib.font_manager import fontManager
    _FONT_CANDIDATES = [
        'SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Sans SC',
        'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans',
    ]
    _available = {f.name for f in fontManager.ttflist}
    _found = [f for f in _FONT_CANDIDATES if f in _available]
    if _found:
        matplotlib.rcParams['font.sans-serif'] = _found + ['sans-serif']
except Exception:
    pass

matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import seaborn as sns

# --- 路径 ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_data(data_path):
    return pd.read_csv(data_path, sep="\t")


def data_overview(df):
    """打印数据集基本信息"""
    texts = df["text"].values
    labels = df["label"].values
    text_lens = np.array([len(str(t).split()) for t in texts])

    all_words = set()
    for t in texts:
        all_words.update(str(t).split())
    vocab_size = len(all_words)

    print("=" * 60)
    print("数据集基本概况")
    print("=" * 60)
    print(f"样本总数:       {df.shape[0]}")
    print(f"列名:           {list(df.columns)}")
    print(f"标签类别数:     {df['label'].nunique()}")
    print(f"标签范围:       {df['label'].min()} - {df['label'].max()}")
    print(f"词汇表大小:     {vocab_size}")
    print(f"序列长度 — 最小: {text_lens.min()}, 最大: {text_lens.max()}, "
          f"均值: {text_lens.mean():.1f}, 中位数: {np.median(text_lens):.1f}")
    print(f"\n标签分布:")
    label_counts = df["label"].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  类别 {label}: {count} ({count / len(df) * 100:.2f}%)")


def plot_label_distribution(df, save_path=None):
    """绘制标签分布柱状图"""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "label_distribution.png")
    _ensure_dir(save_path)

    label_counts = df["label"].value_counts().sort_index()
    n_classes = len(label_counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")
    bars = ax.bar(range(n_classes), label_counts.values,
                  color=[cmap(i % 20) for i in range(n_classes)])
    for bar, count in zip(bars, label_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{count}\n({count / len(df) * 100:.1f}%)", ha="center", fontsize=8)

    ax.set_xlabel("Label")
    ax.set_ylabel("Sample Count")
    ax.set_title("News Text Classification — Label Distribution")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f"Class {i}" for i in range(n_classes)])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 标签分布图 -> {save_path}")


def plot_text_length_distribution(df, save_path=None):
    """绘制文本长度分布"""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "text_length_distribution.png")
    _ensure_dir(save_path)

    text_lens = df["text"].apply(lambda x: len(str(x).split()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(text_lens, bins=100, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(text_lens.median(), color="red", linestyle="--",
                    label=f"Median: {text_lens.median():.0f}")
    axes[0].axvline(text_lens.mean(), color="orange", linestyle="--",
                    label=f"Mean: {text_lens.mean():.0f}")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Text Length Distribution (All Samples)")
    axes[0].legend()

    axes[1].hist(text_lens, bins=100, color="steelblue", edgecolor="white",
                 alpha=0.8, cumulative=True, density=True)
    axes[1].axhline(0.95, color="green", linestyle="--", label="95%")
    axes[1].axvline(np.percentile(text_lens, 95), color="green", linestyle=":",
                    label=f"P95: {np.percentile(text_lens, 95):.0f}")
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Cumulative Proportion")
    axes[1].set_title("Cumulative Distribution of Text Length")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 文本长度分布图 -> {save_path}")


def plot_text_length_by_class(df, save_path=None):
    """绘制各类别的文本长度分布"""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "text_length_by_class.png")
    _ensure_dir(save_path)

    text_lens = df["text"].apply(lambda x: len(str(x).split()))
    df_plot = pd.DataFrame({"label": df["label"], "length": text_lens})
    labels = sorted(df["label"].unique())
    data_by_class = [df_plot[df_plot["label"] == lbl]["length"].values for lbl in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data_by_class, patch_artist=True)
    # 兼容新版 matplotlib: labels 参数已废弃，用 set_xticklabels
    ax.set_xticklabels([f"Class {l}" for l in labels])

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(plt.get_cmap("tab20")(i % 20))

    ax.set_xlabel("Label")
    ax.set_ylabel("Sequence Length")
    ax.set_title("Text Length Distribution by Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 各类别文本长度分布图 -> {save_path}")


def plot_word_frequency(df, top_k=30, save_path=None):
    """绘制高频词汇"""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "word_frequency.png")
    _ensure_dir(save_path)

    word_counts = {}
    for text in df["text"]:
        for w in str(text).split():
            word_counts[w] = word_counts.get(w, 0) + 1

    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    words, counts = zip(*top_words)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(words)), counts, color="steelblue")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels([f"ID:{w}" for w in words])
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {top_k} Most Frequent Words")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 高频词汇图 -> {save_path}")


def run_analysis(data_path):
    """执行完整的数据分析流程"""
    print("加载数据...")
    df = load_data(data_path)

    data_overview(df)

    print(f"\n图表输出目录: {RESULTS_DIR}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_funcs = [
        ("标签分布", plot_label_distribution),
        ("文本长度分布", plot_text_length_distribution),
        ("各类别文本长度分布", plot_text_length_by_class),
        ("高频词汇", plot_word_frequency),
    ]

    for name, func in plot_funcs:
        try:
            func(df)
        except Exception as e:
            print(f"[错误] 生成 {name} 图失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # 验证输出
    generated = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".png")]
    if generated:
        print(f"\n数据分析完成！共生成 {len(generated)} 张图片: {generated}")
    else:
        print(f"\n警告: {RESULTS_DIR}/ 目录中没有生成任何图片，请检查上方错误信息")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "..", "train_set.csv")
    run_analysis(data_path)

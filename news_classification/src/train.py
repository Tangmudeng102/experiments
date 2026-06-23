"""
新闻文本分类 — BERT 风格模型训练与评估

模型: BERT 风格 Transformer
  - [CLS] token + Token/Position/Segment Embedding
  - Post-LN TransformerEncoder
  - [CLS] Pooling → Linear 分类

损失函数: Focal Loss + Label Smoothing
  Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
  其中 gamma=2 降低易分类样本的权重，alpha_t 使用类别权重
  结合 Label Smoothing 防止过拟合

优化器: AdamW + OneCycleLR (Cosine Annealing)
"""

import os
import sys
import time
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- 无头服务器兼容配置（必须在 import pyplot 之前）---
import matplotlib
matplotlib.use('Agg')

_FONT_CANDIDATES = [
    'SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    'Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Sans SC',
    'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans',
]
from matplotlib.font_manager import fontManager
_available = {f.name for f in fontManager.ttflist}
_found = [f for f in _FONT_CANDIDATES if f in _available]
if _found:
    matplotlib.rcParams['font.sans-serif'] = _found + ['sans-serif']
else:
    matplotlib.rcParams['font.sans-serif'] = ['sans-serif']
    warnings.warn("未找到中文字体，图表中文可能显示为方框")
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import create_dataloaders
from model import NewsTextClassifier

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    降低易分类样本的损失权重，使模型聚焦于难分类样本
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)

        # Label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_labels = torch.zeros_like(inputs)
                smooth_labels.fill_(self.label_smoothing / (num_classes - 1))
                smooth_labels.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smooth_labels * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Focal modulation
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        # Class weighting
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class EarlyStopping:
    """早停：监控验证指标，patience 轮无提升则停止"""

    def __init__(self, patience=10, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ExponentialMovingAverage:
    """指数移动平均：对模型参数做平滑，提升泛化能力"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, ema=None, clip_grad=1.0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, masks, labels in loader:
        input_ids = input_ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, masks)
        loss = criterion(outputs, labels)
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        if ema is not None:
            ema.update()

        total_loss += loss.item() * input_ids.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += input_ids.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在测试集上评估"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for input_ids, masks, labels in loader:
        input_ids = input_ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, masks)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * input_ids.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += input_ids.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return total_loss / total, accuracy, precision, recall, f1, all_preds, all_labels


def plot_training_curves(history, save_path=None):
    """绘制训练曲线"""
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "training_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, history["test_loss"], "r-", label="Test Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=1.5)
    axes[1].plot(epochs, history["test_acc"], "r-", label="Test Accuracy", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["test_f1"], "g-", label="Test F1 (macro)", linewidth=1.5)
    axes[2].plot(epochs, history.get("lr", []), "m--", label="LR", linewidth=1)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score / LR")
    axes[2].set_title("F1 Score & Learning Rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 训练曲线 -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path=None):
    """绘制混淆矩阵"""
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[f"C{i}" for i in range(num_classes)],
                yticklabels=[f"C{i}" for i in range(num_classes)],
                annot_kws={"fontsize": 8})
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 混淆矩阵 -> {save_path}")


def plot_per_class_metrics(y_true, y_pred, num_classes, save_path=None):
    """绘制各类别 F1/Precision/Recall"""
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "per_class_metrics.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(num_classes)
    width = 0.25
    ax.bar(x - width, p, width, label="Precision", color="#3498db")
    ax.bar(x, r, width, label="Recall", color="#2ecc71")
    ax.bar(x + width, f1, width, label="F1-Score", color="#e74c3c")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(num_classes)])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 各类别指标图 -> {save_path}")


def train(data_path, epochs=50, batch_size=128, lr=1e-3, wd=1e-2,
          max_len=1024, num_threads=4, d_model=256, nhead=8, num_layers=3,
          dim_feedforward=768):
    """完整训练流程"""
    torch.set_num_threads(num_threads)
    set_seed(42)

    # 设备选择: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 确保输出目录存在
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    # CPU 模式下自动调整线程数
    if device.type == "cpu":
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        torch.set_num_threads(cpu_count)
        print(f"CPU 线程数: {cpu_count}")

    # 数据加载
    print("\n加载数据...")
    train_loader, test_loader, vocab_size, num_classes, class_weights = create_dataloaders(
        data_path, batch_size=batch_size, max_len=max_len, test_size=0.2
    )

    # 模型初始化
    model = NewsTextClassifier(
        vocab_size=vocab_size, num_classes=num_classes,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_feedforward, max_len=max_len, dropout=0.3
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    print(f"  架构: BERT 风格 — [CLS] + Post-LN Transformer Encoder × {num_layers}")

    # Focal Loss + Label Smoothing
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)

    # AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))

    # Cosine Annealing with Warm Restarts
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos"
    )

    early_stopping = EarlyStopping(patience=15, min_delta=0.0005, mode="max")
    ema = ExponentialMovingAverage(model, decay=0.999)

    # 训练循环
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [], "test_f1": [], "lr": []
    }
    best_acc = 0.0

    print(f"\n开始训练 ({epochs} epochs)...")
    print(f"  损失函数: Focal Loss (gamma=2) + Label Smoothing (0.1)")
    print(f"  优化器: AdamW (lr={lr}, weight_decay={wd})")
    print(f"  调度器: OneCycleLR")
    print("-" * 80)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, ema, clip_grad=1.0
        )
        scheduler.step()

        # 评估时使用 EMA 参数
        ema.apply_shadow()
        test_loss, test_acc, precision, recall, f1, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "best_model.pth"))

        ema.restore()

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(f1)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
              f"Macro F1: {f1:.4f} | LR: {lr_now:.2e}")

        if early_stopping(test_acc):
            print(f"\n早停触发！最佳测试准确率: {best_acc:.4f}")
            break

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"训练完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 最终评估 (使用 EMA)
    print("\n加载最佳模型 (EMA) 进行最终评估...")
    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "models", "best_model.pth"), weights_only=True
    ))
    test_loss, test_acc, precision, recall, f1, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n{'='*50}")
    print(f"最终测试结果 (TestSet 40,000 样本):")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}  (macro avg)")
    print(f"  Recall:    {recall:.4f}  (macro avg)")
    print(f"  F1-Score:  {f1:.4f}  (macro avg)")
    print(f"{'='*50}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, digits=4))

    # 保存结果（每个绘图独立 try/except，一个失败不影响其他）
    for name, func in [
        ("训练曲线", lambda: plot_training_curves(history)),
        ("混淆矩阵", lambda: plot_confusion_matrix(all_labels, all_preds, num_classes)),
        ("各类别指标", lambda: plot_per_class_metrics(all_labels, all_preds, num_classes)),
    ]:
        try:
            func()
        except Exception as e:
            print(f"[错误] 生成 {name} 图失败: {e}", file=sys.stderr)

    np.savez(os.path.join(BASE_DIR, "results", "training_history.npz"), **history)
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "final_model.pth"))
    print(f"\n模型已保存到 {os.path.join(BASE_DIR, 'models')}/")

    return model, history, test_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="新闻文本分类训练")
    parser.add_argument("--data", type=str, default=os.path.join(BASE_DIR, "..", "train_set.csv"), help="数据路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--wd", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--max_len", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--threads", type=int, default=4, help="CPU线程数")
    parser.add_argument("--d_model", type=int, default=256, help="模型维度")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=3, help="Transformer 层数")
    parser.add_argument("--dim_feedforward", type=int, default=768, help="FFN 维度")
    args = parser.parse_args()

    train(args.data, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, wd=args.wd, max_len=args.max_len, num_threads=args.threads,
          d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
          dim_feedforward=args.dim_feedforward)

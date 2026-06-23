"""
心跳信号分类预测 — 模型训练与评估

损失函数: 加权交叉熵 (CrossEntropyLoss + class weights)
优化器: Adam + StepLR 学习率衰减
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import create_dataloaders
from model import ECG1DCNN

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 中文显示
_FONT_CANDIDATES = [
    'SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    'Noto Sans CJK SC', 'Noto Sans SC', 'DejaVu Sans',
]
from matplotlib.font_manager import fontManager
_available = {f.name for f in fontManager.ttflist}
_found = [f for f in _FONT_CANDIDATES if f in _available]
if _found:
    plt.rcParams['font.sans-serif'] = _found + ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = criterion(outputs, y)

        total_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += X.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return total_loss / total, accuracy, precision, recall, f1, all_preds, all_labels


def plot_training_curves(history, save_path=None):
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "training_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["test_loss"], "r-", label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Test Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy")
    ax2.plot(epochs, history["test_acc"], "r-", label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Test Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 训练曲线 -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Class {i}" for i in range(4)],
                yticklabels=[f"Class {i}" for i in range(4)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] 混淆矩阵 -> {save_path}")


def train(data_path, epochs=30, batch_size=64, lr=1e-3, weight_decay=0.0,
          num_threads=4, seed=42):
    """完整训练流程"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(num_threads)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 数据加载
    print("\n加载数据...")
    train_loader, test_loader, signal_length, class_weights = create_dataloaders(
        data_path, batch_size=batch_size, test_size=0.2, random_state=42
    )

    # 模型初始化
    model = ECG1DCNN(num_classes=4, dropout=0.5)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,} (可训练: {total_params:,})")

    # 加权交叉熵损失
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"损失函数: CrossEntropyLoss + 类别权重")

    # Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # StepLR: 每 step_size 轮学习率乘以 gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    print(f"优化器: Adam (lr={lr}), 调度器: StepLR(step=10, gamma=0.5)")

    # 训练循环
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    best_epoch = 0

    print(f"\n开始训练 ({epochs} epochs)...")
    print("-" * 70)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc, precision, recall, f1, _, _ = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "best_model.pth"))

        lr_now = optimizer.param_groups[0]["lr"]
        flag = " *" if test_acc >= best_acc - 1e-6 else ""
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
              f"F1: {f1:.4f} | LR: {lr_now:.2e}{flag}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"训练完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"最佳验证准确率: {best_acc:.4f} @ epoch {best_epoch}")

    # 加载最佳模型做最终评估
    print("\n加载最佳模型进行最终评估...")
    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "models", "best_model.pth"), weights_only=True
    ))
    test_loss, test_acc, precision, recall, f1, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n{'='*60}")
    print(f"最终测试结果 (TestSet):")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}  (macro avg)")
    print(f"  Recall:    {recall:.4f}  (macro avg)")
    print(f"  F1-Score:  {f1:.4f}  (macro avg)")
    print(f"{'='*60}")

    # 各类别指标
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    print("\n各类别指标:")
    for i in range(4):
        n_test = sum(np.array(all_labels) == i)
        n_correct = sum((np.array(all_labels) == i) & (np.array(all_preds) == i))
        print(f"  Class {i} (n={n_test}): Acc={n_correct/n_test:.4f}, "
              f"Precision={precisions[i]:.4f}, Recall={recalls[i]:.4f}, F1={f1s[i]:.4f}")

    # 保存结果
    plot_training_curves(history)
    plot_confusion_matrix(all_labels, all_preds)

    np.savez(os.path.join(BASE_DIR, "results", "training_history.npz"), **history)
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "final_model.pth"))
    print(f"\n模型已保存到 {os.path.join(BASE_DIR, 'models')}/")

    return model, history, test_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="心跳信号分类预测训练")
    parser.add_argument("--data", type=str, default=os.path.join(BASE_DIR, "data", "train.csv"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.data, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, num_threads=args.threads, seed=args.seed)

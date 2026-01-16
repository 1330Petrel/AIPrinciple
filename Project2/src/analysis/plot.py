"""训练结果可视化与混淆矩阵绘制模块

提供混淆矩阵热力图和训练历史曲线的绘制与保存工具函数
"""

import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    """计算、绘制并保存混淆矩阵热力图

    Args:
        y_true (np.ndarray): 真实整数标签数组, 一维或扁平化后的标签
        y_pred (np.ndarray): 预测整数标签数组, 与 y_true 形状一致
        class_names (list[str]): 与标签索引一一对应的类别名称列表
        output_path (Path): 保存混淆矩阵 PNG 图像的路径

    Returns:
        None
    """
    # 1. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 2. 创建图像对象, 根据类别数动态设置图像尺寸以保证可读性
    figsize = (max(10, len(class_names) // 3), max(8, len(class_names) // 4))
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # 旋转横轴刻度标签, 避免类别名称过长挤在一起
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # 调整布局防止坐标轴标签被裁剪
    plt.tight_layout()

    # 3. 保存图像到指定路径
    try:
        fig.savefig(output_path, dpi=300)
        logger.info(f"Saved confusion matrix to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix: {e}")

    plt.close(fig)  # 关闭图像以释放内存


def plot_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    """绘制训练和验证的损失与准确率变化曲线

    Args:
        history (dict[str, list[float]]): 训练历史字典, 需包含 train_loss、val_loss、train_acc、val_acc 四个键
        output_path (Path): 保存输出折线图 PNG 文件的路径

    Returns:
        None
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    epochs = range(1, len(train_loss) + 1)  # 以损失长度为基准生成 epoch 序列

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 绘制损失曲线: 训练损失与验证损失
    ax1.plot(epochs, train_loss, "bo-", label="Training Loss")
    ax1.plot(epochs, val_loss, "ro-", label="Validation Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # 绘制准确率曲线: 训练准确率与验证准确率
    ax2.plot(epochs, train_acc, "go-", label="Training Accuracy")
    ax2.plot(epochs, val_acc, "mo-", label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved loss and accuracy plots to {output_path}")

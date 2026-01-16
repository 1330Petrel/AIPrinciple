"""Weights & Biases 日志封装模块

提供针对 wandb 的统一日志接口, 用于记录训练过程中的标量指标、图像以及混淆矩阵等可视化结果
"""

import random
import logging
import numpy as np
from typing import Optional, Any

import torch

from utils.helpers import denormalize_image


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class WandbLogger:
    """Weights & Biases 日志记录封装类

    对 wandb 的常用操作进行简单封装, 提供初始化、指标记录、图像记录与混淆矩阵记录等功能
    """

    @staticmethod
    def init_wandb(config: dict[str, Any]) -> bool:
        """根据配置初始化 wandb

        当配置中启用了 wandb 并且依赖已正确安装时, 调用 wandb.init 创建或恢复一次运行

        Args:
            config (dict[str, Any]): 完整的训练配置字典, 需包含 project_name、run_name、output_dir 和 logging.use_wandb 等字段

        Returns:
            bool: 初始化成功返回 True, 否则返回 False
        """
        if config["logging"]["use_wandb"]:
            if WANDB_AVAILABLE:
                try:
                    wandb.init(
                        project=config["project_name"],
                        name=config["run_name"],
                        config=config,
                        dir=config["output_dir"],
                        save_code=False,
                        resume="allow",
                    )
                    return True
                except Exception as e:
                    logging.error(f"Failed to initialize wandb: {e}")
                logger.info("Weights & Biases integration enabled, run initialized")
            else:
                logger.warning(
                    "Wandb is configured to be used, but the 'wandb' package is not installed. "
                    "Please install it with 'pip install wandb' to enable logging."
                )
        return False

    @staticmethod
    def is_available() -> bool:
        """判断 wandb 是否可用

        Returns:
            bool: 若 wandb 安装可用且当前存在有效运行则返回 True
        """
        return WANDB_AVAILABLE and wandb.run is not None

    @staticmethod
    def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
        """记录标量指标

        常用于记录训练或验证过程中的损失、准确率等标量信息

        Args:
            metrics (dict[str, float]): 指标名称到数值的映射
            step (Optional[int]): 当前记录对应的步数或迭代次数, 为 None 时交由 wandb 自行维护

        Returns:
            None
        """
        if not WandbLogger.is_available():
            return

        if step is not None:
            metrics["step"] = step  # 显式记录 step, 便于对齐曲线
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"W&B log_metrics failed: {e}")

    @staticmethod
    def log_images(
        images: torch.Tensor,
        preds: torch.Tensor,
        gts: torch.Tensor,
        idx_to_class: dict[int, str],
        num_images: int = 16,
    ) -> None:
        """记录一批带有预测和真实标签的图像

        会从当前批次中随机抽取若干张图像, 反归一化后以图像面板形式上传到 wandb

        Args:
            images (torch.Tensor): 图像张量, 形状为 (B, C, H, W)
            preds (torch.Tensor): 模型预测类别索引, 形状为 (B,)
            gts (torch.Tensor): 真实类别索引, 形状为 (B,)
            idx_to_class (dict[int, str]): 类别 id 到类别名的映射字典
            num_images (int): 本次最多记录的图像数量

        Returns:
            None
        """
        if not WandbLogger.is_available() or len(images) == 0:
            return

        # 从批次中随机选择图像进行记录，避免每次都记录相同的图像
        num_to_log = min(num_images, len(images))  # 实际记录的图像数量
        indices = random.sample(range(len(images)), num_to_log)  # 随机挑选索引

        wandb_images: list[Any] = []  # 存放 wandb.Image 对象
        for i in indices:
            image_tensor = images[i]
            # 反归一化图像以获得正确的颜色
            image_np = denormalize_image(image_tensor.unsqueeze(0))

            pred_idx = int(preds[i].item())
            gt_idx = int(gts[i].item())

            pred_label = idx_to_class[pred_idx]
            gt_label = idx_to_class[gt_idx]

            caption = f"Pred: {pred_label}\nGT: {gt_label}"
            wandb_images.append(wandb.Image(image_np, caption=caption))

        # 将图像列表记录到 "Validation Predictions" 面板下
        try:
            WandbLogger.log_metrics({"Validation Predictions": wandb_images})
        except Exception as e:
            logger.warning(f"W&B log_images failed: {e}")

    @staticmethod
    def log_confusion_matrix(
        preds: np.ndarray | list[int],
        gts: np.ndarray | list[int],
        class_names: list[str],
    ) -> None:
        """记录交互式混淆矩阵

        将预测结果与真实标签上传到 wandb, 以交互式混淆矩阵图的形式展示分类性能

        Args:
            preds (np.ndarray | list[int]): 模型的预测类别索引序列
            gts (np.ndarray | list[int]): 真实的类别索引序列
            class_names (list[str]): 按类别 id 对应顺序排列的类别名称列表

        Returns:
            None
        """
        if not WandbLogger.is_available():
            return

        try:
            cm_plot = wandb.plot.confusion_matrix(
                preds=preds, y_true=gts, class_names=class_names
            )
            WandbLogger.log_metrics({"Confusion Matrix": cm_plot})
        except Exception as e:
            logger.warning(f"W&B log_confusion_matrix failed: {e}")

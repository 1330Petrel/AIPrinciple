from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.engine.trainer import Trainer

logger = logging.getLogger(__name__)


class Callback:
    """训练过程回调的抽象基类

    所有具体回调需要继承该类并实现相应的钩子方法
    """

    def __init__(self, trainer: "Trainer") -> None:
        """初始化回调基类

        Args:
            trainer: 训练流程的核心控制对象
        """
        # 保存 trainer 引用, 用于在回调中访问模型与优化器等
        self.trainer: "Trainer" = trainer

    def on_epoch_end(self, epoch: int, logs: dict[str, Any], **kwargs: Any) -> None:
        """在每个 epoch 结束时被调用的钩子

        子类可根据需要重写该方法实现自定义逻辑

        Args:
            epoch: 当前 epoch 序号, 从 0 或 1 开始
            logs: 当前 epoch 的训练与验证指标字典
            **kwargs: 额外的上下文信息

        Returns:
            None
        """
        pass


class ModelCheckpoint(Callback):
    """模型检查点保存回调"""

    def __init__(
        self,
        trainer: "Trainer",
        monitor: str,
    ) -> None:
        """初始化模型检查点回调

        Args:
            trainer: 训练流程控制对象
            monitor: 要监控的指标名称, 例如 'val/acc' 或 'val/loss'

        Returns:
            None
        """
        super().__init__(trainer)
        self.monitor: str = monitor

    def on_epoch_end(self, epoch: int, logs: dict[str, Any], **kwargs: Any) -> None:
        """在每个 epoch 结束时评估并保存检查点

        Args:
            epoch: 当前 epoch 序号
            logs: 包含训练与验证指标的字典
            **kwargs: 预留的额外参数

        Returns:
            None
        """
        # 从日志中获取当前监控指标的取值
        current_val = logs.get(self.monitor)
        if current_val is None:
            logger.warning(
                f"ModelCheckpoint: Monitor '{self.monitor}' not found in logs. Skipping."
            )

        # 通过 ckpt_manager 保存当前检查点
        self.trainer.ckpt_manager.save(
            epoch=epoch,
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            scheduler=self.trainer.scheduler,
            metric=current_val,
        )


class EarlyStopping(Callback):
    """当监控的指标停止改善时, 提前停止训练

    通过记录最优指标和未改善的轮数, 在达到耐心阈值后向 trainer 发出停止信号
    """

    def __init__(
        self,
        trainer: "Trainer",
        monitor: str,
        mode: str,
        patience: int,
    ) -> None:
        """初始化提前停止回调

        Args:
            trainer: 训练流程控制对象
            monitor: 要监控的指标名称
            mode: 指标比较模式, 'min' 表示越小越好, 其他值表示越大越好
            patience: 在指标未改善时最多允许的连续 epoch 数

        Returns:
            None
        """
        super().__init__(trainer)
        self.monitor: str = monitor
        self.mode: str = mode
        self.patience: int = patience
        # 连续未提升的轮数计数器
        self.wait_counter: int = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, Any], **kwargs: Any) -> None:
        """在每个 epoch 结束时检查是否需要提前停止

        Args:
            epoch: 当前 epoch 序号
            logs: 包含训练与验证指标的字典
            **kwargs: 预留的额外参数

        Returns:
            None
        """
        # 从日志中获取当前监控指标的取值
        current_value = logs.get(self.monitor)
        if current_value is None:
            logger.warning(
                f"EarlyStopping: Monitor '{self.monitor}' not found in logs. Skipping."
            )
            return

        # 根据模式判断当前指标是否优于历史最佳
        is_better = (
            (current_value < self.trainer.best_metric_value)
            if self.mode == "min"
            else (current_value > self.trainer.best_metric_value)
        )

        if is_better:
            # 指标改善时更新最佳值并重置等待计数
            self.trainer.best_metric_value = current_value
            self.wait_counter = 0
        else:
            # 未改善时累计等待轮数, 超过耐心阈值则停止训练
            self.wait_counter += 1
            if self.wait_counter >= self.patience:
                logger.info(
                    f"EarlyStopping: Patience of {self.patience} reached. Stopping training."
                )
                logger.info(
                    f"Best '{self.monitor}': {self.trainer.best_metric_value:.2f} at epoch {epoch - self.patience}"
                )
                # 向 trainer 发出停止信号
                self.trainer.stop_training = True


class LRSchedulerStep(Callback):
    """学习率调度器步进回调

    在每个 epoch 结束时调用学习率调度器的 step 方法
    """

    def __init__(self, trainer: "Trainer", monitor: str) -> None:
        """初始化学习率调度回调

        Args:
            trainer: 训练流程控制对象
            monitor: 当使用 ReduceLROnPlateau 时要监控的指标名称

        Returns:
            None
        """
        super().__init__(trainer)
        self.monitor: str = monitor

    def on_epoch_end(self, epoch: int, logs: dict[str, Any], **kwargs: Any) -> None:
        """在每个 epoch 结束时调用调度器步进

        Args:
            epoch: 当前 epoch 序号
            logs: 包含训练与验证指标的字典
            **kwargs: 预留的额外参数

        Returns:
            None
        """
        if self.trainer.scheduler is None:
            return

        # 对于 ReduceLROnPlateau, 需要传入监控指标值
        if isinstance(
            self.trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            metric_value = logs.get(self.monitor)
            if metric_value is not None:
                self.trainer.scheduler.step(metrics=metric_value)
            else:
                logger.warning(
                    f"LRScheduler: Metric '{self.monitor}' not found for ReduceLROnPlateau."
                )
        else:
            # 其他调度器调用 step 即可
            self.trainer.scheduler.step()

"""训练流程控制模块

完整的训练与验证循环, 包括优化器与学习率调度器构建、回调管理、训练历史记录与可视化
"""

import json
import logging
import numpy as np
from tqdm import tqdm
from typing import Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from utils.wandb_logger import WandbLogger
from utils.metrics import MetricsCalculator
from utils.helpers import AttrDict, CheckpointManager
from analysis.plot import plot_training_history
from engine.callbacks import ModelCheckpoint, LRSchedulerStep, EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """通用分类模型训练器

    负责构建优化器和学习率调度器, 管理回调函数, 执行完整的
    训练与验证流程, 并记录与可视化训练历史
    """

    def __init__(
        self,
        config: AttrDict,
        device: torch.device,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        metrics_calculator: MetricsCalculator,
        idx_to_class: dict[int, str],
        use_wandb: bool,
    ) -> None:
        """初始化训练器

        Args:
            config: 解析后的训练配置对象
            device: 训练使用的设备
            model: 需要训练的神经网络模型
            train_loader: 训练集数据加载器
            val_loader: 验证集数据加载器
            metrics_calculator: 用于计算评估指标的工具类实例
            idx_to_class: 类别索引到类别名称的映射
            use_wandb: 是否启用 Weights & Biases 日志记录

        Returns:
            None
        """
        self.cfg: AttrDict = config
        self.device: torch.device = device
        # 将模型移动到指定设备
        self.model: nn.Module = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics_calculator = metrics_calculator
        self.idx_to_class = idx_to_class
        self.class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        self.use_wandb = use_wandb

        # 从配置中解析训练参数
        self.epochs: int = config.training.epochs
        self.grad_clip: float = config.training.get("grad_clip", 0.0)

        # 初始化内部状态
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.stop_training: bool = False
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        # 构建损失函数、优化器、调度器和回调
        self.criterion = CrossEntropyLoss()
        self._build_optimizer()
        self._build_scheduler()
        self._build_callbacks()

        # 如配置中指定, 则从已有检查点恢复训练
        if self.cfg.training.get("resume_from", None):
            self._resume_checkpoint(Path(self.cfg.training.resume_from))

    def _build_optimizer(self) -> None:
        """根据配置构建优化器

        支持 Adam、AdamW 与 SGD 三种优化器类型

        Returns:
            None
        """
        optim_cfg = self.cfg.optimizer
        # 仅对需要更新的参数构建优化器
        params = [p for p in self.model.parameters() if p.requires_grad]
        if optim_cfg.name == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=optim_cfg.lr,
                weight_decay=optim_cfg.weight_decay,
            )
        elif optim_cfg.name == "adamw":
            self.optimizer = optim.AdamW(
                params,
                lr=optim_cfg.lr,
                weight_decay=optim_cfg.weight_decay,
            )
        elif optim_cfg.name == "sgd":
            self.optimizer = optim.SGD(
                params,
                lr=optim_cfg.lr,
                momentum=optim_cfg.momentum,
                weight_decay=optim_cfg.weight_decay,
            )
        else:
            logger.error(f"Unsupported optimizer: {optim_cfg.name}")
            raise ValueError(f"Unsupported optimizer: {optim_cfg.name}")

    def _build_scheduler(self) -> None:
        """根据配置构建学习率调度器

        支持 StepLR、CosineAnnealingLR、ExponentialLR、
        LambdaLR(warmup_cosine) 和 ReduceLROnPlateau 等策略

        Returns:
            None
        """
        sched_cfg = self.cfg.get("scheduler", None)
        if not sched_cfg or sched_cfg.name == "none":
            self.scheduler = None
            return

        name = sched_cfg.name
        warmup_epochs: int = sched_cfg.get("warmup_epochs", 0)

        # 构建主调度器
        if name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.step_size,
                gamma=sched_cfg.gamma,
            )
        elif name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg.max_epochs,
                eta_min=sched_cfg.min_lr,
            )
        elif name == "exp":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=sched_cfg.gamma
            )
        elif name == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.cfg.reporting.monitor_mode,
                factor=sched_cfg.factor,
                patience=sched_cfg.patience,
                min_lr=sched_cfg.min_lr,
            )
        else:
            self.scheduler = None
            logger.warning(f"Unsupported scheduler: {name}. No scheduler will be used.")

        # 可选 warmup: 仅对非 ReduceLROnPlateau 调度器有效, 使用 LinearLR + SequentialLR 组合
        if (
            warmup_epochs > 0
            and name != "reduce_on_plateau"
            and self.scheduler is not None
        ):
            try:
                from torch.optim.lr_scheduler import LinearLR, SequentialLR

                start_factor = float(sched_cfg.get("warmup_start_factor", 0.1))
                warmup_sched = LinearLR(
                    self.optimizer,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                # 将 warmup 调度器串联在主调度器之前
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, self.scheduler],
                    milestones=[warmup_epochs],
                )
            except Exception:
                logger.warning(
                    "Warmup not applied: LinearLR/SequentialLR unavailable. Continuing without warmup."
                )

    def _build_callbacks(self) -> None:
        """根据配置构建训练过程回调

        包括模型检查点保存、提前停止以及学习率调度步进等回调

        Returns:
            None
        """
        report_cfg = self.cfg.reporting
        monitor = report_cfg.monitor_metric
        mode = report_cfg.monitor_mode
        topk = report_cfg.get("save_topk_checkpoints", 0)

        # 根据监控模式初始化最优指标, 以便后续比较
        self.best_metric_value: float = torch.inf if mode == "min" else -torch.inf

        # 构建检查点管理器, 统一管理模型保存与加载
        self.ckpt_manager = CheckpointManager(
            out_dir=Path(self.cfg.output_dir) / "checkpoints",
            monitor=monitor,
            mode=mode,
            save_topk=topk,
        )

        self.callbacks = [
            ModelCheckpoint(self, monitor=monitor),
            (
                EarlyStopping(
                    trainer=self,
                    monitor=monitor,
                    patience=self.cfg.training.get("early_stop_patience"),
                    mode=mode,
                )
                if self.cfg.training.get("early_stop_patience", 0) > 0
                else None
            ),
            LRSchedulerStep(self, monitor=monitor),
        ]
        # 过滤掉未启用的回调
        self.callbacks = [cb for cb in self.callbacks if cb is not None]

    def _train_one_epoch(self) -> tuple[float, float]:
        """执行一个 epoch 的训练循环

        Returns:
            一个二元组, 包含该 epoch 的平均训练损失和准确率
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Train]"
        )

        for batch_idx, (inputs, targets, _) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播与损失计算
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播与参数更新
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # 累积损失与准确率统计
            batch_loss = loss.item()
            train_loss += batch_loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            self.global_step += 1

            # 更新进度条和日志
            progress_bar.set_postfix(
                Loss=f"{batch_loss:.4f}", Acc=f"{100.*correct/total:.2f}%"
            )
            if self.use_wandb and self.global_step % self.cfg.logging.log_interval == 0:
                WandbLogger.log_metrics(
                    {
                        "train/loss": loss.item(),
                        "train/acc": 100.0 * correct / total,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/batch": batch_idx
                        + self.current_epoch * len(self.train_loader),
                        "train/epoch": self.current_epoch,
                    },
                    step=self.global_step,
                )

        train_loss /= len(self.train_loader)
        train_acc = 100.0 * correct / total
        logger.info(
            f"Epoch {self.current_epoch}/{self.epochs} [Train] Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%"
        )
        return train_loss, train_acc

    @torch.inference_mode()
    def _validate_one_epoch(self) -> tuple[float, float, dict[str, Any]]:
        """执行一个 epoch 的验证循环

        Returns:
            三元组, 依次为验证集平均损失、准确率以及详细指标字典
        """
        self.model.eval()
        self.metrics_calculator.reset()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[int] = []
        all_targets: list[int] = []

        progress_bar = tqdm(
            self.val_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Val]"
        )

        image_batch_to_log: torch.Tensor | None = None
        preds_to_log: torch.Tensor | None = None
        gts_to_log: torch.Tensor | None = None

        for i, (inputs, targets, _) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播与损失计算
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            batch_preds = outputs.argmax(dim=1)
            all_preds.extend(batch_preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # 记录首个 batch 的图像与预测用于可视化
            if i == 0:
                image_batch_to_log = inputs.cpu()
                preds_to_log = batch_preds.cpu()
                gts_to_log = targets

            # 累积损失与准确率, 更新指标计算器
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            self.metrics_calculator.update(outputs, targets)

        val_loss /= len(self.val_loader)
        val_acc = 100.0 * correct / total
        metrics = self.metrics_calculator.compute()

        logger.info(
            f"Epoch {self.current_epoch}/{self.epochs} [Val] Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )
        if self.use_wandb:
            # 记录验证指标与混淆矩阵, 以及一批示例图像
            WandbLogger.log_metrics(metrics, step=self.current_epoch)
            WandbLogger.log_confusion_matrix(
                preds=all_preds,
                gts=all_targets,
                class_names=self.class_names,
            )
            if (
                image_batch_to_log is not None
                and preds_to_log is not None
                and gts_to_log is not None
            ):
                WandbLogger.log_images(
                    images=image_batch_to_log,
                    preds=preds_to_log,
                    gts=gts_to_log,
                    idx_to_class=self.idx_to_class,
                )

        return val_loss, val_acc, metrics

    def _resume_checkpoint(self, ckpt_path: Path) -> None:
        """从给定路径恢复训练检查点

        Args:
            ckpt_path: 检查点文件路径

        Returns:
            None
        """
        ckpt = self.ckpt_manager.load(
            ckpt_path, self.model, self.optimizer, self.scheduler
        )
        self.current_epoch = ckpt["epoch"]
        self.best_metric_value = ckpt["best_metric_value"]
        logger.info(f"Resumed training from checkpoint: {ckpt_path}.")

    def fit(self) -> None:
        """执行完整训练流程

        包含多个 epoch 的训练与验证循环, 同时触发回调、
        记录训练历史并在结束后生成可视化与 JSON 结果

        Returns:
            None
        """
        logger.info("Starting training process...")

        for _ in range(self.current_epoch, self.epochs):
            # 同步当前 epoch 计数
            self.current_epoch += 1

            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc, val_metrics = self._validate_one_epoch()

            # 记录当前 epoch 的训练与验证结果
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            epoch_logs: dict[str, Any] = {
                **val_metrics,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
            }
            # 在每个 epoch 结束时依次触发回调
            for cb in self.callbacks:
                cb.on_epoch_end(epoch=self.current_epoch, logs=epoch_logs)

            if self.stop_training:
                break

        # 训练结束后绘制并保存损失和准确率曲线
        self.ckpt_manager.save_best_model(self.device)
        plot_training_history(self.history, Path(self.cfg.output_dir) / "acc_loss.png")
        path = Path(self.cfg.output_dir) / "training_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved training history to {path}.")
        logger.info("Training finished.")

"""训练辅助工具与检查点管理模块

提供随机种子设置、设备选择、图像反归一化等通用工具函数
并封装训练过程中模型检查点的保存、排序与清理逻辑
"""

import os
import json
import time
import shutil
import random
import logging
import numpy as np
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict
from torchvision.models import resnet50

import torch

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42) -> None:
    """设置全局随机种子

    同时设置 Python random、NumPy 和 PyTorch 相关随机源, 在可用时启用 CUDA 的确定性行为

    Args:
        seed (int): 随机种子数值

    Returns:
        None
    """
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # NumPy 随机数
    torch.manual_seed(seed)  # CPU 上的 Torch 随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 开启确定性
        torch.backends.cudnn.benchmark = False  # 关闭 benchmark 保证可复现

    logger.info(f"Global random seed set to: {seed}")


def get_device(device: str) -> torch.device:
    """根据配置字符串选择计算设备

    若传入空字符串则自动在可用 GPU 与 CPU 之间选择, 并对非法配置给出警告后回退

    Args:
        device (str): 期望使用的设备标识, "cuda" 或 "cpu"

    Returns:
        torch.device: 最终实际使用的设备对象
    """
    if device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
    elif device not in ["cuda", "cpu"]:
        logger.warning(f"Unknown device '{device}'.")
        device = "cuda" if torch.cuda.is_available() else "cpu"  # 回退到可用设备
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        device = "cpu"  # 无 GPU 时强制使用 CPU
    torch_device = torch.device(device)
    logger.info(f"Using device: {torch_device}")
    return torch_device


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """将标准化图像张量反归一化并转换为可显示的 NumPy 数组

    默认假设输入图像使用 ImageNet 的均值和方差进行过归一化

    Args:
        tensor (torch.Tensor): 图像张量, 形状为 (1, C, H, W) 或 (C, H, W)

    Returns:
        np.ndarray: 适用于可视化的 uint8 格式图像数组, 形状为 (H, W, C)
    """
    tensor = tensor.clone().squeeze(0).cpu()  # 去除 batch 维度并移动到 CPU
    for t, m, s in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)  # 逐通道反归一化
    tensor = tensor.permute(1, 2, 0)  # C, H, W -> H, W, C
    arr = np.clip(tensor.numpy() * 255, 0, 255)  # 映射到 [0, 255]
    return arr.astype(np.uint8)


class AttrDict(dict):
    """支持属性访问的字典

    可以通过点操作符访问键, 如 obj.key 等价于 obj["key"]
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """构造 AttrDict 实例

        Args:
            *args (Any): 传递给 dict 的位置参数
            **kwargs (Any): 传递给 dict 的关键字参数

        Returns:
            None
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)  # 嵌套字典转换为 AttrDict

    def __getattr__(self, key: str) -> Any:
        """通过属性方式访问字典键

        Args:
            key (str): 字段名称

        Returns:
            Any: 对应键的值
        """
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'") from e

    def __setattr__(self, key: str, value: Any) -> None:
        """通过属性方式设置字典键

        Args:
            key (str): 字段名称
            value (Any): 需要设置的值

        Returns:
            None
        """
        self[key] = value


@dataclass
class _CKPTEntry:
    """单个检查点元数据条目

    记录检查点路径、对应 epoch 与监控指标值
    """

    path: str
    epoch: Optional[int]
    metric: Optional[float]


class CheckpointManager:
    """检查点管理器

    负责保存训练过程中的模型检查点, 根据监控指标维护前 k 个最佳模型并管理其元数据
    """

    def __init__(
        self,
        out_dir: str | Path,
        monitor: str,
        mode: str,
        save_topk: int,
    ):
        """构造检查点管理器

        Args:
            out_dir (str | Path): 存储检查点文件的目录
            monitor (str): 用于排序和选择最佳模型的监控指标名称
            mode (str): 指标模式, "max" 表示越大越好, "min" 表示越小越好
            save_topk (int): 要保留的最佳检查点数量 (>=1)

        Returns:
            None
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_topk = save_topk
        self.metadata_path = self.out_dir / "checkpoints.json"
        self.best_ckpt = self.out_dir / "best.ckpt"
        self.last_best = None
        # maintain list of _CKPTEntry sorted by metric according to mode (best first)
        self._entries: list[_CKPTEntry] = []

    # -------------------------
    # Public API
    # -------------------------
    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metric: Optional[float] = None,
    ) -> Path:
        """保存单个检查点文件并更新内部元数据

        Args:
            epoch (int): 当前训练轮次数
            model (torch.nn.Module): 需要保存参数的模型实例
            optimizer (Optional[torch.optim.Optimizer]): 可选优化器, 若提供则保存其状态
            scheduler (Optional[Any]): 可选调度器, 若提供则保存其状态
            metric (Optional[float]): 用于排序的指标值, 为 None 时不参与最佳模型排序

        Returns:
            Path: 最终保存的检查点文件路径
        """
        # 准备文件名
        ts = int(time.time())  # 当前时间戳
        safe_monitor = str(self.monitor).replace("/", "_").replace("\\", "_")
        safe_metric = f"{metric:.4f}" if metric is not None else "NA"
        filename = f"epoch{epoch if epoch is not None else 'NA'}_{safe_monitor}{safe_metric}.pth"
        tmpfile = self.out_dir / (filename + ".tmp")  # 临时文件路径
        finalfile = self.out_dir / filename  # 最终文件路径

        # 构建要保存的数据负载
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),  # 模型状态字典
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),  # 优化器状态
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),  # 调度器状态
            "best_metric_value": (
                float(metric) if metric is not None else None
            ),  # 指标值
            "saved_at": ts,  # 保存时间戳
        }

        # 原子写入：先写入临时文件然后重命名
        try:
            torch.save(payload, str(tmpfile))  # 保存到临时文件
            os.replace(str(tmpfile), str(finalfile))  # 原子重命名
        except Exception:
            # 回退方案：直接保存
            torch.save(payload, str(finalfile))

        # 创建检查点条目
        entry = _CKPTEntry(
            path=str(finalfile),
            epoch=epoch,
            metric=float(metric) if metric is not None else None,
        )

        # 如果指标存在且 top_k > 0，插入到条目列表中并可能触发 prune
        if metric is not None and self.save_topk > 0:
            self._insert_entry(entry)  # 插入并保持排序
            self._prune_if_needed()  # 如果需要，进行清理
        else:
            # 指标为 None -> 只是追加到列表
            self._entries.append(entry)

        # 保存最佳检查点指针文件
        best_ckpt = self.get_best_checkpoint()
        if best_ckpt is not None and (
            best_ckpt != self.last_best or self.last_best is None
        ):
            self.last_best = best_ckpt
            try:
                shutil.copy2(str(best_ckpt), str(self.best_ckpt))
                logger.info(f"Epoch {epoch}: Updated best ckpt")
            except Exception:
                logger.warning(
                    f"Epoch {epoch}: Failed to update best checkpoint pointer"
                )

        # 持久化元数据
        self._save_metadata()
        return finalfile

    def load(
        self,
        path: str | Path,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> dict[str, Any]:
        """从磁盘加载检查点并可选地恢复模型与优化器状态

        Args:
            path (str | Path): 检查点文件路径
            model (Optional[torch.nn.Module]): 若提供则加载其中的 model_state_dict
            optimizer (Optional[torch.optim.Optimizer]): 若提供且检查点包含优化器状态则加载之
            scheduler (Optional[Any]): 若提供且检查点包含调度器状态则加载之
            map_location (Optional[str]): 传递给 torch.load 的设备映射字符串

        Returns:
            dict[str, Any]: 从磁盘加载的原始检查点数据字典
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        # 加载检查点文件
        ckpt = torch.load(str(p), map_location=map_location or "cpu")

        # 如果提供了模型，加载模型状态
        if (
            model is not None
            and "model_state_dict" in ckpt
            and ckpt["model_state_dict"] is not None
        ):
            try:
                model.load_state_dict(ckpt["model_state_dict"])
            except Exception:
                # 尝试宽松模式回退
                model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # 如果提供了优化器，加载优化器状态
        if (
            optimizer is not None
            and "optimizer_state_dict" in ckpt
            and ckpt["optimizer_state_dict"] is not None
        ):
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                logger.warning(f"Failed to load optimizer state from checkpoint: {p}")

        # 如果提供了调度器，加载调度器状态
        if (
            scheduler is not None
            and "scheduler_state_dict" in ckpt
            and ckpt["scheduler_state_dict"] is not None
        ):
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if "epoch" in ckpt and hasattr(scheduler, "last_epoch"):
                    try:
                        scheduler.last_epoch = ckpt["epoch"]
                    except Exception:
                        logger.warning(
                            "Could not set scheduler.last_epoch from checkpoint epoch"
                        )
            except Exception:
                logger.warning(f"Failed to load scheduler state from checkpoint: {p}")

        return ckpt

    def save_best_model(self, device: torch.device) -> Optional[Path]:
        """提取当前最佳检查点的权重并保存为 best_model.pth

        从 get_best_checkpoint 获取最佳 ckpt 路径, 只加载其中的
        `model_state_dict` 参数并保存为独立的纯权重文件, 便于推理阶段使用

        Args:
            device (torch.device): 加载检查点时使用的设备

        Returns:
            若存在最佳检查点则返回生成的 best_model.pth 路径, 否则返回 None
        """

        best_ckpt = self.get_best_checkpoint()
        if best_ckpt is None:
            logger.warning("No best checkpoint available to export best_model.pth")
            return None

        try:
            ckpt: dict[str, Any] = torch.load(str(best_ckpt), map_location=device)
        except Exception as e:
            logger.error(f"Failed to load best checkpoint from {best_ckpt}: {e}")
            return None

        if "model_state_dict" not in ckpt or ckpt["model_state_dict"] is None:
            logger.error(
                "Best checkpoint does not contain 'model_state_dict', cannot export best_model.pth"
            )
            return None

        model_state = ckpt["model_state_dict"]
        best_model_path = self.out_dir.parent / "best_model.pth"
        try:
            torch.save(model_state, str(best_model_path))
            logger.info(f"Exported best model weights to {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to save best_model.pth: {e}")
            return None

        return best_model_path

    def get_best_checkpoint(self) -> Optional[Path]:
        """根据排序返回当前最佳检查点路径

        Returns:
            Optional[Path]: 若存在有效指标的检查点则返回其路径, 否则返回 None
        """
        if not self._entries:
            return None

        # 过滤出有指标值的条目（已经按最佳在前排序）
        filtered = [e for e in self._entries if e.metric is not None]
        if not filtered:
            return None

        best = filtered[0]
        return Path(best.path)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """返回内部维护的检查点元数据列表

        Returns:
            list[dict[str, Any]]: 检查点元数据字典列表, 有指标值的按最佳在前排序
        """
        return [asdict(e) for e in self._entries]

    def cleanup(self):
        """强制清理无效检查点并应用 topk 约束

        Returns:
            None
        """
        # 移除文件不存在的条目
        new_entries: list[_CKPTEntry] = []
        for e in self._entries:
            if Path(e.path).exists():
                new_entries.append(e)
        self._entries = new_entries

        # 清理到top_k
        self._prune_if_needed()
        self._save_metadata()

    # -------------------------
    # 内部辅助方法
    # -------------------------
    def _insert_entry(self, entry: _CKPTEntry):
        """插入条目并保持内部列表的排序顺序

        Args:
            entry (_CKPTEntry): 要插入的检查点条目

        Returns:
            None
        """
        # 指标为None的条目优先级较低，追加在末尾
        inserted = False
        for i, e in enumerate(self._entries):
            if e.metric is None:
                # 在第一个None（无指标）条目之前插入
                self._entries.insert(i, entry)
                inserted = True
                break
            elif entry.metric is not None:
                # 根据模式比较
                if self.mode == "max":
                    if entry.metric > e.metric:  # 新指标更好
                        self._entries.insert(i, entry)
                        inserted = True
                        break
                else:  # min模式
                    if entry.metric < e.metric:  # 新指标更好
                        self._entries.insert(i, entry)
                        inserted = True
                        break

        if not inserted:
            # 如果没找到插入位置，追加到末尾
            self._entries.append(entry)

    def _prune_if_needed(self):
        """按需删除多余检查点文件, 保留前 save_topk 个条目

        Returns:
            None
        """
        if self.save_topk <= 0:
            return  # 不进行清理

        # 收集有指标的条目
        metric_entries = [e for e in self._entries if e.metric is not None]

        # 如果数量小于等于top_k，不需要清理
        if len(metric_entries) <= self.save_topk:
            return

        # 确定要移除的最差条目：保留前top_k个
        to_remove = metric_entries[self.save_topk :]

        # 移除to_remove的文件并从self._entries中移除
        remove_paths = {e.path for e in to_remove}
        new_entries = [e for e in self._entries if e.path not in remove_paths]

        # 删除文件
        for e in to_remove:
            try:
                p = Path(e.path)
                if p.exists():
                    p.unlink()  # 删除文件
            except Exception:
                logger.warning(f"Failed to delete checkpoint file: {e.path}")

        self._entries = new_entries

    def _save_metadata(self):
        """保存检查点元数据到 JSON 文件

        Returns:
            None
        """
        data = [asdict(e) for e in self._entries]  # 转换为字典列表
        try:
            # 原子写入：先写临时文件再重命名
            tmp = str(self.metadata_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, str(self.metadata_path))
        except Exception:
            # 回退方案：直接写入
            with open(str(self.metadata_path), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

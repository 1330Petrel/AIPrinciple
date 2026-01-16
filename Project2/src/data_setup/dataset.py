"""数据集与数据加载器定义模块

封装蝴蝶分类任务的数据集类与基于配置的 DataLoader 构建逻辑
"""

import json
import logging
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import get_transforms
from utils.helpers import AttrDict

logger = logging.getLogger(__name__)


class ButterflyDataset(Dataset):
    """用于蝴蝶分类任务的自定义数据集

    通过 CSV 中的文件路径和标签信息读取图像, 并应用给定的变换后返回张量和标签
    """

    def __init__(
        self,
        csv_path: Path,
        img_root_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        class_to_idx: dict[str, int],
    ) -> None:
        """初始化数据集

        从 CSV 中读取图像相对路径和标签列, 并检查所有图像文件是否存在

        Args:
            csv_path (Path): 指向 CSV 文件的路径
            img_root_dir (Path): 图像根目录, 与 CSV 中的相对路径拼接得到完整路径
            transform (Callable[[Image.Image], torch.Tensor]): 应用于图像的变换函数或组合
            class_to_idx (dict[str, int]): 将类别名称映射到整数索引的字典

        Returns:
            None
        """
        self.df = pd.read_csv(csv_path)  # 读取包含 filepath 和 label 的表格
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

        # 将相对路径转换为绝对路径
        self.df["filepath"] = self.df["filepath"].apply(lambda p: self.img_root_dir / p)
        missing = self.df["filepath"].apply(lambda p: not Path(p).exists())
        # 检查是否有缺失的图像文件
        if missing.any():
            missing_paths = self.df.loc[missing, "filepath"].tolist()
            logger.error("Missing image files in CSV: %s", missing_paths[:10])
            raise FileNotFoundError(
                f"Missing image files (count={len(missing_paths)})."
            )

    def __len__(self) -> int:
        """返回数据集中样本数量

        Returns:
            int: 数据集中样本的总数
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """按索引获取一个样本

        从表格中取出对应行, 加载图像、应用变换, 并返回张量、类别索引和文件路径字符串

        Args:
            idx (int): 样本索引, 范围为 [0, len(self) - 1]

        Returns:
            tuple[torch.Tensor, int, str]: (转换后的图像张量, 整数类别标签, 图像文件路径字符串)
        """
        row = self.df.iloc[idx]
        label = self.class_to_idx[row["label"]]

        # 加载图像并确保为RGB格式
        path = row["filepath"]
        image = Image.open(path).convert("RGB")

        # 应用变换
        image_t = self.transform(image)

        return image_t, label, str(path)


def create_dataloaders(
    config: AttrDict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """根据配置创建训练、验证和测试集的 DataLoader

    根据配置中的数据根目录、划分 CSV 文件路径和批大小等信息, 构建对应的 Dataset 与 DataLoader

    Args:
        config (AttrDict): 全局配置对象

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
            train_loader 为训练数据加载器
            val_loader 为验证数据加载器
            test_loader 为测试数据加载器
    """
    data_root = Path(config.data.root)
    splits_dir = Path(config.data.splits_dir)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    # 1. 从 JSON 文件加载标签映射
    class_to_idx, _ = get_class_mapping(splits_dir)
    if len(class_to_idx) != config.model.num_classes:
        logger.error(
            f"Class count mismatch: {len(class_to_idx)} in mapping vs {config.model.num_classes} in config."
        )
        raise ValueError("Number of classes in mapping does not match config.")

    # 2. 获取所有变换
    transforms: dict[str, Callable[[Image.Image], torch.Tensor]] = get_transforms(config)  # type: ignore[assignment]

    # 3. 创建 Dataset
    train_dataset = ButterflyDataset(
        csv_path=train_csv,
        img_root_dir=data_root,
        transform=transforms["train"],
        class_to_idx=class_to_idx,
    )
    val_dataset = ButterflyDataset(
        csv_path=val_csv,
        img_root_dir=data_root,
        transform=transforms["val"],
        class_to_idx=class_to_idx,
    )
    test_dataset = ButterflyDataset(
        csv_path=test_csv,
        img_root_dir=data_root,
        transform=transforms["test"],
        class_to_idx=class_to_idx,
    )

    # 4. 创建 DataLoader 实例
    num_workers = config.get("num_workers", 4)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,  # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,  # 验证和测试集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        # num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Train loader: {len(train_loader)} batches, {len(train_dataset)} images."
    )
    logger.info(
        f"Validation loader: {len(val_loader)} batches, {len(val_dataset)} images."
    )
    logger.info(f"Test loader: {len(test_loader)} batches, {len(test_dataset)} images.")

    return train_loader, val_loader, test_loader


def get_class_mapping(class_map_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    """从 JSON 文件加载类别到索引的映射

    读取指定目录下的 class_mapping.json 文件, 返回类别名称到索引以及索引到类别名称的双向映射

    Args:
        class_map_dir (Path): 存放 class_mapping.json 的目录路径

    Returns:
        tuple[dict[str, int], dict[int, str]]:
            第一个元素为 class_to_idx 字典, 将类别名称映射为整数索引
            第二个元素为 idx_to_class 字典, 将整数索引映射回类别名称
    """
    path = Path(class_map_dir) / "class_mapping.json"
    if not path.exists():
        logger.error(f"Class mapping file not found at: {path}")
        raise FileNotFoundError(
            f"Missing class_mapping.json at {path}. Run create_splits.py first"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            class_to_idx = mapping["class_to_idx"]
            idx_to_class = {
                int(i): label for i, label in mapping["idx_to_class"].items()
            }
    except Exception as e:
        logger.error(f"Error loading class mapping: {e}.")
        raise RuntimeError(
            f"Failed to load class_mapping: {e}. Run create_splits.py to regenerate."
        )
    return class_to_idx, idx_to_class

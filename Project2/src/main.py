"""项目主入口脚本

负责解析命令行参数, 加载配置与模型, 构建数据管线,
并根据不同模式执行训练、评估或分析等功能
"""

import time
import logging
import argparse
from typing import Any
from pathlib import Path

import torch

from utils.wandb_logger import WandbLogger
from utils.metrics import MetricsCalculator
from utils.config_parser import load_config
from utils.helpers import set_seed, get_device, AttrDict
from utils.logger import setup_logger, setup_pre_config_logger
from data_setup.create_splits import create_splits
from data_setup.transforms import get_test_transforms
from data_setup.dataset import create_dataloaders, get_class_mapping
from models.model_factory import get_model
from engine.trainer import Trainer
from engine.evaluator import Evaluator
from analysis.interactive_analyzer import ClassAnalyzer, SingleImageAnalyzer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        解析后的命令行参数命名空间
    """

    parser = argparse.ArgumentParser(description="Butterfly Classification Project")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "train",
            "eval",
            "analyze_class",
            "analyze_image",
            "create_splits",
        ],
        default="train",
        help="Operation mode: train, eval, analyze_class, analyze_image, or create_splits",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="./config/base_config.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--exp_config", type=str, required=False, help="Path to experiment config file"
    )
    parser.add_argument(
        "--overrides",
        action="append",
        type=str,
        help="Override config options using the command line"
        "Format: key=value. Nested keys can be specified using dot notation",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resume training",
    )
    parser.add_argument(
        "--best_model",
        type=str,
        default=None,
        help="Path to best model pth for evaluation",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default=None,
        help="Path to single image for analyze_image mode",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help="Target class name or directory for analyze_class mode",
    )
    parser.add_argument(
        "--no_wandb", action="store_true", default=None, help="Disable wandb logging"
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """根据命令行参数执行训练、评估或分析流程

    Args:
        args: 由 parse_args 返回的命令行参数命名空间

    Returns:
        None
    """
    # 在加载配置前初始化基础日志
    setup_pre_config_logger()
    config = load_config(args.base_config, args.exp_config, args.overrides)
    if args.resume is not None:
        config["training"]["resume"] = args.resume
    if args.best_model is not None:
        config["evaluation"]["pth_path"] = args.best_model
    if args.no_wandb is not None:
        config["logging"]["use_wandb"] = not args.no_wandb
    setup_logger(config)
    logger.info(f"User configuration: {config}")
    use_wandb = WandbLogger.init_wandb(config)
    config = AttrDict(config)

    # 设置随机种子
    set_seed(config.seed)

    if args.mode == "create_splits":
        create_splits(
            raw_dir=config.data.root,
            output_dir=config.data.splits_dir,
            split_ratios=config.data.split_ratios,
            seed=config.seed,
        )
        return

    # 获取设备
    device = get_device(config.device)

    # 分割数据
    if not _check_splited(config.data.splits_dir):
        logger.info("Missing split files, starting data splitting...")
        create_splits(
            raw_dir=config.data.root,
            output_dir=config.data.splits_dir,
            split_ratios=config.data.split_ratios,
            seed=config.seed,
        )

    # 加载数据集与类别映射
    train_loader, val_loader, test_loader = create_dataloaders(config)
    class_to_idx, idx_to_class = get_class_mapping(config.data.splits_dir)

    # 创建模型并移动到目标设备
    model = get_model(config.model).to(device)
    logger.info(f"Model '{config.model.name}': {model}")

    # 创建指标计算器
    metrics_calc = MetricsCalculator(
        num_classes=config.model.num_classes,
        top_k=config.evaluation.get("top_k", (1, 5)),
    )

    start = time.time()
    # 根据命令执行相应的功能
    if args.mode == "train":
        logger.info("========== Starting Training Phase ==========")
        trainer = Trainer(
            config=config,
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            metrics_calculator=metrics_calc,
            idx_to_class=idx_to_class,
            use_wandb=use_wandb,
        )
        trainer.fit()
        logger.info("========== Training Phase Finished ==========")
        logger.info("========== Starting Testing Phase  ==========")
        evaluator = Evaluator(
            config=config,
            device=device,
            test_loader=test_loader,
            model=model,
            metrics_calculator=metrics_calc,
            idx_to_class=idx_to_class,
        )
        evaluator.evaluate()
        logger.info("========== Testing Phase Finished ==========")
    elif args.mode == "eval":
        logger.info("========== Starting Testing Phase ==========")
        evaluator = Evaluator(
            config=config,
            device=device,
            test_loader=test_loader,
            model=model,
            metrics_calculator=metrics_calc,
            idx_to_class=idx_to_class,
            load_checkpoint=True,
            pth_path=config.evaluation.get("pth_path"),
        )
        evaluator.evaluate()
        logger.info("========== Testing Phase Finished ==========")
    elif args.mode == "analyze_class":
        if args.class_name is None:
            raise ValueError("--class_name is required for analyze_class mode")
        if config.evaluation.get("pth_path") is None:
            pth_path = Path(config.output_dir) / "best_model.pth"
            logger.warning(f"--best_model is not provided, defaulting to {pth_path}")
        else:
            pth_path = config.evaluation.pth_path
        _load_model_pth(pth_path, model, device)
        logger.info("========== Starting Class Analysis Phase ==========")
        class_analyzer = ClassAnalyzer(
            config=config,
            device=device,
            test_transform=get_test_transforms(config),
            model=model,
            idx_to_class=idx_to_class,
            class_to_idx=class_to_idx,
        )
        class_analyzer.analyze(args.class_name)
        logger.info("========== Class Analysis Phase Finished ==========")
    elif args.mode == "analyze_image":
        if args.img_path is None:
            raise ValueError("--img_path is required for analyze_image mode")
        if config.evaluation.get("pth_path") is None:
            pth_path = Path(config.output_dir) / "best_model.pth"
            logger.warning(f"--best_model is not provided, defaulting to {pth_path}")
        else:
            pth_path = config.evaluation.pth_path
        _load_model_pth(pth_path, model, device)
        logger.info("========== Starting Single Image Analysis Phase ==========")
        analyzer = SingleImageAnalyzer(
            config=config,
            device=device,
            test_transform=get_test_transforms(config),
            model=model,
            idx_to_class=idx_to_class,
        )
        analyzer.analyze(args.img_path)
        logger.info("========== Single Image Analysis Phase Finished ==========")
    logger.info(f"Total time taken: {time.time() - start:.2f} seconds")


def _check_splited(split_dir: Path) -> bool:
    """检查数据划分文件是否已经存在

    Args:
        split_dir: 划分文件所在目录

    Returns:
        若目录下包含必需的划分文件则返回 True, 否则返回 False
    """

    split_dir = Path(split_dir)
    required_files = ["train.csv", "val.csv", "test.csv", "class_mapping.json"]
    for file in required_files:
        if not (split_dir / file).exists():
            return False
    return True


def _load_model_pth(
    pth_path: Path, model: torch.nn.Module, device: torch.device
) -> None:
    """从给定路径加载仅包含权重的 pth 文件

    假设 pth 文件中存储的是纯权重 state_dict, 若传入包含模型等字段的 ckpt 将自动提取

    Args:
        pth_path: 模型权重文件路径
        model: 需要加载权重的模型实例
        device: 加载权重时使用的设备

    Returns:
        None
    """

    model_path = Path(pth_path)
    if not model_path.exists():
        logger.error(f"Best model pth not found at: {model_path}")
        raise FileNotFoundError(f"Best model pth not found at: {model_path}")

    try:
        # 加载权重对象, 并区分纯 state_dict 与 ckpt 字典
        loaded_obj: Any = torch.load(model_path, map_location=device)

        # 如果是 dict, 需要区分两种情形：
        # 1) 完整的 checkpoint 字典
        # 2) 纯粹的 state_dict
        if isinstance(loaded_obj, dict):
            ckpt_like_keys = {
                "model_state_dict",
                "optimizer_state_dict",
                "epoch",
                "state_dict",
                "best_metric",
            }
            if any(k in loaded_obj for k in ckpt_like_keys):
                # 视为 checkpoint-like dict，优先提取 model_state_dict 或 state_dict 字段
                if (
                    "model_state_dict" in loaded_obj
                    and loaded_obj["model_state_dict"] is not None
                ):
                    model_state = loaded_obj["model_state_dict"]
                elif (
                    "state_dict" in loaded_obj and loaded_obj["state_dict"] is not None
                ):
                    model_state = loaded_obj["state_dict"]
                else:
                    logger.error(
                        "Checkpoint appears to be a dict but contains no 'model_state_dict' or 'state_dict'"
                    )
                    raise KeyError(
                        "No 'model_state_dict' or 'state_dict' found in the checkpoint."
                    )
            else:
                # 通过启发式判断为纯 state_dict（参数名 -> tensor）
                model_state = loaded_obj
        else:
            # 若不是 dict，直接把对象当作 state_dict 使用
            model_state = loaded_obj

        model.load_state_dict(model_state)
        logger.info(f"Loaded best model weights from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model pth: {e}")
        raise


if __name__ == "__main__":
    run(parse_args())

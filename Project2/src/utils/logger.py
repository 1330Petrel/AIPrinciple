"""日志工具模块

提供训练过程中的统一日志初始化方法
支持在配置加载前后分别配置控制台与文件日志输出
"""

import sys
import logging
from typing import Any
from pathlib import Path


def setup_pre_config_logger() -> None:
    """初始化预配置阶段的临时日志记录器

    该日志器仅输出到控制台, 主要用于配置文件尚未加载之前的调试输出

    Args:
        None

    Returns:
        None
    """
    # 获取根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除任何已存在的处理器，防止重复输出
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 创建并配置日志格式化器
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 配置控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def setup_logger(config: dict[str, Any]) -> None:
    """根据配置初始化全局日志记录器

    使用训练配置中的输出目录创建日志文件, 并同时向控制台与文件输出统一格式的日志

    Args:
        config (dict[str, Any]): 全局配置字典, 至少需要包含 output_dir 字段

    Returns:
        None
    """
    # 1. 日志输出目录
    log_file_path = Path(config["output_dir"]) / "training.log"  # 日志文件路径

    # 2. 获取根记录器并设置基础配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 3. 清除任何已存在的处理器
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 4. 创建并配置日志格式化器
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 5. 配置控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 6. 配置文件处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info("Logger setup, logging to console and %s", log_file_path)

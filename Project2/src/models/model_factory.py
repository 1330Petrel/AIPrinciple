"""模型工厂与注册表模块

提供统一的模型创建入口、模型注册装饰器, 以及常用归一化层和激活函数的工厂方法
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from utils.helpers import AttrDict

logger = logging.getLogger(__name__)

# 创建模型注册表
_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def get_model(
    config: AttrDict,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """根据配置创建模型实例的主函数

    从模型注册表中查找 config.name 指定的模型构建函数, 按照配置中的参数
    实例化对应的模型, 并将模型移动到指定设备

    Args:
        config (AttrDict): 模型配置对象, 至少包含 name、num_classes 和可选 activation、kwargs
        device (Optional[torch.device]): 目标设备

    Returns:
        nn.Module: 已实例化的模型

    Raises:
        ValueError: 当在模型注册表中找不到指定名称的模型时抛出
    """
    model_name = config.name.lower()
    if model_name not in _MODEL_REGISTRY:
        available_models = ", ".join(_MODEL_REGISTRY.keys())
        logger.error(f"Model '{model_name}' not found in the registry")
        raise ValueError(
            f"Unknown model name: '{model_name}'. Available models are: [{available_models}]"
        )

    # 从注册表中获取模型构建函数
    builder_func = _MODEL_REGISTRY[model_name]
    logger.info(f"Creating model: '{model_name}' with parameters: {config}.")

    # 调用构建函数来创建模型实例
    model = builder_func(
        num_classes=config.num_classes,
        activation=_get_activation(config.get("activation", "none")),
        **config.get("kwargs", {}),
    )
    if device is not None:
        model.to(device)

    return model


def register_model(
    name: str,
) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """装饰器, 用于将模型构建函数注册到模型注册表中

    Args:
        name (str): 模型名称, 作为注册表中的键, 不区分大小写

    Returns:
        Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]: 接受构建函数并返回原函数的装饰器
    """

    def _wrap(build_fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        lname = name.lower()
        if lname in _MODEL_REGISTRY:
            raise KeyError(f"Model name '{name}' already registered.")
        _MODEL_REGISTRY[lname] = build_fn  # 将构建函数保存到全局注册表
        return build_fn

    return _wrap


def list_models() -> dict[str, Callable[..., nn.Module]]:
    """返回当前已注册模型的字典副本

    Returns:
        dict[str, Callable[..., nn.Module]]: 从模型名称到构建函数的映射字典副本
    """
    return dict(_MODEL_REGISTRY)


def get_normalization(
    norm_name: str, num_features: int, num_groups: int = 32
) -> nn.Module:
    """根据名称创建归一化层模块

    支持 batchnorm、layernorm、groupnorm 和 none 四种类型, 对于 GroupNorm
    会自动调整分组数以确保特征维度能够被整除

    Args:
        norm_name (str): 归一化层名称, 支持 "batchnorm"、"layernorm"、"groupnorm"、"none"
        num_features (int): 通道数或特征维度, 传递给对应归一化层
        num_groups (int): GroupNorm 使用的分组数

    Returns:
        nn.Module: 对应配置的归一化层实例

    Raises:
        ValueError: 当分组数非法或归一化名称不在支持列表时抛出
    """
    norm_name = norm_name.lower()
    if norm_name == "batchnorm":
        return nn.BatchNorm2d(num_features)
    elif norm_name == "layernorm":
        return nn.LayerNorm(num_features)
    elif norm_name == "groupnorm":
        try:
            num_groups = int(num_groups)
        except Exception:
            raise ValueError(f"Num_groups must be an integer, got {num_groups}")
        if num_groups < 1:
            num_groups = 1
        # 确保 num_features 可以被 num_groups 整除，否则向下寻找最大的因子
        if num_features % num_groups != 0:
            found = False
            for g in range(min(num_groups, num_features), 0, -1):
                if num_features % g == 0:
                    num_groups = g
                    found = True
                    break
            if not found:
                num_groups = 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    elif norm_name == "none":
        return nn.Identity()
    else:
        raise ValueError(
            f"Invalid normalization '{norm_name}'. Valid normalizations are: {'batchnorm', 'layernorm', 'groupnorm', 'none'}"
        )


def _get_activation(act_name: str) -> nn.Module:
    """根据名称返回对应的激活函数模块

    Args:
        act_name (str): 激活函数名称, 不区分大小写

    Returns:
        nn.Module: 对应的激活函数模块实例

    Raises:
        ValueError: 当名称不在支持列表中时抛出
    """
    act_dict = {
        "elu": nn.ELU(inplace=True),
        "selu": nn.SELU(inplace=True),
        "relu": nn.ReLU(inplace=True),
        "crelu": nn.CELU(inplace=True),
        "lrelu": nn.LeakyReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(inplace=True),
        "mish": nn.Mish(inplace=True),
        "identity": nn.Identity(),
        "none": nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(
            f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}"
        )

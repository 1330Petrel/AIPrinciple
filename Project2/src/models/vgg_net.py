"""基于 VGG 结构的分类模型定义模块

定义类 VGG 风格的卷积网络 ButterflyVGG, 支持两种配置规模, 并提供在模型工厂中注册的构建函数
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from models.model_factory import get_normalization, register_model

logger = logging.getLogger(__name__)

# VGG不同规模的配置
# 整数代表Conv2d的输出通道数, 'M' 代表MaxPool2d
_VGG_CONFIGS: dict[str, list[str | int]] = {
    # 约8个卷积层 + 3个全连接层
    "VGG11_small": [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"],
    # 约13个卷积层
    "VGG16_small": [
        32,
        32,
        "M",
        64,
        64,
        "M",
        128,
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        256,
        256,
        256,
        "M",
    ],
}


class ButterflyVGG(nn.Module):
    """类 VGG 模型实现, 支持不同规模的 VGG 网络结构

    通过配置列表动态堆叠卷积层和池化层, 再接自适应平均池化与多层全连接分类头
    """

    def __init__(
        self,
        num_classes: int,
        config: list[str | int],
        activation: nn.Module,
        normalization: str = "none",
        dropout: float | list[float] | tuple[float, float] = 0.0,
        avgpool_output_size: tuple[int, int] = (7, 7),
        classifier_hidden: int = 512,
        init_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化 ButterflyVGG 网络

        Args:
            num_classes (int): 分类任务的类别数量
            config (list[str | int]): VGG 网络配置列表, 整数为卷积层输出通道数, 'M' 为最大池化
            activation (nn.Module): 用于卷积块和分类器的激活函数模块
            normalization (str): 归一化方法名称, 如 'none'、'batchnorm'、'layernorm'、'groupnorm'
            dropout (float | list[float] | tuple[float, float]): 分类器中 Dropout 比例, 可为单个或两个值
            avgpool_output_size (tuple[int, int]): 自适应平均池化输出的空间尺寸
            classifier_hidden (int): 分类器隐藏层的神经元数量
            init_weights (bool): 是否对权重进行初始化
            **kwargs (Any): 其他可选参数, 如 groupnorm 的 num_groups 等

        Returns:
            None
        """
        num_groups = 32
        if normalization == "groupnorm":
            num_groups = kwargs.pop("num_groups", 32)
        if kwargs:
            logger.warning(
                "ButterflyVGG.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ButterflyVGG, self).__init__()
        # 特征提取部分: 根据配置动态堆叠 Conv-BN/Norm-Activation 和 MaxPool 序列
        self.features = self._make_layers(config, activation, normalization, num_groups)
        # 自适应平均池化: 将特征图统一到固定空间大小, 便于接全连接层
        self.avgpool = nn.AdaptiveAvgPool2d(avgpool_output_size)

        # 分类器: 两层全连接 + 激活 + Dropout, 最后输出到 num_classes 维度
        self.classifier = nn.Sequential(
            nn.Linear(
                int(config[-2]) * avgpool_output_size[0] * avgpool_output_size[1],
                classifier_hidden,
            ),  # config[-2] 为最后一层卷积的输出通道数
            activation,
            nn.Dropout(
                p=(dropout if isinstance(dropout, (int, float)) else dropout[0])
            ),
            nn.Linear(classifier_hidden, classifier_hidden),
            activation,
            nn.Dropout(
                p=(dropout if isinstance(dropout, (int, float)) else dropout[1])
            ),
            nn.Linear(classifier_hidden, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播逻辑

        Args:
            x (torch.Tensor): 输入图像张量, 形状通常为 (batch_size, 3, H, W)

        Returns:
            torch.Tensor: 分类 logits, 形状为 (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 He 初始化，适用于 ReLU
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layers(
        config: list[str | int],
        activation: torch.nn.Module,
        normalization: str = "none",
        num_groups: int = 32,
    ) -> nn.Sequential:
        """根据配置列表构建卷积与池化层序列

        Args:
            config (list[str | int]): VGG 配置列表, 整数为输出通道数, 'M' 为最大池化层
            activation (torch.nn.Module): 卷积块使用的激活函数模块
            normalization (str): 归一化类型名称, 传递给 get_normalization
            num_groups (int): GroupNorm 使用的分组数, 当 normalization 为 'groupnorm' 时生效

        Returns:
            nn.Sequential: 按配置顺序堆叠的卷积 + 归一化 + 激活 + 池化层序列
        """
        layers: list[nn.Module] = []
        in_channels = 3  # 初始输入通道为 3 (RGB 图像)

        for v in config:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 使用 2x2 最大池化进行下采样
            else:
                v = int(v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # 3x3 卷积保持空间尺寸不变
                norm_layer = get_normalization(normalization, v, num_groups)  # 根据配置选择 BatchNorm/LayerNorm/GroupNorm 或 Identity
                layers.extend([conv2d, norm_layer, activation])  # 一个卷积块: Conv -> Norm -> Activation
                in_channels = v

        return nn.Sequential(*layers)


# ===================================================================
# 模型工厂函数
# ===================================================================


@register_model("butterfly_vgg")
def build_vgg(
    num_classes: int, config_name: str, activation: nn.Module, **kwargs: Any
) -> ButterflyVGG:
    """构建并返回一个 ButterflyVGG 模型实例

    Args:
        num_classes (int): 数据集的类别数量
        config_name (str): 要使用的 VGG 配置名称, 如 "VGG11_small"、"VGG16_small"
        activation (nn.Module): 激活函数模块, 作为卷积和分类器的激活
        **kwargs (Any): 传递给 ButterflyVGG 构造函数的其他参数, 如 normalization、dropout 等

    Returns:
        ButterflyVGG: 已根据配置构建好的 VGG 模型实例

    Raises:
        ValueError: 当 config_name 不在预定义配置列表中时抛出
    """
    if config_name not in _VGG_CONFIGS:
        raise ValueError(
            f"Config name '{config_name}' not recognized. Available configs: {list(_VGG_CONFIGS.keys())}"
        )

    model = ButterflyVGG(
        num_classes=num_classes,
        config=_VGG_CONFIGS[config_name],
        activation=activation,
        **kwargs,
    )
    return model

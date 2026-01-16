"""自定义轻量级分类模型定义模块

包含基础版 ButterflyMyNet 与增强版 ButterflyMyNetPro 两个卷积网络, 并提供注册到模型工厂的构建函数
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from models.model_factory import register_model

logger = logging.getLogger(__name__)


class ButterflyMyNet(nn.Module):
    """用于分类任务的简单卷积神经网络

    由两层卷积 + 池化和一个全连接分类头组成, 适用于输入尺寸为 224x224 的图像
    """

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        """初始化 ButterflyMyNet 网络结构

        Args:
            num_classes (int): 分类任务的类别数量
            **kwargs (Any): 预留的额外关键字参数, 会被忽略并给出日志警告

        Returns:
            None
        """
        if kwargs:
            logger.warning(
                "ButterflyMyNet.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ButterflyMyNet, self).__init__()

        self.layers = nn.Sequential(
            # 特征提取部分: 两次卷积 + 最大池化, 最后接 ReLU 激活
            nn.Conv2d(3, 10, kernel_size=(5, 5)),  # 输入通道 3, 输出通道 10, 5x5 卷积核
            nn.MaxPool2d(2),  # 2x2 最大池化, 将空间尺寸减半
            nn.Conv2d(
                10, 20, kernel_size=(5, 5)
            ),  # 通道从 10 提升到 20, 继续提取局部特征
            nn.MaxPool2d(2),  # 再次池化, 进一步下采样
            nn.ReLU(),  # 非线性激活
            # 分类头: 全连接层将展平特征映射到隐含层, 再输出到类别空间
            nn.Flatten(),  # 展平特征图为一维向量
            nn.Linear(20 * 53 * 53, 128),  # 展平后的特征维度映射到 128 维隐藏层
            nn.ReLU(),  # 隐藏层激活
            nn.Linear(128, num_classes),  # 输出到类别数维度, 用于分类
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播逻辑

        Args:
            x (torch.Tensor): 输入图像张量, 形状 (batch_size, 3, H, W)

        Returns:
            torch.Tensor: 分类 logits, 形状为 (batch_size, num_classes)
        """
        x = self.layers(x)
        return x


class ButterflyMyNetPro(nn.Module):
    """用于分类任务的增强版卷积神经网络

    相比基础版增加了卷积层深度、批归一化和 Dropout, 提升表达能力与泛化性能
    """

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        """初始化 ButterflyMyNetPro 网络结构

        Args:
            num_classes (int): 分类任务的类别数量
            **kwargs (Any): 预留的额外关键字参数, 会被忽略并给出日志警告

        Returns:
            None
        """
        if kwargs:
            logger.warning(
                "ButterflyMyNet.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ButterflyMyNetPro, self).__init__()
        # 特征提取部分: 更深的卷积堆叠 + 批归一化 + 池化
        self.features = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=7, stride=2, padding=3
            ),  # 大卷积核和步长用于快速降低分辨率并提取粗粒度特征
            nn.BatchNorm2d(32),  # 对 32 个通道做批归一化
            nn.ReLU(inplace=True),  # ReLU 激活
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3x3 池化进一步下采样
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 通道数提升到 64
            nn.BatchNorm2d(64),  # 对 64 通道做批归一化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            ),  # 再次池化, 压缩特征图尺寸
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 3x3 卷积提取更细粒度特征
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                128, 128, kernel_size=3, padding=1
            ),  # 在 128 通道上再堆叠一层卷积增强表达能力
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            ),  # 最后一层池化得到紧凑特征图
        )
        # 分类头: Dropout + 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 随机丢弃部分节点
            nn.Linear(128 * 14 * 14, 512),  # 将展平特征映射到 512 维高维空间
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 隐藏层后使用 Dropout 增强正则化
            nn.Linear(512, num_classes),  # 输出到类别概率空间
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播逻辑

        Args:
            x (torch.Tensor): 输入图像张量, 形状 (batch_size, 3, H, W)

        Returns:
            torch.Tensor: 分类 logits, 形状为 (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ===================================================================
# 模型工厂函数
# ===================================================================


@register_model("butterfly_mynet")
def build_mynet(num_classes: int, **kwargs: Any) -> ButterflyMyNet:
    """构建并返回基础版 ButterflyMyNet 模型

    Args:
        num_classes (int): 分类任务的类别数量
        **kwargs (Any): 传递给 ButterflyMyNet 构造函数的其他关键字参数

    Returns:
        ButterflyMyNet: 已实例化的基础版模型
    """
    model = ButterflyMyNet(num_classes=num_classes, **kwargs)
    return model


@register_model("butterfly_mynet_pro")
def build_mynet_pro(num_classes: int, **kwargs: Any) -> ButterflyMyNetPro:
    """构建并返回增强版 ButterflyMyNetPro 模型

    Args:
        num_classes (int): 分类任务的类别数量
        **kwargs (Any): 传递给 ButterflyMyNetPro 构造函数的其他关键字参数

    Returns:
        ButterflyMyNetPro: 已实例化的增强版模型
    """
    model = ButterflyMyNetPro(num_classes=num_classes, **kwargs)
    return model

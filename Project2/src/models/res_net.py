"""基于 ResNet 结构的分类模型定义模块

实现简化版 ResNet 残差网络 ButterflyResNet, 并提供在模型工厂中注册的构建函数
"""

import logging
from typing import Type, Optional, Any

import torch
import torch.nn as nn

from models.model_factory import register_model

logger = logging.getLogger(__name__)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """构建一个 3x3 卷积层

    Args:
        in_planes (int): 输入通道数
        out_planes (int): 输出通道数
        stride (int): 步长

    Returns:
        nn.Conv2d: 带有适当 padding 的 3x3 卷积层
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """构建一个 1x1 卷积层

    常用于调整残差分支的通道数或空间尺寸

    Args:
        in_planes (int): 输入通道数
        out_planes (int): 输出通道数
        stride (int): 步长

    Returns:
        nn.Conv2d: 1x1 卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet 的基础残差块

    包含两个 3x3 卷积层及可选的下采样分支, 通过残差连接缓解深层网络的退化问题
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_batch_norm: bool = True,
    ) -> None:
        """初始化 BasicBlock

        Args:
            in_planes (int): 输入通道数
            planes (int): 该残差块主分支卷积的基础通道数
            stride (int): 第一层卷积的步长, 可用于下采样
            downsample (Optional[nn.Module]): 对残差分支进行下采样或通道匹配的模块
            use_batch_norm (bool): 是否在卷积后使用 BatchNorm

        Returns:
            None
        """
        super(BasicBlock, self).__init__()
        # 主分支: 两个 3x3 卷积层, 中间接 BatchNorm 和 ReLU
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if use_batch_norm else nn.Identity()
        # 残差分支: 可选下采样模块, 用于匹配维度
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播逻辑

        Args:
            x (torch.Tensor): 输入特征图, 形状为 (batch_size, C, H, W)

        Returns:
            torch.Tensor: 经过残差块后的特征图, 形状与主分支输出一致
        """
        identity = x

        # 主分支: Conv3x3 -> BN/Identity -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层卷积后再接 BN/Identity
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样或通道匹配, 则对 identity 进行变换
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接: 主分支输出与 identity 相加, 再经过 ReLU
        out += identity
        out = self.relu(out)

        return out


class ButterflyResNet(nn.Module):
    """简化 ResNet 模型

    由一个初始卷积 + 池化层、四个残差阶段以及一个全连接分类头组成
    """

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: list[int],
        num_classes: int = 50,
        base_planes: int = 32,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """初始化 ButterflyResNet 网络

        Args:
            block (Type[BasicBlock]): 残差块类型
            layers (list[int]): 每个残差阶段中残差块的数量列表
            num_classes (int): 分类任务的类别数量
            base_planes (int): 第一阶段卷积的基础通道数
            use_batch_norm (bool): 是否在卷积之后使用 BatchNorm
            dropout_rate (float): 分类器前的 Dropout 比例
            **kwargs (Any): 其他可选参数, 会被忽略并给出日志警告

        Returns:
            None
        """
        if kwargs:
            logger.warning(
                "ButterflyResNet.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ButterflyResNet, self).__init__()

        self.in_planes = base_planes
        self.use_batch_norm = use_batch_norm

        # 初始卷积层: 采用 7x7 大卷积核和步长 2, 快速降低空间分辨率
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )  # 初始最大池化进一步下采样

        # 四个残差阶段: 通道数依次为 base_planes, 2x, 4x, 8x, 后三层通过 stride=2 下采样
        self.layer1 = self._make_layer(block, base_planes, layers[0])
        self.layer2 = self._make_layer(block, base_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_planes * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 将每个通道的特征图汇聚为单个值

        # 分类器: Dropout + 全连接层, 使用平均池化后的通道数作为输入
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(base_planes * 8 * block.expansion, num_classes),
        )

        self._initialize_weights()

    def _make_layer(
        self, block: Type[BasicBlock], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """构建一个残差阶段

        一个阶段由若干个残差块堆叠, 第一个块可以选择下采样, 后续块保持输入输出尺寸一致

        Args:
            block (Type[BasicBlock]): 残差块类型
            planes (int): 该阶段目标通道数
            blocks (int): 残差块数量
            stride (int): 第一个残差块使用的步长, 可用于下采样

        Returns:
            nn.Sequential: 由多个残差块组成的顺序容器
        """
        downsample = None
        # 当需要下采样或通道数变化时，创建 downsample 层
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                (
                    nn.BatchNorm2d(planes * block.expansion)
                    if self.use_batch_norm
                    else nn.Identity()
                ),
            )

        layers = []
        # 第一个 block 负责下采样和通道对齐
        layers.append(
            block(self.in_planes, planes, stride, downsample, self.use_batch_norm)
        )
        self.in_planes = planes * block.expansion
        # 后续的 blocks 不再需要 downsample, 只做特征变换
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, planes, use_batch_norm=self.use_batch_norm)
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """初始化网络权重

        对卷积层使用 He 初始化, 对 BatchNorm 的权重和偏置进行常数初始化

        Returns:
            None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播逻辑

        Args:
            x (torch.Tensor): 输入图像张量, 形状为 (batch_size, 3, H, W)

        Returns:
            torch.Tensor: 分类 logits, 形状为 (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


# ===================================================================
# 模型工厂函数
# ===================================================================
RESNET_CONFIGS = {
    # 类似 ResNet-18 的结构，但通道数减半
    "ResNet18_small": {"block": BasicBlock, "layers": [2, 2, 2, 2], "base_planes": 32},
    # 一个更浅的网络，类似 ResNet-10
    "ResNet10_tiny": {"block": BasicBlock, "layers": [1, 1, 1, 1], "base_planes": 32},
}


@register_model("butterfly_resnet")
def butterfly_resnet(
    num_classes: int, config_name: str, **kwargs: Any
) -> ButterflyResNet:
    """构建并返回一个 ButterflyResNet 模型实例

    Args:
        num_classes (int): 数据集的类别数量
        config_name (str): 要使用的 ResNet 配置名称, 如 "ResNet18_small"、"ResNet10_tiny"
        **kwargs (Any): 传递给 ButterflyResNet 构造函数的其他参数, 如 dropout_rate 等

    Returns:
        ButterflyResNet: 已实例化的 ResNet 模型

    Raises:
        ValueError: 当 config_name 不在预定义配置字典中时抛出
    """
    if config_name not in RESNET_CONFIGS:
        raise ValueError(
            f"Model name '{config_name}' not recognized. Available configs: {list(RESNET_CONFIGS.keys())}"
        )

    config = RESNET_CONFIGS[config_name]

    # 从 kwargs 中提取可选参数或使用默认值
    dropout_rate = kwargs.get("dropout_rate", 0.5)
    use_batch_norm = kwargs.get("use_batch_norm", True)

    model = ButterflyResNet(
        block=config["block"],
        layers=config["layers"],
        base_planes=config["base_planes"],
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        **kwargs,
    )
    return model

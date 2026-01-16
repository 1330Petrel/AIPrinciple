"""Grad-CAM 可视化工具模块

提供 GradCAM 类及辅助函数, 用于对卷积神经网络的决策区域进行可视化
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """实现 Grad-CAM 可视化算法的核心类

    通过在目标卷积层上注册前向与反向钩子, 捕获特征图与梯度, 从而生成类别相关的热力图
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """初始化 GradCAM 实例

        Args:
            model (nn.Module): 已训练的模型
            target_layer (nn.Module): 用于生成 Grad-CAM 的目标卷积层

        Returns:
            None
        """
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """在目标层上注册前向与反向钩子

        前向钩子用于捕获特征图, 反向钩子用于捕获梯度, 供后续计算 Grad-CAM 权重使用

        Returns:
            None
        """

        def forward_hook(
            module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            # 不在此处 detach, 保持计算图, 以便在 backward 时计算梯度
            self.activations = output

        def backward_hook(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
        ) -> None:
            # 捕获从目标层输出回传的梯度
            self.gradients = grad_out[0]

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def _remove_hooks(self) -> None:
        """移除已注册的前向与反向钩子

        Returns:
            None
        """
        self.forward_handle.remove()
        self.backward_handle.remove()

    @staticmethod
    def _find_target_layer(model: nn.Module) -> Optional[nn.Module]:
        """自动查找模型中的最后一层卷积层

        通常将最后一个卷积层作为 Grad-CAM 的目标层, 以获得较高语义级别的特征图

        Args:
            model (nn.Module): 需要进行可视化的模型

        Returns:
            Optional[nn.Module]: 找到的目标卷积层, 如未找到则返回 None
        """
        last_conv_layer = None
        for _, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
        return last_conv_layer

    def __call__(
        self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> tuple[np.ndarray, int]:
        """生成指定类别的 Grad-CAM 热力图

        Args:
            input_tensor (torch.Tensor): 预处理后的输入图像张量, 形状为 (B, C, H, W)
            class_idx (Optional[int]): 需要可视化的类别索引, 为 None 时使用模型预测的类别

        Returns:
            tuple[np.ndarray, int]:
                第一个元素为归一化到 [0, 1] 的热力图数组
                第二个元素为用于可视化的类别索引
        """
        # 将模型设置为评估模式
        self.model.eval()

        # 1. 前向传播得到模型输出
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        # 2. 清零梯度并对指定类别的分数执行反向传播
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=True)

        # 确保梯度与特征图已被成功捕获
        if self.gradients is None or self.activations is None:
            self._remove_hooks()
            raise RuntimeError(
                "Failed to capture gradients or activations. Check hook registration."
            )

        # 3. 通过对梯度做全局平均池化计算通道权重 (alpha)
        weights = torch.mean(
            self.gradients, dim=[2, 3], keepdim=True
        )  # 形状 (B, C, 1, 1)

        # 4. 计算加权特征图: (B, C, H, W) * (B, C, 1, 1) 并在通道维上求和得到 CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # 5. 通过 ReLU 只保留对目标类别有正向贡献的特征
        cam = F.relu(cam)

        # 去掉 batch 与通道维度, 得到单张图的二维热力图
        cam = cam.squeeze(0).squeeze(0).detach().cpu().numpy()

        # 6. 将 CAM 插值到输入图像大小并做归一化
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

        # 清空缓存的特征与梯度, 并移除钩子
        self.activations = None
        self.gradients = None
        self._remove_hooks()
        return cam, class_idx


def show_cam_on_image(
    image: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """将 Grad-CAM 热力图叠加到原始图像上

    Args:
        image (np.ndarray): 原始图像数组, 形状为 (H, W, C), 像素值范围为 [0, 255]
        mask (np.ndarray): 热力图数组, 形状为 (H, W), 数值范围为 [0, 1]
        colormap (int): OpenCV 使用的颜色映射表类型

    Returns:
        np.ndarray: 叠加热力图后的可视化图像, 像素值范围为 [0, 255]
    """
    # 将单通道热力图转换为彩色图
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式以便与原图对齐

    # 将热力图按一定权重叠加到原始图像上
    superimposed_img = np.float32(heatmap) * 0.4 + np.float32(image) * 0.6
    superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))

    return superimposed_img


def generate_grad_cam_visualization(
    model: nn.Module,
    image_path: Path,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    input_size: tuple[int, int],
    target_class_idx: Optional[int] = None,
) -> tuple[Image.Image, int]:
    """为给定模型和图像生成 Grad-CAM 可视化结果

    Args:
        model (nn.Module): 已训练好的模型
        image_path (Path): 输入图像文件路径
        transform (Callable[[Image.Image], torch.Tensor]): 预处理与归一化用的变换函数
        device (torch.device): 运行模型的设备
        input_size (int): 模型期望的输入图像尺寸, 用于可视化时的缩放
        target_class_idx (Optional[int]): 要可视化的目标类别索引, 为 None 时使用模型预测类别

    Returns:
        tuple[Image.Image, int]:
            第一个元素为叠加 Grad-CAM 热力图的可视化图像
            第二个元素为模型预测或指定的类别索引
    """
    # 1. 加载和预处理图像
    original_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # 2. 自动寻找目标卷积层并初始化 GradCAM
    target_layer = GradCAM._find_target_layer(model)
    if target_layer is None:
        raise ValueError("Could not find a Conv2d layer in the model.")
    grad_cam = GradCAM(model, target_layer)

    # 3. 生成热力图遮罩和预测类别索引
    with torch.enable_grad():
        mask, predicted_idx = grad_cam(input_tensor, class_idx=target_class_idx)

    # 4. 创建叠加后的图像, 使用原始图像以获得较好视觉效果
    vis_image = np.array(original_image.resize(input_size))  # 调整为模型输入尺寸
    grad_cam_image = show_cam_on_image(vis_image, mask)

    return Image.fromarray(grad_cam_image), predicted_idx

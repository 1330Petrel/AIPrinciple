"""交互式模型预测与 Grad-CAM 可视化分析模块

提供针对单张图像和指定类别的分析器类, 结合模型预测结果与 Grad-CAM 热力图进行可视化诊断
"""

import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Callable, Optional


import torch
import torch.nn.functional as F

from utils.helpers import AttrDict
from analysis.grad_cam import generate_grad_cam_visualization

logger = logging.getLogger(__name__)


class SingleImageAnalyzer:
    """分析单个图像并可视化模型预测与 Grad-CAM 结果

    负责对一张输入图像执行前向预测、Grad-CAM 可视化以及 Top-k 置信度分析, 并通过 Matplotlib 界面展示
    """

    def __init__(
        self,
        config: AttrDict,
        device: torch.device,
        test_transform: Callable[[Image.Image], torch.Tensor],
        model: torch.nn.Module,
        idx_to_class: dict[int, str],
    ) -> None:
        """初始化单图像分析器

        Args:
            config (AttrDict): 全局配置对象
            device (torch.device): 推理使用的设备
            test_transform (Callable[[Image.Image], torch.Tensor]): 测试阶段的图像预处理
            model (torch.nn.Module): 已训练好的分类模型
            idx_to_class (dict[int, str]): 从类别索引到类别名称的映射字典

        Returns:
            None
        """
        self.config = config
        self.device = device
        self.model = model
        self.test_transform = test_transform
        self.idx_to_class = idx_to_class
        # 用于保存交互式分析结果图像的输出目录
        self.output_dir = Path(config.output_dir) / "interactive_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()

    def analyze(self, image_path: Path, true_class_name: str = "N/A") -> None:
        """对单个图像执行预测与 Grad-CAM 分析并显示结果

        Args:
            image_path (Path): 要分析的图像路径
            true_class_name (str): 图像的真实类别名称, 默认为 "N/A" 表示未知

        Returns:
            None
        """
        image_path = Path(image_path)
        logger.info(f"Analyzing image: {image_path}")

        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        target_class_idx: Optional[int] = None
        if true_class_name != "N/A":
            target_class_idx = None
            for idx, class_name in self.idx_to_class.items():
                if class_name == true_class_name:
                    target_class_idx = idx
                    break
            if target_class_idx is None:
                logger.error(
                    f"True class '{true_class_name}' not found in class mapping."
                )
                raise ValueError(
                    f"True class '{true_class_name}' not found in class mapping."
                )
        # 1. 生成 Grad-CAM 可视化
        try:
            grad_cam_image, predicted_idx = generate_grad_cam_visualization(
                model=self.model,
                image_path=image_path,
                transform=self.test_transform,
                device=self.device,
                input_size=self.config.data.input_size,
                target_class_idx=target_class_idx,
            )
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM for {image_path}: {e}")
            raise

        # 2. 获取置信度分数
        original_image = Image.open(image_path).convert("RGB")
        input_tensor = self.test_transform(original_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        # 3. 获取 Top-5 预测结果
        top5_indices = probabilities.argsort()[-5:][::-1]
        top5_probs = probabilities[top5_indices]
        top5_labels = [self.idx_to_class[i] for i in top5_indices]

        # 4. 计算指标
        predicted_class_name = self.idx_to_class[predicted_idx]
        confidence = probabilities[predicted_idx]
        distinctiveness = (
            top5_probs[0] - top5_probs[1] if len(top5_probs) > 1 else top5_probs[0]
        )

        # 5. 创建并显示 Matplotlib 图形界面
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Single Image Analysis: {image_path.name}", fontsize=16)

        # 子图 1: 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title(f"Original Image\nTrue Class: {true_class_name}")
        axes[0].axis("off")

        # 子图 2: Grad-CAM 热力图
        axes[1].imshow(grad_cam_image)
        axes[1].set_title(f"Grad-CAM Heatmap\nPredicted Class: {predicted_class_name}")
        axes[1].axis("off")

        # 子图 3: 置信度条形图
        bars = axes[2].barh(
            np.arange(len(top5_labels)), top5_probs, align="center", color="skyblue"
        )
        axes[2].set_yticks(np.arange(len(top5_labels)))
        axes[2].set_yticklabels(top5_labels)
        axes[2].invert_yaxis()  # 标签从上到下显示
        axes[2].set_xlabel("Confidence")
        axes[2].set_title("Top-5 Predictions")
        axes[2].set_xlim(0, 1)
        # 在条形图上显示数值
        for bar in bars:
            width = bar.get_width()
            axes[2].text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{width:.2%}",
                va="center",
            )

        # 添加整体分析文本
        analysis_text = (
            f"Prediction: {predicted_class_name}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Distinctiveness (Top-1 - Top-2): {distinctiveness:.2%}"
        )
        fig.text(
            0.5,
            0.02,
            analysis_text,
            ha="center",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
        )

        plt.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(self.output_dir / f"{image_path.stem}_analysis.png")
        plt.show()


class ClassAnalyzer:
    """分析指定类别或目录下所有测试图像的性能

    可对某一类别或指定目录中的所有图像进行批量预测, 统计整体准确率、输出 CSV 报告,
    并从中挑选若干样本调用 SingleImageAnalyzer 做更详细的可视化分析
    """

    def __init__(
        self,
        config: AttrDict,
        device: torch.device,
        test_transform: Callable[[Image.Image], torch.Tensor],
        model: torch.nn.Module,
        idx_to_class: dict[int, str],
        class_to_idx: Optional[dict[str, int]] = None,
    ) -> None:
        """初始化类别分析器

        Args:
            config (AttrDict): 全局配置对象
            device (torch.device): 推理使用的设备
            test_transform (Callable[[Image.Image], torch.Tensor]): 测试阶段使用的预处理
            model (torch.nn.Module): 已训练好的分类模型
            idx_to_class (dict[int, str]): 从类别索引到类别名称的映射
            class_to_idx (Optional[dict[str, int]]): 从类别名称到索引的映射, 若为 None 则从 idx_to_class 反推

        Returns:
            None
        """
        self.config = config
        self.device = device
        self.test_transform = test_transform
        self.model = model
        self.idx_to_class = idx_to_class
        self.class_to_idx = (
            class_to_idx
            if class_to_idx is not None
            else {v: k for k, v in idx_to_class.items()}
        )
        # 输出目录与单图像分析器保持一致
        self.output_dir = Path(config.output_dir) / "interactive_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.single_analyzer = SingleImageAnalyzer(
            config, device, test_transform, model, idx_to_class
        )

        # 加载测试集数据
        test_csv_path = Path(config.data.splits_dir) / "test.csv"
        if not test_csv_path.exists():
            logger.error(f"Failed to find test CSV file at: {test_csv_path}")
            raise FileNotFoundError(f"Test CSV file not found at: {test_csv_path}")
        self.test_df = pd.read_csv(test_csv_path)

    def analyze(self, class_target: str, num_samples_to_show: int = 3) -> None:
        """对一个类别或目录下的所有图像执行批量分析

        Args:
            class_target (str): 目标类别名称, 或者包含该类别图像的文件夹路径
            num_samples_to_show (int): 需要展示详细单图像分析窗口的样本数量

        Returns:
            None
        """
        logger.info(f"Starting analysis for class/dir: {class_target}")
        target_class_name = Path(class_target).name

        # 1. 确定目标图像文件列表
        if Path(class_target).is_dir():
            image_paths = list(Path(class_target).glob("*.[jp][pn]g"))
        elif target_class_name in self.class_to_idx:
            class_files = self.test_df[self.test_df["label"] == target_class_name][
                "filepath"
            ].tolist()
            image_paths = [Path(self.config.data.root) / f for f in class_files]
        else:
            logger.error(
                f"Class or directory '{class_target}' not found in test dataset"
            )
            raise

        if not image_paths:
            logger.warning(f"No images found in '{class_target}'")
            return

        target_class_idx = self.class_to_idx[target_class_name]

        # 2. 遍历所有图像进行预测
        results: list[dict[str, object]] = []  # 存储路径、预测索引和置信度等字段
        correct_count = 0
        with torch.no_grad():
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                input_tensor = self.test_transform(image).unsqueeze(0).to(self.device)
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1).squeeze()

                pred_idx = torch.argmax(probabilities).item()
                confidence_for_target = probabilities[target_class_idx].item()

                if pred_idx == target_class_idx:
                    correct_count += 1

                results.append(
                    {
                        "path": path,
                        "confidence": confidence_for_target,
                        "predicted_idx": pred_idx,
                    }
                )

        # 3. 打印总体统计数据
        success_rate = (correct_count / len(results)) * 100
        logger.info(f"===== Analysis Report for Class '{target_class_name}' =====")
        logger.info(f"Total Images: {len(results)}")
        logger.info(f"Correct Predictions: {correct_count}")
        logger.info(f"Success Rate: {success_rate:.2f}%")
        logger.info("=================================================")

        # 保存结果到 CSV 文件
        try:
            df = pd.DataFrame(
                [
                    {
                        "path": str(r["path"]),
                        "predicted_idx": int(r["predicted_idx"]),  # type: ignore[arg-type]
                        "confidence": float(r["confidence"]),  # type: ignore[arg-type]
                    }
                    for r in results
                ]
            )
            csv_out = self.output_dir / f"class_{target_class_name}_analysis.csv"
            df.to_csv(csv_out, index=False)
            logger.info(f"Saved class {target_class_name} analysis CSV to: {csv_out}")
        except Exception as e:
            logger.error(f"Failed to save class {target_class_name} analysis CSV: {e}")
            raise

        # 4. 选择并展示几个样本的详细分析
        if not results:
            return

        # 按置信度从高到低排序
        results.sort(key=lambda x: x["confidence"], reverse=True)  # type: ignore[call-overload]

        # 选择要展示的样本：置信度最高、最低和随机的
        indices_to_show = set()
        if len(results) > 0:
            indices_to_show.add(0)  # 最高置信度
        if len(results) > 1:
            indices_to_show.add(len(results) - 1)  # 最低置信度
        while len(indices_to_show) < min(num_samples_to_show, len(results)):
            indices_to_show.add(random.randint(0, len(results) - 1))

        for i in sorted(list(indices_to_show)):
            sample = results[i]
            self.single_analyzer.analyze(
                sample["path"], true_class_name=target_class_name
            )

"""模型评估与分析模块

封装测试集评估流程, 并集成混淆矩阵、Grad-CAM 与交互式类别分析等工具
"""

import json
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Any
from sklearn.metrics import classification_report

import torch

from utils.helpers import AttrDict
from utils.metrics import MetricsCalculator
from data_setup.transforms import get_test_transforms
from analysis.plot import plot_confusion_matrix
from analysis.interactive_analyzer import ClassAnalyzer
from analysis.grad_cam import generate_grad_cam_visualization

logger = logging.getLogger(__name__)


class Evaluator:
    """模型评估与分析器

    负责加载最佳模型权重, 在测试集上计算整体指标, 生成分类报告,
    混淆矩阵和 Grad-CAM 可视化, 并调用交互式类别分析工具
    """

    def __init__(
        self,
        config: AttrDict,
        device: torch.device,
        test_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        metrics_calculator: MetricsCalculator,
        idx_to_class: dict[int, str],
        load_checkpoint: bool = False,
        pth_path: Optional[Path] = None,
    ) -> None:
        """初始化评估器

        Args:
            config: 实验配置对象, 包含数据与输出路径等信息
            device: 执行推理使用的设备
            test_loader: 测试集数据加载器
            model: 已训练好的模型或待评估模型
            metrics_calculator: 评估指标计算器实例
            idx_to_class: 类别索引到类别名称的映射
            load_checkpoint: 是否在评估前从检查点加载最佳权重
            pth_path: 可选的模型权重路径, 若为 None 则使用默认 best_model.pth

        Returns:
            None
        """
        self.config = config
        self.device = device
        self.test_loader = test_loader
        self.model = model
        self.metrics_calculator = metrics_calculator
        self.idx_to_class = idx_to_class
        self.output_dir = Path(config.output_dir)
        self.class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        # 构建测试时的数据增强/预处理流水线
        self.test_transforms = get_test_transforms(config)

        # 根据需要加载检查点并切换到评估模式
        if load_checkpoint:
            self._load_model_pth(pth_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model_pth(self, pth_path: Optional[Path]) -> None:
        """加载模型检查点权重

        Args:
            pth_path: 模型权重路径, 若为 None 则使用输出目录下的 best_model.pth

        Returns:
            None
        """
        if pth_path is None:
            model_path = Path(self.output_dir) / "best_model.pth"
            logger.warning(
                "No pth_path provided, defaulting to best_model.pth in output_dir"
            )
        else:
            model_path = Path(pth_path)

        if not model_path.exists():
            logger.error(f"Best model pth not found at: {model_path}")
            raise FileNotFoundError(f"Best model pth not found at: {model_path}")

        try:
            # 加载权重对象, 并区分纯 state_dict 与 ckpt 字典
            loaded_obj: Any = torch.load(model_path, map_location=self.device)

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
                        "state_dict" in loaded_obj
                        and loaded_obj["state_dict"] is not None
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

            self.model.load_state_dict(model_state)
            logger.info(f"Loaded best model weights from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model pth: {e}")
            raise

    def _save_results(self, metrics: dict[str, float], report: str) -> None:
        """保存测试指标与文本分类报告

        Args:
            metrics: 各类评估指标组成的字典
            report: 由 sklearn 生成的分类报告字符串

        Returns:
            None
        """
        metrics_path = self.output_dir / "test_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved test metrics to {metrics_path}")

        report_path = self.output_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved classification report to {report_path}")

    def evaluate(self) -> None:
        """执行完整的测试集评估与分析流程

        遍历测试集、计算整体指标、保存分类报告, 并调用分析工具生成可视化结果

        Returns:
            None
        """
        logger.info("Starting evaluation on the test set...")

        all_targets: list[int] = []
        all_preds: list[int] = []
        all_paths: list[str] = []

        with torch.inference_mode():
            for inputs, targets, paths in tqdm(self.test_loader, desc="[Test]"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                # 更新整体评估指标
                self.metrics_calculator.update(outputs, targets)

                # 保存预测结果和标签以供后续分析工具使用
                top1_preds = outputs.argmax(dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(top1_preds.cpu().numpy())
                all_paths.extend(paths)

        # 1. 计算并输出最终指标
        final_metrics = self.metrics_calculator.compute()
        logger.info("========== Test Results ==========")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("==================================")

        # 2. 生成并保存分类报告
        report = classification_report(
            all_targets, all_preds, target_names=self.class_names, zero_division=0
        )
        self._save_results(final_metrics, str(report))

        # 3. 调用分析工具生成混淆矩阵、Grad-CAM 等结果
        self.run_analysis(targets=all_targets, preds=all_preds, paths=all_paths)

        logger.info("Evaluation finished.")

    def run_analysis(
        self,
        targets: list[int] | np.ndarray,
        preds: list[int] | np.ndarray,
        paths: list[str],
    ) -> None:
        """调用多种分析工具生成可视化与统计结果

        Args:
            targets: 测试集中所有样本的真实标签
            preds: 对应样本的预测标签
            paths: 每个样本对应的图像路径列表

        Returns:
            None
        """
        targets_arr = np.array(targets)
        preds_arr = np.array(preds)

        # 1. 生成并保存混淆矩阵图像
        cm_path = self.output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            y_true=targets_arr,
            y_pred=preds_arr,
            class_names=self.class_names,
            output_path=cm_path,
        )

        # 2. 基于正确/错误样本生成 Grad-CAM 可视化示例
        self._generate_grad_cam_examples(targets_arr, preds_arr, paths)

        # 3. 随机选取一个类别, 启动交互式类别分析器
        try:
            target_class_name = random.choice(self.class_names)
            class_analyzer = ClassAnalyzer(
                self.config,
                self.device,
                self.test_transforms,
                self.model,
                self.idx_to_class,
            )
            class_analyzer.analyze(target_class_name)
        except Exception as e:
            logger.error(f"Error during interactive class analysis: {e}")
            raise

    def _generate_grad_cam_examples(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        paths: list[str],
        num_examples: int = 3,
    ) -> None:
        """选择部分样本图像并生成 Grad-CAM 可视化

        Args:
            targets: 所有测试样本的真实标签数组
            preds: 所有测试样本的预测标签数组
            paths: 每个样本对应的图像路径列表
            num_examples: 从正确和错误样本中各选取的最大可视化数量

        Returns:
            None
        """
        grad_cam_dir = self.output_dir / "grad_cam_test_examples"
        grad_cam_dir.mkdir(exist_ok=True)

        correct_indices = np.where(preds == targets)[0]
        incorrect_indices = np.where(preds != targets)[0]
        sample_correct = np.random.choice(
            correct_indices, min(num_examples, len(correct_indices)), replace=False
        )
        sample_incorrect = np.random.choice(
            incorrect_indices, min(num_examples, len(incorrect_indices)), replace=False
        )
        indices_to_visualize = np.concatenate([sample_correct, sample_incorrect])

        if len(indices_to_visualize) == 0:
            logger.warning("No examples available for Grad-CAM visualization.")
            return

        for idx in indices_to_visualize:
            idx_int = int(idx)
            image_path = Path(paths[idx_int])
            true_label = self.idx_to_class[int(targets[idx_int])]
            pred_label = self.idx_to_class[int(preds[idx_int])]

            try:
                grad_cam_image, _ = generate_grad_cam_visualization(
                    model=self.model,
                    image_path=image_path,
                    transform=self.test_transforms,
                    device=self.device,
                    input_size=self.config.data.input_size,
                    target_class_idx=int(preds[idx_int]),
                )

                # 保存可视化结果, 文件名中包含预测/真实标签与样本文件名
                status = "correct" if true_label == pred_label else "incorrect"
                filename = f"{status}__pred_{pred_label}__true_{true_label}__{image_path.stem}.png"
                output_path = grad_cam_dir / filename
                grad_cam_image.save(output_path)

            except Exception as e:
                logger.error(f"Error generating Grad-CAM for {image_path}: {e}")
                raise

        logger.info(
            f"Saved {len(indices_to_visualize)} Grad-CAM examples to {grad_cam_dir}"
        )

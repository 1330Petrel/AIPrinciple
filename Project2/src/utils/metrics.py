"""分类任务评估指标工具模块

提供多分类任务中 Top-k 准确率以及基于 scikit-learn 的精确率、召回率和 F1 值计算
"""

import torch
from sklearn.metrics import precision_score, recall_score, f1_score


class MetricsCalculator:
    """多分类任务综合指标计算器

    支持 Top-k 准确率以及基于 scikit-learn 的精确率、召回率和 F1 指标
    通过累积多个 batch 的预测与标签统一计算最终结果
    """

    def __init__(self, num_classes: int, top_k: tuple[int, int] = (1, 5)) -> None:
        """初始化指标计算器

        Args:
            num_classes (int): 数据集的类别总数
            top_k (tuple[int, int]): 要计算 Top-k 准确率的 k 值范围 (min_k, max_k)

        Returns:
            None
        """
        self.num_classes = num_classes
        self.top_k = range(min(top_k), max(top_k) + 1)  # 生成闭区间 [min_k, max_k]

        # 检查 top_k 中的最大值是否合理
        if max(self.top_k) > self.num_classes:
            raise ValueError(
                f"Max value in top_k ({max(self.top_k)}) cannot be greater than num_classes ({self.num_classes})."
            )

        self.reset()

    def reset(self) -> None:
        """重置内部状态

        清空累积的预测与标签, 并重置 Top-k 统计量

        Returns:
            None
        """
        # 用于 scikit-learn 计算的数据
        self._all_preds: list[torch.Tensor] = []
        self._all_targets: list[torch.Tensor] = []

        # 用于 Top-k 计算的数据
        self._total_samples: int = 0
        self._top_k_correct: dict[int, int] = {k: 0 for k in self.top_k}

    def update(self, preds_logits: torch.Tensor, targets: torch.Tensor) -> None:
        """用一个批次的数据更新指标

        Args:
            preds_logits (torch.Tensor): 模型输出的原始 logits (未经过 softmax), 形状为 (batch_size, num_classes)
            targets (torch.Tensor): 对应的真实标签, 形状为 (batch_size,)

        Returns:
            None
        """
        # --- 1. 更新 Top-k 准确率 ---
        # 确保数据在 CPU 上
        preds_logits = preds_logits.detach().cpu()
        targets = targets.detach().cpu()

        batch_size = targets.size(0)
        self._total_samples += batch_size  # 累积样本数量

        # 获取 Top-k 预测结果
        _, top_k_preds = preds_logits.topk(
            max(self.top_k), dim=1, largest=True, sorted=True
        )

        # 扩展 targets 以匹配 top_k_preds 的形状, 方便比较
        targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)

        # 比较并计算每个 k 值的正确数量
        correct_matrix = top_k_preds == targets_expanded

        for k in self.top_k:
            # 取前 k 个预测中是否存在正确答案
            self._top_k_correct[k] += correct_matrix[:, :k].any(dim=1).sum().item()

        # --- 2. 存储预测和目标以供 scikit-learn 使用 ---
        # 我们只需要 Top-1 的预测结果
        top1_preds = preds_logits.argmax(dim=1)
        self._all_preds.append(top1_preds)
        self._all_targets.append(targets)

    def compute(self) -> dict[str, float]:
        """计算并返回所有累积数据的最终指标

        Returns:
            dict[str, float]: 指标名称到数值的映射, 包括各类 Top-k 准确率和多种 average 形式的精确率、召回率与 F1 分数
        """
        if self._total_samples == 0:
            return {}

        results: dict[str, float] = {}

        # --- 1. 计算 Top-k 准确率 ---
        for k in self.top_k:
            accuracy = self._top_k_correct[k] / self._total_samples
            results[f"accuracy_top{k}"] = round(accuracy, 4)

        # --- 2. 使用 scikit-learn 计算其他指标 ---
        # 将列表中的张量合并为一个大张量，然后转为 numpy 数组
        all_preds_np = torch.cat(self._all_preds).numpy()
        all_targets_np = torch.cat(self._all_targets).numpy()

        # 计算 precision / recall / f1 的三种 average：macro, micro, weighted
        averages = ["macro", "micro", "weighted"]
        for avg in averages:
            # zero_division=0 表示当一个类的所有样本都被错误分类时，该类的指标记为0，而不是报错
            prec = precision_score(
                all_targets_np, all_preds_np, average=avg, zero_division=0
            )
            rec = recall_score(
                all_targets_np, all_preds_np, average=avg, zero_division=0
            )
            f1 = f1_score(all_targets_np, all_preds_np, average=avg, zero_division=0)

            results[f"precision_{avg}"] = round(prec, 4)
            results[f"recall_{avg}"] = round(rec, 4)
            results[f"f1_score_{avg}"] = round(f1, 4)

        return results

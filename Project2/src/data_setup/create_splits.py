"""数据集划分与统计报告脚本

扫描原始图像目录, 按比例划分为训练/验证/测试集, 生成对应 CSV、类别映射和划分统计报告
"""

import json
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_splits(
    raw_dir: str,
    output_dir: str,
    split_ratios: list[float] | tuple[float, float, float],
    seed: int,
) -> None:
    """执行数据扫描、划分、保存和报告生成

    根据给定的原始图像目录和划分比例, 先收集所有图像及其类别标签, 再按比例划分为
    训练/验证/测试三部分, 同时生成对应的 CSV 和类别映射文件, 最后输出划分统计报告

    Args:
        raw_dir (str): 原始图像数据目录路径
        output_dir (str): 保存 train.csv、val.csv、test.csv 以及报告和映射文件的目录
        split_ratios (List[float] | Tuple[float, float, float]): 训练/验证/测试集划分比例
        seed (int): 随机种子

    Returns:
        None
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 收集所有图像路径和标签
    try:
        all_data = _find_image_files(raw_path)
    except FileNotFoundError as e:
        logger.error(e)
        raise e

    filepaths = [item[0] for item in all_data]
    labels = [item[1] for item in all_data]

    # 2. 从标签生成 class_to_idx 映射
    class_num = _make_class_mapping(labels, output_path)
    logger.info(f"Found {len(filepaths)} images in {class_num} classes")
    logger.info(f"Class mapping saved to '{output_path / 'class_mapping.json'}'")

    # 3. 第一次划分：从总数据中分出测试集
    train_ratio, val_ratio, test_ratio = split_ratios
    logger.info(
        f"Splitting data: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test"
    )

    # 存放训练集和验证集的混合体
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        filepaths,
        labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,  # 确保测试集中的类别分布与总体一致
    )

    # 3. 第二次划分：从剩余数据中分出验证集
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size_adjusted,
        random_state=seed,
        stratify=y_train_val,  # 确保验证集中的类别分布与剩余数据一致
    )

    # 4. 创建 Pandas DataFrames
    train_df = pd.DataFrame({"filepath": X_train, "label": y_train})
    val_df = pd.DataFrame({"filepath": X_val, "label": y_val})
    test_df = pd.DataFrame({"filepath": X_test, "label": y_test})

    # 5. 保存为 CSV 文件
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    # 6. 生成划分报告
    logger.info(f"Splits saved successfully to '{output_path}'")
    _generate_split_report(train_df, val_df, test_df, output_path)


def _find_image_files(raw_dir: Path) -> list[tuple[str, str]]:
    """扫描原始数据目录, 返回包含图像相对路径和类别标签的列表

    Args:
        raw_dir (Path): 原始图像数据的根目录, 每个子目录名视为一个类别标签

    Returns:
        List[Tuple[str, str]]: 每个元素为 (相对路径字符串, 类别名称) 的元组
    """
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    image_data: list[tuple[str, str]] = []  # 存储 (相对路径, 类别名)
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    logger.info(f"Scanning for images in '{raw_dir}'")

    # 遍历所有子目录, 子目录名即为类别标签
    for class_dir in raw_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for file_path in class_dir.iterdir():
                # 检查文件是否为支持的图像格式
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in supported_extensions
                ):
                    # 存储相对于 raw_dir 的路径
                    relative_path = file_path.relative_to(raw_dir)
                    image_data.append((str(relative_path), class_name))

    if not image_data:
        raise FileNotFoundError(
            f"No images found under {raw_dir}. Ensure you have subfolders named by class containing images."
        )

    return image_data


def _make_class_mapping(
    labels: list[str], output_dir: Path
) -> int:
    """根据标签列表生成类别映射并保存为 JSON 文件

    Args:
        labels (list[str]): 所有样本的类别标签列表
        output_dir (Path): 映射文件的输出目录, 将在其中创建 class_mapping.json

    Returns:
        int: 不同类别的数量
    """
    classes_sorted = sorted(list(set(labels)))
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    idx_to_class = {str(i): c for c, i in class_to_idx.items()}

    mapping = {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}
    with open(output_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    return len(class_to_idx)


def _generate_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
):
    """生成数据划分的统计报告和类别分布可视化

    会检查不同划分之间是否存在重复文件路径, 统计各类别在三种划分中的样本数,
    将结果写入文本文件并生成类别分布的可视化图像

    Args:
        train_df (pd.DataFrame): 训练集文件路径和标签的 DataFrame
        val_df (pd.DataFrame): 验证集文件路径和标签的 DataFrame
        test_df (pd.DataFrame): 测试集文件路径和标签的 DataFrame
        output_dir (Path): 报告和图像文件的输出目录

    Returns:
        None
    """
    report_path = output_dir / "split_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=======================================\n")
        f.write("      Data Split Statistical Report    \n")
        f.write("=======================================\n\n")

        # 1. 检查数据泄露 (重复文件)
        train_files = set(train_df["filepath"])
        val_files = set(val_df["filepath"])
        test_files = set(test_df["filepath"])

        if (
            not train_files.isdisjoint(val_files)
            or not train_files.isdisjoint(test_files)
            or not val_files.isdisjoint(test_files)
        ):
            leakage_msg = "WARNING: Duplicate filepaths found between splits! This indicates data leakage"
            logger.warning(leakage_msg)
            f.write(f"{leakage_msg}\n")

        # 2. 类别分布统计
        f.write("--- Class Distribution Analysis ---\n")
        train_counts = train_df["label"].value_counts().sort_index()
        val_counts = val_df["label"].value_counts().sort_index()
        test_counts = test_df["label"].value_counts().sort_index()

        dist_df = (
            pd.DataFrame(
                {"Train": train_counts, "Validation": val_counts, "Test": test_counts}
            )
            .fillna(0)
            .astype(int)
        )
        dist_df["Total"] = dist_df.sum(axis=1)

        f.write("Sample counts per class across splits:\n")
        f.write(dist_df.to_string())
        f.write("\n\n")

        f.write("--- Summary ---\n")
        f.write(f"Total Unique Classes: {len(dist_df)}\n")
        f.write(f"Training Set Size:    {len(train_df)}\n")
        f.write(f"Validation Set Size:  {len(val_df)}\n")
        f.write(f"Test Set Size:        {len(test_df)}\n")

    logger.info(f"Data split report generated at '{report_path}'")

    # 可视化类别分布
    try:
        plt.figure(figsize=(12, max(8, len(dist_df) // 4)))
        dist_df[["Train", "Validation", "Test"]].plot(
            kind="barh", stacked=True, ax=plt.gca()
        )
        plt.title("Class Distribution Across Splits")
        plt.xlabel("Number of Samples")
        plt.ylabel("Class Label")
        plt.tight_layout()
        plt.savefig(output_dir / "class_distribution.png")
        plt.close()
        logger.info(
            f"Class distribution plot saved to '{output_dir / 'class_distribution.png'}'"
        )
    except ImportError:
        logger.warning(
            "matplotlib not installed. Skipping plot generation. Install with 'pip install matplotlib'"
        )


if __name__ == "__main__":
    import argparse
    import yaml
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.helpers import set_seed

    # 1. 配置一个基础的logger
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
    )

    # 2. 定位并加载 base_config.yaml
    config_defaults = {
        "data": {
            "root": "./data/raw",
            "splits_dir": "./data/splits",
            "split_ratios": [0.7, 0.15, 0.15],
        },
        "seed": 42,
    }
    try:
        # 假设 base_config.yaml 位于 ../../configs/base_config.yaml
        base_config_path = (
            Path(__file__).resolve().parent.parent.parent / "config/base_config.yaml"
        )
        with open(base_config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            config_defaults["data"]["root"] = yaml_config["data"]["root"]
            config_defaults["data"]["splits_dir"] = yaml_config["data"]["splits_dir"]
            config_defaults["data"]["split_ratios"] = yaml_config["data"][
                "split_ratios"
            ]
            config_defaults["seed"] = yaml_config["seed"]
            logger.info(f"Loaded default values from '{base_config_path}'")
    except FileNotFoundError:
        logger.warning(
            f"base_config.yaml not found at: {base_config_path}. Using script defaults"
        )
    except KeyError as e:
        logger.error(
            f"Missing expected key in base_config.yaml: {e}. Using script defaults"
        )
    except Exception as e:
        logger.error(f"Error loading base_config.yaml: {e}. Using script defaults")

    # 3. 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="Create train, validation, and test CSV splits from image folder dataset"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=config_defaults["data"]["root"],
        help="Path to the directory containing raw images, organized in class folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config_defaults["data"]["splits_dir"],
        help="Directory where the output CSV files (train.csv, val.csv, test.csv) will be saved",
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=config_defaults["data"]["split_ratios"],
        help="Train, validation, and test split ratios (e.g., 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config_defaults["seed"],
        help="Random seed for shuffling and splitting to ensure reproducibility",
    )

    args = parser.parse_args()

    # --- 3. 验证参数并执行 ---
    ratios = args.split_ratios
    if not (
        len(ratios) == 3
        and all(isinstance(r, float) for r in ratios)
        and all(r >= 0.0 for r in ratios)
        and all(r <= 1.0 for r in ratios)
        and abs(sum(ratios) - 1.0) < 1e-6
    ):
        logger.error(
            "Split ratios must be three positive numbers that sum to 1. "
            f"Received: {ratios} (Sum: {sum(ratios)})"
        )
        exit(1)

    set_seed(args.seed)
    create_splits(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        split_ratios=args.split_ratios,
        seed=args.seed,
    )

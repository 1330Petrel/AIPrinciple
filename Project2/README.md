# 项目二

ButterflyNet 图像分类模型的设计与优化

## 项目简介

ButterflyNet 是基于 PyTorch 的蝴蝶图像分类解决方案，实现了多种经典和自定义卷积神经网络架构，并提供完整的训练、评估、可视化和分析工具链。

蝴蝶种类识别数据集包括 50 类来自世界各地的蝴蝶种类，共计 4479 张图像，从如下链接下载：[https://cloud.tsinghua.edu.cn/d/570a8737f6f740f1bb89/](https://cloud.tsinghua.edu.cn/d/570a8737f6f740f1bb89/)

## 快速开始

以下流程在 **Windows 11 24H2** + **CUDA 12.6** 环境下运行无误

### 安装步骤

1. **创建并激活虚拟环境**

    ```bash
    conda create -n butterfly python=3.14 -y
    conda activate butterfly
    ```

    > 若需 GPU，请确认 `torch` 与 CUDA 版本兼容

2. **安装依赖**

    ```bash
    pip install -r requirements.txt
    ```

### 准备数据集

1. 从[清华云盘](https://cloud.tsinghua.edu.cn/d/570a8737f6f740f1bb89/)下载蝴蝶数据集

2. 解压到 `data/raw/` 目录，确保结构为：

    ```plaintext
    data/raw/
    ├── adonis/
    │   ├── 001.jpg
    │   └── ...
    ├── american_snoot/
    │   └── ...
    └── ...（共 50 个类别文件夹）
    ```

3. 划分数据集

    首次运行前需生成划分/标签映射：

    ```bash
    python ./src/main.py --mode create_splits --base_config ./config/base_config.yaml
    ```

    这将在 `data/splits/` 生成：

    - `train.csv` - 训练集
    - `val.csv` - 验证集
    - `test.csv` - 测试集
    - `class_mapping.json` - 类别映射
    - `split_report.txt` - 划分报告
    - 类别分布可视化图

    如已有既定划分，只需确保文件名一致即可

### 训练模型

`main.py` 通过 `--mode train` 进入训练流程：

- 读取/合并配置（基础 + 实验 + CLI override），并生成 `merged_config.yaml`
- 根据配置检查/创建数据划分及 DataLoader
- 通过模型工厂创建指定架构
- 构建优化器、调度器、回调（Top-K checkpoint、EarlyStopping、Warmup LR 等）
- 训练完成后，在同一进程中跑测试集评估并输出 Grad-CAM、分类报告等

#### 示例命令

1. 使用默认 VGG11 配置

    ```bash
    python ./src/main.py --mode train --base_config ./config/base_config.yaml
    ```

2. 使用实验配置

    ```bash
    python ./src/main.py --mode train \
      --base_config ./config/base_config.yaml \
      --exp_config ./config/experiments/improved.yaml
    ```

3. CLI 参数覆盖

    ```bash
    python ./src/main.py --mode train \
      --base_config ./config/base_config.yaml \
      --overrides model.name=butterfly_resnet \
      --overrides model.kwargs.config_name=ResNet18 \
      --overrides optimizer.lr=0.001
    ```

> WandB：默认关闭，如需启用请在配置或命令行设置 `logging.use_wandb=true`，并提前执行 `wandb login`

#### 训练产物

以 `output/butterfly_vgg/<run_name>/` 为例：

- `best_model.pth`: 最佳模型权重（可直接用于部署/推理）
- `checkpoints/`: Top-K 权重与 `best.ckpt`
- `training_history.json/png`: loss/acc 曲线
- `test_metrics.json`、`classification_report.txt`、`confusion_matrix.png`
- `grad_cam_test_examples/`: 正负样本的 Grad-CAM
- `interactive_analysis/`: 类别与单图分析结果
- `merged_config.yaml`: 最终配置，包含所有覆盖

### 模型评估

```bash
python ./src/main.py --mode eval \
  --base_config ./config/base_config.yaml \
  --best_model ./models/vgg11_best.pth
```

评估会生成：`output/.../test_metrics.json`、`classification_report.txt`、`confusion_matrix.png`、Grad-CAM 示例等

> [!IMPORTANT]
>
> 1. 若不提供 `--best_model`，会使用配置中的 `evaluation.pth_path` 或默认查找 `output/.../best_model.pth`
> 2. 评估不同模型时，需要在配置文件中修改 `model.name` 和 `model.kwargs.config_name` 以匹配模型架构

### 分析工具

- **按类别分析**：

    ```bash
    python ./src/main.py --mode analyze_class \
      --base_config ./config/base_config.yaml \
      --best_model ./models/vgg11_best.pth \
      --class_name "monarch"
    ```

    自动统计该类的成功率、导出 CSV，并在 `output/.../interactive_analysis/` 下保存抽样的可视化

- **单张图片分析**：

    ```bash
    python ./src/main.py --mode analyze_image \
      --base_config ./config/base_config.yaml \
      --best_model ./models/vgg11_best.pth \
      --img_path "./data/raw/monarch/000.jpg"
    ```

    会弹出 Matplotlib 窗口展示原图、Grad-CAM、Top-5 置信度条形图，以及差异分析；同时保存 PNG 到 `interactive_analysis/`

## 命令行参数

| 参数                                                                | 默认值                      | 描述                                                                        |
| ------------------------------------------------------------------- | --------------------------- | --------------------------------------------------------------------------- |
| `--mode {train, eval, analyze_class, analyze_image, create_splits}` | `train`                     | 选择运行阶段：训练+测试、单独评估、类别分析、单图分析或仅生成数据划分       |
| `--base_config PATH`                                                | `./config/base_config.yaml` | 指定基础配置文件，通常包含通用默认值                                        |
| `--exp_config PATH`                                                 | `None`                      | 额外的实验配置，会在基础配置之上进行递归覆盖                                |
| `--overrides key=value`                                             | `None`                      | 可多次传入，使用点号访问嵌套字段（如 `optimizer.lr=1e-4`）以覆盖 YAML 配置  |
| `--resume PATH`                                                     | `None`                      | 恢复训练所用的 checkpoint，等价于直接在配置中设置 `training.resume_from`    |
| `--best_model PATH`                                                 | `None`                      | 测试或分析时显式指定模型权重，未提供则回退到配置中的 `evaluation.ckpt_path` |
| `--img_path PATH`                                                   | `None`                      | `analyze_image` 模式必填，指向待分析的单张图片                              |
| `--class_name NAME/Dir`                                             | `None`                      | `analyze_class` 模式必填，可传类别名或包含该类图片的目录                    |
| `--no_wandb`                                                        | `False`                     | 传入该标志后强制关闭 wandb 记录，即便配置中启用了 `logging.use_wandb`       |

## 配置说明

- `config/base_config.yaml` 给出全局默认值，常见字段：
  - `data`: 数据根目录 / CSV 目录 / 划分比例 / 输入大小 / 归一化 / workers、batch size
  - `augmentations`: Resize、RandomResizedCrop、翻转、旋转、ColorJitter 等，按需启用
  - `model`: 通过 `name` 选择模型（`butterfly_vgg`、`butterfly_resnet`、`butterfly_mynet/pro`），并在 `kwargs` 指定结构化参数
  - `optimizer`、`scheduler`: 支持 `adam/adamw/sgd` 和 `step/cosine/exp/reduce_on_plateau/warmup_cosine/none`
  - `training`: epochs、梯度裁剪、resume 路径、早停设置
  - `reporting`: 监控指标、Top-K checkpoint 数量
  - `logging`: 是否启用 wandb、日志打印频率

- `config/experiments/*` 覆盖对应字段，可堆叠 CLI `--overrides key=value` 继续微调

每次运行都会生成 `output/<model_name>/<run_name>/merged_config.yaml` 方便复现

## 项目结构

```plaintext
Project2/
├── ⚙️ config/                       # 配置文件目录
│   ├── base_config.yaml            # 基础配置（默认参数）
│   └── experiments/                # 实验配置（增量覆盖）
│       └── improved.yaml
│
├── 📊 data/                         # 数据目录
│   ├── raw/                        # 原始图片
│   │   ├── adonis/
│   │   └── ...（50个类别）
│   └── splits/                     # 数据集划分
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── class_mapping.json
│       └── split_report.txt
│
├── 💾 models/                       # 预训练模型权重
│   ├── vgg11_best.pth              # VGG11 最佳模型
│   ├── resnet18_best.pth           # ResNet18 最佳模型
│   ├── resnet10_best.pth           # ResNet10 最佳模型
│   ├── mynetpro_best.pth           # MyNetPro 最佳模型
│   └── mynet_best.pth              # MyNet 基线模型
│
├── 📈 output/                       # 训练输出目录
│   ├── butterfly_vgg/
│   │   └── best_run/
│   │       ├── best_model.pth        # 最佳模型权重
│   │       ├── training_history.json # 训练历史
│   │       ├── training_history.png  # 训练曲线图
│   │       ├── test_metrics.json     # 测试集指标
│   │       ├── classification_report.txt # 分类报告
│   │       ├── confusion_matrix.png  # 混淆矩阵图
│   │       ├── merged_config.yaml    # 完整配置快照
│   │       ├── checkpoints/          # Top-K 检查点
│   │       ├── grad_cam_test_examples/  # Grad-CAM 示例
│   │       └── interactive_analysis/    # 分析结果
│   ├── butterfly_resnet/
│   ├── butterfly_mynet/
│   └── butterfly_mynet_pro/
│
└── 💻 src/                          # 源代码目录
    ├── main.py                     # 主入口脚本
    │
    ├── 🔬 analysis/                 # 分析与可视化模块
    │   ├── __init__.py
    │   ├── grad_cam.py             # Grad-CAM 实现
    │   ├── interactive_analyzer.py # 交互式分析器
    │   └── plot.py                 # 绘图工具
    │
    ├── 📦 data_setup/               # 数据处理模块
    │   ├── __init__.py
    │   ├── create_splits.py        # 数据集划分
    │   ├── dataset.py              # 数据集定义
    │   └── transforms.py           # 数据增强
    │
    ├── 🚂 engine/                   # 训练引擎模块
    │   ├── __init__.py
    │   ├── trainer.py              # 训练器
    │   ├── evaluator.py            # 评估器
    │   └── callbacks.py            # 回调函数
    │
    ├── 🧠 models/                   # 模型定义模块
    │   ├── __init__.py
    │   ├── model_factory.py        # 模型工厂
    │   ├── vgg_net.py              # VGG 系列
    │   ├── res_net.py              # ResNet 系列
    │   └── my_net.py               # 自定义模型
    │
    └── 🛠️ utils/                    # 工具函数模块
        ├── __init__.py
        ├── config_parser.py        # 配置解析
        ├── helpers.py              # 辅助函数
        ├── logger.py               # 日志系统
        ├── metrics.py              # 指标计算
        └── wandb_logger.py         # WandB 集成
```

### `src` 代码概览

| 文件                                   | 说明                                                                               |
| -------------------------------------- | ---------------------------------------------------------------------------------- |
| `src/main.py`                          | 命令行入口，解析配置、创建数据集/模型，并根据 `--mode` 调度训练、评估与分析流程    |
| `src/analysis/grad_cam.py`             | 实现 Grad-CAM 核心逻辑与可视化工具，为评估和交互式分析提供热力图                   |
| `src/analysis/interactive_analyzer.py` | 定义 `ClassAnalyzer` 与 `SingleImageAnalyzer`，支持按类别统计与单图可视化          |
| `src/analysis/plot.py`                 | 输出训练曲线和混淆矩阵的 Matplotlib/Seaborn 绘图工具                               |
| `src/data_setup/create_splits.py`      | 扫描 `data/raw`，生成 train/val/test CSV、类别映射及划分报告                       |
| `src/data_setup/dataset.py`            | 自定义 `ButterflyDataset` 及 `create_dataloaders`，封装 DataLoader 构建逻辑        |
| `src/data_setup/transforms.py`         | 按配置构造 train/val/test 的 torchvision 数据增强/预处理流水线                     |
| `src/engine/callbacks.py`              | 包含 `ModelCheckpoint`、`EarlyStopping`、`LRSchedulerStep` 等训练期回调            |
| `src/engine/trainer.py`                | 训练循环实现，负责优化器/调度器创建、日志记录、历史曲线与 checkpoint 管理          |
| `src/engine/evaluator.py`              | 测试集评估与分析入口，生成指标、报告、Grad-CAM 示例和交互分析                      |
| `src/models/model_factory.py`          | 模型注册/创建工厂，并提供激活函数与归一化层的辅助函数                              |
| `src/models/my_net.py`                 | 简单 CNN (`ButterflyMyNet`/`MyNetPro`) 的定义与注册，作为轻量基线                  |
| `src/models/vgg_net.py`                | 可配置的 VGG 变体实现，支持多种深度、归一化和分类器 Dropout                        |
| `src/models/res_net.py`                | 定义精简版 ResNet 结构及配置字典，支持 BatchNorm/Dropout 可调                      |
| `src/utils/config_parser.py`           | 负责合并 base/experiment 配置、应用 overrides、校验字段并落盘 `merged_config.yaml` |
| `src/utils/helpers.py`                 | 常用工具：随机种子、设备选择、AttrDict、反归一化、CheckpointManager 等             |
| `src/utils/logger.py`                  | 配置控制台/文件日志记录器以及运行前的临时 logger                                   |
| `src/utils/metrics.py`                 | Top-K 准确率与多种 Precision/Recall/F1 统计的累积计算器                            |
| `src/utils/wandb_logger.py`            | Weights & Biases 封装，统一处理 run 初始化、指标/图像/混淆矩阵上传                 |

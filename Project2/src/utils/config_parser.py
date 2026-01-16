"""配置解析与验证工具模块

提供从 YAML 配置文件加载、合并、命令行覆盖以及结构化校验等功能
用于训练与评估流程中的统一配置管理
"""

import json
import yaml
import logging
import datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 允许的配置选项
_ALLOWED_MODELS = [
    "butterfly_mynet",
    "butterfly_mynet_pro",
    "butterfly_vgg",
    "butterfly_resnet",
]
_ALLOWED_ACTIVATIONS = [
    "elu",
    "selu",
    "relu",
    "crelu",
    "lrelu",
    "tanh",
    "sigmoid",
    "softplus",
    "gelu",
    "swish",
    "mish",
    "identity",
    "none",
]
_ALLOWED_NORMALIZATIONS = ["batchnorm", "layernorm", "groupnorm", "none"]
_ALLOWED_CONFIG_NAMES = [
    "VGG16_small",
    "VGG11_small",
    "ResNet18_small",
    "ResNet10_tiny",
]
_ALLOWED_OPTIMIZERS = ["adam", "adamw", "sgd"]
_ALLOWED_SCHEDULERS = [
    "step",
    "cosine",
    "exp",
    "reduce_on_plateau",
    "none",
]


def load_config(
    base_config_path: str | Path,
    exp_config_path: Optional[str | Path] = None,
    overrides: Optional[list[str]] = None,
) -> dict[str, Any]:
    """加载、合并并验证配置

    Args:
        base_config_path (str | Path): 基础配置文件路径，通常位于 `config/base_config.yaml`
        exp_config_path (Optional[str | Path]): 实验配置文件路径，用于在基础配置上进行覆写
        overrides (Optional[list[str]]): 命令行覆盖项列表，格式为 `key.sub_key=value`

    Returns:
        dict[str, Any]: 处理后可直接用于训练/评估流程的完整配置字典
    """
    # 1. 加载基础配置
    base_path = Path(base_config_path)  # 规范化基础配置路径
    if not base_path.exists():  # 基础配置必须存在
        raise FileNotFoundError(f"Base config file not found at: {base_path}")
    logger.info(f"Loading base config from: {base_path}")
    with open(base_path, "r", encoding="utf-8") as f:  # 读取基础配置 YAML
        config: dict[str, Any] = yaml.safe_load(f)

    # 2. 加载实验配置
    if exp_config_path is not None:
        exp_path = Path(exp_config_path)  # 实验配置路径
        if not exp_path.exists():  # 实验配置文件可选，但一旦指定必须存在
            raise FileNotFoundError(f"Experiment config file not found at: {exp_path}")
        logger.info(f"Loading experiment config from: {exp_path}")
        with open(exp_path, "r", encoding="utf-8") as f:  # 读取实验配置 YAML
            experiment_config: dict[str, Any] = yaml.safe_load(f)

        # 3. 合并配置
        config = _recursive_update(config, experiment_config)

    # 4. 应用命令行覆盖
    if overrides:
        for o in overrides:
            if "=" not in o:
                raise ValueError(f"Override '{o}' must be in the form key.subkey=VALUE")
            key, val = o.split("=", 1)
            key = key.strip()  # 去除 key 两侧空白
            val = val.strip()  # 去除 value 两侧空白
            _set_by_path(config, key, val)  # 将覆盖写入配置
        logger.info(f"Applied {len(overrides)} command-line overrides.")

    # 4. 验证配置
    _ConfigValidator().validate(config)

    # 5. 补充实验名称到配置中
    run_name = config.get("run_name", None)  # 若未显式指定 run_name 则自动生成
    if run_name is None:
        now = datetime.datetime.now().astimezone()  # 使用本地有时区时间
        run_name = f"{now.strftime('%Y%m%dT%H%M%S')}"  # 时间戳作为运行名称
        config["run_name"] = run_name

    # 6. 创建输出目录
    output_dir = Path(config["output_dir"]) / config["model"]["name"] / run_name
    config["output_dir"] = str(output_dir)  # 将输出目录转换为字符串存回配置
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    # 7. 保存合并后的配置
    with open(output_dir / "merged_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)  # 将合并后的配置写入输出目录

    return config


def _recursive_update(
    base_dict: dict[str, Any], new_dict: dict[str, Any]
) -> dict[str, Any]:
    """递归更新字典

    将 new_dict 中的键值递归合并到 base_dict 中
    当同一键对应的值均为映射类型时进行递归更新，否则直接覆盖

    Args:
        base_dict (dict[str, Any]): 被更新的基础字典
        new_dict (dict[str, Any]): 提供新值的字典

    Returns:
        dict[str, Any]: 更新后的 ``base_dict`` 引用
    """
    for key, value in new_dict.items():
        if (
            isinstance(value, Mapping)
            and key in base_dict
            and isinstance(base_dict[key], Mapping)
        ):
            base_dict[key] = _recursive_update(
                base_dict[key], value
            )  # 递归合并嵌套字典
        else:
            base_dict[key] = value  # 非映射类型直接覆盖
    return base_dict


def _set_by_path(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """根据点分路径设置配置项

    给定形如 "a.b.c" 的点分键路径，在 cfg 中创建或更新嵌套字典

    Args:
        cfg (dict[str, Any]): 需要写入的配置字典
        dotted_key (str): 点分路径键，例如 "model.kwargs.dropout"
        value (Any): 原始字符串或已解析值，将写入对应路径

    Returns:
        None
    """
    keys = dotted_key.split(".")  # 拆分为层级 key 列表
    cur: dict[str, Any] = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(
            cur[k], dict
        ):  # 若不存在或不是 dict 则创建新字典
            cur[k] = {}
        cur = cur[k]  # 进入下一层级
    cur[keys[-1]] = _coerce_value(value)  # 叶子节点写入并尝试类型转换


def _coerce_value(value: Any) -> Any:
    """将命令行字符串尝试转换为合适类型

    支持自动解析为 None、布尔值、整型、浮点型或 JSON 序列结构
    若转换失败则返回原始值

    Args:
        value (Any): 待转换的原始值，通常为命令行传入的字符串

    Returns:
        Any: 转换后的 Python 对象
    """
    if not isinstance(value, str):  # 非字符串直接返回
        return value
    low = value.lower()
    if low in ("null", "none", "nil", "na"):
        return None
    if low in ("true", "yes", "y"):
        return True
    if low in ("false", "no", "n"):
        return False
    # int
    try:
        iv = int(value)
        return iv
    except Exception:
        pass
    # float
    try:
        fv = float(value)
        return fv
    except Exception:
        pass
    # json parse
    try:
        parsed = json.loads(value)
        return parsed
    except Exception:
        pass

    return value


class _ConfigValidator:
    """配置验证器

    负责对完整配置字典进行结构和取值范围检查，调用 validate 方法
    """

    class _ConfigValidationError(Exception):
        """配置验证错误异常

        用于收集并一次性抛出所有配置错误信息

        Args:
            errors (list[str]): 验证过程中收集的错误消息列表
        """

        def __init__(self, errors: list[str]) -> None:
            message = (
                "Configuration validation failed with the following errors:\n"
                + "\n".join(f"- {e}" for e in errors)
            )
            super().__init__(message)

    def __init__(self) -> None:
        """初始化验证器实例

        创建空错误列表用于存储后续校验错误

        Returns:
            None
        """
        self.errors: list[str] = []

    # -------------------------
    # Public API
    # -------------------------
    def validate(self, cfg: dict[str, Any]) -> bool:
        """验证完整配置字典

        会依次检查顶层结构以及各子模块配置的存在性、类型和值域

        Args:
            cfg (dict[str, Any]): 待验证的完整配置字典

        Returns:
            bool: 若所有检查通过则返回 True，否则抛出异常
        """
        self.errors = []

        if not isinstance(cfg, dict):
            raise self._ConfigValidationError(
                [f"Config must be a dict, got {type(cfg).__name__}"]
            )

        # 顶级字段和类型检查
        self._check_section_type(cfg, "project_name", str, required=True)
        self._check_section_type(cfg, "run_name", str, required=False)
        self._check_section_type(cfg, "seed", int, required=True)
        self._check_section_type(cfg, "device", str, required=False)
        self._check_section_type(cfg, "output_dir", str, required=True)
        self._check_section_type(cfg, "data", dict, required=True)
        self._check_section_type(cfg, "augmentations", dict, required=False)
        self._check_section_type(cfg, "model", dict, required=True)
        self._check_section_type(cfg, "optimizer", dict, required=True)
        self._check_section_type(cfg, "scheduler", dict, required=False)
        self._check_section_type(cfg, "training", dict, required=True)
        self._check_section_type(cfg, "reporting", dict, required=True)
        self._check_section_type(cfg, "logging", dict, required=True)

        # 验证嵌套部分
        self._validate_data(cfg.get("data", {}))
        self._validate_augmentations(cfg.get("augmentations", {}))
        self._validate_model(cfg.get("model", {}))
        self._validate_optimizer(cfg.get("optimizer", {}))
        self._validate_scheduler(cfg.get("scheduler", {}))
        self._validate_training(cfg.get("training", {}))
        self._validate_reporting(cfg.get("reporting", {}))
        self._validate_logging(cfg.get("logging", {}))

        # 报告错误
        if self.errors:
            raise self._ConfigValidationError(self.errors)

        return True

    # -------------------------
    # Helper check primitives
    # -------------------------
    def _check_section_type(
        self, cfg: dict[str, Any], key: str, expected_type: type, required: bool = True
    ) -> None:
        """检查顶级字段类型

        Args:
            cfg (dict[str, Any]): 顶层配置字典
            key (str): 顶级字段名称
            expected_type (type): 期望的 Python 类型
            required (bool): 是否为必需字段

        Returns:
            None
        """
        val = cfg.get(key, None)
        if val is None:
            if required:
                self.errors.append(f"Missing required top-level section '{key}'.")
            return
        if not isinstance(val, expected_type):
            self.errors.append(
                f"Top-level '{key}' expected type {expected_type.__name__}, got {type(val).__name__}."
            )

    def _require(
        self,
        obj: dict[str, Any],
        key: str,
        expected_type: type | tuple[type, ...],
        path: str,
        required: bool = True,
        choices: Optional[list[Any]] = None,
    ) -> None:
        """通用嵌套字段存在与类型检查

        Args:
            obj (dict[str, Any]): 需要检查的子配置字典
            key (str): 目标字段名称
            expected_type (type | tuple[type, ...]): 期望类型或类型元组
            path (str): 用于错误提示的字段路径描述
            required (bool): 是否为必填字段
            choices (Optional[list[Any]]): 若提供则限制取值必须在此列表内

        Returns:
            None
        """
        if key not in obj:
            if required:
                self.errors.append(f"Missing required config field '{path}'.")
            return

        val = obj[key]
        if not isinstance(val, expected_type):
            if isinstance(expected_type, tuple):
                names = ", ".join(t.__name__ for t in expected_type)
            else:
                names = expected_type.__name__
            self.errors.append(
                f"Field '{path}' expected type {names}, got {type(val).__name__}."
            )
            return

        if choices is not None:
            if val not in choices:
                self.errors.append(
                    f"Field '{path}' got '{val}', expected one of {choices}."
                )

    def _check_type_and_range(
        self,
        obj: dict[str, Any],
        key: str,
        path: str,
        expected_type: type | tuple[type, ...],
        min_val: Optional[int | float] = None,
        max_val: Optional[int | float] = None,
        required: bool = True,
    ) -> None:
        """检查字段类型并限定取值范围

        Args:
            obj (dict[str, Any]): 包含待检查字段的字典
            key (str): 字段名称
            path (str): 字段路径描述，用于错误消息
            expected_type (type | tuple[type, ...]): 期望的数值类型
            min_val (Optional[int | float]): 允许的最小值
            max_val (Optional[int | float]): 允许的最大值
            required (bool): 是否为必填字段

        Returns:
            None
        """
        self._require(obj, key, expected_type, path, required=required)
        if key not in obj:
            return
        val = obj[key]
        if isinstance(val, (int, float)):
            if min_val is not None and val < min_val:
                self.errors.append(f"Field '{path}' must be >= {min_val}, got {val}.")
            if max_val is not None and val > max_val:
                self.errors.append(f"Field '{path}' must be <= {max_val}, got {val}.")

    def _check_prob_or_list(
        self,
        obj: dict[str, Any],
        key: str,
        path: str,
        min_val: Optional[int | float] = None,
        max_val: Optional[int | float] = None,
        allow_number: bool = True,
        check_ascending: bool = True,
        required: bool = False,
    ) -> None:
        """检查概率或概率区间配置

        支持单个数值或长度为 2 的列表/元组形式

        Args:
            obj (dict[str, Any]): 包含待检查字段的字典
            key (str): 字段名称
            path (str): 字段路径描述
            min_val (Optional[int | float]): 最小允许值
            max_val (Optional[int | float]): 最大允许值
            allow_number (bool): 是否允许单个数值形式
            check_ascending (bool): 对区间是否要求前者小于等于后者
            required (bool): 是否为必填字段

        Returns:
            None
        """
        if key not in obj:
            if required:
                self.errors.append(f"Missing required config field '{path}'.")
            return

        val = obj[key]
        if val is None:
            self.errors.append(
                f"Field '{path}' is None but a probability or list of probabilities is expected."
            )
            return

        # single numeric
        if allow_number and isinstance(val, (int, float)):
            if min_val is not None and val < min_val:
                self.errors.append(f"Field '{path}' must be >= {min_val}, got {val}.")
            if max_val is not None and val > max_val:
                self.errors.append(f"Field '{path}' must be <= {max_val}, got {val}.")
            return

        # list/tuple of numerics
        if isinstance(val, (list, tuple)):
            if len(val) != 2:
                self.errors.append(
                    f"Field '{path}' expected list/tuple of 2 probabilities, got length {len(val)}."
                )
                return
            if check_ascending and val[0] > val[1]:
                self.errors.append(
                    f"Field '{path}' expected list/tuple with min <= max, got {val}."
                )
            for i, e in enumerate(val):
                if not isinstance(e, (int, float)):
                    self.errors.append(
                        f"Field '{path}[{i}]' expected numeric probability, got {type(e).__name__}."
                    )
                    continue
                if max_val is not None:
                    if check_ascending and e < -max_val:
                        self.errors.append(
                            f"Field '{path}[{i}]' must be >= {-max_val}, got {e}."
                        )
                    if e > max_val:
                        self.errors.append(
                            f"Field '{path}[{i}]' must be <= {max_val}, got {e}."
                        )
                if min_val is not None and not check_ascending and e < min_val:
                    self.errors.append(
                        f"Field '{path}[{i}]' must be >= {min_val}, got {e}."
                    )
            return

        # otherwise invalid type
        self.errors.append(
            f"Field '{path}' expected a probability (float) or list/tuple of probabilities, got {type(val).__name__}."
        )

    # -------------------------
    # Section validators
    # -------------------------
    def _validate_data(self, data: dict[str, Any]) -> None:
        """验证 data 相关配置段

        Args:
            data (dict[str, Any]): 数据加载与预处理相关配置

        Returns:
            None
        """
        path = "data"
        if not data:
            return

        self._require(data, "root", str, f"{path}.root", required=True)
        self._require(data, "splits_dir", str, f"{path}.splits_dir", required=True)

        split_ratios = data.get("split_ratios", None)
        if split_ratios is None:
            self.errors.append(f"Missing required field '{path}.split_ratios'.")
        else:
            if not (
                isinstance(split_ratios, list)
                and len(split_ratios) == 3
                and all(isinstance(r, float) for r in split_ratios)
                and all(r >= 0.0 for r in split_ratios)
                and all(r <= 1.0 for r in split_ratios)
            ):
                self.errors.append(
                    f"'{path}.split_ratios' must be list of 3 floats in [0.0, 1.0]."
                )
            else:
                total = sum(split_ratios)
                if abs(total - 1.0) > 1e-6:
                    self.errors.append(
                        f"'{path}.split_ratios' must sum to 1.0, got sum={total}."
                    )

        self._check_prob_or_list(
            data,
            "input_size",
            f"{path}.input_size",
            min_val=1,
            allow_number=False,
            check_ascending=False,
            required=True,
        )

        # normalization shape check
        norm = data.get("normalization", None)
        if norm is None:
            self.errors.append(f"Missing required field '{path}.normalization'.")
        else:
            if not isinstance(norm, dict):
                self.errors.append(
                    f"Field '{path}.normalization' expected dict with 'mean' and 'std'."
                )
            else:
                mean = norm.get("mean", None)
                std = norm.get("std", None)
                if not (isinstance(mean, list) and isinstance(std, list)):
                    self.errors.append(
                        f"'{path}.normalization.mean/std' must be lists."
                    )
                else:
                    if len(mean) != 3 or len(std) != 3:
                        self.errors.append(
                            f"'{path}.normalization.mean' and 'std' must have length 3."
                        )
                    if not all(isinstance(x, (int, float)) for x in mean + std):
                        self.errors.append(
                            f"'{path}.normalization.mean/std' must be numeric lists."
                        )

        self._check_type_and_range(
            data, "batch_size", f"{path}.batch_size", int, min_val=1, required=True
        )
        self._check_type_and_range(
            data, "num_workers", f"{path}.num_workers", int, min_val=0, required=False
        )

    def _validate_augmentations(self, aug: dict[str, Any]):
        """验证 augmentations 相关配置段

        Args:
            aug (dict[str, Any]): 数据增强相关配置

        Returns:
            None
        """
        path = "augmentations"
        if not aug:
            return

        self._check_prob_or_list(
            aug,
            "resize",
            f"{path}.resize",
            min_val=1,
            check_ascending=False,
            allow_number=False,
            required=False,
        )
        if aug.get("random_crop", None) is not None:
            rc = aug["random_crop"]
            if not isinstance(rc, dict):
                self.errors.append(f"'{path}.random_crop' must dict or null.")
            else:
                if "random_resized_crop" not in rc:
                    self.errors.append(
                        f"Missing required field '{path}.random_crop.random_resized_crop'."
                    )
                if not isinstance(rc["random_resized_crop"], bool):
                    self.errors.append(
                        f"'{path}.random_crop.random_resized_crop' must be boolean."
                    )
                if not rc["random_resized_crop"] and "resize" not in aug:
                    self.errors.append(
                        f"'{path}.resize' must be specified if using random_crop."
                    )
                self._check_prob_or_list(
                    rc,
                    "size",
                    f"{path}.random_crop.size",
                    min_val=1,
                    allow_number=False,
                    check_ascending=False,
                    required=False,
                )
                self._check_prob_or_list(
                    rc,
                    "scale",
                    f"{path}.random_crop.scale",
                    min_val=0.0,
                    allow_number=False,
                    required=False,
                )
                self._check_prob_or_list(
                    rc,
                    "ratio",
                    f"{path}.random_crop.ratio",
                    min_val=0.0,
                    allow_number=False,
                    required=False,
                )
        self._check_type_and_range(
            aug,
            "hflip_prob",
            f"{path}.hflip_prob",
            float,
            min_val=0.0,
            max_val=1.0,
            required=False,
        )
        self._check_type_and_range(
            aug,
            "vflip_prob",
            f"{path}.vflip_prob",
            float,
            min_val=0.0,
            max_val=1.0,
            required=False,
        )
        self._check_prob_or_list(
            aug,
            "rotation_degrees",
            f"{path}.rotation_degrees",
            min_val=0.0,
            max_val=180.0,
            required=False,
        )
        if aug.get("color_jitter", None) is not None:
            cj = aug["color_jitter"]
            if not isinstance(cj, dict):
                self.errors.append(f"'{path}.color_jitter' must be dict or null.")
            else:
                for k in ("brightness", "contrast", "saturation", "hue"):
                    self._check_type_and_range(
                        cj,
                        k,
                        f"{path}.color_jitter.{k}",
                        (int, float),
                        min_val=0.0,
                        max_val=1.0,
                        required=False,
                    )

    def _validate_model(self, model: dict[str, Any]) -> None:
        """验证 model 相关配置段

        Args:
            model (dict[str, Any]): 模型结构与超参数相关配置

        Returns:
            None
        """
        path = "model"
        if not model:
            return

        self._require(
            model, "name", str, f"{path}.name", required=True, choices=_ALLOWED_MODELS
        )
        self._check_type_and_range(
            model, "num_classes", f"{path}.num_classes", int, min_val=1, required=True
        )
        self._require(
            model,
            "activation",
            str,
            f"{path}.activation",
            required=False,
            choices=_ALLOWED_ACTIVATIONS,
        )
        if model.get("kwargs", None) is not None:
            kwargs = model["kwargs"]
            if not isinstance(kwargs, dict):
                self.errors.append(f"'{path}.kwargs' must be a dict if provided.")
            self._require(
                kwargs,
                "normalization",
                str,
                f"{path}.kwargs.normalization",
                required=False,
                choices=_ALLOWED_NORMALIZATIONS,
            )
            self._check_prob_or_list(
                kwargs,
                "dropout",
                f"{path}.kwargs.dropout",
                min_val=0.0,
                max_val=1.0,
                check_ascending=False,
                required=False,
            )
            self._require(
                kwargs,
                "config_name",
                str,
                f"{path}.kwargs.config_name",
                required=model.get("name", None)
                in ["butterfly_vgg", "butterfly_resnet"],
                choices=_ALLOWED_CONFIG_NAMES,
            )
            self._require(
                kwargs,
                "init_weights",
                bool,
                f"{path}.kwargs.init_weights",
                required=False,
            )
        elif model.get("name", None) in ["butterfly_vgg", "butterfly_resnet"]:
            self.errors.append(
                f"Missing required field '{path}.kwargs.config_name' for {model['name']}."
            )

    def _validate_optimizer(self, opt: dict[str, Any]):
        """验证 optimizer 相关配置段

        Args:
            opt (dict[str, Any]): 优化器相关配置

        Returns:
            None
        """
        path = "optimizer"
        if not opt:
            return

        self._require(
            opt,
            "name",
            str,
            f"{path}.name",
            required=True,
            choices=_ALLOWED_OPTIMIZERS,
        )
        self._check_type_and_range(
            opt, "lr", f"{path}.lr", (int, float), min_val=0.0, required=True
        )
        name = opt.get("name", None)
        if name == "adamw" or name == "adam":
            self._check_type_and_range(
                opt,
                "weight_decay",
                f"{path}.weight_decay",
                (int, float),
                min_val=0.0,
                required=True,
            )
        elif name == "sgd":
            self._check_type_and_range(
                opt,
                "weight_decay",
                f"{path}.weight_decay",
                (int, float),
                min_val=0.0,
                required=True,
            )
            self._check_type_and_range(
                opt,
                "momentum",
                f"{path}.momentum",
                (int, float),
                min_val=0.0,
                max_val=1.0,
                required=True,
            )

    def _validate_scheduler(self, sch: dict[str, Any]):
        """验证 scheduler 相关配置段

        Args:
            sch (dict[str, Any]): 学习率调度器相关配置

        Returns:
            None
        """
        path = "scheduler"
        if not sch:
            return

        self._require(
            sch,
            "name",
            str,
            f"{path}.name",
            required=True,
            choices=_ALLOWED_SCHEDULERS,
        )
        """验证 training 相关配置段

        Args:
            tr (dict[str, Any]): 训练流程相关配置

        Returns:
            None
        """
        name = sch.get("name", None)
        if name == "step":
            self._check_type_and_range(
                sch, "step_size", f"{path}.step_size", int, min_val=1, required=True
            )
            self._check_type_and_range(
                sch,
                "gamma",
                f"{path}.gamma",
                (int, float),
                min_val=0.0,
                max_val=1.0,
                required=True,
            )
        elif name == "cosine":
            self._check_type_and_range(
                sch,
                "max_epochs",
                f"{path}.max_epochs",
                int,
                min_val=1,
                required=True,
            )
            self._check_type_and_range(
                sch,
                "min_lr",
                f"{path}.min_lr",
                (int, float),
                min_val=0.0,
                required=True,
            )
        elif name == "exp":
            self._check_type_and_range(
                sch,
                "gamma",
                f"{path}.gamma",
                (int, float),
                min_val=0.0,
                max_val=1.0,
                required=True,
            )
        elif name == "reduce_on_plateau":
            self._check_type_and_range(
                sch,
                "factor",
                f"{path}.factor",
                (int, float),
                min_val=0.0,
                max_val=1.0,
                required=True,
            )
            self._check_type_and_range(
                sch,
                "patience",
                f"{path}.patience",
                int,
                min_val=1,
                required=True,
            )
            self._check_type_and_range(
                sch,
                "min_lr",
                f"{path}.min_lr",
                (int, float),
                min_val=0.0,
                required=True,
            )
        self._check_type_and_range(
            sch,
            "warmup_epochs",
            f"{path}.warmup_epochs",
            int,
            min_val=0,
            required=False,
        )
        self._check_type_and_range(
            sch,
            "warmup_start_factor",
            f"{path}.warmup_start_factor",
            (int, float),
            min_val=0.0,
            max_val=1.0,
            required=False,
        )

    def _validate_training(self, tr: dict[str, Any]):
        path = "training"
        if not tr:
            return

        self._check_type_and_range(
            tr, "epochs", f"{path}.epochs", int, min_val=1, required=True
        )
        self._check_type_and_range(
            tr,
            "grad_clip",
            f"{path}.grad_clip",
            (int, float),
            min_val=0.0,
            required=False,
        )
        self._check_type_and_range(
            tr,
            "resume_from",
            f"{path}.resume_from",
            (str, type(None)),
            required=False,
        )

    def _validate_reporting(self, rep: dict[str, Any]):
        """验证 reporting 相关配置段

        Args:
            rep (dict[str, Any]): 训练日志与指标监控相关配置

        Returns:
            None
        """
        path = "reporting"
        if not rep:
            return

        self._require(
            rep,
            "monitor_metric",
            str,
            f"{path}.monitor_metric",
            required=True,
        )
        self._require(
            rep,
            "monitor_mode",
            str,
            f"{path}.monitor_mode",
            required=True,
            choices=["max", "min"],
        )
        self._check_type_and_range(
            rep,
            "save_topk_checkpoints",
            f"{path}.save_topk_checkpoints",
            int,
            min_val=0,
            required=False,
        )

    def _validate_evaluation(self, ev: dict[str, Any]):
        """验证 evaluation 相关配置段

        Args:
            ev (dict[str, Any]): 评估阶段相关配置

        Returns:
            None
        """
        path = "evaluation"
        if not ev:
            return

        self._check_prob_or_list(
            ev,
            "topk",
            f"{path}.topk",
            min_val=1,
            allow_number=False,
            required=False,
        )
        self._check_type_and_range(
            ev,
            "pth_path",
            f"{path}.pth_path",
            (str, type(None)),
            required=False,
        )

    def _validate_logging(self, log: dict[str, Any]):
        """验证 logging 相关配置段

        Args:
            log (dict[str, Any]): 日志记录相关配置

        Returns:
            None
        """
        path = "logging"
        if not log:
            return

        self._require(
            log,
            "use_wandb",
            bool,
            f"{path}.use_wandb",
            required=True,
        )
        self._check_type_and_range(
            log,
            "log_interval",
            f"{path}.log_interval",
            int,
            min_val=1,
            required=True,
        )

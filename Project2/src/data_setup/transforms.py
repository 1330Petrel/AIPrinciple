"""图像预处理与数据增强配置模块

根据配置对象构建训练、验证与测试阶段所需的 torchvision 变换流水线
"""

import torchvision.transforms as T

from utils.helpers import AttrDict


def get_transforms(
    config: AttrDict, mode: str = ""
) -> T.Compose | dict[str, T.Compose]:
    """根据配置构建图像变换流水线

    当 mode 为空或为 "all" 时, 返回包含 train/val/test 三种模式的变换字典
    否则根据指定模式仅返回对应阶段的变换

    Args:
        config (AttrDict): 包含数据与增强参数的数据配置对象
        mode (str): 需要的变换模式, 可为 "train"、"val"、"test" 或 "all"/空字符串

    Returns:
        T.Compose | dict[str, T.Compose]: 对应模式的变换对象, 或包含所有模式的字典
    """
    if mode == "" or mode == "all" or mode is None:
        # 同时构建 train/val/test 三种模式
        return {
            "train": get_train_transforms(config),
            "val": get_val_transforms(config),
            "test": get_test_transforms(config),
        }
    mode = mode.lower()
    if mode.startswith("train"):
        return get_train_transforms(config)
    elif mode.startswith("val") or mode.startswith("test"):
        return get_val_transforms(config)
    else:
        raise ValueError(f"Unknown mode '{mode}' for get_transforms.")


def get_train_transforms(config: AttrDict) -> T.Compose:
    """根据配置构建训练集图像变换

    根据 config.augmentations 中的设置动态添加随机裁剪、翻转、旋转和颜色抖动等增强

    Args:
        config (AttrDict): 数据与增强相关配置对象

    Returns:
        T.Compose: 训练阶段使用的变换组合
    """
    input_size = tuple(config.data.input_size)
    mean = config.data.normalization.mean
    std = config.data.normalization.std

    # 根据配置决定是否启用数据增强
    train_transforms_list: list[object] = []
    aug_cfg = config.get("augmentations")
    if aug_cfg:
        # 根据配置动态添加增强变换
        crop_size = input_size
        if "resize" in aug_cfg:
            train_transforms_list.append(
                T.Resize(tuple(aug_cfg.resize))
            )
        if aug_cfg.get("random_crop", None) is not None:
            crop_params = aug_cfg.random_crop
            crop_size = tuple(crop_params.get("size", input_size))
            if crop_params.random_resized_crop:
                train_transforms_list.append(
                    T.RandomResizedCrop(
                        size=crop_size,
                        scale=crop_params.get("scale", (0.08, 1.0)),
                        ratio=crop_params.get("ratio", (3 / 4, 4 / 3)),
                    )
                )
            else:
                train_transforms_list.append(
                    T.RandomCrop(size=crop_size)
                )
        if aug_cfg.get("hflip_prob", 0.0) > 0.0:
            train_transforms_list.append(T.RandomHorizontalFlip(p=aug_cfg.hflip_prob))
        if aug_cfg.get("vflip_prob", 0.0) > 0.0:
            train_transforms_list.append(T.RandomVerticalFlip(p=aug_cfg.vflip_prob))
        if aug_cfg.get("rotation_degrees", 0) != 0:
            train_transforms_list.append(
                T.RandomRotation(degrees=aug_cfg.rotation_degrees)
            )
        if aug_cfg.get("color_jitter", None) is not None:
            cj_params = aug_cfg.color_jitter
            b = cj_params.get("brightness", 0.0)
            c = cj_params.get("contrast", 0.0)
            s = cj_params.get("saturation", 0.0)
            h = cj_params.get("hue", 0.0)
            if max(b, c, s, h) > 0.0:
                train_transforms_list.append(
                    T.ColorJitter(
                        brightness=b,
                        contrast=c,
                        saturation=s,
                        hue=h,
                    )
                )
        # 最后添加转为Tensor和归一化
        if crop_size != input_size:
            train_transforms_list.append(T.Resize(input_size))
        train_transforms_list.extend(
            [
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        train_transforms = T.Compose(train_transforms_list)
    else:
        # 如果不启用增强，训练集和验证集的变换相同
        train_transforms = T.Compose(
            [
                T.Resize(input_size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )

    return train_transforms


def get_val_transforms(config: AttrDict) -> T.Compose:
    """根据配置构建验证集图像变换

    验证阶段不使用随机增强, 直接复用测试阶段的变换

    Args:
        config (AttrDict): 数据配置对象

    Returns:
        T.Compose: 验证阶段使用的变换组合
    """
    return get_test_transforms(config)


def get_test_transforms(config: AttrDict) -> T.Compose:
    """根据配置构建测试集图像变换

    仅进行尺寸调整、转张量和归一化, 不包含随机增强

    Args:
        config (AttrDict): 数据配置对象

    Returns:
        T.Compose: 测试阶段使用的变换组合
    """
    input_size = config.data.input_size
    mean = config.data.normalization.mean
    std = config.data.normalization.std

    test_transforms = T.Compose(
        [
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return test_transforms

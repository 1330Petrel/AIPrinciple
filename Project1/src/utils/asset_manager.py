"""
项目资源管理器模块

负责加载、缓存和提供游戏中所有方块的视觉资源
1. 图片：从 'assets/images/' 目录加载
2. 颜色块+数字：当图片资源用尽时，使用预定义的颜色
3. 纯数字：当以上资源都用尽时，直接显示种类ID

通过单例或一次性实例化使用，避免重复加载资源
"""

import os
from PIL import Image, ImageTk
import utils.constants as const


class AssetManager:
    """
    管理游戏中的所有视觉资源，特别是方块的图像和颜色
    """

    def __init__(self):
        """
        初始化资源管理器，并预加载所有图片资源
        """
        # 使用字典缓存加载的PhotoImage对象，防止被Python的垃圾回收机制清除
        self.image_cache = {}
        self._load_tile_images()

    def _load_tile_images(self):
        """
        私有方法，用于从资源目录加载所有方块图片并缓存
        """
        # 检查资源目录是否存在
        if not os.path.isdir(const.IMAGE_DIR):
            print(f"警告: 资源目录 '{const.IMAGE_DIR}' 不存在，将无法加载任何图片方块")
            return

        for i in range(1, const.MAX_TILE_TYPES_WITH_IMAGE + 1):
            image_path = os.path.join(const.IMAGE_DIR, f"{i}.png")
            try:
                # 打开图片文件
                img = Image.open(image_path)

                # 调整图片大小以适应方块尺寸，使用高质量的LANCZOS抗锯齿缩放
                img = img.resize(
                    (const.TILE_SIZE, const.TILE_SIZE), Image.Resampling.LANCZOS
                )

                # 转换为Tkinter兼容的格式并存入缓存
                self.image_cache[i] = ImageTk.PhotoImage(img)

            except FileNotFoundError:
                print(
                    f"警告: 图片文件 '{image_path}' 未找到，种类ID {i} 将无法使用图片显示"
                )
            except Exception as e:
                print(f"错误: 加载或处理图片 '{image_path}' 时发生未知错误: {e}")

    def get_visual_for_type(self, type_id: int) -> dict:
        """
        根据给定的方块种类ID，获取其对应的视觉表现

        Args:
            type_id (int): 方块的种类ID（从1开始）

        Returns:
            dict: 一个包含视觉信息的字典。
                  - 图片: {'type': 'image', 'data': PhotoImage_object}
                  - 颜色块: {'type': 'color_block', 'color': '#hex', 'number': 'str'}
                  - 数字: {'type': 'number', 'number': 'str'}
        """
        # --- 第一级：图片 ---
        # 检查ID是否在图片范围内，并且该图片已成功加载到缓存中
        if (
            1 <= type_id <= const.MAX_TILE_TYPES_WITH_IMAGE
            and type_id in self.image_cache
        ):
            return {"type": "image", "data": self.image_cache[type_id]}

        # --- 第二级：带数字的颜色块 ---
        # ID超出了图片范围，则尝试使用颜色块
        upper_bound_color = (
            const.MAX_TILE_TYPES_WITH_IMAGE + const.MAX_TILE_TYPES_WITH_COLOR
        )
        if const.MAX_TILE_TYPES_WITH_IMAGE < type_id <= upper_bound_color:
            color_index = type_id - const.MAX_TILE_TYPES_WITH_IMAGE - 1
            number_in_color_block = color_index + 1  # 颜色块内的数字从1开始
            return {
                "type": "color_block",
                "color": const.TILE_COLORS[color_index],
                "number": str(number_in_color_block),
            }

        # --- 第三级：纯数字 ---
        # 如果以上所有资源都不可用，或ID超出了所有预设范围，则直接显示ID
        return {
            "type": "number",
            "number": str(type_id - const.MAX_TILE_TYPES_WITH_IMAGE),
        }

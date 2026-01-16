"""
定义了 BoardCanvas 类，一个 tk.Canvas 的子类，用于：
1. 绘制游戏棋盘的网格和方块
2. 处理用户的鼠标点击，并将点击的格子坐标报告给控制器
3. 显示方块选择的高亮效果
4. 演示解法路径的消除动画
"""

import tkinter as tk
from typing import Callable, Tuple, List, Optional

from core.tile import Tile
from core.board import Board
from core.algorithm import AnimationStep
from utils.asset_manager import AssetManager
import utils.constants as const


class BoardCanvas(tk.Canvas):
    """
    负责所有与棋盘渲染和交互的 Canvas 组件
    """

    def __init__(
        self,
        parent,
        asset_manager: AssetManager,
        click_callback: Optional[Callable[[Tuple[int, int]], None]] = None,
        right_click_callback: Optional[Callable[[Tuple[int, int]], None]] = None,
    ):
        """
        初始化棋盘画布

        Args:
            parent: Tkinter 父容器 (通常是 game_frame)
            asset_manager (AssetManager): 用于获取方块视觉资源的实例
            click_callback (Optional[Callable]): 当用户点击棋盘格子时调用的回调函数
        """
        # 计算画布所需的最大尺寸，以容纳最大可能的棋盘
        canvas_dim = (
            const.MAX_BOARD_SIZE * (const.TILE_SIZE + const.TILE_GAP)
            + 2 * const.BOARD_PADDING
        )

        super().__init__(
            parent,
            width=canvas_dim,
            height=canvas_dim,
            bg=const.BOARD_BG,
            highlightthickness=0,
        )

        self.asset_manager = asset_manager
        self.click_callback = click_callback
        self.right_click_callback = right_click_callback

        self.current_board_size = 0
        self.animation_id = None  # 用于存储 .after() 的ID

        # 绑定鼠标点击事件
        self.bind("<Button-1>", self._on_canvas_click)
        self.bind("<Button-3>", self._on_canvas_right_click)

    # --- 回调 ---
    def _on_canvas_click(self, event):
        """
        处理画布上的鼠标点击事件
        """
        if not self.click_callback or self.current_board_size == 0:
            return

        # 将屏幕坐标转换为画布内的绝对坐标
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)

        # 将像素坐标转换为网格坐标
        col = int(
            (canvas_x - const.BOARD_PADDING) // (const.TILE_SIZE + const.TILE_GAP)
        )
        row = int(
            (canvas_y - const.BOARD_PADDING) // (const.TILE_SIZE + const.TILE_GAP)
        )

        # 检查点击是否在有效网格内
        if 0 <= row < self.current_board_size and 0 <= col < self.current_board_size:
            self.click_callback((row, col))

    def _on_canvas_right_click(self, event):
        """
        处理画布上的鼠标右键点击事件
        """
        if not self.right_click_callback or self.current_board_size == 0:
            return

        # 将屏幕坐标转换为画布内的绝对坐标
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)

        # 将像素坐标转换为网格坐标
        col = int(
            (canvas_x - const.BOARD_PADDING) // (const.TILE_SIZE + const.TILE_GAP)
        )
        row = int(
            (canvas_y - const.BOARD_PADDING) // (const.TILE_SIZE + const.TILE_GAP)
        )

        # 检查点击是否在有效网格内
        if 0 <= row < self.current_board_size and 0 <= col < self.current_board_size:
            self.right_click_callback((row, col))

    # --- 公共方法 ---
    def draw_board(self, board: Board):
        """
        根据给定的 Board 对象状态，重绘整个棋盘

        Args:
            board (Board): 要绘制的棋盘对象
        """
        # 1. 清空画布
        self.clear()
        self.current_board_size = board.size

        # 2. 计算棋盘绘制区域的总尺寸，并设置滚动区域
        total_dim = (
            self.current_board_size * (const.TILE_SIZE + const.TILE_GAP)
            + 2 * const.BOARD_PADDING
        )
        self.config(scrollregion=(0, 0, total_dim, total_dim))

        # 3. 绘制坐标标签
        for i in range(self.current_board_size):
            # 顶部列坐标 (0, 1, 2, ...)
            x = (
                const.BOARD_PADDING
                + i * (const.TILE_SIZE + const.TILE_GAP)
                + const.TILE_SIZE / 2
            )
            self.create_text(
                x,
                const.BOARD_PADDING / 2,
                text=str(i),
                font=const.FONT_LABEL,
                fill=const.TEXT_COLOR,
            )
            # 左侧行坐标 (0, 1, 2, ...)
            y = (
                const.BOARD_PADDING
                + i * (const.TILE_SIZE + const.TILE_GAP)
                + const.TILE_SIZE / 2
            )
            self.create_text(
                const.BOARD_PADDING / 2,
                y,
                text=str(i),
                font=const.FONT_LABEL,
                fill=const.TEXT_COLOR,
            )

        # 4. 绘制网格线
        grid_dim = (
            self.current_board_size * (const.TILE_SIZE + const.TILE_GAP)
            - const.TILE_GAP
        )
        for i in range(self.current_board_size + 1):
            pos = (
                const.BOARD_PADDING
                + i * (const.TILE_SIZE + const.TILE_GAP)
                - const.TILE_GAP / 2
            )
            self.create_line(
                const.BOARD_PADDING - const.TILE_GAP / 2,
                pos,
                const.BOARD_PADDING + grid_dim,
                pos,
                fill=const.GRID_LINE_COLOR,
            )
            self.create_line(
                pos,
                const.BOARD_PADDING - const.TILE_GAP / 2,
                pos,
                const.BOARD_PADDING + grid_dim,
                fill=const.GRID_LINE_COLOR,
            )

        # 5. 绘制所有方块
        for tile in board.get_all_tiles():
            self._draw_tile(tile)

    def highlight_tiles(self, positions: List[Tuple[int, int]]):
        """
        在指定的格子位置上绘制高亮边框

        Args:
            positions (List[Tuple[int, int]]): 需要高亮的 (row, col) 坐标列表。
        """
        self.delete("highlight")  # 清除旧的高亮

        for row, col in positions:
            x1 = const.BOARD_PADDING + col * (const.TILE_SIZE + const.TILE_GAP)
            y1 = const.BOARD_PADDING + row * (const.TILE_SIZE + const.TILE_GAP)
            x2, y2 = x1 + const.TILE_SIZE, y1 + const.TILE_SIZE
            self.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=const.TILE_HIGHLIGHT_COLOR,
                width=3,
                tags="highlight",
            )

    def animate_solution(
        self, steps: List[AnimationStep], on_complete: Optional[Callable] = None
    ):
        """
        开始演示消除动画

        Args:
            steps (List[AnimationStep]): 求解器返回的动画步骤
            on_complete (Optional[Callable]): 动画播放完毕后要执行的回调函数
        """
        if self.animation_id:
            self.stop_animation()

        if not steps:
            if on_complete:
                on_complete()
            return

        # 启动动画的第一个步骤
        self._animate_step(steps, 0, on_complete)

    def stop_animation(self):
        """立即停止正在进行的动画"""
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None

    def clear(self):
        """
        清空整个画布并重置状态
        """
        self.delete("all")
        self.current_board_size = 0

    # --- 私有辅助方法 ---
    def _draw_tile(self, tile: Tile):
        """
        在画布上绘制单个方块
        """
        row, col = tile.position

        # 计算方块左上角的像素坐标
        x1 = const.BOARD_PADDING + col * (const.TILE_SIZE + const.TILE_GAP)
        y1 = const.BOARD_PADDING + row * (const.TILE_SIZE + const.TILE_GAP)

        # 为方块创建唯一的标签，方便后续查找和删除
        tile_tag = f"tile_{row}_{col}"

        visual = self.asset_manager.get_visual_for_type(tile.type_id)

        if visual["type"] == "image":
            self.create_image(x1, y1, image=visual["data"], anchor="nw", tags=tile_tag)

        elif visual["type"] == "color_block":
            x2, y2 = x1 + const.TILE_SIZE, y1 + const.TILE_SIZE
            self.create_rectangle(
                x1, y1, x2, y2, fill=visual["color"], outline="", tags=tile_tag
            )
            self.create_text(
                x1 + const.TILE_SIZE / 2,
                y1 + const.TILE_SIZE / 2,
                text=visual["number"],
                font=const.FONT_TILE_NUMBER,
                fill=const.TILE_TEXT_COLOR,
                tags=tile_tag,
            )

        elif visual["type"] == "number":
            x2, y2 = x1 + const.TILE_SIZE, y1 + const.TILE_SIZE
            self.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill="white",
                outline=const.GRID_LINE_COLOR,
                tags=tile_tag,
            )
            self.create_text(
                x1 + const.TILE_SIZE / 2,
                y1 + const.TILE_SIZE / 2,
                text=visual["number"],
                font=const.FONT_TILE_NUMBER,
                fill=const.TILE_TEXT_COLOR,
                tags=tile_tag,
            )

    def _get_pixel_coords(self, position: Tuple[int, int]) -> Tuple[float, float]:
        """将 (row, col) 转换为画布左上角的 (x, y) 像素坐标"""
        row, col = position
        x = const.BOARD_PADDING + col * (const.TILE_SIZE + const.TILE_GAP)
        y = const.BOARD_PADDING + row * (const.TILE_SIZE + const.TILE_GAP)
        return x, y

    def _slide_tile(
        self,
        tile_tag: str,
        dx_per_frame: float,
        dy_per_frame: float,
        frames_left: int,
        on_slide_complete: Callable,
    ):
        """
        递归方法，用于平滑地移动一个方块
        """
        if frames_left <= 0:
            if on_slide_complete:
                on_slide_complete()
            return

        # 移动一小步
        self.move(tile_tag, dx_per_frame, dy_per_frame)

        # 计算每帧的延迟
        total_slide_time = const.ANIMATION_DELAY_MS // 2
        delay_per_frame = max(1, total_slide_time // const.ANIMATION_MOVE_FRAMES)

        # 安排下一次移动
        self.animation_id = self.after(
            delay_per_frame,
            lambda: self._slide_tile(
                tile_tag, dx_per_frame, dy_per_frame, frames_left - 1, on_slide_complete
            ),
        )

    def _animate_step(
        self,
        steps: List[AnimationStep],
        step_index: int,
        on_complete: Optional[Callable],
    ):
        """
        执行动画的单一步骤
        """
        # 动画结束的基本情况
        if step_index >= len(steps):
            self.animation_id = None
            if on_complete:
                on_complete()
            return

        step = steps[step_index]
        action_type = step["action_type"]
        details = step["details"]

        next_step_func = lambda: self._animate_step(steps, step_index + 1, on_complete)

        if action_type == "DIRECT_ELIMINATE":
            tile1, tile2 = details["tile1"], details["tile2"]
            pos1, pos2 = tile1.position, tile2.position

            self.highlight_tiles([pos1, pos2])

            def after_highlight_action():
                self.highlight_tiles([])
                self.delete(f"tile_{pos1[0]}_{pos1[1]}")
                self.delete(f"tile_{pos2[0]}_{pos2[1]}")
                self.animation_id = self.after(
                    const.ANIMATION_DELAY_MS // 2, next_step_func
                )

            self.animation_id = self.after(
                const.ANIMATION_DELAY_MS // 2, after_highlight_action
            )

        elif action_type == "MOVE_AND_ELIMINATE":
            moving_tile = details["moving_tile"]
            start_pos = moving_tile.position
            dest_pos = details["destination"]
            target_tile = details["target_tile"]

            start_x, start_y = self._get_pixel_coords(start_pos)
            end_x, end_y = self._get_pixel_coords(dest_pos)

            dx_total = end_x - start_x
            dy_total = end_y - start_y

            dx_frame = dx_total / const.ANIMATION_MOVE_FRAMES
            dy_frame = dy_total / const.ANIMATION_MOVE_FRAMES

            tile_tag = f"tile_{start_pos[0]}_{start_pos[1]}"

            # 定义滑动完成后的动作
            def on_slide_complete():
                # 在新位置和目标位置高亮
                self.highlight_tiles([dest_pos, target_tile.position])

                # 高亮后的消除动作
                def after_highlight_action():
                    self.highlight_tiles([])
                    self.delete(tile_tag)  # 删除已移动到位的方块
                    r_target, c_target = target_tile.position
                    self.delete(f"tile_{r_target}_{c_target}")  # 删除目标方块
                    # 进入下一个动画步骤
                    self.animation_id = self.after(
                        const.ANIMATION_DELAY_MS // 4, next_step_func
                    )

                # 安排消除动作
                self.animation_id = self.after(
                    const.ANIMATION_DELAY_MS // 2, after_highlight_action
                )

            self._slide_tile(
                tile_tag,
                dx_frame,
                dy_frame,
                const.ANIMATION_MOVE_FRAMES,
                on_slide_complete,
            )

"""
定义了 GameFrame 类，作为游戏界面的主容器和控制器：
1. 布局 ControlPanel, BoardCanvas, 和 StatusBar
2. 管理游戏的核心状态（如 Board 对象、代价设置）
3. 协调 UI 事件（如按钮点击）与核心逻辑（如棋盘生成、算法求解）之间的通信
"""

import tkinter as tk
from tkinter import messagebox, ttk
import random
import time
from typing import Dict, Callable, List, Tuple

from gui.game.components.control_panel import ControlPanel
from gui.game.components.board_canvas import BoardCanvas
from gui.game.components.status_bar import StatusBar
from gui.game.components.cost_editor import _CostEditorWindow
from gui.game.components.batch_placement import _BatchPlacementWindow
from core.board import Board
from core.tile import Tile
from core.algorithm import Solver
from utils.asset_manager import AssetManager
import utils.constants as const


class GameFrame(tk.Frame):
    """
    游戏界面的主框架，协调所有子组件和游戏逻辑
    """

    def __init__(
        self, parent, back_to_menu_callback: Callable, asset_manager: AssetManager
    ):
        super().__init__(parent, bg=const.WINDOW_BG)

        self.back_to_menu_callback = back_to_menu_callback
        self.asset_manager = asset_manager

        # --- 初始化游戏状态 ---
        self.board: Board = None
        self.board_size = 0
        self.num_pairs = 0
        self.tile_costs: Dict[int, int] = {}
        self.last_tile_costs: Dict[int, int] = {}
        self.is_custom_mode = False
        self.custom_placements: List[Tuple[int, int]] = []
        self.current_custom_type = 0
        self.unique_id_counter = 0

        # --- 定义回调字典，传递给 ControlPanel ---
        control_callbacks = {
            "random_generate": self._random_generate,
            "custom_generate": self._start_custom_generate,
            "set_cost_mode": self._set_cost_mode,
            "show_cost_editor": self._show_cost_editor_window,
            "set_cost_mode_custom": self._set_cost_mode_custom,
            "finish_current_type": self._finish_current_type,
            "finish_all_custom": self._finish_all_types,
            "undo_last_placement": self._undo_last_placement,
            "batch_placement": self._show_batch_placement_window,
            "solve": self._solve,
            "reset": self._reset,
            "back_to_menu": self.back_to_menu_callback,
        }

        # --- 创建并布局子组件 ---
        self.control_panel = ControlPanel(self, control_callbacks)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.status_bar = StatusBar(self)
        self.status_bar.pack(side=tk.TOP, fill=tk.X)

        # 创建一个框架来容纳棋盘画布和其滚动条
        board_area_frame = tk.Frame(self, bg=const.BOARD_BG)
        board_area_frame.pack(
            side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(0, 10), pady=10
        )

        v_scrollbar = ttk.Scrollbar(board_area_frame, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(board_area_frame, orient=tk.HORIZONTAL)

        self.board_canvas = BoardCanvas(
            board_area_frame,
            self.asset_manager,
            self._on_board_click,
            self._on_board_right_click,
        )

        v_scrollbar.config(command=self.board_canvas.yview)
        h_scrollbar.config(command=self.board_canvas.xview)
        self.board_canvas.config(
            yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set
        )

        # 使用 grid 布局来放置画布和滚动条
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.board_canvas.grid(row=0, column=0, sticky="nsew")

        # 让画布在框架内可以缩放
        board_area_frame.grid_rowconfigure(0, weight=1)
        board_area_frame.grid_columnconfigure(0, weight=1)

    # --- 棋盘生成与设置 ---
    def _validate_inputs(self, size_str: str, pairs_str: str) -> bool:
        """验证输入的棋盘大小和方块总对数"""
        try:
            self.board_size = int(size_str)
            self.num_pairs = int(pairs_str)
            if not (const.MIN_BOARD_SIZE <= self.board_size <= const.MAX_BOARD_SIZE):
                raise IndexError(
                    f"棋盘大小必须在 {const.MIN_BOARD_SIZE} 和 {const.MAX_BOARD_SIZE} 之间"
                )
            if self.num_pairs < const.MIN_PAIRS:
                raise IndexError(f"方块总对数至少为 {const.MIN_PAIRS}")
            if self.num_pairs * 2 > self.board_size**2:
                raise IndexError(
                    f"棋盘空间不足，总对数/种类数不能超过 {self.board_size**2 // 2}"
                )
            return True
        except ValueError as e:
            messagebox.showerror("输入错误", f"{e}。请输入有效的正整数")
            return False
        except IndexError as e:
            messagebox.showerror("输入错误", str(e))
            return False

    def _random_generate(self, size_str: str, pairs_str: str):
        """随机生成棋盘"""
        if not self._validate_inputs(size_str, pairs_str):
            return

        self._reset_board_state()
        self._update_costs_for_new_board()

        self.board = Board.create_random_board(
            self.board_size, self.num_pairs, self.tile_costs
        )
        self.board_canvas.draw_board(self.board)

        self.control_panel.set_solver_active(True)
        self.control_panel.set_cost_editor_active(True)
        self.status_bar.update_text(
            f"已生成 {self.board_size}x{self.board_size} 棋盘，包含 {self.board.get_num_of_type()} 种共 {self.num_pairs} 对方块"
        )

    def _start_custom_generate(self, size_str: str, pairs_str: str):
        """开始自定义棋盘生成流程"""
        if not self._validate_inputs(size_str, pairs_str):
            return

        self._reset_board_state()
        self.is_custom_mode = True
        self.current_custom_type = 1

        self.board = Board(self.board_size)
        self.board_canvas.draw_board(self.board)
        self.board_canvas.highlight_tiles([])

        self.control_panel.set_board_creation_active(False)
        self.control_panel.show_custom_placement_layout()
        self._update_custom_panel_prompt()
        self.status_bar.update_text("自定义棋盘：点击棋盘选择方块位置，右键撤销选择")

    def _on_board_click(self, pos: Tuple[int, int]):
        """处理来自 board_canvas 的左键点击事件"""
        if not self.is_custom_mode:
            return
        if self.board.get_tile_at(pos):
            self.status_bar.update_text("该位置已被其他种类方块占据，请选择一个空格子")
            return
        if pos in self.custom_placements:
            self.status_bar.update_text("该位置已被选择，请选择其他格子")
            return

        self.custom_placements.append(pos)

        self.board_canvas.highlight_tiles(self.custom_placements)
        self.control_panel.update_custom_panel(
            self._get_custom_prompt(), self.custom_placements
        )
        self.status_bar.update_text(
            f"放置第 {len(self.custom_placements)} 个方块于 {pos}"
        )

    def _on_board_right_click(self, pos: Tuple[int, int]):
        """处理来自 board_canvas 的右键点击事件"""
        if not self.is_custom_mode:
            return
        if self.board.get_tile_at(pos):
            self.status_bar.update_text("该位置已被占据，无法撤销")
            return
        # 检查点击的位置是否是已选中的位置
        if pos in self.custom_placements:
            self._remove_custom_placement(pos)
        else:
            self.status_bar.update_text("该位置未被选择，无法撤销")

    def _undo_last_placement(self):
        """撤销上一步操作"""
        if len(self.custom_placements) == 0:
            messagebox.showerror("操作无效", "没有可以撤销的放置")
            return

        # 移除最后一个坐标
        self._remove_custom_placement(self.custom_placements[-1])

    def _show_batch_placement_window(self):
        """打开批量操作窗口"""
        if not self.is_custom_mode:
            return

        _BatchPlacementWindow(
            self,
            self.board,
            self.custom_placements,
            self._save_batch_placements,
        )

    def _save_batch_placements(self, new_coords: List[Tuple[int, int]]):
        """验证并保存来自批量操作窗口的坐标"""
        self.custom_placements = new_coords

        # 更新UI
        self.board_canvas.highlight_tiles(self.custom_placements)
        self._update_custom_panel_prompt()
        self.status_bar.update_text(f"批量操作：已放置 {len(new_coords)} 个方块")

    def _finish_current_type(self):
        """完成当前种类方块的放置"""
        if not self.is_custom_mode:
            return

        num_current = len(self.custom_placements)
        if num_current % 2 != 0:
            messagebox.showerror("操作无效", "当前种类的方块数量必须为偶数")
            return
        if num_current == 0:
            messagebox.showerror("操作无效", "请至少选择两个位置来放置方块")
            return
        if (
            self.current_custom_type < self.num_pairs
            and num_current == self.board.get_num_of_empty()
        ):
            if messagebox.askyesno(
                "确认", "棋盘空间已满，放置后无法继续添加新种类，是否继续？"
            ):
                self._place_custom_tiles(num_current)
                self._finish_custom_generate()
            else:
                return

        self._place_custom_tiles(num_current)
        if self.current_custom_type > self.num_pairs:
            self._finish_custom_generate()
        else:
            self._update_custom_panel_prompt()

    def _finish_all_types(self):
        """完成所有种类方块的放置"""
        num_current = len(self.custom_placements)
        if num_current > 0:
            if num_current % 2 != 0:
                if messagebox.askyesno(
                    "确认", "当前种类的方块数量为奇数，是否放弃这些方块并继续？"
                ):
                    self.custom_placements = []
                    self.board_canvas.highlight_tiles([])
                    self._update_custom_panel_prompt()
                else:
                    return
            else:
                if (
                    self.current_custom_type < self.num_pairs
                    and num_current == self.board.get_num_of_empty()
                ):
                    if not messagebox.askyesno(
                        "确认", "棋盘空间已满且有未放置的方块种类，确认完成吗？"
                    ):
                        return

                self._place_custom_tiles(num_current)
                if (
                    self.current_custom_type <= self.num_pairs
                    and self.board.get_num_of_empty() > 0
                ):
                    self._update_custom_panel_prompt()
                else:
                    self._finish_custom_generate()
                    return

        if self.current_custom_type <= const.MIN_PAIRS:
            messagebox.showerror("操作无效", f"方块种类数至少为 {const.MIN_PAIRS}")
            return
        if self.current_custom_type <= self.num_pairs:
            if not messagebox.askyesno("确认", "还有未放置的方块种类，确认完成吗？"):
                return

        self._finish_custom_generate()

    def _finish_custom_generate(self):
        """完成自定义棋盘的生成"""
        self.is_custom_mode = False
        self.control_panel.show_initial_layout()

        self.num_pairs = self.board.get_num_of_type()
        self._update_costs_for_new_board()
        self.control_panel.set_solver_active(True)
        self.control_panel.set_cost_editor_active(True)
        self.status_bar.update_text(
            f"已生成 {self.board_size}x{self.board_size} 棋盘，包含 {self.num_pairs} 种共 {(self.board_size**2 - self.board.get_num_of_empty()) // 2} 对方块"
        )

    # --- 代价管理 ---
    def _set_cost_mode(self, mode: str):
        """当更改代价模式时更新代价字典"""
        if not self.board:
            self.status_bar.update_text("请先生成一个棋盘，再设置代价模式")
            self.control_panel.cost_mode_var.set("default")
            return
        self._update_costs_for_new_board()
        self.status_bar.update_text(f"代价模式已设置为 '{mode}'，重新求解以应用更新")

    def _set_cost_mode_custom(self):
        """当点击“自定义代价”时直接打开编辑器"""
        if not self.board:
            messagebox.showerror("操作无效", "没有棋盘，无法编辑代价")
            self.control_panel.set_cost_mode_variable("default")
            return

        self.status_bar.update_text(f"代价模式已设置为 'custom'")
        self._show_cost_editor_window()

    def _save_edited_costs(self, new_costs: Dict[int, int]):
        """保存来自代价编辑窗口的自定义代价"""
        self.tile_costs = new_costs
        self.last_tile_costs = self.tile_costs.copy()
        self.control_panel.set_cost_mode_variable("custom")
        self.status_bar.update_text("自定义代价已保存，重新求解以应用更新")

    def _update_costs_for_new_board(self):
        """根据当前选择的代价模式，为新生成的棋盘更新或创建代价字典"""
        mode = self.control_panel.cost_mode_var.get()
        if mode == "default":
            self.tile_costs = {i: 1 for i in range(1, self.num_pairs + 1)}
        elif mode == "random":
            self.tile_costs = {
                i: random.randint(1, 100) for i in range(1, self.num_pairs + 1)
            }
        elif self.last_tile_costs and len(self.last_tile_costs) == self.num_pairs:
            self.tile_costs = self.last_tile_costs.copy()
        else:
            self.tile_costs = {i: 1 for i in range(1, self.num_pairs + 1)}

    def _show_cost_editor_window(self):
        """打开代价编辑窗口"""
        if not self.board:
            messagebox.showerror("无棋盘", "没有棋盘，无法编辑代价")
            self.control_panel.set_cost_mode_variable("default")
            return

        # 创建并显示窗口，传入保存代价的回调函数
        _CostEditorWindow(
            self,
            self.asset_manager,
            self.tile_costs,
            self.board.get_num_of_type(),
            self._save_edited_costs,
        )

    # --- 求解与控制 ---
    def _solve(self, solve_type: str):
        """根据选择的求解类型，调用相应的算法求解并演示结果"""
        if not self.board:
            messagebox.showerror("错误", "请先生成一个棋盘")
            return

        self.control_panel.set_animation_lock(True)
        self.status_bar.update_text("正在求解中，请稍候...")

        board_to_solve = self.board.clone()

        solver = Solver()
        start = time.time()
        if solve_type == "shortest_path":
            result = solver.solve_shortest_path(board_to_solve)
        elif solve_type == "min_cost":
            for tile in board_to_solve.get_all_tiles():
                tile.cost_per_move = self.tile_costs.get(tile.type_id, 1)
            result = solver.solve_min_cost(board_to_solve)
        elif solve_type == "shortest_path_greedy":
            result = solver.solve_shortest_path_greedy(board_to_solve)
        elif solve_type == "min_cost_greedy":
            for tile in board_to_solve.get_all_tiles():
                tile.cost_per_move = self.tile_costs.get(tile.type_id, 1)
            result = solver.solve_min_cost_greedy(board_to_solve)
        end = time.time()

        if result is None:
            self.status_bar.update_text("无解！此棋盘无法完成消除")
            messagebox.showinfo(
                "无解", f"耗时: {end - start:.4f} 秒\n此棋盘无法完成消除"
            )
            self.control_panel.set_animation_lock(False)
        else:
            steps = result["steps"]
            total_cost = result["total_cost"]
            if solve_type in ["shortest_path", "shortest_path_greedy"]:
                self.status_bar.update_text(
                    f"找到解！序列长: {total_cost}。需演示 {len(steps)} 步，开始演示..."
                )
            else:
                self.status_bar.update_text(
                    f"找到解！总代价: {total_cost}。需演示 {len(steps)} 步，开始演示..."
                )
            self.board_canvas.animate_solution(
                steps,
                on_complete=lambda: self._on_animation_complete(
                    end - start, result, solve_type
                ),
            )

    def _on_animation_complete(self, time: float, result: Dict, mode: str):
        """动画正常结束后，显示结果弹窗并恢复棋盘"""
        self._show_result_popup(time, result, mode)

        self.board_canvas.draw_board(self.board)
        self.control_panel.set_animation_lock(False)
        self.status_bar.update_text("演示完成！可以重新求解或重置")

    def _show_result_popup(self, time: float, result: Dict, mode: str):
        """结果用弹窗显示"""
        steps = result["steps"]
        total_cost = result["total_cost"]

        title = "求解结果"

        if mode == "shortest_path":
            mode_desc = "最短序列\n最短序列长度"
        elif mode == "min_cost":
            mode_desc = "最小代价\n最小代价"
        elif mode == "shortest_path_greedy":
            mode_desc = "贪心最短序列\n序列长度"
        else:
            mode_desc = "贪心最小代价\n总代价"

        path_details = []
        i = 0
        for step in steps:
            action_type = step["action_type"]
            details = step["details"]
            cost = step["cost"]

            if action_type == "DIRECT_ELIMINATE":
                t1, t2 = details["tile1"], details["tile2"]
                path_details.append(
                    f"-  直接消除: 方块 {t1.position} 与 {t2.position} (种类 {t1.type_id})"
                )
            elif action_type == "MOVE_AND_ELIMINATE":
                i += 1
                mt, dest, tt = (
                    details["moving_tile"],
                    details["destination"],
                    details["target_tile"],
                )
                if mode in ["shortest_path", "shortest_path_greedy"]:
                    path_details.append(
                        f"{i}. 移动消除: 方块 {mt.position} -> {dest} 以消除 {tt.position} (种类 {mt.type_id})"
                    )
                else:
                    path_details.append(
                        f"{i}. 移动消除: 方块 {mt.position} -> {dest} 以消除 {tt.position} (种类 {mt.type_id}, 代价: {cost})"
                    )

        path_str = "\n".join(path_details)
        if not path_str:
            path_str = "无需移动，棋盘已清空"

        message = f"模式: {mode_desc}: {total_cost}\n耗时: {time:.4f} 秒\n\n详细步骤:\n{path_str}"

        messagebox.showinfo(title, message)

    # --- 重置 ---
    def _reset(self):
        self.board_canvas.stop_animation()
        self.control_panel.set_animation_lock(False)
        self._reset_board_state()
        self.control_panel.set_board_creation_active(True)
        self.control_panel.set_solver_active(False)
        self.control_panel.set_cost_editor_active(False)
        self.control_panel.show_initial_layout()
        self.status_bar.update_text("已重置，请在左侧面板生成新棋盘")

    # --- 辅助方法 ---
    def _reset_board_state(self):
        self.board_canvas.clear()
        self.board = None
        self.is_custom_mode = False
        self.custom_placements = []
        self.current_custom_type = 0
        self.unique_id_counter = 0
        self.tile_costs = {}

    def _remove_custom_placement(self, to_remove: Tuple[int, int]):
        """
        从自定义放置列表中移除一个指定的坐标
        """
        self.custom_placements.remove(to_remove)

        # 更新显示
        self.board_canvas.highlight_tiles(self.custom_placements)
        self.control_panel.update_custom_panel(
            self._get_custom_prompt(), self.custom_placements
        )
        self.status_bar.update_text(f"已撤销放置在 {to_remove} 的方块")

    def _place_custom_tiles(self, num_current: int):
        """将当前种类的方块放置到棋盘上"""
        tile_cost = self.tile_costs.get(self.current_custom_type, 1)
        for pos in self.custom_placements:
            tile = Tile(
                self.current_custom_type, self.unique_id_counter, pos, tile_cost
            )
            self.board.place_tile(tile)
            self.unique_id_counter += 1

        self.current_custom_type += 1
        self.custom_placements = []

        self.board_canvas.highlight_tiles([])
        self.board_canvas.draw_board(self.board)
        self.status_bar.update_text(
            f"已完成种类 {self.current_custom_type - 1} 的方块 {num_current} 个"
        )

    def _get_custom_prompt(self) -> str:
        return f"请为种类 {self.current_custom_type} 选择偶数个位置:"

    def _update_custom_panel_prompt(self):
        self.control_panel.update_custom_panel(
            self._get_custom_prompt(), self.custom_placements
        )

"""
定义了 ControlPanel 类，它创建并管理游戏界面左侧的所有UI控件：
- 棋盘生成设置（大小、种类数）
- 自定义棋盘的辅助面板
- 代价设置选项
- 求解和游戏控制按钮
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Callable, List, Tuple

import utils.constants as const


class ControlPanel(tk.Frame):
    """
    游戏界面左侧的控制面板，包含所有用户交互控件
    """

    def __init__(self, parent, callbacks: Dict[str, Callable]):
        """
        初始化控制面板

        Args:
            parent: Tkinter 父容器 (通常是 game_frame)
            callbacks (Dict[str, Callable]): 一个包含所有回调函数的字典，
                                             用于将用户操作通知给控制器
        """
        super().__init__(
            parent, width=const.CONTROL_PANEL_WIDTH, bg=const.CONTROL_PANEL_BG
        )
        self.pack_propagate(False)  # 防止框架自动缩放

        self.callbacks = callbacks

        self.scrollable_frame = tk.Frame(self, bg=const.CONTROL_PANEL_BG)
        self.scrollable_frame.pack(fill="both", expand=True)

        # --- 创建并布局所有UI组件 ---
        self._create_game_controls_frame()
        self._create_generation_frame()
        self._create_cost_frame()
        self._create_solver_frame()
        self._create_custom_panel()

        # 初始化为标准布局
        self.show_initial_layout()

    def _create_generation_frame(self):
        """创建“棋盘生成”部分的UI"""
        self.generation_frame = tk.LabelFrame(
            self.scrollable_frame,
            text=" 1. 棋盘生成 ",
            font=const.FONT_LABEL,
            bg=const.CONTROL_PANEL_BG,
            padx=10,
            pady=10,
        )
        self.generation_frame.pack(fill=tk.X, padx=10, pady=10)

        # 棋盘大小
        tk.Label(
            self.generation_frame,
            text="棋盘大小 (NxN):",
            font=const.FONT_LABEL,
            bg=const.CONTROL_PANEL_BG,
        ).grid(row=0, column=0, sticky="w", pady=2)
        self.size_var = tk.StringVar(value=str(const.DEFAULT_BOARD_SIZE))
        self.size_entry = ttk.Entry(self.generation_frame, textvariable=self.size_var, width=10)
        self.size_entry.grid(row=0, column=1, sticky="e", pady=2)

        # 种类/对数
        tk.Label(
            self.generation_frame,
            text="方块种类/对数:",
            font=const.FONT_LABEL,
            bg=const.CONTROL_PANEL_BG,
        ).grid(row=1, column=0, sticky="w", pady=2)
        self.pairs_var = tk.StringVar(value=str(const.DEFAULT_NUM_PAIRS))
        self.pairs_entry = ttk.Entry(self.generation_frame, textvariable=self.pairs_var, width=10)
        self.pairs_entry.grid(row=1, column=1, sticky="e", pady=2)

        # 按钮
        self.random_gen_btn = tk.Button(
            self.generation_frame,
            text="随机生成",
            font=const.FONT_BUTTON,
            command=self._on_random_generate,
        )
        self.random_gen_btn.grid(row=2, column=0, pady=(10, 0), sticky="ew")

        self.custom_gen_btn = tk.Button(
            self.generation_frame,
            text=" 自定义 ",
            font=const.FONT_BUTTON,
            command=self._on_custom_generate,
        )
        self.custom_gen_btn.grid(row=2, column=1, pady=(10, 0), sticky="ew")

    def _create_custom_panel(self):
        """创建用于自定义棋盘放置的辅助面板（默认隐藏）"""
        self.custom_frame = tk.LabelFrame(
            self.scrollable_frame,
            text=" 自定义放置 ",
            font=const.FONT_LABEL,
            bg=const.CONTROL_PANEL_BG,
            padx=10,
            pady=10,
        )

        self.custom_prompt_label = tk.Label(
            self.custom_frame,
            text="",
            font=const.FONT_LABEL,
            bg=const.CONTROL_PANEL_BG,
            fg=const.LABEL_HEADER_COLOR,
        )
        self.custom_prompt_label.pack(fill=tk.X, pady=5)

        # 创建一个有固定高度的容器来放置滚动区域
        scroll_container = tk.Frame(
            self.custom_frame, height=180, bd=1, relief=tk.SUNKEN
        )
        scroll_container.pack(fill=tk.X, expand=False, pady=5)
        scroll_container.pack_propagate(False)

        # 在固定高度的容器内创建 Canvas 和 Scrollbar
        coords_canvas = tk.Canvas(
            scroll_container, bg=const.CONTROL_PANEL_BG, highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            scroll_container, orient="vertical", command=coords_canvas.yview
        )
        coords_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        coords_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建容纳内容的、可以滚动的 Frame
        scrollable_content_frame = tk.Frame(coords_canvas, bg=const.CONTROL_PANEL_BG)
        coords_canvas.create_window(
            (0, 0), window=scrollable_content_frame, anchor="nw"
        )

        scrollable_content_frame.bind(
            "<Configure>",
            lambda e: coords_canvas.configure(scrollregion=coords_canvas.bbox("all")),
        )

        self.custom_coords_label = tk.Label(
            scrollable_content_frame,
            text="已选: []",
            font=const.FONT_LABEL,
            bg=const.CONTROL_PANEL_BG,
            justify=tk.LEFT,
            wraplength=const.CONTROL_PANEL_WIDTH - 70,
        )
        self.custom_coords_label.pack(fill=tk.X, pady=5)

        def _on_mousewheel(event):
            if event.num == 4:
                delta = -1
            elif event.num == 5:
                delta = 1
            else:
                delta = -1 * int(event.delta / 120)
            coords_canvas.yview_scroll(delta, "units")

        # 将事件绑定到所有相关组件上
        for widget in [
            coords_canvas,
            scrollable_content_frame,
            self.custom_coords_label,
        ]:
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel)
            widget.bind("<Button-5>", _on_mousewheel)

        # 创建一个容器 Frame 用于 grid 布局
        btn_container = tk.Frame(self.custom_frame, bg=const.CONTROL_PANEL_BG)
        btn_container.pack(fill=tk.X, pady=5)
        btn_container.columnconfigure(0, weight=1)
        btn_container.columnconfigure(1, weight=1)

        self.undo_btn = tk.Button(
            btn_container,
            text="撤销上步",
            command=lambda: self.callbacks["undo_last_placement"](),
        )
        self.batch_op_btn = tk.Button(
            btn_container,
            text="批量操作",
            command=lambda: self.callbacks["batch_placement"](),
        )
        self.finish_type_btn = tk.Button(
            btn_container,
            text="完成当前种类",
            command=lambda: self.callbacks["finish_current_type"](),
        )
        self.finish_all_btn = tk.Button(
            btn_container,
            text="完成所有放置",
            command=lambda: self.callbacks["finish_all_custom"](),
        )

        self.undo_btn.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=(0, 4))
        self.batch_op_btn.grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=(0, 4))
        self.finish_type_btn.grid(row=1, column=0, sticky="ew", padx=(0, 2))
        self.finish_all_btn.grid(row=1, column=1, sticky="ew", padx=(2, 0))

    def _create_cost_frame(self):
        """创建“代价设置”部分的UI"""
        self.cost_frame = tk.LabelFrame(
            self.scrollable_frame,
            text=" 2. 代价设置 ",
            font=const.FONT_LABEL,
            bg=const.CONTROL_PANEL_BG,
            padx=10,
            pady=10,
        )
        self.cost_frame.pack(fill=tk.X, padx=10, pady=10)

        self.cost_mode_var = tk.StringVar(value="default")

        # 将按钮放在一个内部Frame中，以便布局
        rb_frame = tk.Frame(self.cost_frame, bg=const.CONTROL_PANEL_BG)
        rb_frame.pack(fill=tk.X)

        ttk.Radiobutton(
            rb_frame,
            text="默认代价 (全为1)",
            variable=self.cost_mode_var,
            value="default",
            command=lambda: self.callbacks["set_cost_mode"]("default"),
        ).pack(anchor="w")
        ttk.Radiobutton(
            rb_frame,
            text="随机代价",
            variable=self.cost_mode_var,
            value="random",
            command=lambda: self.callbacks["set_cost_mode"]("random"),
        ).pack(anchor="w")
        ttk.Radiobutton(
            rb_frame,
            text="自定义代价",
            variable=self.cost_mode_var,
            value="custom",
            command=lambda: self.callbacks["set_cost_mode_custom"](),
        ).pack(anchor="w")

        # 查看/编辑代价按钮
        self.edit_cost_btn = tk.Button(
            self.cost_frame,
            text="查看/编辑代价",
            font=const.FONT_BUTTON,
            command=lambda: self.callbacks["show_cost_editor"](),
        )
        self.edit_cost_btn.pack(fill=tk.X, pady=(10, 0))

        # 初始时禁用编辑按钮
        self.set_cost_editor_active(False)

    def _create_solver_frame(self):
        """创建“求解”部分的UI"""
        self.solver_frame = tk.LabelFrame(
            self.scrollable_frame,
            text=" 3. 求解 ",
            font=const.FONT_LABEL,
            bg=const.CONTROL_PANEL_BG,
            padx=10,
            pady=10,
        )
        self.solver_frame.pack(fill=tk.X, padx=10, pady=10)

        self.solve_dist_btn = tk.Button(
            self.solver_frame,
            text="最短序列求解",
            font=const.FONT_BUTTON,
            command=lambda: self.callbacks["solve"]("shortest_path"),
        )
        self.solve_dist_btn.pack(fill=tk.X, pady=5)

        self.solve_cost_btn = tk.Button(
            self.solver_frame,
            text="最小代价求解",
            font=const.FONT_BUTTON,
            command=lambda: self.callbacks["solve"]("min_cost"),
        )
        self.solve_cost_btn.pack(fill=tk.X, pady=5)

        self.solve_dist_greedy_btn = tk.Button(
            self.solver_frame,
            text="最短序列求解 (贪心)",
            font=const.FONT_BUTTON,
            command=lambda: self.callbacks["solve"]("shortest_path_greedy"),
        )
        self.solve_dist_greedy_btn.pack(fill=tk.X, pady=5)

        self.solve_cost_greedy_btn = tk.Button(
            self.solver_frame,
            text="最小代价求解 (贪心)",
            font=const.FONT_BUTTON,
            command=lambda: self.callbacks["solve"]("min_cost_greedy"),
        )
        self.solve_cost_greedy_btn.pack(fill=tk.X, pady=5)

        # 初始时禁用求解按钮
        self.set_solver_active(False)

    def _create_game_controls_frame(self):
        """创建“游戏控制”部分的UI"""
        self.controls_frame = tk.LabelFrame(
            self.scrollable_frame,
            text=" 游戏控制 ",
            font=const.FONT_LABEL,
            bg=const.CONTROL_PANEL_BG,
            padx=10,
            pady=10,
        )
        self.controls_frame.pack(fill=tk.X, padx=10, pady=(10, 20))

        tk.Button(
            self.controls_frame,
            text="重置 / 新局",
            font=const.FONT_BUTTON,
            command=lambda: self.callbacks["reset"](),
        ).pack(fill=tk.X, pady=5)
        tk.Button(
            self.controls_frame,
            text="返回主菜单",
            font=const.FONT_BUTTON,
            command=lambda: self.callbacks["back_to_menu"](),
        ).pack(fill=tk.X, pady=5)

    # --- 事件处理器 ---
    def _on_random_generate(self):
        size_str = self.size_var.get()
        pairs_str = self.pairs_var.get()
        self.callbacks["random_generate"](size_str, pairs_str)

    def _on_custom_generate(self):
        size_str = self.size_var.get()
        pairs_str = self.pairs_var.get()
        self.callbacks["custom_generate"](size_str, pairs_str)

    # --- 公共方法 (由外部控制器调用) ---
    def show_initial_layout(self):
        """显示标准的初始布局"""
        self.controls_frame.pack(fill=tk.X, padx=10, pady=(10, 20))
        self.generation_frame.pack(fill=tk.X, padx=10, pady=10)
        self.cost_frame.pack(fill=tk.X, padx=10, pady=10)
        self.solver_frame.pack(fill=tk.X, padx=10, pady=10)
        self.custom_frame.pack_forget()  # 隐藏自定义面板

    def show_custom_placement_layout(self):
        """显示自定义放置时的简化布局"""
        self.generation_frame.pack_forget()
        self.cost_frame.pack_forget()
        self.solver_frame.pack_forget()

        self.controls_frame.pack(fill=tk.X, padx=10, pady=(10, 20))
        self.custom_frame.pack(fill=tk.X, padx=10, pady=10)

    def update_custom_panel(self, prompt: str, coords: List[Tuple[int, int]]):
        self.custom_prompt_label.config(text=prompt)
        self.custom_coords_label.config(text=f"已选: {coords}")

    def set_board_creation_active(self, is_active: bool):
        """启用或禁用与棋盘创建相关的控件"""
        state = tk.NORMAL if is_active else tk.DISABLED
        self.size_entry.config(state=state)
        self.pairs_entry.config(state=state)
        self.random_gen_btn.config(state=state)
        self.custom_gen_btn.config(state=state)

    def set_cost_editor_active(self, is_active: bool):
        """启用或禁用代价编辑器按钮"""
        state = tk.NORMAL if is_active else tk.DISABLED
        self.edit_cost_btn.config(state=state)

    def set_cost_mode_variable(self, mode: str):
        """设置代价模式的选中项"""
        self.cost_mode_var.set(mode)

    def set_solver_active(self, is_active: bool):
        """启用或禁用求解器按钮"""
        state = tk.NORMAL if is_active else tk.DISABLED
        self.solve_dist_btn.config(state=state)
        self.solve_cost_btn.config(state=state)
        self.solve_dist_greedy_btn.config(state=state)
        self.solve_cost_greedy_btn.config(state=state)

    def set_animation_lock(self, is_locked: bool):
        """
        根据动画播放状态，锁定或解锁大部分UI控件
        """
        self.set_board_creation_active(not is_locked)
        self.set_cost_editor_active(not is_locked)
        self.set_solver_active(not is_locked)

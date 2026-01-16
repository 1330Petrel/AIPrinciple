"""
定义了 MainMenuFrame 类，它创建并管理主菜单的所有UI元素，
包括标题和功能按钮（开始游戏、游戏说明、退出游戏）
通过回调函数与主窗口进行通信
"""

import tkinter as tk
from tkinter import messagebox
from typing import Callable

import utils.constants as const


class MainMenuFrame(tk.Frame):
    """
    应用程序的主菜单界面
    """

    def __init__(
        self, parent, start_game_callback: Callable[[], None], exit_game_callback: Callable[[], None]
    ):
        """
        初始化主菜单框架

        Args:
            parent: Tkinter 父容器 (通常是主窗口)
            start_game_callback (callable): 当点击“开始游戏”按钮时要调用的函数
            exit_game_callback (callable): 当点击“退出游戏”按钮时要调用的函数
        """
        super().__init__(parent, bg=const.WINDOW_BG)

        self.start_game_callback = start_game_callback
        self.exit_game_callback = exit_game_callback

        self._setup_widgets()

    def _setup_widgets(self):
        """
        创建并布局此框架中的所有小部件
        """
        # 使用一个中央框架来容纳所有控件，并使其在窗口中居中
        center_frame = tk.Frame(self, bg=const.WINDOW_BG)
        center_frame.pack(expand=True)

        # 1. 应用程序标题
        title_label = tk.Label(
            center_frame,
            text=const.WINDOW_TITLE,
            font=const.FONT_TITLE,
            bg=const.WINDOW_BG,
            fg=const.LABEL_HEADER_COLOR,
        )
        title_label.pack(pady=(0, 60))  # 标题下方留出较大间距

        # 2. 开始游戏按钮
        start_button = tk.Button(
            center_frame,
            text="开始游戏",
            font=const.FONT_BUTTON,
            bg=const.BUTTON_BG,
            fg=const.BUTTON_FG,
            activebackground=const.BUTTON_ACTIVE_BG,
            activeforeground=const.BUTTON_FG,
            width=20,
            pady=10,
            relief=tk.FLAT,
            command=self.start_game_callback,  # 绑定回调函数
        )
        start_button.pack(pady=10)

        # 3. 游戏说明按钮
        instructions_button = tk.Button(
            center_frame,
            text="使用说明",
            font=const.FONT_BUTTON,
            bg=const.BUTTON_BG,
            fg=const.BUTTON_FG,
            activebackground=const.BUTTON_ACTIVE_BG,
            activeforeground=const.BUTTON_FG,
            width=20,
            pady=10,
            relief=tk.FLAT,
            command=self._show_instructions,
        )
        instructions_button.pack(pady=10)

        # 4. 退出游戏按钮
        exit_button = tk.Button(
            center_frame,
            text="退出游戏",
            font=const.FONT_BUTTON,
            bg=const.BUTTON_BG,
            fg=const.BUTTON_FG,
            activebackground=const.BUTTON_ACTIVE_BG,
            activeforeground=const.BUTTON_FG,
            width=20,
            pady=10,
            relief=tk.FLAT,
            command=self.exit_game_callback,  # 绑定回调函数
        )
        exit_button.pack(pady=10)

    def _show_instructions(self):
        """
        显示一个包含使用说明的弹窗
        """
        instructions_title = "使用说明"
        instructions_text = """
欢迎使用“砖了个砖求解器”！

游戏目标:
通过滑动消除棋盘上所有的方块。

基本规则:
1. 每次选择一个方块，可以向上下左右四个方向滑动。
2. 只有当两个相同的方块之间没有其他任何方块阻挡时，才能成功完成移动并被消除。

求解器功能:
1. 生成棋盘：可以随机生成，或自定义棋盘大小和方块分布 (点击棋盘)。
2. 代价设置：可以为不同种类的方块设置不同的滑动代价。
3. 求解：
   - 最短序列：找到最短的滑动序列。
   - 最小代价：找到使总代价（移动 × 代价）最小的滑动路径。
   - 贪心算法：提供基于贪心策略的快速求解选项。
        """
        # 使用 tkinter 的 messagebox 显示信息
        messagebox.showinfo(instructions_title, instructions_text.strip())

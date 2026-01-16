"""
该文件定义了 StatusBar 类，用于在游戏界面的顶部显示文本信息，
如操作提示、求解状态或结果
"""

import tkinter as tk

import utils.constants as const


class StatusBar(tk.Frame):
    """
    显示在窗口底部的状态栏
    """

    def __init__(self, parent):
        """
        初始化状态栏框架

        Args:
            parent: Tkinter 父容器 (通常是 game_frame)
        """
        # 初始化父框架，并设置背景色和高度
        super().__init__(parent, bg=const.STATUS_BAR_BG, height=const.STATUS_BAR_HEIGHT)

        # 防止框架因内部小部件的大小而收缩
        self.pack_propagate(False)

        # 用于显示文本的标签
        self.status_label = tk.Label(
            self,
            text="欢迎来到游戏！请在左侧面板生成棋盘",
            font=const.FONT_STATUS,
            bg=const.STATUS_BAR_BG,
            fg=const.STATUS_TEXT_COLOR,
            anchor="w",  # 文本左对齐
        )
        self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

    def update_text(self, message: str):
        """
        更新状态栏中显示的文本

        Args:
            message (str): 要显示的新消息
        """
        self.status_label.config(text=message)
        # 强制 Tkinter 立即更新界面，这在执行长任务前更新文本时很有用
        self.update_idletasks()

"""
定义了 MainWindow 类，作为整个应用程序的主窗口
1. 创建和管理主窗口的基本属性
2. 初始化并持有所有主要的界面框架
3. 处理不同界面之间的切换逻辑
"""

import tkinter as tk

from gui.main_menu_frame import MainMenuFrame
from gui.game.game_frame import GameFrame
from utils.asset_manager import AssetManager
import utils.constants as const


class MainWindow(tk.Tk):
    """
    应用程序的主窗口，负责管理和切换不同的界面框架
    """

    def __init__(self):
        super().__init__()

        # 1. 首先初始化资源管理器
        self.asset_manager = AssetManager()

        # 2. 配置主窗口
        self.title(const.WINDOW_TITLE)
        self.geometry(f"{const.WINDOW_WIDTH}x{const.WINDOW_HEIGHT}")
        self.resizable(False, False)  # 禁止调整窗口大小
        self.configure(bg=const.WINDOW_BG)
        self._center_window()

        # 3. 创建一个容器框架，用于容纳所有其他界面
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # 4. 初始化所有界面框架，并将它们放入一个字典中方便管理
        self.frames = {}

        # 遍历需要创建的框架类
        for F in (MainMenuFrame, GameFrame):
            frame_name = F.__name__

            # 根据框架类型，传递不同的参数进行实例化
            if frame_name == "MainMenuFrame":
                frame = F(
                    container,
                    start_game_callback=self._show_game_frame,
                    exit_game_callback=self._exit_game,
                )
            elif frame_name == "GameFrame":
                frame = F(
                    container,
                    back_to_menu_callback=self._show_main_menu,
                    asset_manager=self.asset_manager,
                )

            self.frames[frame_name] = frame
            # 将所有框架都放置在同一个网格位置，它们会重叠在一起
            frame.grid(row=0, column=0, sticky="nsew")

        # 5. 默认显示主菜单界面
        self._show_frame("MainMenuFrame")

    def _center_window(self):
        """将主窗口居中于屏幕"""
        self.update_idletasks()  # 确保窗口尺寸已更新
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _show_frame(self, frame_name: str):
        """
        将指定的框架提升到最顶层，使其可见

        Args:
            frame_name (str): 要显示的框架的类名
        """
        frame = self.frames[frame_name]
        frame.tkraise()

    # --- 回调函数 (用于传递给子框架) ---

    def _show_main_menu(self):
        """切换到主菜单界面"""
        # 在返回主菜单时，重置游戏界面的状态，以确保下次进入是全新的
        game_frame = self.frames["GameFrame"]
        if game_frame:
            game_frame._reset()

        self._show_frame("MainMenuFrame")

    def _show_game_frame(self):
        """切换到游戏界面"""
        self._show_frame("GameFrame")

    def _exit_game(self):
        """关闭应用程序"""
        self.destroy()

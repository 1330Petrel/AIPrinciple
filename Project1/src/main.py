"""
1. 创建主窗口的一个实例
2. 启动 Tkinter 的主事件循环，开始运行并响应用户事件
"""

from gui.main_window import MainWindow

if __name__ == "__main__":
    # 实例化主窗口
    app = MainWindow()

    # 启动 Tkinter 的事件循环
    app.mainloop()

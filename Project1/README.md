# 项目一

基于 A* 搜索算法的“砖了个砖”游戏自动求解器

## 项目简介

在“砖了个砖”游戏中，棋盘是一个二维矩阵，棋盘上每种图案成对出现。玩家可以选择某个图案向上下左右任一方向滑动（不限格子数），如果两个相同图案之间没有其他图案阻挡，它们将被消除。游戏目标是通过合理的滑动顺序，将棋盘上的所有图案消除。

本项目实现一个求解器，能够自动找到游戏的最优解。采用**A* 搜索算法**作为核心求解引擎，提供多种求解模式（最优最短路径、最优最小代价、贪心等），并配备可视化的 GUI 界面以直观地查看求解过程。

## 项目结构

```plaintext
Project1/
├── assets/
│   └── images/                      # 游戏图片资源
│
├── src/
│   ├── main.py                      # 应用程序入口
│   │
│   ├── core/                        # 核心算法模块
│   │   ├── algorithm.py             # A* 求解算法实现
│   │   ├── board.py                 # 游戏棋盘状态管理
│   │   └── tile.py                  # 游戏方块类定义
│   │
│   ├── gui/                         # 用户界面模块
│   │   ├── main_window.py           # 主窗口框架
│   │   ├── main_menu_frame.py       # 主菜单界面
│   │   └── game/
│   │       ├── game_frame.py        # 游戏界面主框架
│   │       └── components/          # GUI 组件
│   │           ├── board_canvas.py  # 棋盘画布渲染
│   │           ├── control_panel.py # 控制面板
│   │           ├── cost_editor.py   # 代价编辑器
│   │           ├── batch_placement.py # 批量放置工具
│   │           └── status_bar.py    # 状态栏
│   │
│   └── utils/                       # 工具模块
│       ├── asset_manager.py         # 资源管理器
│       └── constants.py             # 全局常量配置
│
└── report.pdf                       # 项目技术报告
```

## 运行命令

```bash
python src/main.py
```

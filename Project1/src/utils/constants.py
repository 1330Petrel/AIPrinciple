"""
项目常量模块
"""

# -----------------
# 1. 窗口与布局尺寸
# -----------------
WINDOW_TITLE = "砖了个砖"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 840

# 左侧控制面板宽度
CONTROL_PANEL_WIDTH = 280

# 底部状态栏高度
STATUS_BAR_HEIGHT = 40

# -----------------
# 2. 棋盘与方块尺寸
# -----------------
# 棋盘画布的内边距
BOARD_PADDING = 40

# 每个方块的尺寸
TILE_SIZE = 60

# 方块之间的间隙
TILE_GAP = 4

# -----------------
# 3. 颜色定义
# -----------------
# 背景色
WINDOW_BG = "#F0F0F0"  # 窗口主背景色 (浅灰色)
CONTROL_PANEL_BG = "#EAEAEA"  # 控制面板背景色
STATUS_BAR_BG = "#333333"  # 状态栏背景色 (深灰色)
BOARD_BG = "#DCDCDC"  # 棋盘画布背景色 (亮灰色)

# 组件颜色
BUTTON_BG = "#4CAF50"  # 按钮背景色 (绿色)
BUTTON_FG = "#FFFFFF"  # 按钮前景色 (白色)
BUTTON_ACTIVE_BG = "#45a049"  # 按钮激活背景色

# 文本颜色
TEXT_COLOR = "#111111"  # 主要文本颜色 (近黑色)
LABEL_HEADER_COLOR = "#005f73"  # 标签标题颜色 (深青色)
STATUS_TEXT_COLOR = "#FFFFFF"  # 状态栏文本颜色 (白色)

# 棋盘与方块颜色
GRID_LINE_COLOR = "#BBBBBB"  # 棋盘网格线颜色
TILE_TEXT_COLOR = "#000000"  # 方块内数字/文字颜色
TILE_HIGHLIGHT_COLOR = "red"  # 方块选中时的高亮边框颜色

# 当图片资源不够用时，备选的10种颜色块
TILE_COLORS = [
    "#ffadad",  # Light Pink
    "#ffd6a5",  # Light Orange
    "#fdffb6",  # Light Yellow
    "#caffbf",  # Light Green
    "#9bf6ff",  # Light Cyan
    "#a0c4ff",  # Light Blue
    "#bdb2ff",  # Light Purple
    "#ffc6ff",  # Light Magenta
    "#e0e1dd",  # Light Grey
    "#ffc8dd",  # Light Rose
]

# -----------------
# 4. 字体定义
# -----------------
# (字体家族, 字号, 样式)
FONT_FAMILY = "Microsoft YaHei UI"
FONT_TITLE = (FONT_FAMILY, 16, "bold")
FONT_BUTTON = (FONT_FAMILY, 12, "bold")
FONT_LABEL = (FONT_FAMILY, 11, "normal")
FONT_STATUS = (FONT_FAMILY, 10, "normal")
FONT_TILE_NUMBER = (FONT_FAMILY, 20, "bold")  # 方块内的数字字体

# -----------------
# 5. 游戏逻辑与资源常量
# -----------------
# 资源限制
MAX_TILE_TYPES_WITH_IMAGE = 38  # 有对应图片的方块种类数
MAX_TILE_TYPES_WITH_COLOR = 10  # 颜色块方块种类数

# 棋盘参数限制
MIN_BOARD_SIZE = 4
MAX_BOARD_SIZE = 12
MIN_PAIRS = 1  # 棋盘上最少需要的方块对数

# 默认棋盘参数
DEFAULT_BOARD_SIZE = 5
DEFAULT_NUM_PAIRS = 3

# 动画演示速度 (单位: 毫秒)
ANIMATION_DELAY_MS = 1000
# 滑动动画的帧数，数值越高滑动越平滑
ANIMATION_MOVE_FRAMES = 15

# -----------------
# 6. 资源文件路径
# -----------------
ASSET_DIR = "assets"
IMAGE_DIR = f"{ASSET_DIR}/images"

"""
该文件定义了 Board 类，用于表示和管理游戏棋盘的状态
它包含一个二维网格来存放 Tile 对象，并提供了操作棋盘所需的核心方法，
如放置/移除方块、检查路径、克隆状态等
"""

from typing import List, Tuple, Optional, Dict

from core.tile import Tile


class Board:
    """
    表示游戏棋盘，管理所有方块的状态和位置

    Attributes:
        size (int): 棋盘的边长
        grid (List[List[Optional[Tile]]]): 表示棋盘状态的二维列表，
                                           每个单元格可以是一个 Tile 对象或 None (表示为空)
        tiles_by_type (Dict[int, List[Tile]]): 按种类ID分组的方块字典，用于快速查找配对；
                                                键为 type_id, 值为该种类的 Tile 对象列表
    """

    def __init__(self, size: int):
        """
        初始化一个指定大小的空棋盘

        Args:
            size (int): 棋盘的边长

        Raises:
            ValueError: 如果 size 不是一个正整数
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("棋盘大小必须是一个正整数")

        self.size: int = size
        self.grid: List[List[Optional[Tile]]] = [
            [None for _ in range(size)] for _ in range(size)
        ]
        self.tiles_by_type: Dict[int, List[Tile]] = {}

    def place_tile(self, tile: Tile):
        """
        在棋盘上的指定位置放置一个方块

        Args:
            tile (Tile): 要放置的方块对象

        Raises:
            IndexError: 如果方块的位置超出了棋盘边界
            ValueError: 如果目标位置已经被其他方块占据
        """
        row, col = tile.position
        if not self.is_position_valid((row, col)):
            raise IndexError(f"位置 ({row}, {col}) 超出棋盘边界")
        if self.grid[row][col] is not None:
            raise ValueError(f"位置 ({row}, {col}) 已被占据")

        self.grid[row][col] = tile

        # 将方块添加到按种类分组的字典中
        if tile.type_id not in self.tiles_by_type:
            self.tiles_by_type[tile.type_id] = []
        self.tiles_by_type[tile.type_id].append(tile)

    def remove_tile(self, tile: Tile):
        """
        从棋盘上移除一个方块

        Args:
            tile (Tile): 要移除的方块对象

        Raises:
            ValueError: 如果在指定位置找不到要移除的方块
        """
        row, col = tile.position
        if self.get_tile_at((row, col)) != tile:
            raise ValueError(f"在位置 ({row}, {col}) 未找到指定的方块实例 {tile}")

        self.grid[row][col] = None

        # 从按种类分组的字典中移除方块
        if tile.type_id in self.tiles_by_type:
            # 安全地移除指定的方块实例
            self.tiles_by_type[tile.type_id] = [
                t
                for t in self.tiles_by_type[tile.type_id]
                if t.unique_id != tile.unique_id
            ]
            # 如果该种类的方块已全部移除，则从字典中删除这个键
            if not self.tiles_by_type[tile.type_id]:
                del self.tiles_by_type[tile.type_id]

    def get_tile_at(self, position: Tuple[int, int]) -> Optional[Tile]:
        """根据坐标获取方块对象"""
        row, col = position
        if self.is_position_valid(position):
            return self.grid[row][col]
        return None

    def is_position_valid(self, position: Tuple[int, int]) -> bool:
        """检查给定坐标是否在棋盘边界内"""
        row, col = position
        return 0 <= row < self.size and 0 <= col < self.size

    def is_path_clear(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """
        检查两个位置之间的直线路径（水平或垂直）是否没有障碍物
        """
        r1, c1 = pos1
        r2, c2 = pos2

        # 路径必须是严格的水平或垂直线
        if r1 != r2 and c1 != c2:
            return False

        # 检查水平路径
        if r1 == r2:
            start_col, end_col = min(c1, c2), max(c1, c2)
            for col in range(start_col + 1, end_col):
                if self.grid[r1][col] is not None:
                    return False
        # 检查垂直路径
        else:  # c1 == c2
            start_row, end_row = min(r1, r2), max(r1, r2)
            for row in range(start_row + 1, end_row):
                if self.grid[row][c1] is not None:
                    return False

        return True

    def get_num_of_type(self) -> int:
        """返回当前棋盘上方块的种类数"""
        return len(self.tiles_by_type)

    def get_num_of_tiles(self) -> int:
        """返回当前棋盘上方块的总数"""
        return sum(len(tile_list) for tile_list in self.tiles_by_type.values())

    def get_num_of_empty(self) -> int:
        """返回当前棋盘上空位置的总数"""
        return self.size * self.size - sum(
            len(tile_list) for tile_list in self.tiles_by_type.values()
        )

    def get_empty_squares(self) -> List[Tuple[int, int]]:
        """返回当前棋盘上所有空位置的坐标列表"""
        empty_squares = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] is None:
                    empty_squares.append((r, c))
        return empty_squares

    def get_all_tiles(self) -> List[Tile]:
        """返回当前棋盘上所有方块对象的列表"""
        all_tiles = []
        for tile_list in self.tiles_by_type.values():
            all_tiles.extend(tile_list)
        return all_tiles

    def clone(self) -> "Board":
        """
        创建并返回当前棋盘状态的一个深层副本
        """
        new_board = Board(self.size)
        for tile in self.get_all_tiles():
            # 为每个方块创建新的 Tile 实例，以避免对象引用问题
            new_tile = Tile(
                type_id=tile.type_id,
                unique_id=tile.unique_id,
                position=tile.position,
                cost_per_move=tile.cost_per_move,
            )
            new_board.place_tile(new_tile)
        return new_board

    def to_hashable(self) -> tuple:
        """
        将当前棋盘状态转换为一个不可变的、可哈希的元组
        使棋盘状态可以被添加到集合中，用于在搜索算法中记录已访问的节点
        """
        hashable_grid = []
        for r in range(self.size):
            # 用方块的 type_id (或0代表空) 来代表一行
            row_tuple = tuple(
                self.grid[r][c].type_id if self.grid[r][c] else 0
                for c in range(self.size)
            )
            hashable_grid.append(row_tuple)
        return tuple(hashable_grid)

    def __repr__(self) -> str:
        """
        返回棋盘的文本表示形式，方便调试
        """
        if self.size == 0:
            return "空棋盘"

        col_headers = " ".join([f"{i:^3}" for i in range(self.size)])
        header = f"   {col_headers}\n"
        board_str = header

        for i, row in enumerate(self.grid):
            row_str = [f"{tile.type_id:^3}" if tile else " . " for tile in row]
            board_str += f"{i:<2} {''.join(row_str)}\n"

        return board_str.strip()

    @staticmethod
    def create_random_board(
        size: int, num_pairs: int, costs: Optional[Dict[int, int]] = None
    ) -> "Board":
        """
        创建一个具有随机布局的新棋盘

        随机地在棋盘上放置指定数量的方块对，不保证生成的棋盘一定有解

        Args:
            size (int): 棋盘的边长
            num_pairs (int): 要生成的方块对的数量
            costs (Optional[Dict[int, int]], optional):
                一个字典，映射 type_id 到其移动代价
                如果未提供或某个 type_id 不在字典中，则代价默认为 1

        Returns:
            Board: 一个包含了随机方块布局的新 Board 实例

        Raises:
            ValueError: 如果棋盘空间不足以容纳所有方块
        """
        import random
        from collections import Counter
        import utils.constants as const

        if size * size < num_pairs * 2:
            raise ValueError(
                f"棋盘空间不足 ({size}x{size})，无法容纳 {num_pairs} 对（共 {num_pairs*2}个）方块"
            )

        # 1. 随机抽样阶段，直到生成的唯一种类数在允许的范围内
        while True:
            pool_size = num_pairs + const.DEFAULT_NUM_PAIRS  # 增加一些余量
            raw_samples = [random.randint(1, pool_size) for _ in range(num_pairs)]

            # 统计生成的唯一种类数
            raw_counts = Counter(raw_samples)

            # 检查唯一种类数是否满足最小要求
            if len(raw_counts) >= const.MIN_PAIRS:
                break

        # 2. 统计与ID重映射阶段
        # 创建映射关系：原始ID -> 新的连续ID (1, 2, 3...)
        id_map = {
            raw_id: new_id for new_id, raw_id in enumerate(sorted(raw_counts.keys()), 1)
        }

        # 生成最终要放置的配对列表
        pair_type_ids_to_place = []
        for raw_id, count in raw_counts.items():
            new_type_id = id_map[raw_id]
            # 将该种类的 ID 添加 'count' 次到列表中
            pair_type_ids_to_place.extend([new_type_id] * count)

        # 3. 准备棋盘和位置
        board = Board(size)
        all_positions = [(r, c) for r in range(size) for c in range(size)]
        random.shuffle(all_positions)

        # 4. 逐对放置方块
        unique_id_counter = 0
        for type_id in pair_type_ids_to_place:
            # 获取该种类方块的代价
            tile_cost = costs.get(type_id, 1) if costs else 1

            # 弹出两个随机位置用于放置一对新方块
            try:
                pos1 = all_positions.pop()
                pos2 = all_positions.pop()
            except IndexError:
                raise RuntimeError("从位置列表中获取坐标时出错")

            # 创建并放置第一个方块
            tile1 = Tile(
                type_id=type_id,
                unique_id=unique_id_counter,
                position=pos1,
                cost_per_move=tile_cost,
            )
            board.place_tile(tile1)
            unique_id_counter += 1

            # 创建并放置第二个方块
            tile2 = Tile(
                type_id=type_id,
                unique_id=unique_id_counter,
                position=pos2,
                cost_per_move=tile_cost,
            )
            board.place_tile(tile2)
            unique_id_counter += 1

        return board

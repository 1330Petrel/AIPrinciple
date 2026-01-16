"""
定义 Tile 类，用于表示棋盘上的一个方块
每个 Tile 对象都包含了其种类、唯一标识、在棋盘上的位置以及移动代价等信息
"""

from typing import Tuple


class Tile:
    """
    表示棋盘上的一个方块

    Attributes:
        type_id (int): 方块的种类ID。相同ID的方块可以配对消除
        unique_id (int): 方块的唯一ID。用于区分种类相同但实例不同的方块
        position (Tuple[int, int]): 方块在棋盘上的坐标，格式为 (row, col)
        cost_per_move (int): 该种类方块每移动一个格子的代价
    """

    def __init__(
        self,
        type_id: int,
        unique_id: int,
        position: Tuple[int, int],
        cost_per_move: int = 1,
    ):
        """
        初始化一个 Tile 对象

        Args:
            type_id (int): 方块的种类ID
            unique_id (int): 此方块的唯一标识符
            position (Tuple[int, int]): 方块的初始位置 (row, col)
            cost_per_grid (int, optional): 移动此方块一个格子的代价，默认为 1
        """
        if not isinstance(type_id, int) or type_id <= 0:
            raise ValueError("type_id 必须是一个正整数")
        if not isinstance(unique_id, int) or unique_id < 0:
            raise ValueError("unique_id 必须是一个非负整数")

        self.type_id: int = type_id
        self.unique_id: int = unique_id
        self.position: Tuple[int, int] = position
        self.cost_per_move: int = cost_per_move

    def __repr__(self) -> str:
        """
        返回一个清晰表示 Tile 对象状态的字符串
        """
        return (
            f"Tile(Type: {self.type_id}, "
            f"UID: {self.unique_id}, "
            f"Pos: {self.position}, "
            f"Cost: {self.cost_per_move})"
        )

    def __eq__(self, other) -> bool:
        """
        判断两个 Tile 对象是否相等
        只有当它们的 unique_id 相同时，才认为是同一个方块实例
        """
        if not isinstance(other, Tile):
            return NotImplemented
        return self.unique_id == other.unique_id

    def __hash__(self) -> int:
        """
        返回 Tile 对象的哈希值，使其可以被放入集合或作为字典的键
        """
        return hash(self.unique_id)

    def update_position(self, new_position: Tuple[int, int]):
        """
        更新方块的位置

        Args:
            new_position (Tuple[int, int]): 新的位置坐标 (row, col)
        """
        self.position = new_position

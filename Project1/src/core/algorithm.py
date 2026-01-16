"""
求解算法模块，实现 A* 搜索算法

提供了四种求解模式：
1. 最优最短路径: 保证找到移动次数最少的解
2. 最优最小代价: 保证找到总移动代价最小的解
3. 贪心最短路径: 快速寻找一个解，但不保证移动次数最少
4. 贪心最小代价: 快速寻找一个解，但不保证总代价最小

核心设计思想：
- 使用 A* 算法作为基础搜索框架
- 通过传入不同的“启发函数”和“后继状态生成器”来实现不同的求解模式
- 将路径查找与动画数据包的构建分离，以提升搜索性能
- 包含了针对特定模式的死锁检测剪枝，以优化搜索效率
"""

import heapq
from typing import List, Tuple, Dict, Optional, Any, Generator, Callable
from itertools import combinations, count
from collections import defaultdict

from core.tile import Tile
from core.board import Board

# 轻量级动作：在搜索时用于记录路径，避免存储完整的 Tile 对象以节省内存
# 格式: (动作类型字符串, 包含UID和坐标等最小信息的元组)
Action = Tuple[str, tuple]
# 动画步骤：在搜索结束后，根据轻量级动作重建的、包含完整信息的数据包，供UI使用
AnimationStep = Dict[str, Any]

# 后继状态生成器"函数签名：输入一个 Board，产出所有可能的 (后继Board, 动作, 代价)
SuccessorGenerator = Callable[[Board], Generator[Tuple[Board, Action, int], None, None]]
# 启发函数签名：输入一个 Board，返回一个估计的未来成本
HeuristicFunction = Callable[[Board], int]


class Solver:

    # --- 1. 公开的 API 方法 ---

    # --- 1a. 最优解求解器 (使用可接受的启发函数) ---

    def solve_shortest_path(self, initial_board: Board) -> Optional[Dict[str, Any]]:
        """[最优版] 使用可接受的启发函数，保证找到移动次数最少的解"""
        return self._solve_astar_base(
            initial_board,
            self._generate_successors_shortest_path,
            self._heuristic_optimal_shortest_path,
        )

    def solve_min_cost(self, initial_board: Board) -> Optional[Dict[str, Any]]:
        """[最优版] 使用可接受的启发函数，保证找到移动总代价最小的解"""
        return self._solve_astar_base(
            initial_board,
            self._generate_successors_min_cost,
            self._heuristic_optimal_min_cost,
        )

    # --- 1b. 贪心版求解器 (使用非可接受的启发函数) ---

    def solve_shortest_path_greedy(
        self, initial_board: Board
    ) -> Optional[Dict[str, Any]]:
        """
        [贪心版] 使用'剩余对数'作为启发函数，会非常快速地寻找一个解，
        但不保证该解是移动次数最少的
        """
        return self._solve_astar_base(
            initial_board,
            self._generate_successors_shortest_path,
            self._heuristic_greedy_remaining_pairs,
        )

    def solve_min_cost_greedy(self, initial_board: Board) -> Optional[Dict[str, Any]]:
        """
        [贪心版] 使用'加权剩余对数'作为启发函数，会非常快速地寻找一个低代价解，
        但不保证该解是总代价最小的
        """
        return self._solve_astar_base(
            initial_board,
            self._generate_successors_min_cost,
            self._heuristic_greedy_weighted_pairs,
        )

    # --- 2. 私有的基础 A* 循环 ---

    def _solve_astar_base(
        self,
        initial_board: Board,
        successor_generator: SuccessorGenerator,
        heuristic_function: HeuristicFunction,
    ) -> Optional[Dict[str, Any]]:
        """通用的 A* 搜索框架"""
        # 唯一的、递增的计数器，用于在优先队列中当 f_score 相同时打破平局，防止比较 Board 对象。
        unique_counter = count()
        # 优先队列（开放列表），存储待探索的节点。
        # 格式: (f_score, g_score, unique_id, board_state)
        open_list = []

        # 计算初始状态的启发值 h(n)
        initial_h = heuristic_function(initial_board)
        # 将初始节点加入优先队列。g_score (初始代价) 为 0
        heapq.heappush(open_list, (initial_h, 0, next(unique_counter), initial_board))

        # came_from 字典用于在找到解后回溯路径
        # 格式: {子状态哈希: (父状态哈希, 导致转变的轻量级动作)}
        came_from: Dict[tuple, Tuple[tuple, Action]] = {}
        # state_cache 用于在路径重建时，通过哈希快速找到完整的 Board 对象
        state_cache: Dict[tuple, Board] = {initial_board.to_hashable(): initial_board}
        # g_scores 记录从起点到每个状态的已知最低成本
        g_scores: Dict[tuple, int] = {initial_board.to_hashable(): 0}

        # A* 主循环，当开放列表不为空时持续进行
        while open_list:
            # 弹出 f_score (g_score + h_score) 最小的节点进行探索
            _, g_score, _, current_board = heapq.heappop(open_list)
            current_hash = current_board.to_hashable()

            # 如果找到了另一条到达此状态的更优路径，则忽略当前路径
            if g_score > g_scores.get(current_hash, float("inf")):
                continue

            # 剪枝策略：死锁检测。如果当前状态是注定无解的，则放弃此分支
            if self._detect_corner_deadlock(current_board):
                continue

            # 目标检查：如果棋盘上没有方块了，说明找到解
            if not current_board.get_all_tiles():
                # 进入路径重建阶段
                total_cost = g_score
                steps = self._reconstruct_animation_steps(
                    came_from, state_cache, current_hash
                )
                return {"steps": steps, "total_cost": total_cost}

            # 生成所有可能的后继状态
            # 调用作为参数传入的特定模式的生成器函数
            for next_board, light_action, move_cost in successor_generator(
                current_board
            ):
                next_hash = next_board.to_hashable()
                # 计算经由当前节点到达后继状态的新路径成本
                new_g_score = g_score + move_cost

                # 如果新路径成本更低，则更新信息并加入开放列表
                if new_g_score < g_scores.get(next_hash, float("inf")):
                    g_scores[next_hash] = new_g_score
                    # 调用作为参数传入的特定模式的启发函数
                    new_h_score = heuristic_function(next_board)
                    new_f_score = new_g_score + new_h_score

                    heapq.heappush(
                        open_list,
                        (new_f_score, new_g_score, next(unique_counter), next_board),
                    )

                    # 记录路径信息和缓存状态
                    came_from[next_hash] = (current_hash, light_action)
                    state_cache[next_hash] = next_board

        # 如果开放列表为空仍未找到解，则说明无解
        return None

    # --- 3. 启发函数库 ---

    # --- 3a. 最优版 (可接受的) 启发函数 ---
    def _heuristic_optimal_shortest_path(self, board: Board) -> int:
        """[最优/可接受] 计算所有种类方块的'孤立'启发值之和"""
        total_h = 0
        for tiles in board.tiles_by_type.values():
            isolated_count = self._calculate_isolated_blocks(tiles)
            total_h += (isolated_count + 1) // 2
        return total_h

    def _heuristic_optimal_min_cost(self, board: Board) -> int:
        """[最优/可接受] 计算所有种类方块的'孤立'启发值，并乘以各自的代价"""
        total_h = 0
        for tiles in board.tiles_by_type.values():
            isolated_count = self._calculate_isolated_blocks(tiles)
            cost_per_move = tiles[0].cost_per_move
            total_h += ((isolated_count + 1) // 2) * cost_per_move
        return total_h

    def _calculate_isolated_blocks(self, tiles: List[Tile]) -> int:
        """
        一个方块是孤立的，当且仅当它是其所在行和列中唯一的该种类方块
        """
        row_counts = defaultdict(int)
        col_counts = defaultdict(int)
        for t in tiles:
            row_counts[t.position[0]] += 1
            col_counts[t.position[1]] += 1

        isolated_count = 0
        for t in tiles:
            if row_counts[t.position[0]] == 1 and col_counts[t.position[1]] == 1:
                isolated_count += 1
        return isolated_count

    # --- 3b. 贪心版 (非可接受的) 启发函数 ---
    # 以下启发函数会高估实际成本，不保证找到最优解，但通常速度很快
    def _heuristic_greedy_remaining_pairs(self, board: Board) -> int:
        """[贪心/非可接受] 启发函数：返回剩余方块对的总数"""
        return board.get_num_of_tiles() // 2

    def _heuristic_greedy_weighted_pairs(self, board: Board) -> int:
        """[贪心/非可接受] 启发函数：返回 剩余对数 * 场上最小单位代价"""
        all_tiles = board.get_all_tiles()
        if not all_tiles:
            return 0
        remaining_pairs = len(all_tiles) // 2
        min_cost = min(tile.cost_per_move for tile in all_tiles)
        return remaining_pairs * min_cost

    # --- 4. 辅助函数 ---
    def _count_stuck_tiles(self, board: Board) -> int:
        """
        [核心辅助函数] 计算棋盘上“被困”方块的总数
        一个方块是“被困的”如果它无法与任何其他同类方块形成直接消除路径
        """
        stuck_tiles_count = 0
        all_tiles = board.get_all_tiles()

        for tile in all_tiles:
            can_be_freed = False
            # 检查此方块是否能与任何其他同类方块直接消除
            for other_tile in all_tiles:
                # 确保是不同实例但同种类的方块
                if (
                    tile.unique_id != other_tile.unique_id
                    and tile.type_id == other_tile.type_id
                ):
                    if board.is_path_clear(tile.position, other_tile.position):
                        can_be_freed = True
                        break  # 找到了一个可行的消除

            if not can_be_freed:
                stuck_tiles_count += 1

        return stuck_tiles_count

    def _analyze_stuck_tiles(self, board: Board) -> Tuple[int, float]:
        """
        [核心辅助函数 - 已优化] 在一次遍历中，计算并返回：
        1. “被困”方块的总数
        2. 在所有“被困”方块中的最小移动代价
        """
        stuck_tiles_count = 0
        min_stuck_cost = float("inf")
        all_tiles = board.get_all_tiles()

        for tile in all_tiles:
            can_be_freed = False
            for other_tile in all_tiles:
                if (
                    tile.unique_id != other_tile.unique_id
                    and tile.type_id == other_tile.type_id
                ):
                    if board.is_path_clear(tile.position, other_tile.position):
                        can_be_freed = True
                        break

            if not can_be_freed:
                stuck_tiles_count += 1
                min_stuck_cost = min(min_stuck_cost, tile.cost_per_move)

        return stuck_tiles_count, min_stuck_cost

    def _generate_successors_shortest_path(
        self, board: Board
    ) -> Generator[Tuple[Board, Action, int], None, None]:
        """后继状态生成器：用于最短路径模式，移动代价恒为 1"""
        # --- 类型 1: 直接消除 (代价为 0) ---
        tiles_to_check = board.get_all_tiles()
        processed_uids = set()
        for tile1, tile2 in combinations(tiles_to_check, 2):
            if tile1.unique_id in processed_uids or tile2.unique_id in processed_uids:
                continue
            if tile1.type_id == tile2.type_id and board.is_path_clear(
                tile1.position, tile2.position
            ):
                next_board = board.clone()
                next_board.remove_tile(next_board.get_tile_at(tile1.position))
                next_board.remove_tile(next_board.get_tile_at(tile2.position))
                action: Action = (
                    "DIRECT_ELIMINATE",
                    (tile1.unique_id, tile2.unique_id),
                )
                processed_uids.update([tile1.unique_id, tile2.unique_id])
                yield (next_board, action, 0)

        # --- 类型 2: 移动并消除 (代价为 1) ---
        all_tiles = board.get_all_tiles()
        empty_squares = board.get_empty_squares()
        for moving_tile in all_tiles:
            targets = [
                t
                for t in all_tiles
                if t.type_id == moving_tile.type_id
                and t.unique_id != moving_tile.unique_id
            ]
            for target_tile in targets:
                for dest_pos in empty_squares:
                    # 检查移动路径和消除路径是否都通畅
                    if board.is_path_clear(
                        moving_tile.position, dest_pos
                    ) and board.is_path_clear(dest_pos, target_tile.position):
                        cost = 1
                        action: Action = (
                            "MOVE_AND_ELIMINATE",
                            (
                                moving_tile.unique_id,
                                dest_pos,
                                target_tile.unique_id,
                                cost,
                            ),
                        )
                        next_board = board.clone()
                        next_board.remove_tile(
                            next_board.get_tile_at(moving_tile.position)
                        )
                        next_board.remove_tile(
                            next_board.get_tile_at(target_tile.position)
                        )
                        yield (next_board, action, cost)

    def _generate_successors_min_cost(
        self, board: Board
    ) -> Generator[Tuple[Board, Action, int], None, None]:
        """后继状态生成器：用于最小代价模式，移动代价根据方块决定"""
        tiles_to_check = board.get_all_tiles()
        processed_uids = set()
        for tile1, tile2 in combinations(tiles_to_check, 2):
            if tile1.unique_id in processed_uids or tile2.unique_id in processed_uids:
                continue
            if tile1.type_id == tile2.type_id and board.is_path_clear(
                tile1.position, tile2.position
            ):
                next_board = board.clone()
                next_board.remove_tile(next_board.get_tile_at(tile1.position))
                next_board.remove_tile(next_board.get_tile_at(tile2.position))
                action: Action = (
                    "DIRECT_ELIMINATE",
                    (tile1.unique_id, tile2.unique_id),
                )
                processed_uids.update([tile1.unique_id, tile2.unique_id])
                yield (next_board, action, 0)

        all_tiles = board.get_all_tiles()
        empty_squares = board.get_empty_squares()
        for moving_tile in all_tiles:
            targets = [
                t
                for t in all_tiles
                if t.type_id == moving_tile.type_id
                and t.unique_id != moving_tile.unique_id
            ]
            for target_tile in targets:
                for dest_pos in empty_squares:
                    if board.is_path_clear(
                        moving_tile.position, dest_pos
                    ) and board.is_path_clear(dest_pos, target_tile.position):
                        cost = moving_tile.cost_per_move  # 核心区别：代价来自方块属性
                        action: Action = (
                            "MOVE_AND_ELIMINATE",
                            (
                                moving_tile.unique_id,
                                dest_pos,
                                target_tile.unique_id,
                                cost,
                            ),
                        )
                        next_board = board.clone()
                        next_board.remove_tile(
                            next_board.get_tile_at(moving_tile.position)
                        )
                        next_board.remove_tile(
                            next_board.get_tile_at(target_tile.position)
                        )
                        yield (next_board, action, cost)

    def _detect_corner_deadlock(self, board: Board) -> bool:
        """
        高效检测两种只剩一对的方块是否形成对角死锁
        如果A对在(r1,c1),(r2,c2)，而B对恰好在(r1,c2),(r2,c1)，则形成死锁
        """
        # 筛选出所有只剩下一对的方块种类
        single_pairs = [
            tuple(tiles) for tiles in board.tiles_by_type.values() if len(tiles) == 2
        ]
        if len(single_pairs) < 2:
            return False
        # 遍历这些“单对”种类的所有组合
        for pair_A, pair_B in combinations(single_pairs, 2):
            tile_A1, tile_A2 = pair_A
            pos_A1, pos_A2 = tile_A1.position, tile_A2.position
            tile_B1, tile_B2 = pair_B
            # 使用集合进行无序比较，更高效健壮
            pos_B_set = {tile_B1.position, tile_B2.position}
            # 计算A对形成的对角坐标
            corners_A_set = {(pos_A1[0], pos_A2[1]), (pos_A2[0], pos_A1[1])}
            if pos_B_set == corners_A_set:
                return True
        return False

    def _reconstruct_animation_steps(
        self, came_from, state_cache, final_hash
    ) -> List[AnimationStep]:
        """
        在找到解后，从 came_from 字典回溯路径，并构建详细的动画步骤列表供UI使用
        """
        steps = []
        current_hash = final_hash
        while current_hash in came_from:
            parent_hash, light_action = came_from[current_hash]
            parent_board = state_cache[parent_hash]
            action_type, details = light_action

            # 根据轻量级动作中的信息，从父状态的 Board 中找到完整的 Tile 对象，
            # 从而构建出包含丰富信息的 AnimationStep 字典
            anim_step = {"action_type": action_type, "cost": 0}
            if action_type == "DIRECT_ELIMINATE":
                uid1, uid2 = details
                tile1 = next(
                    t for t in parent_board.get_all_tiles() if t.unique_id == uid1
                )
                tile2 = next(
                    t for t in parent_board.get_all_tiles() if t.unique_id == uid2
                )
                anim_step["details"] = {"tile1": tile1, "tile2": tile2}
            elif action_type == "MOVE_AND_ELIMINATE":
                moving_uid, dest_pos, target_uid, cost = details
                moving_tile = next(
                    t for t in parent_board.get_all_tiles() if t.unique_id == moving_uid
                )
                target_tile = next(
                    t for t in parent_board.get_all_tiles() if t.unique_id == target_uid
                )
                anim_step["details"] = {
                    "moving_tile": moving_tile,
                    "destination": dest_pos,
                    "target_tile": target_tile,
                }
                anim_step["cost"] = cost
            steps.append(anim_step)
            current_hash = parent_hash
        # 因为是回溯得到的，所以需要反转列表以得到正确的执行顺序
        return steps[::-1]

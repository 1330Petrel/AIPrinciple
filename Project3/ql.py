import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import PhotoImage

np.random.seed(42)
os.makedirs("./output", exist_ok=True)
logger = logging.getLogger(__name__)


# ==========================================
# 1. Configuration (配置管理)
# ==========================================
MAZE_H = 6  # 迷宫的高度（格子数）
MAZE_W = 6  # 迷宫的宽度（格子数）


class Config:
    # 状态与动作
    N_STATES = MAZE_H * MAZE_W  # 状态数量：6x6=36个位置
    N_ACTIONS = 4  # 上下左右四个动作

    # 强化学习超参数
    LEARNING_RATE = 0.1  # alpha
    REWARD_DECAY = 0.9  # gamma

    # Epsilon 贪心策略参数
    EPSILON_START = 0.3
    EPSILON_END = 0.05  # End < 0 表示不使用衰减
    EPSILON_DECAY = 0.98  # 每次 episode 衰减比例

    # 训练参数
    MAX_EPISODES = 100  # 总训练回合数
    RENDER_TRAINING = False  # 训练时是否渲染画面
    RENDER_EVAL = True  # 评估时是否渲染画面
    PLOT_WINDOW_SIZE = 10  # 每多少个Episode计算一次平均值

    # 路径配置
    LOG_FILE = "./output/logging.log"
    MODEL_PATH = "q_table.npy"
    PLOT_PATH = "./output/training_curve.png"
    CSV_PATH = "./output/q_table.csv"
    EVOLUTION_PATH = "./output/path_evolution.png"


# ==========================================
# 2. Environment (环境定义)
# ==========================================
UNIT = 100  # 迷宫中每个格子的像素大小
OFFSET = UNIT / 2  # 像素坐标偏移量
INIT_POS = [0, 0]  # 劳拉的起始位置
GOAL_POS = [3, 3]  # 宝藏的位置
TRAP_POS = [[2, 3], [3, 2], [2, 4], [4, 3]]  # 陷阱的位置


class Maze(tk.Tk, object):
    def __init__(self) -> None:
        super(Maze, self).__init__()
        self.title("Q-learning")
        self.geometry("{0}x{1}".format(MAZE_W * UNIT, MAZE_H * UNIT))
        self.init_pos = np.array(INIT_POS) * UNIT + OFFSET
        self.goal = self._pixel_to_index(np.array(GOAL_POS) * UNIT + OFFSET)
        self.trap_list = [
            self._pixel_to_index(np.array(pos) * UNIT + OFFSET) for pos in TRAP_POS
        ]
        self._build_maze()

    def _build_maze(self) -> None:
        """初始化迷宫图形界面"""
        self.canvas = tk.Canvas(
            self, bg="white", height=MAZE_H * UNIT, width=MAZE_W * UNIT
        )

        for c in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_H * UNIT, fill="black")
        for r in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_W * UNIT, r, fill="black")

        origin = np.array([OFFSET, OFFSET])

        self.bm_laura = PhotoImage(file="./assets/laura.png")
        self.laura = self.canvas.create_image(
            origin[0] + INIT_POS[0] * UNIT,
            origin[1] + INIT_POS[1] * UNIT,
            image=self.bm_laura,
        )

        self.bm_trap = PhotoImage(file="./assets/trap.png")
        for i in range(len(TRAP_POS)):
            self.canvas.create_image(
                origin[0] + UNIT * TRAP_POS[i][0],
                origin[1] + UNIT * TRAP_POS[i][1],
                image=self.bm_trap,
            )

        self.bm_goal = PhotoImage(file="./assets/treasure.png")
        self.canvas.create_image(
            origin[0] + GOAL_POS[0] * UNIT,
            origin[1] + GOAL_POS[1] * UNIT,
            image=self.bm_goal,
        )

        self.canvas.pack()

    def reset(self) -> int:
        """重置环境，返回初始状态索引"""
        self.canvas.delete(self.laura)
        self.laura = self.canvas.create_image(
            self.init_pos[0], self.init_pos[1], image=self.bm_laura
        )
        # 返回初始状态的索引
        return self._pixel_to_index(self.canvas.coords(self.laura))

    def step(self, action: int) -> tuple[int, float, bool]:
        """
        执行动作
        Returns: next_state_index, reward, done
        """
        s = self.canvas.coords(self.laura)
        base_action = np.array([0, 0])

        # 动作逻辑: 0上, 1下, 2右, 3左
        if action == 0:  # Up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # Down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # Right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # Left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.laura, base_action[0], base_action[1])
        s_ = self._pixel_to_index(self.canvas.coords(self.laura))
        done = False
        reward = 0

        # 检查是否到达宝藏
        if s_ == self.goal:
            reward = 1
            done = True
        # 检查是否掉入陷阱
        elif s_ in self.trap_list:
            reward = -1
            done = True

        return s_, reward, done

    def render(self) -> None:
        self.update()
        time.sleep(0.1)

    def _pixel_to_index(self, coords) -> int:
        """将像素坐标转换为 0~35 的状态索引"""
        # 网格坐标 (col, row)
        col = int((coords[0] - OFFSET) / UNIT)
        row = int((coords[1] - OFFSET) / UNIT)
        # 线性索引: Index = Row * Width + Col
        return row * MAZE_W + col


# ==========================================
# 3. Utils & Processors (工具与数据处理)
# ==========================================
class Utils:
    @staticmethod
    def setup_logger() -> None:
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(Config.LOG_FILE, mode="w"),
                logging.StreamHandler(),
            ],
        )
        logger.info(f"Log file located at {Config.LOG_FILE}")

    @staticmethod
    def reward_process(raw_reward: float, state: int, state_: int) -> float:
        """回报函数处理
        raw_reward: 环境返回的原始回报
        state: 当前状态索引
        state_: 下一个状态索引
        Returns: 处理后的回报值
        """
        if raw_reward > 0:
            return 5.0  # 找到宝藏
        elif raw_reward < 0:
            return -5.0  # 掉入陷阱
        elif state == state_:
            return -0.1  # 撞墙惩罚
        return 0.0  # 普通移动

    @staticmethod
    def plot_training(
        rewards: list[float],
        steps: list[int],
        wins: list[int],
        episode_epsilons: list[float],
        path: str = Config.PLOT_PATH,
        window_size: int = Config.PLOT_WINDOW_SIZE,
    ) -> None:
        """绘制训练指标: reward/steps 的原始曲线 + 滑动平均，累计胜场与胜率"""
        episodes = np.arange(1, len(rewards) + 1)
        rewards_arr = np.asarray(rewards, dtype=float)
        steps_arr = np.asarray(steps, dtype=int)
        wins_arr = np.asarray(wins, dtype=int)
        cum_win_rate = np.cumsum(wins_arr) / episodes

        def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
            if data.size < window_size:
                return np.array([])
            return np.convolve(
                data, np.full(window_size, 1 / window_size), mode="valid"
            )

        ma_rewards = moving_average(rewards_arr, window_size)
        ma_steps = moving_average(steps_arr, window_size)
        ma_win_rate = moving_average(wins_arr, window_size)

        fig_num = 4 if episode_epsilons else 3
        fig, axes = plt.subplots(fig_num, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Training Metrics", fontsize=16)

        # 1. Reward
        axes[0].plot(episodes, rewards_arr, color="tab:blue", alpha=0.4, label="Reward")
        if ma_rewards.size:
            axes[0].plot(
                episodes[window_size - 1 :],
                ma_rewards,
                color="tab:blue",
                label=f"MA Reward ({window_size})",
            )
        axes[0].set_ylabel("Reward")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # 2. Steps
        axes[1].plot(episodes, steps_arr, color="tab:orange", alpha=0.4, label="Steps")
        if ma_steps.size:
            axes[1].plot(
                episodes[window_size - 1 :],
                ma_steps,
                color="tab:orange",
                label=f"MA Steps ({window_size})",
            )
        axes[1].set_ylabel("Steps")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # 3. Win Rate
        axes[2].plot(
            episodes, cum_win_rate, color="tab:purple", linestyle="--", label="Win Rate"
        )
        if ma_win_rate.size:
            axes[2].plot(
                episodes[window_size - 1 :],
                ma_win_rate,
                color="tab:green",
                label=f"MA Win Rate ({window_size})",
            )
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_ylabel("Win Rate")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)

        # 4. Epsilon (if applicable)
        if episode_epsilons:
            axes[3].plot(
                np.arange(1, len(episode_epsilons) + 1),
                episode_epsilons,
                color="tab:brown",
                label="Epsilon",
            )
            axes[3].set_ylabel("Epsilon")
            axes[3].set_xlabel("Episodes")
            axes[3].legend(loc="upper right")
            axes[3].grid(True, alpha=0.3)
        else:
            axes[2].set_xlabel("Episodes")

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(path)
        logger.info(f"Saved training plot to {path}")
        plt.close()

    @staticmethod
    def print_q_table(q_table: np.ndarray, path: str = Config.CSV_PATH) -> None:
        """将 Q-Table 转换为带坐标标签的 DataFrame 并保存 CSV"""
        row_labels: list[str] = []
        for i in range(q_table.shape[0]):
            r, c = divmod(i, MAZE_W)
            row_labels.append(f"({r},{c})")
        df = pd.DataFrame(
            q_table,
            index=pd.Index(row_labels, name="State"),
            columns=["Up", "Down", "Right", "Left"],
        )
        df.to_csv(path)
        logger.info(f"Saved Q-Table csv to {path} with {len(df)} states")
        logger.info(f"[Q-Table]:\n{df}")

    @staticmethod
    def plot_path_evolution(path_history: list[dict]) -> None:
        """绘制路径演化汇总图
        path_history: list of dict {'episode': int, 'path': list of (row, col), 'success': bool}
        """
        if not path_history:
            return

        # 起点, 25%, 50%, 75%, 终点
        total = len(path_history)
        indices = [
            0,
            int(total * 0.25),
            int(total * 0.50),
            int(total * 0.75),
            total - 1,
        ]
        indices = sorted(list(set([i for i in indices if i < total])))

        n_plots = len(indices)
        _, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for idx, (ax, hist_idx) in enumerate(zip(axes, indices)):
            record = path_history[hist_idx]
            ep = record["episode"]
            path = record["path"]
            success = record["success"]

            # 1. 设置坐标轴
            ax.set_xlim([-0.5, MAZE_W - 0.5])
            ax.set_ylim([MAZE_H - 0.5, -0.5])
            ax.set_aspect("equal")
            ax.set_xticks(range(MAZE_W))
            ax.set_yticks(range(MAZE_H))
            ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

            # 2. 绘制静态元素 (陷阱、起点、终点)
            for tr in TRAP_POS:
                ax.add_patch(
                    plt.Rectangle(
                        (tr[0] - 0.5, tr[1] - 0.5), 1, 1, color="red", alpha=0.3
                    )
                )
            ax.add_patch(
                plt.Rectangle(
                    (INIT_POS[0] - 0.5, INIT_POS[1] - 0.5),
                    1,
                    1,
                    color="green",
                    alpha=0.3,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (GOAL_POS[0] - 0.5, GOAL_POS[1] - 0.5),
                    1,
                    1,
                    color="gold",
                    alpha=0.5,
                )
            )

            # 3. 绘制路径
            if path:
                rows = [p[0] for p in path]
                cols = [p[1] for p in path]

                color = "blue" if success else "gray"
                ax.plot(
                    cols,
                    rows,
                    marker="o",
                    color=color,
                    markersize=5,
                    linewidth=2,
                    alpha=0.7,
                )
                ax.plot(cols[0], rows[0], "go", markersize=8)
                ax.plot(cols[-1], rows[-1], "X", color="black", markersize=8)

            status = "Success" if success else "Fail"
            ax.set_title(f"Ep: {ep}\n{status} (Len: {len(path) - 1})")
            ax.set_xlabel("Col")
            if idx == 0:
                ax.set_ylabel("Row")

        plt.suptitle("Path Evolution during Training", fontsize=16)
        plt.tight_layout()
        plt.savefig(Config.EVOLUTION_PATH)
        logger.info(f"Saved path evolution plot to {Config.EVOLUTION_PATH}")
        plt.close()


# ==========================================
# 4. Algorithm (Q-learning 算法)
# ==========================================
class QLearning:

    def __init__(
        self, n_states: int, n_actions: int, learning_rate: float, gamma: float
    ) -> None:
        """Q-learning 算法实现

        n_states: 状态数量
        n_actions: 动作数量
        learning_rate: 学习率 alpha
        gamma: 折扣因子 gamma
        """
        self.lr = learning_rate
        self.gamma = gamma

        # 初始化 Q-Table
        # 行：状态索引 (0-35), 列：动作索引 (0-3)
        self.q_table = np.zeros((n_states, n_actions))

    def learn(self, s: int, a: int, r: float, s_: int, done: bool) -> None:
        """Q-learning 更新

        s: 当前状态索引
        a: 执行动作索引
        r: 获得的回报
        s_: 下一个状态索引
        done: 是否为终止状态

        公式:
        Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        """
        q_predict = self.q_table[s, a]

        if not done:
            q_target = r + self.gamma * np.max(self.q_table[s_, :])
        else:
            # 终止状态下，目标值为即时奖励
            q_target = r

        self.q_table[s, a] += self.lr * (q_target - q_predict)


# ==========================================
# 5. Agent (智能体封装)
# ==========================================
class Agent:
    def __init__(
        self,
        n_states: int = Config.N_STATES,
        n_actions: int = Config.N_ACTIONS,
        learning_rate: float = Config.LEARNING_RATE,
        gamma: float = Config.REWARD_DECAY,
        epsilon: float = Config.EPSILON_START,
    ) -> None:
        """智能体封装, 包含学习算法实例与与环境交互的方法"""
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.algorithm = QLearning(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=learning_rate,
            gamma=gamma,
        )
        Utils.setup_logger()
        config_dict = {
            k: v for k, v in Config.__dict__.items() if not k.startswith("__")
        }
        logger.info(f"Usr conf: {config_dict}")

        if Config.EPSILON_END >= 0:
            logger.info(
                f"Epsilon decays from {Config.EPSILON_START} to {Config.EPSILON_END} with decay rate {Config.EPSILON_DECAY}"
            )
        else:
            logger.info(
                f"Epsilon decay disabled; using constant epsilon {Config.EPSILON_START}"
            )

    def decay_epsilon(self) -> float:
        """衰减 epsilon"""
        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)
        return self.epsilon

    def predict(self, state: int) -> int:
        """训练模式: ε-贪心算法用于动作选择"""
        if np.random.uniform() > self.epsilon:
            # 利用：选择 Q 值最大的动作
            # 引入随机扰动打破相同值的僵局
            state_action = self.algorithm.q_table[state, :]
            # np.where 返回最大值的索引列表，随机选一个
            action = np.random.choice(np.where(state_action == np.max(state_action))[0])
        else:
            # 探索：随机选择
            action = np.random.choice(self.n_actions)
        return action

    def learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """接收交互信息并调用学习模块进行参数更新

        state: 当前状态索引
        action: 执行动作索引
        reward: 获得的回报
        next_state: 下一步状态索引
        done: 是否为终止状态
        """
        return self.algorithm.learn(state, action, reward, next_state, done)

    def exploit(self, state: int) -> int:
        """评估模式: 直接选择 Q 值最大的动作"""
        return int(np.argmax(self.algorithm.q_table[state, :]))

    def get_best_path(self) -> tuple[list[tuple[int, int]], bool]:
        """基于当前Q表模拟贪婪路径"""
        curr_row, curr_col = INIT_POS[1], INIT_POS[0]
        curr_state = curr_row * MAZE_W + curr_col
        path: list[tuple[int, int]] = [(curr_row, curr_col)]
        success = False

        for _ in range(Config.N_STATES):
            # 贪婪选择动作
            q_values = self.algorithm.q_table[curr_state]
            action = np.random.choice(np.where(q_values == np.max(q_values))[0])

            # 计算下一个坐标
            next_row, next_col = curr_row, curr_col
            if action == 0:
                next_row = max(0, curr_row - 1)  # Up
            elif action == 1:
                next_row = min(MAZE_H - 1, curr_row + 1)  # Down
            elif action == 2:
                next_col = min(MAZE_W - 1, curr_col + 1)  # Right
            elif action == 3:
                next_col = max(0, curr_col - 1)  # Left

            next_state = next_row * MAZE_W + next_col
            path.append((next_row, next_col))

            # 检查终止条件
            if [next_col, next_row] == GOAL_POS:
                success = True
                break
            if [next_col, next_row] in TRAP_POS:
                success = False
                break

            curr_row, curr_col = next_row, next_col
            curr_state = next_state

        return path, success

    def save_model(self, path: str = Config.MODEL_PATH) -> None:
        """保存模型为 numpy.ndarray"""
        try:
            with open(path, "wb") as f:
                np.save(f, self.algorithm.q_table)
            logger.info(f"Saved model to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, path: str = Config.MODEL_PATH) -> None:
        """从文件加载模型"""
        try:
            with open(path, "rb") as f:
                self.algorithm.q_table = np.load(f)
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            exit(1)


# ==========================================
# 6. Workflow (工作流控制)
# ==========================================
def train(env: Maze, agent: Agent, visualize: bool = Config.RENDER_TRAINING) -> None:
    """训练模式工作流"""
    start = time.time()
    logger.info(f"{'='*20} Start Training {'='*20}")

    if not visualize:
        env.withdraw()

    path_evolution_history: list[dict] = []
    episode_epsilons: list[float] = []
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_wins: list[int] = []
    win_cnt = 0

    for episode in range(Config.MAX_EPISODES):
        # 回合开始前记录当前策略下的路径
        if (
            episode == 0
            or (episode + 1) % 20 == 0
            or episode == Config.MAX_EPISODES - 1
        ):
            sim_path, sim_success = agent.get_best_path()
            path_evolution_history.append(
                {"episode": episode + 1, "path": sim_path, "success": sim_success}
            )

        # 重置并获取初始状态
        state = env.reset()
        if visualize:
            env.render()
        total_reward = 0.0
        step = 0

        while True:
            # 1. 智能体根据状态选择动作
            action = agent.predict(state)

            # 2. 环境执行动作，返回反馈
            state_, reward, done = env.step(action)

            # 3. 回报处理
            processed_reward = Utils.reward_process(reward, state, state_)

            # 4. 智能体学习
            agent.learn(state, action, processed_reward, state_, done)

            # 5. 更新状态
            state = state_
            total_reward += processed_reward
            step += 1
            # 刷新渲染
            if visualize:
                env.render()

            # 结束条件
            if done:
                # 衰减 epsilon
                if Config.EPSILON_END >= 0:
                    episode_epsilons.append(agent.decay_epsilon())
                # 记录本回合数据
                is_win = 1 if reward > 0 else 0
                episode_rewards.append(total_reward)
                episode_steps.append(step)
                episode_wins.append(is_win)
                win_cnt += is_win
                logger.info(
                    f"Episode {episode+1:3d} | Reward: {total_reward:+.1f} | Steps: {step:3d} | Win: {is_win} | WinRate: {win_cnt / (episode + 1):.2%}"
                )
                break

    logger.info(f"{'='*20} Training Finished {'='*20}")
    agent.save_model()
    try:
        Utils.print_q_table(agent.algorithm.q_table)
        Utils.plot_training(
            episode_rewards, episode_steps, episode_wins, episode_epsilons
        )
        Utils.plot_path_evolution(path_evolution_history)
    except Exception as e:
        plt.close()
        logger.warning(f"Post-training processing failed: {e}")
    evaluate(env, agent)
    logger.info(f"Total Time: {time.time() - start:.2f} seconds")


def evaluate(
    env: Maze,
    agent: Agent,
    visualize: bool = Config.RENDER_EVAL,
    load_path: str | None = None,
) -> None:
    """评估模式：使用 exploit 策略"""
    logger.info(f"{'='*20} Start Evaluation {'='*20}")
    if visualize:
        env.deiconify()
    else:
        env.withdraw()
    if load_path:
        agent.load_model(load_path)

    EPISODES = 3
    episode_steps = []
    win_cnt = 0

    for i in range(EPISODES):
        state = env.reset()
        if visualize:
            env.render()
        step = 0

        while True:
            action = agent.exploit(state)

            state_, reward, done = env.step(action)

            state = state_
            step += 1
            if visualize:
                env.render()

            if step >= Config.N_STATES:
                episode_steps.append(step)
                logger.info(f"Eval Episode {i+1} | Result: Failure | Note: step limit")
                break

            if done:
                if reward > 0:
                    result = "Success"
                    win_cnt += 1
                else:
                    result = "Failure"
                episode_steps.append(step)
                logger.info(f"Eval Episode {i+1} | Result: {result} | Steps: {step}")
                break

    win_rate = win_cnt / EPISODES
    avg_steps = sum(episode_steps) / EPISODES
    logger.info(f"Win Rate: {win_rate:.2%} | Average Steps: {avg_steps:.2f}")
    logger.info(f"{'='*20} Evaluation Finished {'='*20}")
    env.destroy()


# ==========================================
# 7. Main Entry (主程序入口)
# ==========================================
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation using a pre-trained model",
    )
    argparser.add_argument(
        "--model-path",
        type=str,
        default=Config.MODEL_PATH,
        help="Path to the pre-trained model to load, effective in evaluation mode",
    )
    args = argparser.parse_args()

    # 创建迷宫环境实例
    env = Maze()

    # 创建智能体实例
    agent = Agent()

    # 根据模式选择训练或评估
    if args.eval_only:
        evaluate(env, agent, load_path=args.model_path)
    else:
        env.after(100, lambda: train(env, agent))

    # 启动主循环
    env.mainloop()

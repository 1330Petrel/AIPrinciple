# 项目三

Q_learning 算法求解迷宫宝藏问题

## 项目简介

实现 **Q-Learning 强化学习算法** 来求解迷宫寻宝问题。智能体需要在 6×6 的网格迷宫中找到宝藏，同时避开陷阱。包含两个实现版本：

- [`ql.py`](ql.py)：标准 Q-Learning 算法实现
- [`ql2.py`](ql2.py)：N-Step Q-Learning 算法实现，考虑多步折扣奖励

## 使用指南

1. 训练智能体

   ```bash
   python ql.py
   # 或
   python ql2.py
   ```

2. 评估预训练模型

   使用 `--eval-only` 参数评估已保存的模型，可选地通过 `--model-path` 指定模型路径：

   ```bash
   python ql.py --eval-only --model-path <optional_path_to_model>
   # 或
   python ql2.py --eval-only --model-path <optional_path_to_model>
   ```

## 项目结构

```plaintext
Project3/
├── ql.py                  # 标准 Q-Learning 实现
├── ql2.py                 # N-Step Q-Learning 实现
├── q_table.npy            # 预训练的 Q-Table
├── assets/                # 图片资源文件
└── output/                # 输出目录
    ├── logging.log        # 训练日志
    ├── q_table.csv        # Q-Table 表格
    ├── training_curve.png # 训练曲线
    └── path_evolution.png # 路径演化图
```

## 参数配置

- 学习相关

   ```python
   LEARNING_RATE = 0.1      # 学习率 α
   REWARD_DECAY = 0.9       # 折扣因子 γ
   N_STEP = 2               # N-Step 中的步数（仅 ql2.py）
   ```

- 探索策略 (ε-贪心)

   ```python
   EPSILON_START = 0.3      # 初始探索率
   EPSILON_END = 0.05       # 最终探索率
   EPSILON_DECAY = 0.98     # 每个 Episode 的衰减系数
   ```

- 训练设置

   ```python
   MAX_EPISODES = 100       # 总训练回合数
   RENDER_TRAINING = False  # 训练时是否显示可视化
   RENDER_EVAL = True       # 评估时是否显示可视化
   ```

## 接口说明

### Maze 类（环境）

- `__init__()`: 初始化 6×6 迷宫，加载图片资源
- `reset()`: 重置到初始状态
- `step(action)`: 执行动作，返回 (next_state, reward, done)
- `render()`: 刷新显示

### Agent 类（智能体）

- `__init__()`: 初始化学习算法和参数
- `predict(state)`: ε-贪心策略选择动作（训练模式）
- `exploit(state)`: 贪心选择最优动作（评估模式）
- `learn()`: 更新 Q-Table
- `save_model()`: 保存 Q-Table
- `load_model()`: 加载 Q-Table

### QLearning 类（算法）

- `learn(s, a, r, s_, done)`: 实现 Q-Learning 更新公式

### Utils 类（工具函数）

- `plot_training()`: 绘制训练曲线
- `plot_path_evolution()`: 绘制路径演化
- `print_q_table()`: 保存 Q-Table 为 CSV
- `reward_process()`: 奖励处理和塑形

# 基于值迭代的简单倒立摆控制

本项目实现了一个基于网格值迭代的简单倒立摆控制器。程序会在离散化的状态空间上计算全局代价函数，并从代价函数中提取查表式反馈控制策略，用于完成单摆的摆起和倒立点附近稳定。

项目采用 Python 脚本实现，不依赖 Jupyter Notebook。运行 `main.py` 后会自动生成仿真数据、结果图和 GIF 动画。

## 功能

- 建立带输入力矩的非线性单摆模型。
- 将角度和角速度组成的状态空间离散化。
- 使用 9 个离散控制输入，输入范围为 `-3 Nm` 到 `3 Nm`。
- 使用 `numpy` 对值迭代过程进行向量化计算。
- 支持两种代价函数对比：
  - `minimum_time`
  - `quadratic`
- 从自然下垂附近的初始状态进行闭环仿真。
- 自动生成以下结果：
  - 最优代价函数热力图
  - 控制策略热力图
  - 状态轨迹曲线
  - 控制输入曲线
  - 值迭代收敛曲线
  - 单摆摆起 GIF 动画

## 模型

单摆状态定义为：

```text
x = [theta, theta_dot]
```

在本项目中，`theta = 0` 表示倒立向上位置，`theta = pi` 或 `theta = -pi` 表示自然下垂位置。角度会被周期化到 `[-pi, pi)`。

动力学模型为：

```text
theta_dot = omega
omega_dot = g / l * sin(theta) + u / (m * l^2) - b * omega
```

默认参数集中放在 `main.py` 文件顶部：

```python
N_THETA = 81
N_OMEGA = 81
OMEGA_MAX = 8.0

MASS = 1.0
LENGTH = 0.5
GRAVITY = 9.81
DAMPING = 0.05

INPUT_LIMIT = 3.0
INPUT_GRID = tuple(np.linspace(-INPUT_LIMIT, INPUT_LIMIT, 9))

DT = 0.05
SIMULATION_TIME = 8.0
MAX_ITERATIONS = 700
```

对应的输入力矩网格为：

```text
[-3.00, -2.25, -1.50, -0.75, 0.00, 0.75, 1.50, 2.25, 3.00]
```

## 方法

程序先将状态空间离散为二维网格 `(theta, theta_dot)`。对于每个状态网格点和每个控制输入，程序使用四阶 Runge-Kutta 方法积分一个时间步，并将下一状态映射回最近的网格点。

值迭代使用 Bellman 更新：

```text
J_new(x) = min_u [g(x, u) + gamma * J(f(x, u))]
```

值函数收敛后，对每个状态选择使 Bellman 目标最小的输入：

```text
policy(x) = argmin_u [g(x, u) + gamma * J(f(x, u))]
```

最终得到的是一个查表式反馈控制器。闭环仿真时，连续状态会先映射到最近的网格点，然后读取该网格点对应的控制输入。

## 项目结构

```text
.
├── main.py                  # 主程序入口和实验参数
├── pendulum.py              # 单摆动力学、RK4 积分和网格辅助函数
├── value_iteration.py       # 向量化值迭代算法
├── costs.py                 # 代价函数和目标区域判断
├── plotting.py              # 结果绘图和 GIF 动画导出
├── requirements.txt         # Python 依赖
├── results/                 # 自动生成的实验结果
└── README.md
```

## 安装

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 运行

运行默认实验：

```bash
.venv/bin/python main.py
```

程序会自动运行 `minimum_time` 和 `quadratic` 两组代价函数实验。如果需要修改网格数量、物理参数、仿真时长或输入力矩范围，直接编辑 `main.py` 顶部的参数即可。

## 输出结果

所有结果会保存到 `results/` 目录。

每种代价函数都会生成：

- `*_trajectory.csv`：闭环仿真轨迹数据
- `*_value_function.png`：最优代价函数热力图
- `*_policy.png`：反馈控制策略热力图
- `*_states.png`：角度误差和角速度随时间变化曲线
- `*_torque.png`：控制输入随时间变化曲线
- `*_convergence.png`：Bellman 更新量收敛曲线
- `*_animation.gif`：单摆摆起动画

汇总文件：

- `results/summary.md`
- `results/summary.json`

默认参数下的结果示例：

```text
minimum_time:
  final theta error = -0.1178 rad
  final theta_dot   = -0.1199 rad/s
  first entered target neighborhood = 2.95 s

quadratic:
  final theta error = -0.1462 rad
  final theta_dot   = -0.1941 rad/s
  first entered target neighborhood = 3.0 s
```

## 图片含义

- `value_function.png` 表示每个状态到达倒立目标附近所需的最优代价。颜色越小，说明该状态越接近目标或越容易被控制到目标区域。
- `policy.png` 表示每个状态下控制器选择的输入力矩。红蓝区域明显时，说明策略具有接近 bang-bang 控制的特点。
- `states.png` 展示角度误差和角速度是否逐渐接近 0，用于判断单摆是否摆起并减速。
- `torque.png` 展示控制器在每个时刻选择的离散输入力矩。
- `convergence.png` 展示每次值迭代中最大 Bellman 更新量的变化，曲线下降说明值函数逐渐收敛。
- `animation.gif` 直观展示闭环控制下单摆从下垂位置摆起到倒立附近的过程。

## 说明

由于状态空间和输入空间都经过离散化，控制器得到的是近似最优策略，因此结果通常会收敛到倒立平衡点附近，而不是精确停在 `theta = 0, theta_dot = 0`。如果希望进一步减小稳态误差，可以在靠近倒立点时切换到局部 LQR 或 PD 控制器。

## 参考

- MIT Underactuated Robotics, Dynamic Programming: <https://underactuated.mit.edu/dp.html>

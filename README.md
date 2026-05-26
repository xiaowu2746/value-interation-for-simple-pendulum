# Project 2: 基于值迭代的倒立摆全局最优控制

本项目对应机器人力学课程设计的 **Project 2: Value Iteration for Pendulum**。目标是针对单摆系统建立离散状态空间和输入空间，通过动态规划中的值迭代方法求解近似全局最优控制策略，并比较不同代价函数对控制效果的影响。

## 项目目标

- 对简单单摆系统进行建模，状态为角度和角速度：

  ```text
  x = [theta, theta_dot]
  ```

- 将状态空间 `[theta, theta_dot]` 和输入空间 `[u]` 网格化。
- 在离散网格上实现 Value Iteration，计算最优代价函数 `J*`。
- 根据最优代价函数提取控制策略，例如 bang-bang 控制或状态反馈控制。
- 对比不同成本函数，例如 Minimum Time Cost 和 Quadratic Cost，分析生成的控制策略差异。
- 通过仿真动画、状态曲线和控制输入曲线验证控制效果。

## 作业要求

根据课程设计 PDF，本项目需要完成以下核心任务：

1. **状态空间离散化**
   - 研究对象为简单单摆。
   - 将角度 `theta`、角速度 `theta_dot` 和控制输入 `u` 离散化。
   - 合理设置网格范围、网格分辨率和输入约束。

2. **值迭代实现**
   - 在离散网格上实现 Value Iteration。
   - 使用动态规划 Bellman 更新计算最优代价函数 `J*`。
   - 设置收敛阈值、折扣因子或终止条件。

3. **控制策略提取**
   - 从最优代价函数中提取最优控制动作。
   - 支持 bang-bang 控制或近似反馈控制。
   - 在仿真中验证控制器能将单摆摆起并稳定到目标状态附近。

4. **仿真实验与对比**
   - 改变成本函数并进行对比实验。
   - 至少比较以下两类成本函数之一组：
     - Minimum Time Cost
     - Quadratic Cost
   - 分析不同成本函数下的状态轨迹、控制输入和收敛速度。

## 考察重点

- 动态规划方法在非线性系统全局最优控制中的应用。
- 状态空间和输入空间离散化的设计。
- Value Iteration 的收敛性与误差分析。
- 最优控制策略的提取和仿真验证。
- 不同代价函数对控制行为的影响。

## 数学模型

简单单摆动力学可写为：

```text
theta_dot = omega
omega_dot = -g / l * sin(theta) + u / (m * l^2)
```

其中：

- `theta`：摆杆角度
- `omega` 或 `theta_dot`：角速度
- `u`：输入力矩
- `m`：摆杆质量
- `l`：摆长
- `g`：重力加速度

目标状态通常设为倒立平衡点：

```text
theta = pi, theta_dot = 0
```

为了便于数值计算，角度需要做周期化处理，例如将 `theta` 映射到 `[-pi, pi]` 或 `[0, 2pi)`。

## 推荐实现流程

1. **建立参数配置**
   - 设置 `m, l, g, dt, u_max` 等物理参数。
   - 设置状态空间范围和网格数量。
   - 设置控制输入集合，例如：

     ```text
     u in {-u_max, 0, u_max}
     ```

2. **离散化状态空间**
   - 构造二维网格 `(theta_i, omega_j)`。
   - 为每个状态建立索引。
   - 实现连续状态到最近网格点的映射。

3. **构造一步转移**
   - 对每个状态和每个输入，使用 Euler 或 Runge-Kutta 方法积分一个时间步。
   - 将积分后的连续状态映射回离散网格。
   - 对角度进行 wrap 处理。

4. **定义代价函数**
   - Minimum Time Cost 示例：

     ```text
     cost = 0, if state is near target
     cost = 1, otherwise
     ```

   - Quadratic Cost 示例：

     ```text
     cost = q_theta * angle_error^2 + q_omega * omega^2 + r * u^2
     ```

5. **执行值迭代**
   - 对所有状态执行 Bellman 更新：

     ```text
     J_new(x) = min_u [cost(x, u) + gamma * J(f(x, u))]
     ```

   - 当最大更新误差小于阈值时停止迭代。

6. **提取控制策略**
   - 对每个状态保存使 Bellman 目标最小的输入。
   - 得到查表式控制策略：

     ```text
     policy[x] = argmin_u [cost(x, u) + gamma * J(f(x, u))]
     ```

7. **仿真验证**
   - 从多个初始状态出发进行闭环仿真。
   - 绘制 `theta(t)`、`theta_dot(t)`、`u(t)`。
   - 生成单摆运动动画。

8. **对比实验**
   - 分别使用 Minimum Time Cost 和 Quadratic Cost。
   - 比较控制是否更激进、是否出现 bang-bang 行为、是否更平滑、到达目标所需时间和能量消耗。

## 项目结构

```text
.
├── README.md
├── main.py                  # 主程序入口
├── pendulum.py              # 单摆动力学和仿真函数
├── value_iteration.py       # 值迭代算法
├── costs.py                 # 成本函数
├── plotting.py              # 使用 matplotlib 输出图像和 GIF 动画
├── requirements.txt         # Python 依赖
└── results/                 # 运行后保存图像、动画和实验结果
```

当前实现采用 Python 脚本，使用 `numpy` 加速值迭代，使用 `matplotlib` 和 `pillow` 生成结果图与 GIF 动画。

## 最终提交内容

课程设计要求最终提交：

1. **代码实现**
   - 提交 `.ipynb` 或 Python 脚本。
   - 代码需包含完整的仿真动画演示。

2. **报告文档**
   - 说明单摆动力学推导。
   - 说明状态空间、输入空间和参数选择依据。
   - 说明 Value Iteration 的 Bellman 更新过程。
   - 展示并解释状态曲线、控制输入曲线和仿真结果。
   - 分析失败案例，例如网格太粗、输入力矩不足、代价函数设置不合理等。

3. **对比实验**
   - 至少在一个参数上进行消融或对比实验。
   - 本项目建议比较 Minimum Time Cost 和 Quadratic Cost。
   - 需要说明不同设置对策略、轨迹和控制输入的影响。

## 结果分析建议

报告中建议至少包含以下图表：

- 最优代价函数 `J*` 的二维热力图。
- 控制策略 `policy(theta, theta_dot)` 的二维图。
- 闭环仿真的角度曲线 `theta(t)`。
- 闭环仿真的角速度曲线 `theta_dot(t)`。
- 控制输入曲线 `u(t)`。
- Minimum Time Cost 与 Quadratic Cost 的对比结果。
- 单摆摆起过程动画。

## 运行方式

第一次运行前建议创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

运行默认实验：

```bash
.venv/bin/python main.py
```

所有实验参数都集中在 `main.py` 文件顶部，便于直接修改：

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

其中 `INPUT_GRID` 对应 9 个离散控制输入：

```text
[-3.00, -2.25, -1.50, -0.75, 0.00, 0.75, 1.50, 2.25, 3.00]
```

默认会分别运行：

- `minimum_time`
- `quadratic`

输出文件保存在 `results/` 目录，包括：

- `summary.md`：实验摘要
- `summary.json`：实验摘要数据
- `minimum_time_trajectory.csv` 和 `quadratic_trajectory.csv`：闭环轨迹数据
- `*_value_function.png`：最优代价函数热力图
- `*_policy.png`：控制策略图
- `*_states.png`：状态曲线
- `*_torque.png`：控制输入曲线
- `*_convergence.png`：值迭代收敛曲线
- `*_animation.gif`：单摆摆起动画

## 参考

- 课程设计要求 PDF：`课程设计要求.pdf`
- MIT Underactuated Robotics: <https://underactuated.mit.edu/>

# mujoco_6

导航库：**A\* 全局规划 + acados MPC 路径跟踪**，纯 Python，主逻辑直接 `import` 使用。

## 用法

```python
from nav import Navigation

nav = Navigation(map_yaml='maps/rc2027.yaml')

# 设目标：返回 False 表示不可达
if not nav.set_goal(2.3, -1.3, yaw=0.0):
    print('目标不可达')

# 每帧（建议 20~100 Hz）
cmd = nav.update(x, y, yaw)
if cmd is None:
    print('到达并停稳')          # 阶段 ARRIVED
else:
    vx, vy, wz = cmd
    send_serial(vx, vy, wz)
```

## 接口

| 方法 | 说明 |
|---|---|
| `Navigation(map_yaml, **params)` | 加载地图、建膨胀、初始化 MPC（首次约 1~2 秒，acados 要编译） |
| `set_goal(x, y, yaw, pose=None)` | 规划到目标，返回 `bool`。`pose` 省略时用上次 `update` 的位姿 |
| `update(x, y, yaw)` | 喂位姿，返回 `(vx, vy, wz)`；`None` = 已到达 |
| `cancel()` | 放弃当前目标 |
| `nav.phase` | `IDLE` / `TRACK` / `ALIGN` / `ARRIVED` |
| `nav.path` | 最近一次的规划路径 `(N, 2)`，可视化用 |

**注意**：`update()` 内部会调一次 MPC 求解，约 **1.3 ms**。如果在 asyncio 协程里调用，会阻塞事件循环 1.3 ms，一般可接受；如果敏感，可以降到 20~50 Hz 调用。

## 运动策略：三阶段

```
① 平移   车头锁死（MPC 内部 vω = 0），靠 α 让底盘朝任意方向移动
         反向移动靠倒车（v < 0），避免 α 卡在 ±π 的约束边界
② 到位   位置误差 < goal_tolerance (0.15 m)
③ 自转   原地 P 控制转到目标朝向，误差 < rotate_tolerance (2.9°)
```

## 目录

```
mujoco_6/
├── nav/                    ← 主逻辑 import 这个
│   ├── __init__.py         懒加载入口
│   ├── navigation.py       ★ 对外主类
│   ├── map.py              地图加载 + 膨胀（scipy EDT）
│   ├── grid.py             栅格坐标
│   ├── astar.py            栅格 A*（自带，numpy + heapq）
│   └── mpc_core/
│       ├── mpc_acados.py   acados MPC（含横向偏差惩罚）
│       └── linear.py       SplinePlanner
├── maps/                   rc2027.yaml + pgm
├── sim/                    MuJoCo 仿真桥（仅本地验证用）
└── tests/
```

## 依赖

| 库 | 用途 |
|---|---|
| `numpy` | 栅格运算 |
| `scipy` | 距离变换（精确膨胀） |
| `pyyaml` | 读地图配置 |
| `casadi` | MPC 符号建模 |
| `acados_template` | MPC 求解器生成 |

A\* 是自己实现的（`nav/astar.py`），不依赖第三方包。

## 参数

```python
Navigation(
    map_yaml,
    inscribed_radius=0.42,    # 车中心到障碍的最小距离
    inflation_radius=0.68,    # 膨胀衰减到 0 的距离
    cost_weight=2.0,          # A* 对膨胀代价的权重
    ref_speed=0.5,            # 参考点推进速度（实际决定车速）
    mpc_dt=0.05,
    mpc_horizon=25,           # 视野 = 0.05 × 25 = 1.25 s
    v_max=1.5,                # 输出限幅
    goal_tolerance=0.15,
    rotate_tolerance=0.05,
)
```

## 已知问题

| 问题 | 影响 | 说明 |
|---|---|---|
| **A\* 跨全场要 70 ms** | 每次设目标多等 0.07 s | 同侧路径只要 2~3 ms；跨全场（14 m 路径）70 ms。C++ 版是 11.6 ms，Python 慢 6 倍 |
| 传递区开口窄 | 通过时余量只有 175 mm | 物理尺寸，改不了 |
| 车头方向锁死在起点 | 斜向通道车体投影变宽 | 45° 通道时投影 0.92 m，正交时 0.65 m |

## 和 mujoco_5 的区别

| | mujoco_5 | mujoco_6 |
|---|---|---|
| A\* | C++ 节点（11.6 ms） | Python 自实现（2~70 ms） |
| 接口 | ROS topic / action | **Python API（import）** |
| MPC | ROS 节点调用 | **直接调用** |
| 主逻辑对接 | 发话题 | **import** |

## 本地验证

### ① 纯库自检（不需要 ROS）

```bash
cd mujoco_6
python3 tests/selftest.py
```

检查依赖、地图、膨胀、A*、MPC 初始化、三阶段闭环。全绿说明库本身没问题。

**给主逻辑做验收就用这个。**

### ② 接仿真（需要 ROS + MuJoCo）

首次要建工作区并编译：

```bash
cd mujoco_6
mkdir -p src
ln -sfn ../../sim/nav_interfaces src/nav_interfaces
ln -sfn ../../sim/mujoco_ros2_bridge src/mujoco_ros2_bridge
colcon build --symlink-install --packages-select nav_interfaces mujoco_ros2_bridge
```

然后：

```bash
# 终端 1：仿真 + Foxglove
source install/setup.bash
ros2 launch mujoco_ros2_bridge bridge.launch.py

# 终端 2：把库包成节点（可选，想在 Foxglove 里点目标时需要）
source install/setup.bash
python3 tests/nav_node.py

# 或者用脚本直接跑一个目标
python3 tests/sim_bridge.py 2.3 -1.3 0
```

Foxglove 连 `ws://127.0.0.1:8765`。

**环境变量必须一致**，否则节点互相看不见：

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_LOCALHOST_ONLY=1
```

### 三个测试脚本的分工

| 脚本 | 需要 ROS | 用途 |
|---|---|---|
| `tests/selftest.py` | ❌ | 验证库本身，给主逻辑验收 |
| `tests/nav_node.py` | ✅ | 把库包成 ROS 节点，在 Foxglove 里点目标调试 |
| `tests/sim_bridge.py` | ✅ | 脚本化发一个目标并跟踪 |

**这三个都不是库的一部分**，主逻辑只需要 `nav/` 和 `maps/`。

## 交付给主逻辑

只拷两个目录：

```
nav/
maps/
```

**外加** `tests/selftest.py` 供他验收。

依赖：`numpy`、`scipy`、`pyyaml`、`casadi`、`acados_template`。

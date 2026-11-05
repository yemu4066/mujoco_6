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
    print('到达并停稳')          # phase == 'ARRIVED'
else:
    vx, vy, wz = cmd
    send_serial(vx, vy, wz)
```

**初始化耗时约 1.4 秒**，因为 acados 要生成并编译求解器。建议在主逻辑启动时初始化一次，不要每次任务重建。

**`update()` 内部会调一次 MPC 求解，约 1.3 ms。** 如果在 asyncio 协程里调用，会阻塞事件循环 1.3 ms；一般可接受，敏感场景可降到 20~50 Hz 调用。

## 接口

| 方法 | 说明 |
|---|---|
| `Navigation(map_yaml, **params)` | 加载地图、建膨胀、初始化 MPC |
| `set_goal(x, y, yaw, pose=None)` | 规划到目标，返回 `bool`。`pose` 省略时用上次 `update` 的位姿 |
| `update(x, y, yaw)` | 喂位姿，返回 `(vx, vy, wz)`；`None` = 已到达 |
| `cancel()` | 放弃当前目标 |
| `nav.phase` | `'IDLE'` / `'TRACK'` / `'ALIGN'` / `'ARRIVED'` |
| `nav.path` | 最近一次规划路径 `(N, 2)`，可视化用 |
| `nav.goal` | 当前目标 `(x, y, yaw)` 或 `None` |

## 运动策略：三阶段

```
① 平移   车头锁死（MPC 内部 vω = 0），靠 α 让底盘朝任意方向移动
         反向移动靠倒车（v < 0），避免 α 卡在 ±π 的约束边界
② 到位   位置误差 < goal_tolerance (0.15 m)
③ 自转   原地 P 控制转到目标朝向，误差 < rotate_tolerance (2.9°)
```

舵轮底盘的车头方向与运动方向解耦，所以平移时不需要转头。

## 目录

```
mujoco_6/
├── nav/
│   ├── __init__.py         懒加载入口（只 import map 不会触发 casadi）
│   ├── navigation.py       ★ 对外主类
│   ├── astar.py            栅格 A*（numpy + heapq）
│   ├── map.py              地图加载 + 膨胀（scipy 距离变换）
│   ├── grid.py             栅格坐标
│   └── mpc_core/
│       ├── mpc_acados.py   acados MPC（含横向偏差惩罚）
│       └── linear.py       SplinePlanner
├── maps/                   rc2027.yaml + pgm
└── README.md
```

## 依赖

| 库 | 用途 |
|---|---|
| `numpy` | 栅格运算 |
| `scipy` | 距离变换（精确膨胀） |
| `pyyaml` | 读地图配置 |
| `casadi` | MPC 符号建模 |
| `acados_template` | MPC 求解器生成 |

A\* 是自己实现的，不依赖第三方包。

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
    rotate_gain=2.5,
    rotate_max=1.2,
)
```

## 性能实测

| 操作 | 耗时 |
|---|---|
| 初始化（含 acados 编译） | 1.4 s |
| 地图膨胀 | 60 ms |
| A\* 同侧短路径（7 m） | 2~3 ms |
| A\* 跨场上 L1（14 m） | 57~70 ms |
| MPC 单帧求解 | 1.3 ms |
| 闭环到达（上 L1） | 586 帧，位置误差 13 cm，朝向 2.5° |

## 已知问题

| 问题 | 说明 |
|---|---|
| A\* 跨全场要 70 ms | 同侧只要 2~3 ms。C++ 版是 11.6 ms，Python 慢 6 倍 |
| 传递区开口窄 | 通过时余量只有 175 mm（开口 1.0 m，车 0.65 m） |
| 车头方向锁死在起点 | 斜向通道时车体投影变宽：45° 时 0.92 m，正交时 0.65 m |

## 首次运行会产生什么

跑 `Navigation(...)` 时，acados 会在**当前工作目录**生成：

```
c_generated_code/     求解器 C 代码
mpc_ocp.json          求解器配置
```

这两个已经在 `.gitignore` 里。**只会在初始化时生成一次**，之后走缓存。

**注意**：生成位置是「当前工作目录」，所以在哪里 import 就在哪里生成。建议固定在工作区根目录跑。

## 交付

只需要两个目录：

```
nav/
maps/
```

依赖装好后即可 import 使用，不需要 ROS、不需要编译。

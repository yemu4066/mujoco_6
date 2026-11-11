# mujoco_6 — 导航与仿真调试手册

舵轮底盘（四轮独立转向）的 **A\* 全局规划 + acados MPC 路径跟踪**，纯 Python。
本文件是唯一的操作手册：快速开始、地图切换、功能开关、参数、调试手段全在这里。

历史分析文档（代码解构、诊断报告、架构图）已移到工作区 `../docs_mujoco6/`，不在代码库里。

---

## 一、快速开始

### 仿真（MuJoCo）

```bash
bash tests/start_all.sh
```

### 纯导航库（自己写主逻辑）

```python
from nav import Navigation

nav = Navigation(map_yaml='maps/rc2027.yaml')     # 初始化约 1.4 s（编译 acados）

if not nav.set_goal(2.3, -1.3, yaw=0.0):          # False = 不可达
    print(nav.last_error)

cmd = nav.update(x, y, yaw)                       # 每帧调用，返回 (vx, vy, wz)
if cmd is None:
    pass                                          # 已到达并停稳
else:
    vx, vy, wz = cmd
```

`update()` 内部调一次 MPC 求解，约 1.3 ms。

**acados 的生成物**（`c_generated_code/`、`mpc_ocp.json`）落在**当前工作目录**，已在 `.gitignore` 里，只生成一次。建议固定在工作区根目录跑。

---

## 二、地图切换

### 切换方式

地图是 nav2 格式（`.yaml` + `.pgm`），运行时用 ROS 参数指定：

```bash
# 默认（不传参）→ maps/rc2027.yaml
ros2 run <pkg> nav_node

# 指定地图，三种写法都行
ros2 run <pkg> nav_node --ros-args -p map:=maps/debug_arena_10m.yaml
ros2 run <pkg> nav_node --ros-args -p map:=debug_arena_10m     # 自动补 maps/ 和 .yaml
ros2 run <pkg> nav_node --ros-args -p map:=/abs/path/xxx.yaml
```

启动日志第一行会打 `使用地图: xxx.yaml`。路径写错会**立刻抛异常并打印已尝试的完整路径**，不会静默用默认地图。

### 现有地图

| 文件 | 尺寸 | 内容 | 用途 |
|---|---|---|---|
| `rc2027.yaml` | 11.2 m | 完整赛场（围栏、中心区、斜坡） | 仿真主场景 |
| `debug_empty_10m.yaml` | 10 m | **纯空白，连边界都没有** | 看 A\* 在无障碍下的行为 |
| **`debug_arena_10m.yaml`** | 10 m | 空白 + 10 cm 围栏 | **实车调试推荐** |
| `debug_arena_20m.yaml` | 20 m | 空白 + 10 cm 围栏 | 大场地 |
| `empty_20x20/30x30/100x100.yaml` | 20/30/100 m | 纯空白 | 早期遗留 |

**建议用 `debug_arena_*`**：纯空白没有边界，A\* 会往外规划，实车会一直开出去。

### 自己造地图

1. 画一张 PGM（0=障碍，254=空闲，205=未知），分辨率 0.05 m/格
2. 写同名 `.yaml`：

```yaml
image: your_map.pgm
mode: trinary
resolution: 0.05
origin: [-5.0, -5.0, 0.0]    # 世界坐标左下角
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

3. `-p map:=your_map` 启动

**注意**：PGM 和 MuJoCo 的 XML **没有任何关联**。改 XML 不会自动改地图，反之亦然。两者必须手工对齐，否则会出现"导航以为能过、物理上撞墙"。

---

## 三、功能开关

### 动态避障

```python
# nav/navigation.py:173
self.enable_dynamic_obstacles = True     # False 全关
```

打开后才会接收 `/scan`（真雷达）或 `/obstacles`（仿真真值），并做重规划。

| 参数 | 默认 | 位置 | 含义 |
|---|---|---|---|
| `replan_interval` | 0.4 s | `navigation.py:182` | 最快多久重规划一次 |
| `blocked_retry_interval` | 0.6 s | `navigation.py:183` | 被堵死后多久重试 |
| `obstacle_clearance` | 0.05 m | `navigation.py:184` | 动态障碍额外膨胀（叠加在 `inscribed_radius` 之上） |
| 挡路阈值 | 0.45 m | `_obstacle_blocks_path()` | 障碍到**未走路径**的最近距离 |

**挡路阈值必须小于 A\* 的膨胀半径**（`_inscribed_cells + obstacle_clearance/res` = 0.50 m），否则 A\* 怎么绕都满足"挡路"，会无限重规划。

**数据链路**：

```
真机 /scan  ──┐
              ├─→ nav.set_scan / set_obstacle_points ─→ _dynamic_points
仿真 /obstacles┘                                          ↓
                            手动 /debug_obstacle ─→ _manual_points（独立层，不被覆盖）
                                                   ↓
                                    _planning_cost()：膨胀成 LETHAL
                                                   ↓
                                    _obstacle_blocks_path() → 触发 set_goal
```

### 路径平滑

**没有开关，永远开启。** 三步：视线拉直 → 圆弧倒角 → 5 cm 重采样。

| 参数 | 默认 | 含义 |
|---|---|---|
| `smooth_loss` | 0.03 m | 允许比原路径再贴墙多少（**别调大**，实测 0.06 就开始蹭墙） |
| `smooth_radius` | 0.35 m | 圆弧半径，放不下时自动缩到 0.75/0.5/0.35 倍 |
| `smooth_adaptive` | 0.58 m | 折点附近余量小于它就不倒角（窄区保护） |

平滑失败时**整条回退 A\* 原样**，这是设计——宁可锯齿也不给一条更急的路径。

### 速度限制

限速由五项叠加，**逐点取最小值**：

```python
# ① 横向余量映射
margin = 到墙距离 - 车体外接圆(0.478 m)
v = speed_floor + (speed_wide - speed_floor) × clip((margin - min)/(max - min))

# ② 直道豁免（曲率小的窄道按直道算）
straight = (kappa < straight_kappa) & (margin > 0.03)
margin_eff = margin + straight_boost

# ③ 曲率限速（两个约束取较严）
v_curve = min(sqrt(lat_acc_max × R),  dalpha_max × R)

# ④ 掉头/折返（单点转角 > 30°）改用制动距离
v_brake = sqrt(2 × dv_max × margin)

# ⑤ 区间最小前瞻
ref_speed = vmax[s_now : s_now + speed_lookahead].min()
```

| 参数 | 默认 | 位置 | 含义 |
|---|---|---|---|
| `ref_speed` | 1.2 m/s | 构造参数 | 直道最高速（= `speed_wide`） |
| `speed_margin_min` | 0.01 m | `navigation.py:146` | 余量 ≤ 此值 → 降到 floor |
| `speed_margin_max` | 0.15 m | `navigation.py:147` | 余量 ≥ 此值 → 跑满 |
| `speed_floor` | 0.35 m/s | `navigation.py:148` | 最低速 |
| `straight_kappa` | 0.3 | 构造参数 | κ < 此值算直道（R > 3.3 m） |
| `straight_boost` | 0.05 m | 构造参数 | 直道等效余量加成 |
| `lat_acc_max` | 3.0 m/s² | `navigation.py:177` | 弯道横向加速度上限 |
| `dalpha_max` | 3.0 rad/s | 构造参数 | 弯道转向速率上限 |
| `dv_max` | 2.0 m/s² | 构造参数 | 折返制动能力 |
| `speed_lookahead` | 1.2 m | 构造参数 | 区间最小前瞻长度 |

**`dalpha_max` 和 `dv_max` 必须与 `mpc_acados.py` 的约束同步**（`lbu/ubu` 的对应分量），否则规划层说"能过"但控制器做不到。

### 脱困

```python
stuck_frames_threshold = 30    # 连续多少帧没有进展就触发
```

触发后用 `_kick_direction()` 算出指向"前方最近参考点"的方向，强制推一下。

### 规划器选择

```python
# nav/astar.py
GridPlanner(costmap, theta=False)    # False = 标准 8 邻域 A*（默认）
```

Theta\*（`theta=True`）能砍掉一半折点，但实测：规划慢 2~3 倍、路径更贴墙、平滑后曲率忽好忽坏。**当前用标准 A\***。

---

## 四、MPC 参数

```python
# nav/mpc_core/mpc_acados.py
mpc_dt = 0.01        # 100 Hz，必须与 odom 发布频率一致
mpc_horizon = 175    # 预测 1.75 s
```

**这两个值必须匹配实际调用频率。** 原来 `dt=0.05` 但按 100 Hz 调用，时间尺度差 5 倍：`dv=2` 实际是 10 m/s²，表现是"敢在门前全速然后刹不住"。

| 约束 | 值 | 位置 | 说明 |
|---|---|---|---|
| `v` | ±4.0 m/s | `mpc_acados.py:211-212` | 速度上限（一般够不到） |
| `α` | ±π | 同上 | 允许倒车 |
| `ω` | 0 | 同上 | 车头锁死 |
| `dv` | ±2.0 | `mpc_acados.py:216-217` | 与 `dv_max` 同步 |
| `dα` | ±3.0 | 同上 | 与 `dalpha_max` 同步 |

**改了这些必须重新编译 acados**：

```bash
rm -rf c_generated_code mpc_ocp.json
python3 -c "from nav import Navigation; Navigation('maps/rc2027.yaml')"
```

---

## 五、调试手段

### Foxglove 实时注入障碍

测试重规划能力，不用改代码：

| Topic | 类型 | 作用 |
|---|---|---|
| `/debug_obstacle` | `geometry_msgs/PointStamped` | 点一下加一个障碍（累加） |
| `/debug_obstacle_clear` | `std_msgs/Empty` | 清空 |
| `/debug_obstacles` | `visualization_msgs/MarkerArray` | 可视化：橙柱=实体，黄盘=膨胀范围 |

在 Foxglove 3D 面板用 "Publish point" 发到 `/debug_obstacle` 即可。

手动障碍存在**独立一层**（`_manual_points`），不会被 `/obstacles` 每 100 ms 一帧冲掉。

**黄盘半径 = 0.50 m**，就是 A\* 实际看到的挡路范围。看到橙点但车没反应，说明黄盘还没压到路径（阈值 0.45 m）。

### 关键可视化话题

| Topic | 内容 |
|---|---|
| `/costmap` | 静态 + 动态障碍的代价地图 |
| `/plan` | 平滑后的参考路径 |
| `/odom` | 位姿 |
| `/obstacles` | 仿真障碍真值 |
| `/debug_obstacles` | 手动障碍 |

### 监控量

```python
nav.phase                 # IDLE / TRACK / ALIGN / ARRIVED
nav.blocked               # 是否被堵死
nav.last_error            # 最近一次失败原因
nav.last_lateral          # 横向偏差 [m]，正=偏左
nav.mpc.last_aug_state    # (v, α, ω)
nav.mpc.ref_speed         # 当前参考推进速度
nav.raw_path              # A* 原始输出
nav.path                  # 平滑后
```

**`last_lateral` 是最重要的一个**：门处余量只有 2.2 cm，超过它就会蹭墙。

---

## 六、坐标系与单位

| 项 | 约定 |
|---|---|
| 世界系 | `map`，右手系，x 向东、y 向北 |
| 车体系 | `base_link`，x 向前、y 向左 |
| 角度 | 弧度，`atan2` 约定 |
| 距离 | 米，地图分辨率 0.05 m/格 |
| 速度输出 | `(vx, vy)` 世界系速度 + `wz` 角速度 |

---

## 七、已知限制

**① 窄通道无法平滑**
L1 通道约 1.0 m 宽，路径中心离墙 0.45~0.50 m，车体外接圆 0.478 m，余量 0~2.2 cm。倒角需要 5~15 cm 空间，**放不下**。表现是那段保留 A\* 原折线。

**② `inscribed_radius` 卡在 0.43**
现有场地 0.44 时斜坡还能过，0.45 以上会把中心区切成 4 块互不连通。换场地要重新标定。

**③ 重规划同步阻塞**
`set_goal()` 在 `update()` 里直接调用，A\* + 平滑要 110~150 ms。期间控制输出冻结，1.2 m/s 下走 18 cm。障碍加得太近会来不及。

**④ 车头锁死**
`ω ≡ 0`，转向完全靠 `α`。好处是全向移动，坏处是弯道受 `dα ≤ 3 rad/s` 限制——半径 0.35 m 的弯最高只能 1.05 m/s。

---

## 八、目录结构

```
nav/
  navigation.py      主逻辑：规划、限速、状态机、动态避障
  astar.py           栅格 A*
  path_smooth.py     平滑：拉直 → 倒角 → 重采样
  map.py             nav2 格式地图加载
  grid.py            栅格索引/坐标转换
  mpc_core/
    mpc_acados.py    MPC 建模与求解
    linear.py        路径参数化（SplinePlanner）
maps/                地图（.yaml + .pgm）
sim/                 MuJoCo 仿真桥与 ROS 接口
tests/               启动脚本、导航节点、自检
```

---

## 九、自检

```bash
# 语法 + 未定义名称检查（能抓住 NameError 类 bug）
python3 tests/check_names.py

# 地图能否加载
python3 -c "from nav.map import load_map; load_map('maps/debug_arena_10m.yaml')"
```

`check_names.py` 值得每次改完代码跑一次——之前有过一次编辑失误漏了变量定义，导致 `nav_node` 启动即崩、Foxglove 收不到地图。

# mujoco_6

舵轮底盘（四轮独立转向、车头锁死）的 **A\* 全局规划 + acados MPC 路径跟踪**，纯 Python，
配套 MuJoCo 仿真桥。

---

## 一、快速开始

### 作为库使用

```python
from nav import Navigation

nav = Navigation(map_yaml='maps/rc2027.yaml')   # 首次初始化会编译 acados，约 1.4 s

if not nav.set_goal(2.3, -1.3, yaw=0.0):        # False = 不可达，原因见 nav.last_error
    print(nav.last_error)

cmd = nav.update(x, y, yaw)                     # 每帧调用，约 1.3 ms
if cmd is not None:
    vx, vy, wz = cmd
```

acados 的生成物（`c_generated_code/`、`mpc_ocp.json`）落在当前工作目录，已在 `.gitignore` 里，
只生成一次。建议固定在工作区根目录运行。

### 仿真

需要 ROS 2 与 `~/ros2_ws` 工作区（`src/mujoco_ros2_bridge` 指向本仓库的 `sim/mujoco_ros2_bridge`）。

```bash
cd ~/ros2_ws && colcon build --packages-select mujoco_ros2_bridge && source install/setup.bash
```

| 场景 | 命令 |
|---|---|
| **rc2027 完整赛场** | `ros2 launch mujoco_ros2_bridge bridge.launch.py` |
| **20×20 空白场** | `ros2 launch mujoco_ros2_bridge bridge_empty.launch.py` |
| **L2 对抗场（"申"字形）** | `ros2 launch mujoco_ros2_bridge bridge_l2.launch.py` |

三个 launch 都会自动起 `nav_node` 和 Foxglove。

---

## 二、场景与地图

**PGM 地图和 MuJoCo 的 XML 没有任何关联**——改一个不会自动改另一个，必须手工保持一致。

| 场景 | 地图 | 尺寸 | 内容 |
|---|---|---|---|
| **rc2027** | `maps/rc2027.yaml` | 11.2 m | 完整赛场（围栏、L1/L2 台面、坡道、台阶） |
| **L2 对抗** | `maps/l2_arena.yaml` | 5.5 × 3.1 m | "申"字形：中央 3×3 + 东西两翼 1.2×1.2 |
| **空白场** | `maps/empty_20x20.yaml` | 20 m | 纯空白 |

调试用地地图：

| 文件 | 内容 |
|---|---|
| `debug_arena_10m/20m` | 空白 + 10 cm 围栏，**实车调试推荐**（纯空白没有边界，A\* 会往外规划） |
| `debug_empty_10m` | 纯空白，用来看 A\* 在无障碍下的行为 |

### 自造地图

1. 画一张 PGM（0 = 障碍，254 = 空闲，205 = 未知），分辨率 0.05 m/格
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

3. `-p map:=your_map` 启动（自动补 `maps/` 和 `.yaml`，也接受绝对路径）

---

## 三、参数

### 车体与静态膨胀

| 参数 | 默认 | 含义 |
|---|---|---|
| `car_length` / `car_width` | 0.60 / 0.60 m | 车体外廓 |
| `car_sweep_radius` | 0.424 m | 外接圆 = `hypot(L,W)/2`。「车体到障碍」要按它算 |
| `inscribed_radius` | 0.45 m | 小于此距离 → `LETHAL`（不可通行） |
| `inflation_radius` | 0.90 m | 大于此距离 → 无代价；中间按 2.5 次幂衰减 |

`inscribed_radius` 是「车体最坏朝向下的外接圆半径」，不是车宽一半。取小了会让 A\* 认为
能走的地方车体实际侵入。

### 路径平滑

三步：视线拉直 → 圆弧倒角 → 5 cm 稠密重采样。**没有开关，永远开启。**

| 参数 | 默认 | 含义 |
|---|---|---|
| `smooth_loss` | 0.03 m | 允许比原路径再贴墙多少。调大会让圆弧切得更狠但余量被吃掉 |
| `smooth_radius` | 0.35 m | 圆弧半径，放不下时自动缩到 0.75/0.5/0.35 倍 |
| `smooth_adaptive` | 0.58 m | 折点附近余量小于它就不倒角（窄区保护） |

平滑失败时**整条回退 A\* 原样**——宁可锯齿也不给一条更急的路径。

### 速度剖面

速度只由路径的**拐弯**决定：直道一律 `speed_wide`，弯道按曲率/转向速率压，掉头按制动距离压。
逐点取最小值后再做一次「后向刹车 + 前向加速」包络扫描。

| 参数 | 默认 | 含义 |
|---|---|---|
| `ref_speed` / `speed_wide` | 4.0 m/s | 直道最高速 |
| `speed_min` | 1.20 m/s | 弯道速度下限。视野 = `v·dt·N`，压低它会缩短前视距离 |
| `lat_acc_max` | 7.0 m/s² | 横向加速度上限 |
| `k_steer_limit` | 3.4 | 转向速率限速系数，`v <= k·R` |
| `dalpha_max` | 6.0 rad/s | 转向速率上限，**必须与 `mpc_acados.py` 的 `dα` 约束一致** |
| `dv_max` | 3.0 m/s² | 折返制动能力，**必须与 `dv` 约束一致** |
| `speed_lookahead` | 2.4 m | 取 `[s, s+它]` 区间内的最小限速 |

### 动态避障

```python
self.enable_dynamic_obstacles = True     # 关掉后 set_scan / set_obstacle_points 直接 return
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `dynamic_inflate_radius` | 0.40 m | 障碍周围按 `LETHAL` 处理的半径 → A\* 绕它多远 |
| `block_threshold` | 0.30 m | 障碍表面距**剩余路径**多近才触发重规划 |
| `replan_interval` | 0.25 s | 最快重规划间隔 |
| `blocked_retry_interval` | 0.60 s | 被堵死时多久重试 |
| `dyn_stop_gap` | 0.10 m | 净间隙 ≤ 此值 → 速度 0 |
| `dyn_full_gap` | 2.00 m | 净间隙 ≥ 此值 → 不限速 |
| `dyn_brake_a` | 3.0 m/s² | 中间段 `v = sqrt(2·a·(gap − stop_gap))` |

**两个判据用的距离口径不同，别混：**

| 模块 | 口径 | 管什么 |
|---|---|---|
| `_speed_cap_from_obstacles` | **我方车体 ↔ 对方车体** | 两车贴多近 → 该多慢 |
| `_obstacle_blocks_path` | **障碍 ↔ 剩余路径** | 路径是否被挡 → 要不要重规划 |

**约束：`block_threshold` 必须小于 `dynamic_inflate_radius`**，否则 A\* 刚绕开又判挡路，
变成每 `replan_interval` 重规划一次、路径乱变。

**净间隙 ↔ 实际距离**：`车心距 = gap + car_sweep_radius + 对方外接圆半径`。
对方是 650×650 方块时外接圆 0.46 m，所以 `gap=0.10 ⇔ 0.98 m`、`gap=2.00 ⇔ 2.88 m`。

数据链路：

```
真机 /scan  ──┐
              ├─→ nav.set_scan / set_obstacle_points ─→ _dynamic_points
仿真 /obstacles┘                                            ↓
                             手动 /debug_obstacle ─→ _manual_points（独立层）
                                                    ↓
                                    _planning_cost()：膨胀成 LETHAL
                                                    ↓
                    _obstacle_blocks_path() → 触发 set_goal(is_replan=True)
```

### 脱困

`stuck_frames_threshold = 30`：连续 30 帧没有进展就触发，用 `_kick_direction()` 沿
「指向前方最近参考点」的方向强制推一下。

### 规划器

```python
GridPlanner(costmap, theta=False)    # False = 标准 8 邻域 A*（当前使用）
```

Theta\* 能砍掉一半折点，但规划慢 2~3 倍、路径更贴墙、平滑后曲率不稳定。

---

## 四、MPC

```python
mpc_dt = 0.01        # 100 Hz，必须与实际调用频率一致
mpc_horizon = 175    # 预测 1.75 s
```

时间尺度必须匹配调用频率，否则模型里的加速度会被放大若干倍。

| 约束 | 值 | 说明 |
|---|---|---|
| `v` | ±5.0 m/s | 速度上限（一般够不到） |
| `α` | ±20π | 只作数值防爆，不是物理约束。**内部状态不做归一化** |
| `ω` | 0 | 车头锁死 |
| `dv` | ±3.0 | 与 `dv_max` 同步 |
| `dα` | ±6.0 | 与 `dalpha_max` 同步 |

**权重（`W`，10 项）**：`[pos_x, pos_y, heading, lateral, v, α, ω, dv, dα, dω]`

```python
W   = [2.0, 2.0, 1.47, 12.0, 0, 0, 0, 0.01, 0.05, 0.2]
W_e = [4.0, 4.0, 0.2, 4.0, 0.2, 0.2, 0.2]
```

- **横向项（`W[3]`）** 是抑制切内角的主力
- **方向项（`W[2]`）** 本质是横向回线的阻尼：`τ = 1/(v·sqrt(q_lat/W[2]))`，
  `q_lat = W[0]+W[3]`。`τ` 必须大于转舵滞后 `t_lag = Δα/dα_max`（30° @ 6 rad/s → 0.087 s）
- **`W[8]`（dα）** 要和方向项一起看：打满舵代价 `W[8]·dα²` 大于歪 90° 代价
  `W[2]·(π/2)²` 时，优化器宁可歪着也不打舵

改 `W`、`dα`、`n_horizon` 都要重新编译：

```bash
rm -rf c_generated_code mpc_ocp.json
python3 -c "from nav import Navigation; Navigation('maps/rc2027.yaml')"
```

---

## 五、调试

### Foxglove 可视化话题

| 话题 | 类型 | 内容 |
|---|---|---|
| `/costmap` | `OccupancyGrid` | 静态 + 动态障碍的代价地图 |
| `/plan` | `Path` | 平滑后的参考路径（**重规划后会刷新**） |
| `/plan_corridor` | `MarkerArray` | 路径两侧警戒边界（宽度 = `dynamic_inflate_radius`） |
| `/brake_zone` | `MarkerArray` | 车周围刹车圈（红）与预警圈（黄） |
| `/mpc_ref` | `MarkerArray` | MPC 参考点 |
| `/debug_obstacles` | `MarkerArray` | 手动注入的障碍 |
| `/odom` / `/cmd_vel` | — | 位姿 / 控制输出 |
| `/obstacles` | `ObstacleArray` | 仿真障碍真值 |

### 手动注入障碍

| 话题 | 类型 | 作用 |
|---|---|---|
| `/debug_obstacle` | `PointStamped` | 点一下加一个障碍（累加） |
| `/debug_obstacle_clear` | `Empty` | 清空 |

在 Foxglove 3D 面板用 "Publish point" 发到 `/debug_obstacle`。手动障碍存在独立一层
（`_manual_points`），不会被 `/obstacles` 每帧冲掉。

### 监控量

```python
nav.phase                 # IDLE / TRACK / ALIGN / ARRIVED
nav.blocked               # 是否被堵死
nav.last_error            # 最近一次 set_goal 失败的原因
nav.last_lateral          # 横向偏差 [m]，正 = 偏左
nav.path_version          # 每次规划 +1，用于检测重规划
nav.mpc.last_aug_state    # (v, α, ω)，α 不归一化
nav.mpc.ref_speed         # 当前参考推进速度
nav.raw_path              # A* 原始输出
nav.path                  # 平滑后
```

---

## 六、坐标系

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
通道中心离墙余量小于车体外接圆时，圆弧放不下，那段保留 A\* 原折线。

**② `inscribed_radius` 与场地绑定**
设得过大（如 0.45 以上遇到窄口）会把连通区域切成互不连通的几块，A\* 直接无解。
换场地要重新标定。

**③ 重规划同步阻塞**
`set_goal()` 在 `update()` 里直接调用，A\* + 平滑约 70~150 ms，期间控制输出冻结。
障碍加得太近会来不及。

**④ 车头锁死**
`ω ≡ 0`，转向完全靠 `α`。好处是全向移动，坏处是弯道受 `dα` 限制——
半径 0.35 m 的弯最高只能约 1 m/s。

---

## 八、目录结构

```
nav/
  navigation.py      主逻辑：规划、限速、状态机、动态避障
  astar.py           栅格 A*
  path_smooth.py     平滑：拉直 → 倒角 → 重采样
  map.py             nav2 格式地图加载与膨胀
  grid.py            栅格索引/坐标转换
  mpc_core/
    mpc_acados.py    MPC 建模与求解
    linear.py        路径参数化（SplinePlanner）
maps/                地图（.yaml + .pgm）
sim/                 MuJoCo 仿真桥与 ROS 接口
tests/               导航节点、自检脚本
```

---

## 九、自检

```bash
# 语法 + 未定义名称检查（能抓住 NameError 类问题）
python3 tests/check_names.py

# 地图能否加载
python3 -c "from nav.map import load_map; load_map('maps/rc2027.yaml')"
```

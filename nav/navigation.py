"""导航主类：对外只暴露三个方法，主逻辑直接 import 使用。

    from nav import Navigation
    nav = Navigation(map_yaml='...')
    nav.set_goal(x, y, yaw)          # 规划
    cmd = nav.update(x, y, yaw)      # 每帧喂位姿，拿 (vx, vy, wz)

运动策略（三阶段）：

    ① 平移   车头锁死（MPC 内部 vw = 0），靠 alpha 让底盘朝任意方向移动；
             反向移动靠倒车（v < 0），避免 alpha 卡在 ±pi 的约束边界
    ② 到位   位置误差 < goal_tolerance
    ③ 自转   原地 P 控制转到目标朝向，误差 < rotate_tolerance

为什么车头要锁死：舵轮底盘的车头方向与运动方向解耦，平移时不需要转头。
"""
from __future__ import annotations

import math
import time

import numpy as np

from .astar import GridPlanner, nearest_free
from .map import LETHAL, Costmap, inflate, load_map
from .mpc_core.mpc_acados import AugmentedSwerveMPC
from .path_smooth import smooth_path

# 阶段常量
IDLE = 'IDLE'
TRACK = 'TRACK'
ALIGN = 'ALIGN'
ARRIVED = 'ARRIVED'


def wrap_angle(a: float) -> float:
    """把角度归一化到 [-pi, pi]。"""
    return math.atan2(math.sin(a), math.cos(a))


class Navigation:
    """A* 全局规划 + acados MPC 局部跟踪，纯 Python，不依赖 ROS。"""

    def __init__(self, map_yaml: str, *,
                 car_length: float = 0.60,
                 car_width: float = 0.60,
                 inscribed_radius: float = 0.45,
                 inflation_radius: float = 0.90,
                 cost_weight: float = 2.0,
                 ref_speed: float = 4.0,
                 mpc_dt: float = 0.01,
                 mpc_horizon: int = 175,
                 v_max: float = 5.0,
                 goal_tolerance: float = 0.05,
                 rotate_tolerance: float = 0.05,
                 rotate_gain: float = 2.5,
                 rotate_max: float = 1.2,
                 smooth_loss: float = 0.03,
                 smooth_radius: float = 0.35,
                 smooth_adaptive: float = 0.58,
                 speed_lookahead: float = 1.2,
                 dalpha_max: float = 4.0,
                 dv_max: float = 3.0,) -> None:
        # ---- 地图 ----
        base, occupied = load_map(map_yaml)
        cost = inflate(occupied, base.spec.resolution,
                       inscribed_radius=inscribed_radius,
                       inflation_radius=inflation_radius)
        cost[base.data == 255] = LETHAL
        self.costmap = Costmap(data=cost, spec=base.spec)
        # 静态图保持不变；动态障碍叠加在副本上，不改原图
        self._static_cost = cost.copy()
        # ★ inscribed_radius 的物理含义是"车体最坏朝向下的外接圆半径"，不是车宽一半。
        #   车体 0.70x0.65 → 外接圆 0.478 m。车头锁死时车身可能斜 45° 移动，
        #   此时需要 0.478；只有正交移动才是 0.325。
        #   之前设 0.35/0.40 意味着 A* 认为"能走"的地方，车体实际侵入 13/8 cm ——
        #   这是"过直角墙就撞"的根因，跟平滑无关。
        self.inscribed_radius = float(inscribed_radius)
        self._inscribed_cells = max(1, int(round(inscribed_radius / base.spec.resolution)))
        self.cost_weight = float(cost_weight)

        # ---- 路径平滑 ----
        # A* 是八邻域搜索，产物必然是 0/45/90° 的阶梯。CubicSpline 是**插值**，
        # 会忠实复现这个阶梯 —— 实测最小曲率半径只有 2.6 cm，而车长 0.70 m，
        # 物理上跟不出来（表现就是拐角处切内角、横向偏 15 cm）。
        #
        # 平滑器把阶梯换成"直线段 + 圆弧"，曲率能到 40~50 cm。三步：
        #   ① 视线拉直 ② 圆弧倒角 ③ 5 cm 稠密重采样
        # 圆弧上的每一点都检查离障 >= smooth_margin，窄处干脆不倒角。
        #
        # smooth_loss 是"允许平滑后比原路径再贴墙多少"。
        # 两个方向都有代价，务必小心：
        #   loss 大 → 圆弧能切得更狠、曲率更好，但实测撞墙。
        #     对角任务 A* 离障 0.55 m，车头锁死斜 45° 扫掠半宽 0.477 m，
        #     本来只剩 7 cm；loss=0.06 直接把余量吃到 2 cm，跟踪误差一上来就撞。
        #   loss 小 → 保住了余量，但圆弧放不下，曲率只改善到 20 cm 左右。
        #
        # 实测（1.1 m 门 + 0.65 m 车）：
        #   loss=0.06 → 离障 0.50 m  曲率 43 cm   ← 撞墙（实测）
        #   loss=0.03 → 离障 0.55 m  曲率 35 cm
        #   loss=0.00 → 离障 0.55 m  曲率 13 cm   ← 严格不劣化，保底方案
        #
        # ★ 直角拐角要特别注意：R 的圆弧在 90° 角上会往墙里切 0.414·R。
        #   R=0.50 → 切 20.7 cm；R=0.35 → 切 14.5 cm。远大于 loss。
        #   所以直角处的圆弧基本都会被拒（退化成尖角），这是**正确**的行为。
        self.smooth_loss = float(smooth_loss)          # 允许比原路径再让出多少 [m]
        self.smooth_radius = float(smooth_radius)      # 圆弧半径 [m]
        self.smooth_adaptive = float(smooth_adaptive)   # 折点附近余量小于它就不倒角 [m]

        # 到最近障碍的距离场（启动时算一次，约 60 ms），用来判断通道宽窄
        from scipy import ndimage
        self._dist_field = ndimage.distance_transform_edt(~occupied) * base.spec.resolution

        self.speed_wide = float(ref_speed)
        self.car_length = float(car_length)
        self.car_width = float(car_width)

        # ★ 车体外接圆半径 = 车头锁死、车身斜 45° 时需要的包络。
        #   这才是"车体到障碍"真正要的距离，比车宽一半（0.325）大 47%。
        #   限速的余量必须按它算，否则会高估 3 倍（实测：门处真实余量 7 cm，
        #   而按车宽算出来是 22.5 cm，于是给了 0.75 m/s —— 一偏就撞）。
        self.car_sweep_radius = 0.5 * math.hypot(self.car_length, self.car_width)

        # 余量 = 到墙距离 − 车体外接圆 = **还能横向偏多少不碰车体**
        # 限速前瞻长度 [m]：取 [s_now, s_now+它] 区间内的最小限速
        self.speed_lookahead = float(speed_lookahead)

        # ---- α 抑制：车头没对正路径方向就不许提速 ----
        #   限速表只看"路径本身"（余量、曲率），看不到车头还歪着。
        #   U 型弯出弯后路径已经变直、限速立刻放开，而 α 还要 1 s 才转完，
        #   车就会斜着加速冲出去。这里用 α 直接压一档。
        #   |α| <= free 不惩罚；超过 free 后线性衰减，到 free+span 时压到 floor。
        # 只在曲率大于 alpha_curve_kappa（R < 2 m）的地方启用 —— 直道不参与
        self.alpha_curve_kappa = 0.50
        self.alpha_speed_free = 0.20     # rad（约 11°）
        self.alpha_speed_span = 0.90     # rad
        self.alpha_speed_floor = 0.35    # 最低压到 35%
        # 转向角速率的物理上限 [rad/s]，要和 mpc_acados.py 里 dα 的约束一致。
        # 车头锁死时拐弯全靠 α：v/R <= dα → 弯道速度上限 = dα·R。
        self.dalpha_max = float(dalpha_max)

        # 纵向加速度上限 [m/s²]，要和 mpc_acados.py 里 dv 的约束一致。
        # 掉头/折返不靠转弯，靠 v 变号（α 不变），真正的约束是制动距离：
        #     v <= sqrt(2 · dv · 可用横向空间)
        self.dv_max = float(dv_max)


        # 最近一帧的横向偏差 [m]，正=偏左（路径法向）。供外部监控/可视化用。
        self.last_lateral = 0.0
        # 转角限速：v <= sqrt(a_lat / kappa)。
        # 参考点是按时间等距推进的，转角处路径弯曲而参考点"直线前进"，
        # MPC 会在"贴路径"和"追参考点"之间反复权衡，表现为来回修正、绕圈。
        # 按曲率把推进速度降下来，两者就不冲突了。
        # 曲率限速：v <= sqrt(a_lat / κ)
        # 原来设 100，导致 sqrt(100/2.85)=5.9 m/s，比限速上限还高 —— 这一项等于没写，
        # 拐角处照样 1.2 m/s 冲过去。降到 2.5 后 κ=2.85 的弯限到 1.03 m/s。
        # 实测总时间几乎不变（拐角弧长占比极小），但拐角是真降速了。
        self.lat_acc_max = 3.0          # 允许的最大横向加速度 [m/s²]
        self.speed_min = 0.20         # 再慢也不低于这个

        # 路径上每点的限速（set_goal 时算）
        self._path_s: np.ndarray | None = None
        self._path_vmax: np.ndarray | None = None
        self._path_kappa: np.ndarray | None = None

        # ---- 动态障碍（雷达）----
        # 动态避障总开关。默认关闭 —— 底盘还没接雷达时不会有任何数据，
        # 保持关闭也能避免"订阅了不存在的话题"这类困惑。
        # 接上 MS200 之后设 True（或者直接调 set_scan，会自动打开）。
        # 动态避障总开关。打开后 nav.set_scan / set_obstacle_points 才会存点，
        # update() 里才会做「障碍挡路 → 重规划」。
        # 数据源：真机是 LaserScan；仿真是 mujoco_node 发布的 /obstacles
        # （两个 mocap 对手），由 nav_node._on_obstacles 转成世界坐标点喂进来。
        self.enable_dynamic_obstacles = False
        self._dynamic_points: np.ndarray | None = None   # (N, 2) 世界坐标
        # 调试层：手动加的障碍（Foxglove 点选）。和 _dynamic_points 分开存，
        # 因为 set_scan / set_obstacle_points 是**全量替换**，会把手动加的冲掉。
        self._manual_points: np.ndarray | None = None
        # 手动障碍的**中心**（x, y, radius），给可视化用；_manual_points 是圆周采样
        self._manual_centers: list[tuple[float, float, float]] = []
        self._scan_dirty = False
        self._last_replan = 0.0
        self.replan_interval = 0.4        # 最快重规划间隔 [s]
        self.blocked_retry_interval = 0.6 # 被堵死时多久重试一次 [s]
        self.obstacle_clearance = 0.05    # 动态障碍额外余量 [m]

        # ---- MPC ----
        self.mpc = AugmentedSwerveMPC(dt=mpc_dt, n_horizon=mpc_horizon)
        self.ref_speed = float(ref_speed)
        self.v_max = float(v_max)
        self.goal_tolerance = float(goal_tolerance)
        self.rotate_tolerance = float(rotate_tolerance)
        self.rotate_gain = float(rotate_gain)
        self.rotate_max = float(rotate_max)

        # ---- 状态 ----
        self._phase = IDLE
        self._goal: tuple[float, float, float] | None = None
        self._path: np.ndarray | None = None
        self._raw_path: np.ndarray | None = None      # 平滑前的 A* 栅格路径
        self._blocked = False       # 当前是否因障碍无法前进
        # ---- 死锁脱困 ----
        # MPC 的动力学 f=[v·cos(θ+α), v·sin(θ+α)] 在 v=0、α=0 处，
        # 若路径方向和车头差 90°，两个控制量都没有梯度，会永久卡住。
        # 不放宽 MPC 内部（改一次崩一次），只在外层检测到卡死时
        # 直接给一个小速度把车"推"出死锁，之后 MPC 自己接管。
        self._stuck_frames = 0
        self._anchor = None        # 判断位移的锚点
        self.stuck_frames_threshold = 30   # 约 0.3 s（10 ms 一帧）
        self.kick_speed = 0.15             # 脱困速度 [m/s]
        self.last_error = ''        # 最近一次 set_goal 失败的原因

    # ================================================================
    # 对外接口
    # ================================================================
    @property
    def phase(self) -> str:
        """当前阶段：IDLE / TRACK / ALIGN / ARRIVED"""
        return self._phase

    @property
    def goal(self) -> tuple[float, float, float] | None:
        return self._goal

    def set_goal(self, x: float, y: float, yaw: float = 0.0,
                 pose: tuple[float, float, float] | None = None) -> bool:
        """规划到目标。返回 False 表示不可达（此时状态不变）。

        ``pose`` 是当前位姿，省略时用上一次 ``update`` 传入的值。
        """
        start = pose if pose is not None else self._last_pose
        if start is None:
            raise RuntimeError('还没有位姿，请先调 update() 或显式传 pose')

        spec = self.costmap.spec
        si, sj = spec.world_to_index(start[0], start[1])
        gi, gj = spec.world_to_index(x, y)

        # 用「静态 + 动态障碍」的图来规划
        plan_cost = self._planning_cost()
        height, width = plan_cost.shape
        if not (0 <= si < width and 0 <= sj < height):
            self.last_error = '起点在图外'
            self._blocked = True
            return False

        # 起点**强制**可通行：车就在这儿，不管图上怎么说都必须从这儿出发。
        # 之前用 nearest_free 找 40 cm 内的空格，车停在围栏旁或被动态障碍
        # 的膨胀覆盖时，找不到空格就直接规划失败 —— 表现为"发了目标车不动"。
        # ★ 开洞半径要覆盖车体外接圆，不能只开 3x3（15 cm）。
        #   动态障碍的膨胀半径是 0.50 m：车一旦冲进去，3x3 的洞外面全是
        #   LETHAL，A* 从起点就迈不出去 → 无解 → _blocked 永久卡死。
        r_open = max(1, int(math.ceil(self.car_sweep_radius / spec.resolution)))
        plan_cost[max(0, sj - r_open):min(height, sj + r_open + 1),
                  max(0, si - r_open):min(width, si + r_open + 1)] = 0
        s = (si, sj)

        # 终点允许吸附：用户可能点到了围栏上
        passable = plan_cost < 254
        g = nearest_free(passable, (gi, gj), 20)      # 半径 8 -> 20（1 m）
        if g is None:
            self.last_error = '目标附近 1 m 内没有可通行格（点到障壁里了？）'
            self._blocked = True
            return False

        planner = GridPlanner(Costmap(data=plan_cost, spec=spec),
                              cost_weight=self.cost_weight)
        indices = planner.plan(s, g)
        if indices is None:
            self.last_error = 'A* 无解（起点被围死，或目标不可达）'
            self._blocked = True
            return False

        points = np.array([spec.index_to_world(i, j) for i, j in indices])
        self._raw_path = points
        points = smooth_path(points, self._dist_field, spec,
                             min_clearance=self.inscribed_radius,
                             loss=self.smooth_loss,
                             radius=self.smooth_radius,
                             adaptive=self.smooth_adaptive)
        self.mpc.set_path(points, target_yaw=float(yaw), ref_speed=self.ref_speed)

        self._goal = (float(x), float(y), float(yaw))
        self._path = points
        self._build_speed_profile(points)
        self._phase = TRACK
        self._blocked = False
        self.last_error = ''
        self._stuck_frames = 0
        self._anchor = None
        self._last_replan = time.time()
        return True

    def update(self, x: float, y: float, yaw: float):
        """喂当前位姿，返回 ``(vx, vy, wz)``；返回 None 表示已到达并停稳。"""
        self._last_pose = (float(x), float(y), float(yaw))

        if self._phase == IDLE:
            return None
        if self._phase == ARRIVED:
            return None

        # ---- 阶段 ③：原地自转 ----
        if self._phase == ALIGN:
            err = wrap_angle(self._goal[2] - yaw)
            if abs(err) <= self.rotate_tolerance:
                self._phase = ARRIVED
                self.mpc.following = False
                return None
            wz = max(-self.rotate_max, min(self.rotate_max, self.rotate_gain * err))
            return (0.0, 0.0, float(wz))

        # ---- 阶段 ②：到达判定 ----
        end = self.mpc.path_planner
        if len(end.x_path) > 0:
            dist = math.hypot(end.x_path[-1] - x, end.y_path[-1] - y)
            if dist <= self.goal_tolerance:
                self.mpc.following = False
                err = wrap_angle(self._goal[2] - yaw)
                if abs(err) <= self.rotate_tolerance:
                    self._phase = ARRIVED
                    return None
                self._phase = ALIGN
                return (0.0, 0.0, 0.0)      # 先停一帧再转

        # ---- 动态障碍重规划 ----
        # 只在「有新的雷达数据 + 距上次规划够久 + 障碍确实挡在路径上」时才算，
        # 否则每帧都跑 A* 会把 CPU 吃满（一次 50~70 ms）。
        if (self._scan_dirty
                and time.time() - self._last_replan > self.replan_interval
                and self._obstacle_blocks_path()):
            self._scan_dirty = False
            self._last_replan = time.time()
            if not self.set_goal(self._goal[0], self._goal[1], self._goal[2],
                                 pose=self._last_pose):
                # 被堵死：先停下来等障碍移开，主逻辑可通过 nav.blocked 判断
                self._blocked = True
            else:
                self._blocked = False

        if self._blocked:
            # ★ 被堵住时必须定期重试，否则永久卡死。
            #   障碍移开、或者车被推到别处之后，这里不重试就永远醒不过来。
            if (self._goal is not None
                    and time.time() - self._last_replan > self.blocked_retry_interval):
                self._last_replan = time.time()
                if self.set_goal(self._goal[0], self._goal[1], self._goal[2],
                                 pose=self._last_pose):
                    self._blocked = False
                else:
                    return (0.0, 0.0, 0.0)
            else:
                return (0.0, 0.0, 0.0)

        # ---- 横向偏差监控 ----
        # 这是"车实际偏离规划路径多少"，和 MPC 代价里那个 lateral_err
        # （预测未来的偏差）不是一回事。外部可以读 nav.last_lateral 做告警/降速。
        self.last_lateral = self._lateral_error(x, y)

        # ---- 按横向余量调整参考速度 ----
        self._apply_speed_limit()

        # ---- 阶段 ①：MPC 跟踪 ----
        u = self.mpc.update(np.array([x, y, yaw]))
        vx, vy, wz = float(u[0]), float(u[1]), float(u[2])

        # ---- 死锁脱困 ----
        # 判定用「位移」而不是「速度」：
        #   - 完全卡死时速度恒 0
        #   - 还有一种是来回振荡（+0.09/-0.07/+0.03...），速度不为 0
        #     但净位移始终为 0
        # 两种都要能抓到，所以看"最近 N 帧内位移有没有超过阈值"。
        if self._anchor is None:
            self._anchor = (x, y)
        moved = math.hypot(x - self._anchor[0], y - self._anchor[1])
        if moved < 0.02:
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0
            self._anchor = (x, y)

        if self._stuck_frames >= self.stuck_frames_threshold:
            kick = self._kick_direction(x, y, yaw)
            if kick is not None:
                vx, vy = kick[0] * self.kick_speed, kick[1] * self.kick_speed
                # 保持触发：只踢一两帧的话，车刚动起来 MPC 又振荡回原点，
                # 会变成"走两步停三步"。这里让踢的动作持续到车真的开始
                # 移动（位移超过 2 cm，_stuck_frames 会被重置），
                # 但设个上限防止永远踢下去。
                self._kick_frames = getattr(self, '_kick_frames', 0) + 1
                if self._kick_frames < 120:      # 最多踢 1.2 秒
                    self._stuck_frames = self.stuck_frames_threshold
                else:
                    self._kick_frames = 0
        else:
            self._kick_frames = 0

        speed = math.hypot(vx, vy)
        if speed > self.v_max and speed > 1e-9:
            k = self.v_max / speed
            vx, vy = vx * k, vy * k

        return (vx, vy, wz)

    # ================================================================
    # 按通道宽度限速
    # ================================================================
    def _build_speed_profile(self, points: np.ndarray) -> None:
        """给路径上每个点算一个速度上限，越窄越慢。

        宽度 = 2 × 到最近障碍的距离。1 m 的门对 0.65 m 的车只有
        17.5 cm 单边余量，全速通过会因侧滑贴着围栏走。
        """
        spec = self.costmap.spec
        res = spec.resolution
        height, width = self._dist_field.shape

        i = np.round((points[:, 0] - spec.origin_x) / res).astype(np.int64)
        j = np.round((points[:, 1] - spec.origin_y) / res).astype(np.int64)
        ok = (i >= 0) & (i < width) & (j >= 0) & (j < height)
        dist = np.zeros(len(points), dtype=np.float64)
        dist[ok] = self._dist_field[j[ok], i[ok]]

        # 允许的横向偏差 = 到墙距离 − 车体外接圆（折返制动还要用）
        margin = dist - self.car_sweep_radius

        # 曲率（曲率限速、折返判定都要用）
        kappa = self._curvature(points)
        self._path_kappa = kappa          # 供 _apply_speed_limit 的 α 抑制判据用

        # ★ 已移除"窄路限速"（横向余量 → 速度的映射）和它的直道豁免。
        #   原来窄处降到 speed_floor、宽处跑满，是"离墙多近"在限制速度。
        #   现在速度只由路径的**拐弯**决定：直道一律 speed_wide，弯道按
        #   曲率/转向速率压，掉头按制动距离压。
        vmax = np.full(len(points), self.speed_wide, dtype=np.float64)


        # ---- 曲率限速：两个约束取较严的那个 ----
        #   ① 横向加速度   v <= sqrt(a_lat · R)
        #   ② 转向速率     v <= dα · R
        #
        # ★ ② 才是本项目的瓶颈：车头锁死（ω=0），拐弯只能靠 α，
        #   而 α 的变化率受 dα ≤ 2 rad/s 限制。半径 0.35 m 的弯：
        #       ① 给 0.94 m/s，② 只给 0.70 m/s
        #   原来只按 ① 算（而且 a_lat 还写到 100，完全失效），
        #   车以 1.0+ m/s 进弯，α 转不过来 → 冲出弯道 → 就是"过直角墙撞墙"。
        radius = 1.0 / np.maximum(kappa, 1e-3)
        v_lat = np.sqrt(self.lat_acc_max * radius)      # 横向加速度
        v_steer = self.dalpha_max * radius              # 转向速率
        v_curve = np.minimum(v_lat, v_steer)

        # ---- 掉头 / 折返：不能用曲率限速 ----
        # 全向车折返靠 v 变号（α 不用变、车头不转、扫掠宽度不变），
        # 根本不存在"转弯半径"。而曲率在这种地方会算出 R≈5 cm
        # （5 cm 采样步长 + 两段平行反向造成的数值假象），
        # v_steer = dα·R ≈ 0.15，再被 speed_min 兜底到 0.20 —— 表现就是"刹停"。
        #
        # 折返真正的约束是**制动距离**：把 v 从 +v 变成 −v，
        #     v <= sqrt(2 · dv · 可用横向空间)
        # 宽处（余量 0.37 m）给 1.22 m/s，窄门（余量 0.07 m）给 0.53 m/s。
        theta = np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0]))
        turn = np.zeros(len(points))
        if len(theta) > 1:
            # theta 有 n-1 个，diff 之后 n-2 个，正好对应 turn[1..n-2]
            # （turn[k] = 第 k-1 段与第 k 段之间的转角）
            turn[1:-1] = np.abs(np.diff(np.unwrap(theta)))
        # 阈值取 30°：平滑出来的圆弧拐角，单点转角只有 arctan 步长/半径
        # ≈ 0.05/0.35 ≈ 8°；而真正 "平滑器放不下圆弧、被迫留下的尖角"
        # 单点转角是 45°~180°。两者差一个数量级，30° 能干净分开。
        sharp = turn > np.radians(30.0)
        if sharp.any():
            v_brake = np.sqrt(2.0 * self.dv_max * np.maximum(margin, 1e-3))
            v_curve = np.where(sharp, v_brake, v_curve)

        vmax = np.minimum(vmax, np.maximum(v_curve, self.speed_min))

        # 前后各 0.6 m 取最小：弯道的低速往两侧扩散，避免"到了才减"。
        # 放在曲率限速之后才有意义——这之前 vmax 是常数 speed_wide。
        from scipy.ndimage import minimum_filter1d
        look = max(1, int(round(0.6 / res)))
        vmax = minimum_filter1d(vmax, size=2 * look + 1, mode='nearest')

        # 沿弧长累计，供 update 查表
        seg = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
        s = np.concatenate(([0.0], np.cumsum(seg)))
        self._path_s = s
        self._path_vmax = vmax

    @staticmethod
    def _curvature(points: np.ndarray) -> np.ndarray:
        """离散曲率 kappa = |x'y'' - y'x''| / (x'^2+y'^2)^1.5。"""
        if len(points) < 3:
            return np.zeros(len(points))
        x, y = points[:, 0], points[:, 1]
        dx, dy = np.gradient(x), np.gradient(y)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        denom = (dx * dx + dy * dy) ** 1.5
        num = np.abs(dx * ddy - dy * ddx)
        return np.where(denom > 1e-9, num / np.maximum(denom, 1e-9), 0.0)

    def _lateral_error(self, x: float, y: float) -> float:
        """车相对规划路径的横向偏差 [m]（路径法向，左正右负）。"""
        end = self.mpc.path_planner
        if len(end.x_path) == 0:
            return 0.0
        try:
            s_now = end.get_nearest_s(x, y)
            ref = end.get_state_by_s(s_now)
        except (ValueError, IndexError):
            return 0.0
        nx, ny = -math.sin(ref[2]), math.cos(ref[2])   # 路径法向
        return float((x - ref[0]) * nx + (y - ref[1]) * ny)

    def _apply_speed_limit(self) -> None:
        """按车当前弧长位置设置 MPC 的参考推进速度。"""
        if self._path_s is None or self._last_pose is None:
            return
        end = self.mpc.path_planner
        if len(end.x_path) == 0:
            return
        s_now = end.get_nearest_s(self._last_pose[0], self._last_pose[1])

        # ★ 取"当前弧长 → 前方 speed_lookahead 米"这一段里的**最小**限速。
        #
        # 为什么不能用单点：单点只是换个位置去看，如果前瞻末端恰好是宽档，
        #   看到的还是高速（实测把单点前瞻从 0.6 拉到 1.5，总时间一秒没变）。
        #   而"直道后紧跟急转弯"的场景里，前方 1.2 m 处限速只有 0.57，
        #   单点 0.6 m 完全看不到 → 车在直道全速加速 → 到弯口来不及 → 冲出去。
        #
        # 为什么是 1.2 m：1.2→0.3 m/s 的减速距离 0.34 m，加 MPC 的响应余量 0.3 m，
        #   约 0.65 m 是理论下限；实测 0.6→1.0 还有 4% 差别，1.2 m 是稳妥折中
        #   （总时间 +6%，换来弯前真正降速）。
        s_end = min(s_now + self.speed_lookahead, self._path_s[-1])
        k0 = int(np.searchsorted(self._path_s, max(0.0, s_now)))
        k1 = int(np.searchsorted(self._path_s, s_end))
        k0 = max(0, min(k0, len(self._path_vmax) - 1))
        k1 = max(k1, k0 + 1)
        k1 = min(k1, len(self._path_vmax))
        ref = float(self._path_vmax[k0:k1].min())

        # ★ α 抑制（见 __init__ 里的说明），**只在弯道附近启用**。
        #   直道上车头朝向不该影响速度：起步时车头没对准、或刚出弯 α 还没
        #   归零，都会让整段直道被误压。
        kappa_local = (float(self._path_kappa[k0])
                       if self._path_kappa is not None
                       and 0 <= k0 < len(self._path_kappa) else 0.0)
        if kappa_local > self.alpha_curve_kappa:
            try:
                alpha = float(self.mpc.last_aug_state[1])
            except Exception:                  # noqa: BLE001 - 平滑绝不该让控制失败
                alpha = 0.0
            over = abs(alpha) - self.alpha_speed_free
            if over > 0.0:
                ref *= max(self.alpha_speed_floor,
                           1.0 - over / self.alpha_speed_span)

        self.mpc.ref_speed = ref

    # ================================================================
    # 动态障碍
    # ================================================================
    def set_scan(self, msg) -> None:
        """传入一帧 ``sensor_msgs/LaserScan``（车体坐标系）。

        内部会：过滤无效值 → 极坐标转车体系 → 用最近位姿转世界系 →
        存成动态障碍点。**下一帧会全量替换**，所以障碍走开后自动消失。
        """
        if not self.enable_dynamic_obstacles:
            return
        if self._last_pose is None:
            return

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            return

        angles = msg.angle_min + msg.angle_increment * np.arange(ranges.size)
        rmin = float(getattr(msg, 'range_min', 0.0)) or 0.01
        rmax = float(getattr(msg, 'range_max', 100.0)) or 100.0

        valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
        if not valid.any():
            self._dynamic_points = None
            return

        r = ranges[valid]
        a = angles[valid]

        # 极坐标 → 车体系 → 世界系
        xb = r * np.cos(a)
        yb = r * np.sin(a)
        px, py, yaw = self._last_pose
        c, s = math.cos(yaw), math.sin(yaw)
        xw = px + c * xb - s * yb
        yw = py + s * xb + c * yb

        self._dynamic_points = np.column_stack([xw, yw])
        self._scan_dirty = True

    def set_obstacle_points(self, points) -> None:
        """直接设置动态障碍的世界坐标点（测试/仿真用，跳过坐标转换）。"""
        if not self.enable_dynamic_obstacles:
            return
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            self._dynamic_points = None
            return
        self._dynamic_points = pts
        self._scan_dirty = True

    def add_manual_obstacle(self, x: float, y: float,
                            radius: float = 0.12) -> None:
        """调试用：在 (x, y) 加一个圆形障碍，**累加**，不会被感知数据冲掉。

        给 Foxglove 点选测试用。内部按圆周采样成点，和感知进来的障碍
        走同一套膨胀和挡路判定。
        """
        if not self.enable_dynamic_obstacles:
            return
        n = max(8, int(round(2 * math.pi * radius / 0.05)))
        a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        ring = np.column_stack([x + radius * np.cos(a), y + radius * np.sin(a)])
        if self._manual_points is None or len(self._manual_points) == 0:
            self._manual_points = ring
        else:
            self._manual_points = np.vstack([self._manual_points, ring])
        self._manual_centers.append((float(x), float(y), float(radius)))
        self._scan_dirty = True
        # 手动障碍是"立刻出现"的，不该再等 replan_interval —— 等 0.4 s
        # 车已经开过去 0.5 m 了。
        self._last_replan = 0.0

    def clear_manual_obstacles(self) -> None:
        """清空手动加的障碍（不动感知层的）。"""
        self._manual_points = None
        self._manual_centers.clear()
        self._scan_dirty = True

    def _all_obstacle_points(self) -> np.ndarray | None:
        """感知障碍 + 手动障碍合并后的点集。"""
        parts = [p for p in (self._dynamic_points, self._manual_points)
                 if p is not None and len(p)]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else np.vstack(parts)

    def clear_obstacles(self) -> None:
        """清空动态障碍。"""
        self._dynamic_points = None
        self._manual_points = None
        self._manual_centers.clear()
        self._scan_dirty = False

    @property
    def blocked(self) -> bool:
        """当前是否因为障碍无法规划（主逻辑可据此决定停等还是换目标）。"""
        return self._blocked

    def _planning_cost(self) -> np.ndarray:
        """静态代价图 + 动态障碍（只膨胀到 inscribed_radius，避免双重膨胀）。"""
        cost = self._static_cost.copy()
        pts = self._all_obstacle_points()
        if pts is None:
            return cost

        spec = self.costmap.spec
        res = spec.resolution
        height, width = cost.shape
        radius = self._inscribed_cells + max(
            0, int(round(self.obstacle_clearance / res)))

        # 世界坐标 → 栅格索引
        i = np.round((pts[:, 0] - spec.origin_x) / res).astype(np.int64)
        j = np.round((pts[:, 1] - spec.origin_y) / res).astype(np.int64)
        inside = (i >= 0) & (i < width) & (j >= 0) & (j < height)
        if not inside.any():
            return cost
        i, j = i[inside], j[inside]

        # 圆形膨胀：把圆内所有偏移量一次性铺上去（numpy 向量化）
        r2 = radius * radius
        di = np.arange(-radius, radius + 1)
        dj = np.arange(-radius, radius + 1)
        ddi, ddj = np.meshgrid(di, dj, indexing='ij')
        mask = (ddi * ddi + ddj * ddj) <= r2
        off_i = ddi[mask]
        off_j = ddj[mask]

        ii = (i[:, None] + off_i[None, :]).ravel()
        jj = (j[:, None] + off_j[None, :]).ravel()
        keep = (ii >= 0) & (ii < width) & (jj >= 0) & (jj < height)
        cost[jj[keep], ii[keep]] = LETHAL
        return cost

    def _obstacle_blocks_path(self) -> bool:
        """动态障碍是否挡在「车当前位置之后」的路径上。

        不能用「看路径后半段」这种近似 —— 障碍挂在 1/3 处时会漏掉。
        正确做法是把车投影到路径上取弧长 s0，只看 s >= s0 的那部分。
        """
        pts = self._all_obstacle_points()
        if pts is None:
            return False

        end = self.mpc.path_planner
        if len(end.x_path) == 0:
            return False

        path = np.column_stack([end.x_path, end.y_path])

        # 只保留还没走过的那段
        if self._last_pose is not None and len(end.s_samples) == len(path):
            s0 = end.get_nearest_s(self._last_pose[0], self._last_pose[1])
            idx = int(np.searchsorted(end.s_samples, s0))
            path = path[max(0, idx - 1):]
        if len(path) == 0:
            return False

        # ★ 阈值必须**略小于 A* 自己保证的间隔**，否则永远判"挡路"。
        #
        # _dynamic_points 是障碍**边缘**的圆周采样点（半径 = 障碍半径），
        # 所以"采样点到路径的最短距离" = "障碍边缘到路径的间隙"，
        # 注意它**没有方向性**：正前方和侧方一样算。
        #
        # _planning_cost() 里动态障碍的膨胀半径是
        #     radius = _inscribed_cells + obstacle_clearance/res
        #            = 9 + 1 = 10 格 = 0.50 m
        # 也就是说 A* 规划出来的路径，离每个障碍边缘至少 0.50 m。
        #
        # 原来阈值取 inscribed*1.5 = 0.645 m > 0.50 —— A* 再怎么绕都够不到，
        # 于是每 0.4 s 判一次"挡路"、重规划一次，路径无限变化 → 车走直线也在摆。
        # 取 0.45 m（比 0.50 小 5 cm，留出栅格量化余量）才能真正区分
        # "绕开了"和"障碍压过来了"。
        threshold = 0.45
        for p in pts:
            d = np.hypot(path[:, 0] - p[0], path[:, 1] - p[1]).min()
            if d < threshold:
                return True
        return False

    def _kick_direction(self, x, y, yaw):
        """算出脱困方向（车体系单位向量）：指向「前方最近的参考点」。

        参考点比目标点好：它已经在路径上，不用再考虑绕障。
        """
        end = self.mpc.path_planner
        if len(end.x_path) == 0:
            return None
        s_now = end.get_nearest_s(x, y)
        s_look = min(s_now + 0.5, float(end.s_samples[-1]))
        st = end.get_states_batch(np.array([s_look]))[0]
        dx, dy = float(st[0]) - x, float(st[1]) - y
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return None
        # 世界系 → 车体系
        c, s = math.cos(yaw), math.sin(yaw)
        bx = c * dx + s * dy
        by = -s * dx + c * dy
        bn = math.hypot(bx, by)
        if bn < 1e-9:
            return None
        return (bx / bn, by / bn)

    def cancel(self) -> None:
        """放弃当前目标，回到 IDLE。"""
        self._phase = IDLE
        self._goal = None
        self._path = None
        self.mpc.following = False

    # ================================================================
    # 辅助
    # ================================================================
    _last_pose: tuple[float, float, float] | None = None

    @property
    def path(self) -> np.ndarray | None:
        """最近一次的规划路径 (N, 2)，供可视化用（平滑后）。"""
        return self._path

    @property
    def raw_path(self) -> np.ndarray | None:
        """平滑前的 A* 栅格路径，和 ``path`` 叠在一起就能看出平滑做了什么。"""
        return self._raw_path

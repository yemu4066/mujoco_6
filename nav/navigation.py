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
                 inscribed_radius: float = 0.35,
                 inflation_radius: float = 0.68,
                 cost_weight: float = 2.0,
                 ref_speed: float = 1.2,
                 mpc_dt: float = 0.05,
                 mpc_horizon: int = 35,
                 v_max: float = 4.0,
                 goal_tolerance: float = 0.15,
                 rotate_tolerance: float = 0.05,
                 rotate_gain: float = 2.5,
                 rotate_max: float = 1.2) -> None:
        # ---- 地图 ----
        base, occupied = load_map(map_yaml)
        cost = inflate(occupied, base.spec.resolution,
                       inscribed_radius=inscribed_radius,
                       inflation_radius=inflation_radius)
        cost[base.data == 255] = LETHAL
        self.costmap = Costmap(data=cost, spec=base.spec)
        # 静态图保持不变；动态障碍叠加在副本上，不改原图
        self._static_cost = cost.copy()
        self._inscribed_cells = max(1, int(round(inscribed_radius / base.spec.resolution)))
        self.cost_weight = float(cost_weight)
        self.planner = GridPlanner(self.costmap, cost_weight=cost_weight)

        # 到最近障碍的距离场（启动时算一次，约 60 ms），用来判断通道宽窄
        from scipy import ndimage
        self._dist_field = ndimage.distance_transform_edt(~occupied) * base.spec.resolution

        # 按通道宽度分档限速。**默认关闭**——让 MPC 自己按 4 m/s 的约束决定速度。
        # 需要重新打开时，把 speed_widths 填回去：
        #     ((1.30, 0.30), (2.00, 0.45))
        # 窄门必须降速：1 m 门、0.65 m 车，超过 0.5 m/s 时下坡侧滑会让车偏 15 cm
        self.speed_widths = ((1.30, 0.30), (2.00, 0.80))
        self.speed_wide = float(ref_speed)
        # 转角限速：v <= sqrt(a_lat / kappa)。
        # 参考点是按时间等距推进的，转角处路径弯曲而参考点"直线前进"，
        # MPC 会在"贴路径"和"追参考点"之间反复权衡，表现为来回修正、绕圈。
        # 按曲率把推进速度降下来，两者就不冲突了。
        self.lat_acc_max = 100.0        # 允许的最大横向加速度 [m/s²]
        self.speed_min = 0.20         # 再慢也不低于这个

        # 路径上每点的限速（set_goal 时算）
        self._path_s: np.ndarray | None = None
        self._path_vmax: np.ndarray | None = None

        # ---- 动态障碍（雷达）----
        self._dynamic_points: np.ndarray | None = None   # (N, 2) 世界坐标
        self._scan_dirty = False
        self._last_replan = 0.0
        self.replan_interval = 0.4        # 最快重规划间隔 [s]
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
        self._blocked = False       # 当前是否因障碍无法前进
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
        plan_cost[max(0, sj - 1):min(height, sj + 2),
                  max(0, si - 1):min(width, si + 2)] = 0
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
        self.mpc.set_path(points, target_yaw=float(yaw), ref_speed=self.ref_speed)

        self._goal = (float(x), float(y), float(yaw))
        self._path = points
        self._build_speed_profile(points)
        self._phase = TRACK
        self._blocked = False
        self.last_error = ''
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
            return (0.0, 0.0, 0.0)

        # ---- 按通道宽度调整参考速度 ----
        self._apply_speed_limit()

        # ---- 阶段 ①：MPC 跟踪 ----
        u = self.mpc.update(np.array([x, y, yaw]))
        vx, vy, wz = float(u[0]), float(u[1]), float(u[2])

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

        clearance = dist * 2.0        # 近似通道宽度
        vmax = np.full(len(points), self.speed_wide, dtype=np.float64)
        for thr, v in self.speed_widths:
            vmax = np.where(clearance < thr, np.minimum(vmax, v), vmax)

        # 曲率限速
        kappa = self._curvature(points)
        v_curve = np.sqrt(self.lat_acc_max / np.maximum(kappa, 1e-3))
        vmax = np.minimum(vmax, np.maximum(v_curve, self.speed_min))

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

    def _apply_speed_limit(self) -> None:
        """按车当前弧长位置设置 MPC 的参考推进速度。"""
        if self._path_s is None or self._last_pose is None:
            return
        end = self.mpc.path_planner
        if len(end.x_path) == 0:
            return
        s_now = end.get_nearest_s(self._last_pose[0], self._last_pose[1])
        # 往前看 2 m 取最严格的限制。
        # 视野只有 1.5 m 时是"到了窄道才减"，来不及；
        # 提前 2 m 减速，车到弯道/窄门前速度已经降下来了。
        look = s_now + 0.6
        k = int(np.searchsorted(self._path_s, min(look, self._path_s[-1])))
        k = max(0, min(k, len(self._path_vmax) - 1))
        self.mpc.ref_speed = float(self._path_vmax[k])

    # ================================================================
    # 动态障碍
    # ================================================================
    def set_scan(self, msg) -> None:
        """传入一帧 ``sensor_msgs/LaserScan``（车体坐标系）。

        内部会：过滤无效值 → 极坐标转车体系 → 用最近位姿转世界系 →
        存成动态障碍点。**下一帧会全量替换**，所以障碍走开后自动消失。
        """
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
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            self._dynamic_points = None
            return
        self._dynamic_points = pts
        self._scan_dirty = True

    def clear_obstacles(self) -> None:
        """清空动态障碍。"""
        self._dynamic_points = None
        self._scan_dirty = False

    @property
    def blocked(self) -> bool:
        """当前是否因为障碍无法规划（主逻辑可据此决定停等还是换目标）。"""
        return self._blocked

    def _planning_cost(self) -> np.ndarray:
        """静态代价图 + 动态障碍（只膨胀到 inscribed_radius，避免双重膨胀）。"""
        cost = self._static_cost.copy()
        if self._dynamic_points is None or len(self._dynamic_points) == 0:
            return cost

        spec = self.costmap.spec
        res = spec.resolution
        height, width = cost.shape
        radius = self._inscribed_cells + max(
            0, int(round(self.obstacle_clearance / res)))

        # 世界坐标 → 栅格索引
        i = np.round((self._dynamic_points[:, 0] - spec.origin_x) / res).astype(np.int64)
        j = np.round((self._dynamic_points[:, 1] - spec.origin_y) / res).astype(np.int64)
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
        if self._dynamic_points is None or len(self._dynamic_points) == 0:
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

        threshold = self._inscribed_cells * self.costmap.spec.resolution * 1.5
        for p in self._dynamic_points:
            d = np.hypot(path[:, 0] - p[0], path[:, 1] - p[1]).min()
            if d < threshold:
                return True
        return False

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
        """最近一次的规划路径 (N, 2)，供可视化用。"""
        return self._path

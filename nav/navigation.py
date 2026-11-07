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
                 inscribed_radius: float = 0.42,
                 inflation_radius: float = 0.68,
                 cost_weight: float = 2.0,
                 ref_speed: float = 0.5,
                 mpc_dt: float = 0.05,
                 mpc_horizon: int = 25,
                 v_max: float = 1.5,
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
        passable = plan_cost < 254
        s = nearest_free(passable, (si, sj), 8)
        g = nearest_free(passable, (gi, gj), 8)
        if s is None or g is None:
            self._blocked = True
            return False

        planner = GridPlanner(Costmap(data=plan_cost, spec=spec),
                              cost_weight=self.cost_weight)
        indices = planner.plan(s, g)
        if indices is None:
            self._blocked = True
            return False

        points = np.array([spec.index_to_world(i, j) for i, j in indices])
        self.mpc.set_path(points, target_yaw=float(yaw), ref_speed=self.ref_speed)

        self._goal = (float(x), float(y), float(yaw))
        self._path = points
        self._phase = TRACK
        self._blocked = False
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

        # ---- 阶段 ①：MPC 跟踪 ----
        u = self.mpc.update(np.array([x, y, yaw]))
        vx, vy, wz = float(u[0]), float(u[1]), float(u[2])

        speed = math.hypot(vx, vy)
        if speed > self.v_max and speed > 1e-9:
            k = self.v_max / speed
            vx, vy = vx * k, vy * k

        return (vx, vy, wz)

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
        """动态障碍是否落在「当前位置之后」的路径附近。"""
        if self._dynamic_points is None or len(self._dynamic_points) == 0:
            return False

        end = self.mpc.path_planner
        if len(end.x_path) == 0:
            return False

        path = np.column_stack([end.x_path, end.y_path])
        # 只检查路径后半段（已经走过的部分不用管）
        half = path[len(path) // 2:]
        threshold = self._inscribed_cells * self.costmap.spec.resolution * 1.5

        # 每个障碍点到最近路径点的距离
        for p in self._dynamic_points:
            d = np.hypot(half[:, 0] - p[0], half[:, 1] - p[1]).min()
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

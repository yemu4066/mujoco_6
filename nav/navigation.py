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
        self.planner = GridPlanner(self.costmap, cost_weight=cost_weight)

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

        passable = self.costmap.data < 254
        s = nearest_free(passable, (si, sj), 8)
        g = nearest_free(passable, (gi, gj), 8)
        if s is None or g is None:
            return False

        indices = self.planner.plan(s, g)
        if indices is None:
            return False

        points = np.array([spec.index_to_world(i, j) for i, j in indices])
        self.mpc.set_path(points, target_yaw=float(yaw), ref_speed=self.ref_speed)

        self._goal = (float(x), float(y), float(yaw))
        self._path = points
        self._phase = TRACK
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

        # ---- 阶段 ①：MPC 跟踪 ----
        u = self.mpc.update(np.array([x, y, yaw]))
        vx, vy, wz = float(u[0]), float(u[1]), float(u[2])

        speed = math.hypot(vx, vy)
        if speed > self.v_max and speed > 1e-9:
            k = self.v_max / speed
            vx, vy = vx * k, vy * k

        return (vx, vy, wz)

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

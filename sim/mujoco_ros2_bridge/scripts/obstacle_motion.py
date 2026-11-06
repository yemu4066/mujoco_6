"""移动障碍的轨迹：折线往返与匀速圆周。

只依赖 numpy / math，方便脱离 ROS 与 MuJoCo 做单元测试。
坐标均为场地世界系（米），z 由调用方按所在台面高度决定。
"""
from __future__ import annotations

import math

import numpy as np

# L1 台面上的巡逻路径：对方半场（x<0）西侧走廊 + 北侧公共边缘
L1_PATROL = ((-2.2, -2.5), (-2.2, 2.5), (-0.4, 2.5))


def patrol_state(points, speed: float, t: float):
    """沿折线匀速往返，返回 (x, y, vx, vy)。

    走到终点后原路返回，因此速度方向在折返点会翻转；
    在拐角处方向也会变，障碍层用瞬时速度做短时预测，影响很小。
    """
    pts = np.asarray(points, dtype=np.float64)
    segment = np.diff(pts, axis=0)
    lengths = np.hypot(segment[:, 0], segment[:, 1])
    total = float(lengths.sum())
    if total <= 0.0:
        return float(pts[0, 0]), float(pts[0, 1]), 0.0, 0.0

    cycle = 2.0 * total
    distance = (speed * t) % cycle
    forward = distance <= total
    if not forward:
        distance = cycle - distance

    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    index = int(np.searchsorted(cumulative, distance, side='right') - 1)
    index = max(0, min(index, len(lengths) - 1))
    local = (distance - cumulative[index]) / max(lengths[index], 1e-9)
    position = pts[index] + local * segment[index]
    direction = segment[index] / max(lengths[index], 1e-9)
    sign = 1.0 if forward else -1.0
    return (float(position[0]), float(position[1]),
            float(direction[0] * speed * sign),
            float(direction[1] * speed * sign))


def orbit_state(radius: float, omega: float, t: float):
    """绕原点匀速圆周，返回 (x, y, vx, vy)。"""
    angle = omega * t
    return (radius * math.cos(angle), radius * math.sin(angle),
            -radius * omega * math.sin(angle), radius * omega * math.cos(angle))

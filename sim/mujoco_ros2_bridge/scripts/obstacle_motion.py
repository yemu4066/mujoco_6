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

# ============================================================
# L2 对抗场专用：对方小车往返"东翼 ↔ L2 四角"
# ============================================================

def build_l2_timeline(cycles: int = 60, speed: float = 0.6,
                      home: tuple[float, float] = (2.1, 0.0),
                      corners=((-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)),
                      wait_home: float = 10.0, wait_corner: float = 3.0,
                      seed: int = 20271116):
    """预生成一条"东翼等待 → 随机角 → 等待 → 回东翼"的时间表。

    为什么预生成而不是每帧现算：
        mujoco_node 每帧都调这个函数，同一时刻 t 必须给出同一位置（否则
        障碍会跳）。而"随机选角"和"每段时长依赖距离"叠在一起，没法用简单
        的 t % period 表达。预生成几十个周期后按时间查表最省事，而且种子
        固定 → 可复现，调试时不会每次跑出不同的剧本。

    一个周期：
        东翼等 wait_home 秒
        匀速走到随机选中的角（时长 = 距离 / speed）
        在角上等 wait_corner 秒
        匀速走回东翼（时长同上）

    返回 (segments, total_time)，segments 每项是 (t_start, t_end, kind, data)：
        kind='wait'  data=位置
        kind='move'  data=(起点, 终点)
    """
    import random
    rng = random.Random(seed)
    hx, hy = home
    segs = []
    t = 0.0
    for _ in range(int(cycles)):
        cx, cy = rng.choice(list(corners))
        dt = math.hypot(cx - hx, cy - hy) / max(speed, 1e-6)

        segs.append((t, t + wait_home, 'wait', (hx, hy)))
        t += wait_home
        segs.append((t, t + dt, 'move', ((hx, hy), (cx, cy))))
        t += dt
        segs.append((t, t + wait_corner, 'wait', (cx, cy)))
        t += wait_corner
        segs.append((t, t + dt, 'move', ((cx, cy), (hx, hy))))
        t += dt
    return segs, t


def timeline_state(segs, total: float, speed: float, t: float):
    """按时间查表，返回 (x, y, vx, vy)。"""
    if not segs or total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    tt = t % total
    lo, hi = 0, len(segs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if segs[mid][1] <= tt:
            lo = mid + 1
        else:
            hi = mid
    t0, t1, kind, data = segs[lo]
    if kind == 'wait':
        return float(data[0]), float(data[1]), 0.0, 0.0
    p0, p1 = data
    span = max(t1 - t0, 1e-9)
    frac = min(max((tt - t0) / span, 0.0), 1.0)
    x = p0[0] + frac * (p1[0] - p0[0])
    y = p0[1] + frac * (p1[1] - p0[1])
    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if d < 1e-9:
        return float(x), float(y), 0.0, 0.0
    return (float(x), float(y),
            float((p1[0]-p0[0])/d*speed), float((p1[1]-p0[1])/d*speed))

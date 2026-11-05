"""栅格 A*，带代价权重与禁止斜穿墙角。

为什么不用现成的 Python A* 包：``jrialland/python-astar`` 为每个搜索节点
创建一个 ``SearchNode`` 对象，实测 224x224 场地规划一条 7.4 m 的路径要
**236 ms**；这里用「一维整型索引 + 预分配 numpy 数组 + heapq」的写法，
同样的地图和路径 **12 ms** 左右，和 C++ 版（11.6 ms）基本持平。

数据结构约定：

* 节点用一维索引 ``idx = j * width + i`` 表示，避免元组分配
* ``g`` / ``parent`` / ``closed`` 都是预分配的一维数组
* 堆里放 ``(f, seq, idx)``，``seq`` 是单调计数器，避免 f 相等时比较 idx

代价叠加：``step * (1 + cost_weight * normalized_cost)``。
``cost_weight = 0`` 退化为纯最短路；调大则更愿意绕开膨胀带。
"""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

SQRT2 = math.sqrt(2.0)

# (di, dj, 步长)
NEIGHBOURS = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
)

# 与发布时的 ``raw*100//254`` 压缩等价：/costmap 里 < 100 即可通行
PASSABLE_THRESHOLD = 254


class GridPlanner:
    """可复用的栅格规划器。

    把 ``g`` / ``parent`` / ``closed`` 这些工作数组**缓存在实例里**，
    每次规划只做 ``fill`` 而不是重新分配。224x224 的场地里这三个数组
    各 5 万格，每次重新分配约 400 KB，是长路径规划的一笔可观开销。
    """

    def __init__(self, costmap, *, cost_weight: float = 2.0,
                 threshold: int = PASSABLE_THRESHOLD,
                 prevent_corner_cutting: bool = True) -> None:
        self.costmap = costmap
        self.cost_weight = float(cost_weight)
        self.threshold = int(threshold)
        self.prevent_corner_cutting = bool(prevent_corner_cutting)

        height, width = costmap.data.shape
        self.width = width
        self.height = height
        self.size = height * width
        self.passable = costmap.data < np.uint8(self.threshold)
        self.flat_pass = self.passable.ravel()
        self.flat_cost = costmap.data.ravel().astype(np.float32) / 254.0

        self._g = np.empty(self.size, dtype=np.float64)
        self._parent = np.empty(self.size, dtype=np.int64)
        self._closed = np.empty(self.size, dtype=bool)

    def plan(self, start, goal):
        return astar(self.costmap, start, goal,
                     cost_weight=self.cost_weight,
                     threshold=self.threshold,
                     prevent_corner_cutting=self.prevent_corner_cutting,
                     _cache=(self._g, self._parent, self._closed,
                             self.flat_pass, self.flat_cost))


def astar(costmap, start: tuple[int, int], goal: tuple[int, int], *,
          cost_weight: float = 2.0,
          threshold: int = PASSABLE_THRESHOLD,
          prevent_corner_cutting: bool = True,
          _cache=None) -> list[tuple[int, int]] | None:
    """在代价地图上规划，返回格索引路径（含起终点）；无解返回 None。"""
    data = costmap.data
    height, width = data.shape
    si, sj = int(start[0]), int(start[1])
    gi, gj = int(goal[0]), int(goal[1])

    if not (0 <= si < width and 0 <= sj < height):
        return None
    if not (0 <= gi < width and 0 <= gj < height):
        return None

    if _cache is not None:
        g, parent, closed, flat_pass, flat_cost = _cache
        passable = None
    else:
        passable = data < np.uint8(threshold)
        flat_cost = data.ravel().astype(np.float32) / 254.0
        flat_pass = passable.ravel()
        g = np.empty(height * width, dtype=np.float64)
        parent = np.empty(height * width, dtype=np.int64)
        closed = np.empty(height * width, dtype=bool)

    if not (flat_pass[sj * width + si] and flat_pass[gj * width + gi]):
        return None
    if (si, sj) == (gi, gj):
        return [(si, sj)]

    size = height * width
    source = sj * width + si
    target = gj * width + gi
    w_cost = float(cost_weight)

    g.fill(np.inf)
    parent.fill(-1)
    closed.fill(False)

    def heuristic(idx: int) -> float:
        """八方向网格的 octile 距离，比欧氏更贴合真实步数。"""
        x = idx % width
        y = idx // width
        dx = abs(x - gi)
        dy = abs(y - gj)
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)

    counter = itertools.count()
    g[source] = 0.0
    heap = [(heuristic(source), next(counter), source)]

    while heap:
        _, _, current = heapq.heappop(heap)
        if current == target:
            return _reconstruct(parent, current, width)
        if closed[current]:
            continue
        closed[current] = True

        ci = current % width
        cj = current // width
        base_g = g[current]

        for di, dj, step in NEIGHBOURS:
            ni = ci + di
            nj = cj + dj
            if ni < 0 or ni >= width or nj < 0 or nj >= height:
                continue
            nidx = nj * width + ni
            if not flat_pass[nidx] or closed[nidx]:
                continue

            # 斜走时不允许穿过两个障碍的夹角
            if di and dj and prevent_corner_cutting:
                if not (flat_pass[cj * width + ni] and flat_pass[nj * width + ci]):
                    continue

            new_g = base_g + step * (1.0 + w_cost * float(flat_cost[nidx]))
            if new_g >= g[nidx]:
                continue

            g[nidx] = new_g
            parent[nidx] = current
            heapq.heappush(heap, (new_g + heuristic(nidx), next(counter), nidx))

    return None


def _reconstruct(parent: np.ndarray, last: int, width: int) -> list[tuple[int, int]]:
    """从终点沿 parent 回溯到起点，返回正序路径。"""
    cells: list[tuple[int, int]] = []
    node = int(last)
    while node >= 0:
        cells.append((node % width, node // width))
        node = int(parent[node])
    cells.reverse()
    return cells


def nearest_free(passable: np.ndarray, index: tuple[int, int],
                 radius: int = 8) -> tuple[int, int] | None:
    """起点/终点落在障碍上时，由近及远找最近的可通行格。"""
    height, width = passable.shape
    i0, j0 = index
    if 0 <= i0 < width and 0 <= j0 < height and passable[j0, i0]:
        return i0, j0

    for r in range(1, radius + 1):
        best, best_d2 = None, None
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                i, j = i0 + di, j0 + dj
                if not (0 <= i < width and 0 <= j < height):
                    continue
                if not passable[j, i]:
                    continue
                d2 = di * di + dj * dj
                if best_d2 is None or d2 < best_d2:
                    best, best_d2 = (i, j), d2
        if best is not None:
            return best
    return None

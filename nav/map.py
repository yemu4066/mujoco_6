"""代价地图：从 PGM 加载 + 按车体半径膨胀。

刻意只保留两条路径：
  * `load_pgm`  读 nav2 格式的 PGM（P5）
  * `inflate`   用 scipy 的精确欧氏距离场做膨胀

不做动态层、不做多分辨率、不做滚动窗口——那些等真需要了再加。
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import yaml

from .grid import GridSpec

FREE = 0
LETHAL = 254
UNKNOWN = 255


@dataclass
class Costmap:
    data: np.ndarray          # (H, W) uint8，0 可通行 / 254 障碍 / 255 未知
    spec: GridSpec

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def at(self, x: float, y: float) -> int:
        i, j = self.spec.world_to_index(x, y)
        if not self.spec.contains(i, j):
            return UNKNOWN
        return int(self.data[j, i])

    def passable(self, threshold: int = LETHAL) -> np.ndarray:
        """可通行掩码。未知（255）同样不可通行。"""
        return self.data < np.uint8(threshold)


def _read_pgm(path: pathlib.Path) -> np.ndarray:
    """解析二进制 PGM（P5），跳过注释。"""
    data = path.read_bytes()
    tokens: list[bytes] = []
    idx = 0
    while len(tokens) < 4:
        while data[idx:idx + 1].isspace():
            idx += 1
        if data[idx:idx + 1] == b'#':
            while data[idx:idx + 1] not in (b'\n', b''):
                idx += 1
            continue
        start = idx
        while data[idx:idx + 1] and not data[idx:idx + 1].isspace():
            idx += 1
        tokens.append(data[start:idx])
    idx += 1
    height, width = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data[idx:], dtype=np.uint8).reshape(height, width)


def load_map(yaml_path: str) -> tuple[Costmap, np.ndarray]:
    """加载 nav2 格式地图。

    返回 ``(代价地图, 原始障碍掩码)``。

    PGM 里 0 = 占据、254 = 空闲、205 = 未知。代价地图按 nav2 的取值放：
    占据 -> LETHAL(254)，未知 -> UNKNOWN(255)，其余 -> FREE(0)。
    """
    yaml_path = pathlib.Path(yaml_path)
    cfg = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    image = (yaml_path.parent / cfg['image']).resolve()
    raw = _read_pgm(image)
    # PGM 第 0 行是图像**顶部**（视觉坐标 y 向下），而 OccupancyGrid /
    # map_server 要求第 0 行是 **y 最小那行**（左下角起）。不翻转的话，
    # 地图在 Foxglove 里会上下颠倒，A* 也会按镜像后的障碍规划。
    raw = raw[::-1, :]

    occupied = raw < 100          # PGM 黑 = 障碍
    unknown = (raw >= 100) & (raw < 250)

    data = np.full(raw.shape, FREE, dtype=np.uint8)
    data[occupied] = LETHAL
    data[unknown] = UNKNOWN

    origin = cfg['origin']
    spec = GridSpec(float(cfg['resolution']), raw.shape[1], raw.shape[0],
                    float(origin[0]), float(origin[1]))
    return Costmap(data=data, spec=spec), occupied


def inflate(occupied: np.ndarray, resolution: float, inscribed_radius: float,
            inflation_radius: float, decay_power: float = 2.5) -> np.ndarray:
    """按到最近障碍的距离生成膨胀代价。

    * 距离 < ``inscribed_radius``  -> LETHAL（车中心绝不能进）
    * 距离 > ``inflation_radius``  -> FREE
    * 中间按 ``decay_power`` 衰减

    用 scipy 的精确欧氏距离变换，几十毫秒级别；不需要自己写两遍扫描。
    """
    from scipy import ndimage

    if not occupied.any():
        return np.zeros(occupied.shape, dtype=np.uint8)

    dist = ndimage.distance_transform_edt(~occupied) * float(resolution)
    cost = np.zeros(occupied.shape, dtype=np.float64)

    span = max(float(inflation_radius) - float(inscribed_radius), 1e-6)
    ramp = (dist - float(inscribed_radius)) / span
    ramp = np.clip(ramp, 0.0, 1.0)
    cost = (1.0 - ramp) ** float(decay_power) * (LETHAL - 1)

    cost[dist >= float(inflation_radius)] = 0.0
    cost[occupied] = LETHAL
    return np.clip(cost, 0, LETHAL).astype(np.uint8)

"""栅格坐标：世界坐标 <-> 栅格索引。

约定与 nav2 的 map_server 一致：
  * 行优先（row-major），行号 j 沿 +y 增大
  * origin 是地图**左下角**
  * index = (i, j)，world = (x, y)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float

    def contains(self, i: int, j: int) -> bool:
        return 0 <= i < self.width and 0 <= j < self.height

    def contains_world(self, x: float, y: float) -> bool:
        i, j = self.world_to_index(x, y)
        return self.contains(i, j)

    def world_to_index(self, x: float, y: float) -> tuple[int, int]:
        return (int((x - self.origin_x) / self.resolution),
                int((y - self.origin_y) / self.resolution))

    def index_to_world(self, i: int, j: int) -> tuple[float, float]:
        """格中心的世界坐标。"""
        return (self.origin_x + (i + 0.5) * self.resolution,
                self.origin_y + (j + 0.5) * self.resolution)

    @property
    def center(self) -> tuple[float, float]:
        return (self.origin_x + self.width * self.resolution / 2.0,
                self.origin_y + self.height * self.resolution / 2.0)

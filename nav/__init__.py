"""nav：A* 全局规划 + acados MPC 路径跟踪，纯 Python 库。

主入口：

    from nav import Navigation
    nav = Navigation(map_yaml='maps/rc2027.yaml')
    nav.set_goal(2.3, -1.3, yaw=0.0)
    cmd = nav.update(x, y, yaw)      # (vx, vy, wz) 或 None

``Navigation`` 依赖 casadi / acados，所以这里是**懒加载**：
只 import ``nav.map`` 或 ``nav.astar_adapter`` 时不会触发 MPC 的依赖检查。
"""
from __future__ import annotations

__all__ = ['Navigation', 'wrap_angle', 'IDLE', 'TRACK', 'ALIGN', 'ARRIVED']

_LAZY = {'Navigation', 'wrap_angle', 'IDLE', 'TRACK', 'ALIGN', 'ARRIVED'}


def __getattr__(name: str):
    if name in _LAZY:
        from . import navigation as _nav
        return getattr(_nav, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

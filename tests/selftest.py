#!/usr/bin/env python3
"""纯库自检：不依赖 ROS，验证地图 / A* / MPC / 三阶段闭环。

用法：
    cd mujoco_6
    python3 tests/selftest.py

全绿说明库本身没问题；红了说明环境缺库或代码有 bug。
"""
import math
import sys
import time
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

MAP = ROOT / 'maps' / 'rc2027.yaml'

failures = []


def check(name, cond, detail=''):
    mark = '✓' if cond else '✗'
    print(f'  {mark} {name}' + (f'   {detail}' if detail else ''))
    if not cond:
        failures.append(name)


def main():
    print('=' * 60)
    print('1. 依赖检查')
    print('=' * 60)
    for mod in ('numpy', 'scipy', 'yaml', 'casadi', 'acados_template'):
        try:
            __import__(mod)
            check(mod, True)
        except ImportError as e:
            check(mod, False, str(e))
    if failures:
        print('\n依赖不全，后续测试跳过。')
        return 1

    print()
    print('=' * 60)
    print('2. 地图加载')
    print('=' * 60)
    from nav.map import load_map, inflate, Costmap
    base, occ = load_map(str(MAP))
    spec = base.spec
    check('地图尺寸', spec.width == 224 and spec.height == 224,
          f'{spec.width}x{spec.height}')
    check('分辨率', abs(spec.resolution - 0.05) < 1e-6, f'{spec.resolution}')
    check('有障碍', int(occ.sum()) > 1000, f'{int(occ.sum())} 格')

    print()
    print('=' * 60)
    print('3. 膨胀')
    print('=' * 60)
    t0 = time.time()
    cost = inflate(occ, spec.resolution, 0.42, 0.68)
    dt = (time.time() - t0) * 1000
    cm = Costmap(data=cost, spec=spec)
    free = int((cost == 0).sum())
    check('耗时 < 500ms', dt < 500, f'{dt:.0f} ms')
    check('有自由区', free > 1000, f'{free} 格')

    print()
    print('=' * 60)
    print('4. A* 规划')
    print('=' * 60)
    from nav.astar import GridPlanner
    planner = GridPlanner(cm, cost_weight=2.0)
    idx = spec.world_to_index

    for name, a, b, tmax in [
        ('同侧短路径', (4.75, -4.8), (4.6, 2.5), 50),
        ('跨场上 L1', (4.75, -4.8), (2.3, -1.3), 200),
    ]:
        ts = []
        path = None
        for _ in range(3):
            t0 = time.time()
            path = planner.plan(idx(*a), idx(*b))
            ts.append((time.time() - t0) * 1000)
        dt = min(ts)
        check(f'{name} 有解', path is not None,
              f'{len(path) if path else 0} 点, {dt:.0f} ms')
        if path:
            check(f'{name} 耗时 < {tmax}ms', dt < tmax, f'{dt:.0f} ms')

    # 不可达（中心 L2 台面上）
    unreachable = planner.plan(idx(4.75, -4.8), idx(0.0, 0.0))
    check('中心台面判定为不可达', unreachable is None,
          'None' if unreachable is None else f'{len(unreachable)} 点')

    print()
    print('=' * 60)
    print('5. MPC 初始化')
    print('=' * 60)
    t0 = time.time()
    from nav import Navigation
    nav = Navigation(str(MAP))
    dt = time.time() - t0
    check('初始化 < 10s', dt < 10, f'{dt:.1f} s (首次编译 acados 会慢)')

    print()
    print('=' * 60)
    print('6. 三阶段闭环（用简单运动学推位姿，不是真仿真）')
    print('=' * 60)
    for name, goal in [('上 L1', (2.3, -1.3, 0.0)), ('同侧', (4.6, 2.5, 0.0))]:
        nav.cancel()
        pose = (4.75, -4.8, math.pi / 2)
        t0 = time.time()
        ok = nav.set_goal(*goal, pose=pose)
        plan_ms = (time.time() - t0) * 1000
        if not ok:
            check(f'{name} 规划', False, '无解')
            continue

        dt_ = 0.05
        arrived = False
        for i in range(1000):
            cmd = nav.update(*pose)
            if cmd is None:
                arrived = True
                break
            vx, vy, wz = cmd
            c, s = math.cos(pose[2]), math.sin(pose[2])
            pose = (pose[0] + (vx * c - vy * s) * dt_,
                    pose[1] + (vx * s + vy * c) * dt_,
                    pose[2] + wz * dt_)

        dist = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
        dyaw = abs(math.degrees(math.atan2(
            math.sin(pose[2] - goal[2]), math.cos(pose[2] - goal[2]))))
        check(f'{name} 到达', arrived, f'{i} 帧, 规划 {plan_ms:.0f}ms')
        check(f'{name} 位置误差 < 20cm', dist < 0.20, f'{dist*100:.1f} cm')
        check(f'{name} 朝向误差 < 5°', dyaw < 5.0, f'{dyaw:.1f}°')

    print()
    print('=' * 60)
    if failures:
        print(f'✗ {len(failures)} 项失败：')
        for f in failures:
            print(f'    - {f}')
        return 1
    print('✓ 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())

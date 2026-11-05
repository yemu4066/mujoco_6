#!/usr/bin/env python3
"""仿真验证：把 nav 库接到 MuJoCo 仿真上跑真实运动。

前置：mujoco_bridge 已经在跑（会发布 /odom、执行 /cmd_vel）。

    # 终端 1：起仿真
    ros2 launch mujoco_ros2_bridge bridge.launch.py

    # 终端 2：跑本脚本
    python3 tests/sim_bridge.py 2.3 -1.3 0.0

参数：目标 x y yaw（yaw 单位是度）
"""
import math
import sys
import pathlib

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from nav import Navigation                      # noqa: E402


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SimBridge(Node):
    def __init__(self, goal):
        super().__init__('nav6_sim_bridge')

        print('初始化导航库（首次要编译 acados，约 1~2 秒）...')
        self.nav = Navigation(str(ROOT / 'maps' / 'rc2027.yaml'))
        print('导航库就绪')

        self.goal = goal
        self.pose = None
        self.arrived = False
        self._frames = 0

        self.create_subscription(Odometry, '/odom', self._on_odom, 20)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self.pose = (float(p.x), float(p.y), yaw_of(msg.pose.pose.orientation))
        self._frames += 1

        # 第一次收到 odom 时规划
        if self._frames == 1:
            ok = self.nav.set_goal(*self.goal, pose=self.pose)
            if ok:
                print(f'目标 {self.goal}  规划成功，开始跟踪')
            else:
                print(f'目标 {self.goal}  不可达')
                self.arrived = True
                return

        if self.arrived:
            self.cmd_pub.publish(Twist())
            return

        cmd = self.nav.update(*self.pose)
        if cmd is None:
            print(f'到达  位置 ({self.pose[0]:.2f},{self.pose[1]:.2f}) '
                  f'yaw {math.degrees(self.pose[2]):+.1f}°  phase={self.nav.phase}')
            self.cmd_pub.publish(Twist())
            self.arrived = True
            return

        vx, vy, wz = cmd
        t = Twist()
        t.linear.x = vx
        t.linear.y = vy
        t.angular.z = wz
        self.cmd_pub.publish(t)

        if self._frames % 200 == 0:
            print(f'  [{self._frames}] ({self.pose[0]:.2f},{self.pose[1]:.2f}) '
                  f'phase={self.nav.phase} cmd=({vx:.2f},{vy:.2f},{wz:.2f})')


def main():
    if len(sys.argv) < 4:
        print('用法: python3 tests/sim_bridge.py <x> <y> <yaw_deg>')
        return 1
    goal = (float(sys.argv[1]), float(sys.argv[2]),
            math.radians(float(sys.argv[3])))

    rclpy.init()
    node = SimBridge(goal)
    try:
        while rclpy.ok() and not node.arrived:
            rclpy.spin_once(node, timeout_sec=0.1)
        # 到达后再转 1 秒，确保零速度发下去
        import time
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < 1.0:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())

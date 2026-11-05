#!/usr/bin/env python3
"""把 nav 库包成 ROS 节点（示例 / 联调用）。

这不是库的一部分，只是演示「怎么用 Navigation 类」：

    /goal_pose  ──┐
    /odom       ──┼──> 本节点 ──> /cmd_vel
                  │         └──> /plan      （路径可视化）
                  │         └──> /costmap   （地图可视化）
                  └──> nav.update()

用法：

    ros2 run ... 或者在容器里直接跑：
    python3 tests/nav_node.py

然后在 Foxglove 里：
    - 发 /goal_pose（PoseStamped，frame_id=map）
    - 看 /plan 和 /costmap

主逻辑不需要这个节点 —— 它直接 `from nav import Navigation` 就行。
"""
import math
import pathlib
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from nav import Navigation                      # noqa: E402


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NavNode(Node):
    def __init__(self, map_yaml: str):
        super().__init__('nav6_node')

        self.nav = Navigation(map_yaml)
        self.get_logger().info('导航库已初始化')

        # 地图用 transient_local，Foxglove 晚连也能拿到
        map_qos = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.plan_pub = self.create_publisher(Path, '/plan', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/costmap', map_qos)

        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 20)

        self.pose = None
        self._publish_costmap()
        self.get_logger().info('就绪：发 /goal_pose 即可')

    def _publish_costmap(self) -> None:
        cm = self.nav.costmap
        spec = cm.spec
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.info.resolution = spec.resolution
        grid.info.width = spec.width
        grid.info.height = spec.height
        grid.info.origin.position.x = spec.origin_x
        grid.info.origin.position.y = spec.origin_y
        grid.info.origin.orientation.w = 1.0
        # 地图行序：第 0 行是 y 最小
        data = (cm.data.astype(np.int32) * 100 // 254)
        grid.data = data.ravel().tolist()
        self.map_pub.publish(grid)
        self.get_logger().info(
            f'/costmap 已发布 {spec.width}x{spec.height}，自由格 {int((cm.data == 0).sum())}')

    def _on_goal(self, msg: PoseStamped) -> None:
        if self.pose is None:
            self.get_logger().warn('还没有 /odom，忽略目标')
            return
        yaw = yaw_of(msg.pose.orientation)
        ok = self.nav.set_goal(msg.pose.position.x, msg.pose.position.y, yaw,
                               pose=self.pose)
        if not ok:
            self.get_logger().warn('目标不可达')
            return
        self.get_logger().info(
            f'目标 ({msg.pose.position.x:.2f},{msg.pose.position.y:.2f}) '
            f'yaw {math.degrees(yaw):.0f}°，开始跟踪')

        # 发布路径供可视化
        path = self.nav.path
        if path is not None:
            p = Path()
            p.header.frame_id = 'map'
            p.header.stamp = self.get_clock().now().to_msg()
            for x, y in path:
                ps = PoseStamped()
                ps.header = p.header
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(y)
                ps.pose.orientation.w = 1.0
                p.poses.append(ps)
            self.plan_pub.publish(p)
            self.get_logger().info(f'/plan 已发布 {len(path)} 点')

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.pose = (float(p.x), float(p.y), yaw_of(msg.pose.pose.orientation))

        if self.nav.phase == 'IDLE':
            return

        cmd = self.nav.update(*self.pose)
        t = Twist()
        if cmd is not None:
            t.linear.x, t.linear.y, t.angular.z = cmd
        else:
            self.get_logger().info(
                f'到达 ({self.pose[0]:.2f},{self.pose[1]:.2f}) '
                f'yaw {math.degrees(self.pose[2]):+.1f}°')
        self.cmd_pub.publish(t)


def main():
    rclpy.init()
    node = NavNode(str(ROOT / 'maps' / 'rc2027.yaml'))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

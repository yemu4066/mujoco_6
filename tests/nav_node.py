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
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from nav import Navigation                      # noqa: E402
from nav_interfaces.msg import ObstacleArray    # noqa: E402


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
        # 真机：激光雷达（MS200 等 2D 单线）
        self.create_subscription(LaserScan, 'scan', self._on_scan, 10)
        # 仿真：障碍真值（没有雷达时用它测避障逻辑）
        self.create_subscription(ObstacleArray, '/obstacles', self._on_obstacles, 10)

        self.pose = None
        self._scan_logged = False
        self._has_scan = False
        self.obstacle_z_max = 1.0    # 高于这个高度的障碍忽略（够不着）
        self._map_msg = self._build_costmap()
        # 周期性重发：Foxglove 默认用 VOLATILE 订阅，晚连上收不到
        # 启动时发的那一条 TRANSIENT_LOCAL 历史消息，所以隔一秒再发一次。
        self.create_timer(1.0, lambda: self.map_pub.publish(self._map_msg))
        self.get_logger().info(
            f'/costmap {self._map_msg.info.width}x{self._map_msg.info.height}，'
            f'自由格 {int((self.nav.costmap.data == 0).sum())}')
        self.get_logger().info('就绪：发 /goal_pose 即可')

    def _build_costmap(self) -> OccupancyGrid:
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
        return grid

    def _on_goal(self, msg: PoseStamped) -> None:
        if self.pose is None:
            self.get_logger().warn('还没有 /odom，忽略目标')
            return
        yaw = yaw_of(msg.pose.orientation)
        ok = self.nav.set_goal(msg.pose.position.x, msg.pose.position.y, yaw,
                               pose=self.pose)
        if not ok:
            reason = getattr(self.nav, 'last_error', '') or '未知原因'
            self.get_logger().warn(f'目标不可达：{reason}')
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

    def _on_scan(self, msg: LaserScan) -> None:
        """真机激光雷达：原样交给库处理（库内部做坐标转换）。"""
        self.nav.set_scan(msg)
        self._has_scan = True
        if not self._scan_logged:
            self._scan_logged = True
            self.get_logger().info(
                f'收到激光：{len(msg.ranges)} 点，'
                f'{math.degrees(msg.angle_min):.0f}°~'
                f'{math.degrees(msg.angle_min + msg.angle_increment * len(msg.ranges)):.0f}°')

    def _on_obstacles(self, msg: ObstacleArray) -> None:
        """仿真障碍真值：转成世界坐标点喂给库（仅在没雷达时用）。"""
        if self._has_scan:
            return          # 有真雷达就不看仿真障碍
        pts = []
        for obs in msg.obstacles:
            if obs.confidence < 0.5:
                continue
            # 按高度过滤：车在地面/L1，够不着 L2（z≈1.175）上的物体。
            # 真雷达也有同样的问题——装在低处的雷达扫不到高处的物体。
            if obs.position.z > self.obstacle_z_max:
                continue
            r = max(0.05, float(obs.radius))
            # 圆周采样，近似一个圆
            for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                pts.append((obs.position.x + r * math.cos(a),
                            obs.position.y + r * math.sin(a)))
        if pts:
            self.nav.set_obstacle_points(pts)

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

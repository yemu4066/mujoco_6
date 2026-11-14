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
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist
from std_msgs.msg import Empty, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
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
    def __init__(self, map_yaml: str | None = None):
        super().__init__('nav6_node')

        # 地图可运行时指定，默认 rc2027：
        #   --ros-args -p map:=maps/debug_arena_10m.yaml
        # 也接受相对仓库根的路径，或者只写地图名。
        if map_yaml is None:
            self.declare_parameter('map', str(ROOT / 'maps' / 'rc2027.yaml'))
            map_yaml = str(self.get_parameter('map').value)
        mp = pathlib.Path(map_yaml)
        if not mp.is_absolute():
            mp = (ROOT / mp).resolve()
        if not mp.exists():
            for cand in (ROOT / 'maps' / map_yaml,
                         ROOT / 'maps' / f'{map_yaml}.yaml'):
                if cand.exists():
                    mp = cand.resolve()
                    break
        if not mp.exists():
            raise FileNotFoundError(f'找不到地图: {map_yaml}（已尝试 {mp}）')
        self.map_path = mp
        self.nav = Navigation(str(mp))
        self.get_logger().info(f'使用地图: {mp.name}')
        self.get_logger().info('导航库已初始化')

        # 地图用 transient_local，Foxglove 晚连也能拿到
        map_qos = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.plan_pub = self.create_publisher(Path, '/plan', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/costmap', map_qos)
        # 手动障碍可视化：Foxglove 里加完能看见在哪儿
        self.debug_obs_pub = self.create_publisher(
            MarkerArray, '/debug_obstacles', 10)
        # 实时速率：数值给 Plot 面板，文字浮在车顶给 3D 面板
        self.speed_pub = self.create_publisher(Float32MultiArray, '/speed', 10)
        self.speed_label_pub = self.create_publisher(
            MarkerArray, '/speed_label', 10)
        # MPC 可视化：参考点 + 预测轨迹
        self.ref_pub = self.create_publisher(MarkerArray, '/mpc_ref', 10)
        self.pred_pub = self.create_publisher(MarkerArray, '/mpc_pred', 10)

        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 20)
        # 真机：激光雷达（MS200 等 2D 单线）
        self.create_subscription(LaserScan, 'scan', self._on_scan, 10)
        # 仿真：障碍真值（没有雷达时用它测避障逻辑）
        self.create_subscription(ObstacleArray, '/obstacles', self._on_obstacles, 10)
        # 调试：Foxglove 3D 面板点一下就在那儿放个障碍，用来测重规划。
        # 发 geometry_msgs/PointStamped 到 /debug_obstacle（累加），
        # 发 std_msgs/Empty 到 /debug_obstacle_clear 清空。
        self.create_subscription(PointStamped, '/debug_obstacle',
                                 self._on_debug_obstacle, 10)
        self.create_subscription(Empty, '/debug_obstacle_clear',
                                 self._on_debug_clear, 10)

        self.pose = None
        self._scan_logged = False
        self._has_scan = False
        self.obstacle_z_max = 1.0    # 高于这个高度的障碍忽略（够不着）
        self._map_msg = self._build_costmap()
        # 周期性重发：Foxglove 默认用 VOLATILE 订阅，晚连上收不到
        # 启动时发的那一条 TRANSIENT_LOCAL 历史消息，所以隔一秒再发一次。
        self.create_timer(1.0, lambda: self.map_pub.publish(self._map_msg))
        self.create_timer(0.5, self._publish_debug_obstacles)
        self.create_timer(0.2, self._publish_mpc_viz)   # 5 Hz，减轻 GIL 争用
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

    def _on_debug_obstacle(self, msg: PointStamped) -> None:
        """Foxglove 点选：在 (x, y) 累加一个圆形障碍。"""
        x, y = float(msg.point.x), float(msg.point.y)
        self.nav.add_manual_obstacle(x, y)
        n = 0 if self.nav._manual_points is None else len(self.nav._manual_points)
        self.get_logger().info(f'[debug] 加障碍 ({x:+.2f}, {y:+.2f})，手动障碍点累计 {n}')

    def _on_debug_clear(self, msg: Empty) -> None:
        """清空手动障碍。"""
        self.nav.clear_manual_obstacles()
        self.get_logger().info('[debug] 已清空手动障碍')

    def _publish_speed(self, vx: float, vy: float, speed: float) -> None:
        """发布实时速率：数值 /speed（画曲线）+ 文字 /speed_label（3D 里看）。"""
        try:
            ref = float(self.nav.mpc.ref_speed)
        except Exception:                      # noqa: BLE001
            ref = 0.0
        try:
            v_mpc = float(self.nav.mpc.last_aug_state[0])
        except Exception:                      # noqa: BLE001
            v_mpc = 0.0

        # 数值：Plot 面板可以直接画
        arr = Float32MultiArray()
        arr.data = [speed, vx, vy, ref, v_mpc]
        self.speed_pub.publish(arr)

        # 文字：浮在车顶上方
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'speed'
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(self.pose[0])
        m.pose.position.y = float(self.pose[1])
        m.pose.position.z = 1.20
        m.pose.orientation.w = 1.0
        m.scale.z = 0.20                       # 字高
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.3, 1.0
        m.text = (f'{speed:.2f} m/s\n'
                  f'ref {ref:.2f}  v_mpc {v_mpc:.2f}')
        arr2 = MarkerArray()
        arr2.markers.append(m)
        self.speed_label_pub.publish(arr2)

    def _publish_mpc_viz(self) -> None:
        """把 MPC 这帧用的参考点和预测轨迹画出来。只读，不参与控制。"""
        mpc = self.nav.mpc
        now = self.get_clock().now().to_msg()

        # ① 参考点：小球，越靠后越大（颜色由蓝到青，表示"时间上越远"）
        ref = getattr(mpc, 'last_ref_points', None)
        if ref is not None:
            arr = MarkerArray()
            n = len(ref)
            for k in range(n):
                x, y = float(ref[k, 0]), float(ref[k, 1])
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = now
                m.ns = 'mpc_ref'
                m.id = k
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = x
                m.pose.position.y = y
                m.pose.position.z = 0.05
                m.pose.orientation.w = 1.0
                # 起点小、终点略大，便于看出方向
                frac = k / max(n - 1, 1)
                m.scale.x = m.scale.y = m.scale.z = 0.02 + 0.02 * frac
                m.color.r = 0.1
                m.color.g = 0.6 + 0.4 * frac
                m.color.b = 1.0 - 0.3 * frac
                m.color.a = 0.85
                arr.markers.append(m)
            self.ref_pub.publish(arr)

        # ② 预测轨迹：一条橙色折线，直接看 MPC 想怎么走
        pred = getattr(mpc, 'last_pred_traj', None)
        if pred is not None and len(pred) >= 2:
            arr2 = MarkerArray()
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'mpc_pred'
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.02
            m.color.r = 1.0
            m.color.g = 0.55
            m.color.b = 0.0
            m.color.a = 0.95
            for k in range(len(pred)):
                pt = Point()
                pt.x = float(pred[k, 0])
                pt.y = float(pred[k, 1])
                pt.z = 0.03
                m.points.append(pt)
            arr2.markers.append(m)

            # 终点再放一个大球，标出预测尽头
            me = Marker()
            me.header.frame_id = 'map'
            me.header.stamp = now
            me.ns = 'mpc_pred'
            me.id = 1
            me.type = Marker.SPHERE
            me.action = Marker.ADD
            me.pose.position.x = float(pred[-1, 0])
            me.pose.position.y = float(pred[-1, 1])
            me.pose.position.z = 0.03
            me.pose.orientation.w = 1.0
            me.scale.x = me.scale.y = me.scale.z = 0.06
            me.color.r = 1.0
            me.color.g = 0.2
            me.color.b = 0.0
            me.color.a = 0.9
            arr2.markers.append(me)
            self.pred_pub.publish(arr2)

    def _publish_debug_obstacles(self) -> None:
        """把手动障碍画成 Marker，让人在 Foxglove 里看得见。"""
        centers = getattr(self.nav, '_manual_centers', None) or []
        arr = MarkerArray()
        # 先发一条 DELETEALL，避免清空后残留旧 marker
        clear_m = Marker()
        clear_m.header.frame_id = 'map'
        clear_m.header.stamp = self.get_clock().now().to_msg()
        clear_m.action = Marker.DELETEALL
        arr.markers.append(clear_m)

        # A* 实际看到的是"中心点膨胀 (inscribed + obstacle_clearance)"的圆，
        # 这里就按这个半径画，看到的范围就是真正挡路的范围。
        try:
            res = self.nav.costmap.spec.resolution
            r_eff = (self.nav._inscribed_cells
                     + max(0, int(round(self.nav.obstacle_clearance / res)))) * res
        except Exception:                      # noqa: BLE001
            r_eff = 0.5

        for k, (x, y, _rad) in enumerate(centers):
            # ① 真实大小（半透明小球）
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'manual_obstacle'
            m.id = 2 * k
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = max(0.1, 2.0 * 0.12)
            m.scale.z = 1.2
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.45, 0.0, 0.85
            arr.markers.append(m)

            # ② 实际挡路半径（线框圆环，落在 z=0.02 地面上）
            ring = Marker()
            ring.header.frame_id = 'map'
            ring.header.stamp = m.header.stamp
            ring.ns = 'manual_obstacle_inflated'
            ring.id = 2 * k + 1
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position.x = float(x)
            ring.pose.position.y = float(y)
            ring.pose.position.z = 0.02
            ring.pose.orientation.w = 1.0
            ring.scale.x = ring.scale.y = 2.0 * r_eff
            ring.scale.z = 0.02
            ring.color.r, ring.color.g, ring.color.b, ring.color.a = 1.0, 0.85, 0.1, 0.28
            arr.markers.append(ring)

        self.debug_obs_pub.publish(arr)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.pose = (float(p.x), float(p.y), yaw_of(msg.pose.pose.orientation))

        # ---- 实时速率 ----
        # odom.twist 是世界系的（仿真桥直接取 body 的世界系线速度）
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        speed = math.hypot(vx, vy)
        self._publish_speed(vx, vy, speed)

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
    node = NavNode()          # 地图由 -p map:=... 决定，默认 maps/rc2027.yaml
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

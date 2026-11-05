#!/usr/bin/env python3
"""MuJoCo 仿真桥。

职责边界刻意收窄：

* **仿真**：加载场景、驱动物理、按正弦轨迹移动 mocap 障碍；
* **发布**：带噪声的 odom / TF、障碍真值 ``nav_interfaces/ObstacleArray``；
* **不参与导航决策**：不再发布 ``/dynamic_obstacle`` 这类"圆柱专用"话题，
  也不规定局部规划该怎么避障。障碍怎么用，由代价地图的障碍层决定。
"""
from __future__ import annotations

import math
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from nav_interfaces.msg import Obstacle, ObstacleArray
from visualization_msgs.msg import Marker, MarkerArray

from obstacle_motion import L1_PATROL, orbit_state, patrol_state
from odom_noise_node import OdomNoiseConfig, OdomNoiseGenerator
from swerve_solver import SwerveSolver


class MujocoSimNode(Node):
    def __init__(self) -> None:
        super().__init__('mujoco_sim_node')

        # ---------------- 参数 ----------------
        self.declare_parameter('scene', 'rc2027/scene_nav.xml')
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('steer_rate_limit', 8.0)
        self.declare_parameter('steer_noise_std', 0.002)
        self.declare_parameter('drive_noise_std', 0.02)
        # 动态障碍：沿 x 正弦往返，y 固定
        # 两个 650×650 移动对手：L1 台面 L 形折返、L2 台面绕中央基座环行
        self.declare_parameter('obstacle_l1_speed', 0.5)
        self.declare_parameter('obstacle_l2_speed', 0.2)
        self.declare_parameter('obstacle_l2_orbit_radius', 0.85)
        self.declare_parameter('obstacle_height', 0.5)
        # 是否把场景里的可动比赛用品也作为障碍发布（它们会被机器人推动）
        self.declare_parameter('publish_movable_objects', True)
        self.declare_parameter('obstacle_rate', 20.0)
        # 仿真里没有定位系统，用 identity 把 map 与 odom 接起来
        self.declare_parameter('publish_map_to_odom', True)
        gp = self.get_parameter

        self.use_viewer = bool(gp('use_viewer').value)
        self.steer_rate_limit = float(gp('steer_rate_limit').value)
        self.steer_noise_std = float(gp('steer_noise_std').value)
        self.drive_noise_std = float(gp('drive_noise_std').value)
        self.obstacle_l1_speed = float(gp('obstacle_l1_speed').value)
        self.obstacle_l2_speed = float(gp('obstacle_l2_speed').value)
        self.obstacle_l2_orbit = float(gp('obstacle_l2_orbit_radius').value)
        self.obstacle_height = float(gp('obstacle_height').value)
        self.publish_movable_objects = bool(gp('publish_movable_objects').value)
        self.obstacle_rate = float(gp('obstacle_rate').value)

        # ---------------- 模型 ----------------
        model_path = self._resolve_scene(str(gp('scene').value))
        self.get_logger().info(f'加载 MuJoCo 场景: {model_path}')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.chassis_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'chassis')

        # 动态障碍（mocap body，纯视觉，不参与物理）
        # 两个移动对手：L1 台面 (z=0.625) 与 L2 台面 (z=0.925)
        self.moving_obstacles: list[dict] = []
        for name, kind, z in (('obstacle_l1', 'l1', 0.875),
                              ('obstacle_l2', 'l2', 1.175)):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                self.get_logger().warning(f'场景缺少 {name}，跳过该移动障碍')
                continue
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id < 0:
                self.get_logger().warning(f'{name} 不是 mocap body，跳过')
                continue
            self.moving_obstacles.append({
                'name': name,
                'kind': kind,
                'body_id': body_id,
                'mocap_id': mocap_id,
                'z': z,
                'radius': self._horizontal_radius(body_id),
            })

        # 可动比赛用品：由 freejoint 驱动，这里只读取位姿
        self.movable_bodies: list[int] = []
        if self.publish_movable_objects:
            for body_id in range(1, self.model.nbody):
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ''
                if (name.startswith('skyblock')
                        or name.startswith('earthblock')
                        or name == 'mustika'):
                    self.movable_bodies.append(body_id)
        self.movable_radii = {
            body_id: self._horizontal_radius(body_id)
            for body_id in self.movable_bodies
        }

        # ---------------- 通信 ----------------
        self.create_subscription(Twist, 'cmd_vel', self._on_cmd_vel, 10)
        self.odom_publisher = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.base_link_state_publisher = self.create_publisher(
            Vector3Stamped, '/state/base_link', 10)
        self.real_vel_publisher = self.create_publisher(
            Vector3Stamped, '/mujoco/real_vel', 10)
        self.state_test_publisher = self.create_publisher(
            Vector3Stamped, '/state/test', 10)
        self.obstacle_publisher = self.create_publisher(
            ObstacleArray, 'obstacles', 10)
        # Foxglove / RViz 可视化：底盘与两个移动对手的方块
        self.marker_publisher = self.create_publisher(MarkerArray, 'nav_markers', 10)

        if bool(gp('publish_map_to_odom').value):
            self._static_tf = tf2_ros.StaticTransformBroadcaster(self)
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'map'
            transform.child_frame_id = 'odom'
            transform.transform.rotation.w = 1.0
            self._static_tf.sendTransform(transform)

        # ---------------- 底盘 ----------------
        self.target_v_x = 0.0
        self.target_v_y = 0.0
        self.target_v_yaw = 0.0
        self.prev_test_time = None
        self.prev_test_pos = None

        self.wheels_pos = [(0.325, 0.325), (0.325, -0.325),
                           (-0.325, 0.325), (-0.325, -0.325)]
        self.wheel_radius = 0.058
        self.steer_qposadr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f'wheel{i}_steer')])
            for i in range(4)
        ]

        self.noise_gen = OdomNoiseGenerator(OdomNoiseConfig(
            std_pos_100hz=0.0002, std_ori_100hz=0.002,
            std_pos_10hz=0.0001, std_ori_10hz=0.001, std_vel=0.02,
        ))
        self.swerve_solver = SwerveSolver(
            wheels_pos=self.wheels_pos,
            wheel_radius=self.wheel_radius,
            steer_lag_alpha=0.0,
            drive_lag_alpha=0.0,
            steer_noise_std=self.steer_noise_std,
            drive_noise_std=self.drive_noise_std,
            steer_rate_limit=self.steer_rate_limit,
        )

        self.create_timer(0.01, self._publish_truth)
        self.create_timer(1.0 / max(self.obstacle_rate, 1e-3),
                          self._publish_obstacles)
        self.create_timer(0.1, self._publish_markers)

        # 真值缓存：所有 MjData 访问都在仿真线程完成，发布线程只读缓存，
        # 避免 mj_objectVelocity 与 mj_step 并发改写同一个 data 造成抖动。
        self._truth = None
        self._movable_truth: list[tuple[float, float, float, float, float]] = []
        self._truth_step = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()

        self.get_logger().info(
            f'仿真桥就绪：可动件 {len(self.movable_bodies)} 个，'
            f'移动障碍 {len(self.moving_obstacles)} 个，'
            f'障碍发布 {self.obstacle_rate:.0f} Hz')

    # ---------------- 场景解析 ----------------
    def _resolve_scene(self, scene: str) -> str:
        candidates = []
        if os.path.isabs(scene) or os.path.exists(scene):
            candidates.append(scene)
        try:
            share = get_package_share_directory('mujoco_ros2_bridge')
            candidates.append(os.path.join(share, 'model', scene))
        except Exception:
            pass
        candidates.append(os.path.join('src', 'mujoco_ros2_bridge', 'model', scene))
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'找不到场景: {scene}（已尝试 {candidates}）')

    # ---------------- 几何 ----------------
    def _horizontal_radius(self, body_id: int) -> float:
        """近似的水平外接半径，用于把 body 变成圆形障碍。"""
        address = int(self.model.body_geomadr[body_id])
        count = int(self.model.body_geomnum[body_id])
        if count <= 0:
            return 0.2
        radius = 0.05
        for gid in range(address, address + count):
            geom_type = int(self.model.geom_type[gid])
            size = self.model.geom_size[gid]
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                value = math.hypot(float(size[0]), float(size[1]))
            elif geom_type in (int(mujoco.mjtGeom.mjGEOM_SPHERE),
                               int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                               int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                value = float(size[0])
            else:
                value = float(self.model.geom_rbound[gid])
            radius = max(radius, value)
        return radius

    def _obstacle_state(self, item: dict, t: float):
        if item['kind'] == 'l1':
            return patrol_state(L1_PATROL, self.obstacle_l1_speed, t)
        omega = self.obstacle_l2_speed / max(self.obstacle_l2_orbit, 1e-3)
        return orbit_state(self.obstacle_l2_orbit, omega, t)

    # ---------------- 回调 ----------------
    def _on_cmd_vel(self, msg: Twist) -> None:
        self.target_v_x = float(msg.linear.x)
        self.target_v_y = float(msg.linear.y)
        self.target_v_yaw = float(msg.angular.z)

    # ---------------- 仿真 ----------------
    def _simulation_loop(self) -> None:
        viewer = None
        if self.use_viewer and os.environ.get('DISPLAY'):
            try:
                viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.get_logger().info('MuJoCo viewer 已启动')
            except Exception as exc:  # pragma: no cover - 取决于显示环境
                self.get_logger().error(f'viewer 启动失败，转无头模式: {exc}')
                viewer = None
        else:
            self.get_logger().info('无头模式运行（未启用 viewer）')

        while not self.stop_event.is_set():
            if viewer is not None and not viewer.is_running():
                break
            step_start = time.time()

            for item in self.moving_obstacles:
                x, y, _, _ = self._obstacle_state(item, self.data.time)
                self.data.mocap_pos[item['mocap_id']] = [x, y, item['z']]

            current_steer = [float(self.data.qpos[self.steer_qposadr[i]])
                             for i in range(4)]
            motor_commands = self.swerve_solver.get_actuator_commands(
                self.target_v_x, self.target_v_y, self.target_v_yaw,
                current_steer, dt=self.model.opt.timestep)
            for i, (steer_ctrl, drive_ctrl) in enumerate(motor_commands):
                self.data.actuator(f'steer{i}').ctrl[0] = steer_ctrl
                self.data.actuator(f'drive{i}').ctrl[0] = drive_ctrl

            mujoco.mj_step(self.model, self.data)

            self._truth_step += 1
            if self._truth_step % 10 == 0:
                position = self.data.body('chassis').xpos.copy()
                quaternion = self.data.body('chassis').xquat.copy()
                body_velocity = np.zeros(6, dtype=float)
                mujoco.mj_objectVelocity(
                    self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                    self.chassis_body_id, body_velocity, 1)
                self._truth = (position, quaternion, body_velocity)

                movable = []
                for body_id in self.movable_bodies:
                    item_pos = self.data.xpos[body_id]
                    item_vel = np.zeros(6, dtype=float)
                    mujoco.mj_objectVelocity(
                        self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                        body_id, item_vel, 0)
                    movable.append((float(item_pos[0]), float(item_pos[1]),
                                    float(item_pos[2]), float(item_vel[3]),
                                    float(item_vel[4])))
                self._movable_truth = movable

            # 软件渲染下每步 sync 会拖慢物理循环、画面反而卡；
            # 50 Hz 刷新对肉眼足够，物理仍跑 1 kHz。
            if viewer is not None and self._truth_step % 20 == 0:
                viewer.sync()

            remaining = self.model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        if viewer is not None:
            viewer.close()

    # ---------------- 发布 ----------------
    def _publish_truth(self) -> None:
        stamp = self.get_clock().now().to_msg()
        if self._truth is None:
            return
        pos, quat, body_velocity = self._truth
        pos = pos.copy()
        quat = quat.copy()

        truth = Odometry()
        truth.header.stamp = stamp
        truth.pose.pose.position.x = pos[0]
        truth.pose.pose.position.y = pos[1]
        truth.pose.pose.position.z = pos[2]
        truth.pose.pose.orientation.w = quat[0]
        truth.pose.pose.orientation.x = quat[1]
        truth.pose.pose.orientation.y = quat[2]
        truth.pose.pose.orientation.z = quat[3]
        truth.twist.twist.linear.x = body_velocity[3]
        truth.twist.twist.linear.y = body_velocity[4]
        truth.twist.twist.linear.z = body_velocity[5]
        truth.twist.twist.angular.x = body_velocity[0]
        truth.twist.twist.angular.y = body_velocity[1]
        truth.twist.twist.angular.z = body_velocity[2]

        noisy_pos, noisy_quat, noisy_v_lin, noisy_v_ang = \
            self.noise_gen.apply_to_truth(truth)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = noisy_pos[0]
        odom.pose.pose.position.y = noisy_pos[1]
        odom.pose.pose.position.z = noisy_pos[2]
        odom.pose.pose.orientation.x = noisy_quat[0]
        odom.pose.pose.orientation.y = noisy_quat[1]
        odom.pose.pose.orientation.z = noisy_quat[2]
        odom.pose.pose.orientation.w = noisy_quat[3]
        odom.twist.twist.linear.x = noisy_v_lin[0]
        odom.twist.twist.linear.y = noisy_v_lin[1]
        odom.twist.twist.linear.z = noisy_v_lin[2]
        odom.twist.twist.angular.x = noisy_v_ang[0]
        odom.twist.twist.angular.y = noisy_v_ang[1]
        odom.twist.twist.angular.z = noisy_v_ang[2]
        self.odom_publisher.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = noisy_pos[0]
        transform.transform.translation.y = noisy_pos[1]
        transform.transform.translation.z = noisy_pos[2]
        transform.transform.rotation.x = noisy_quat[0]
        transform.transform.rotation.y = noisy_quat[1]
        transform.transform.rotation.z = noisy_quat[2]
        transform.transform.rotation.w = noisy_quat[3]
        self.tf_broadcaster.sendTransform(transform)

        state_base_link = Vector3Stamped()
        state_base_link.header.stamp = stamp
        state_base_link.header.frame_id = 'base_link'
        state_base_link.vector.x = body_velocity[3]
        state_base_link.vector.y = body_velocity[4]
        state_base_link.vector.z = body_velocity[2]
        self.base_link_state_publisher.publish(state_base_link)

        real_vel = Vector3Stamped()
        real_vel.header.stamp = stamp
        real_vel.header.frame_id = 'base_link'
        real_vel.vector.x = body_velocity[3]
        real_vel.vector.y = body_velocity[4]
        real_vel.vector.z = body_velocity[2]
        self.real_vel_publisher.publish(real_vel)

    def _publish_markers(self) -> None:
        """发布 3D 方块，Foxglove 3D 面板加 MarkerArray(/nav_markers) 即可看到车和对手。"""
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'robot'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        if self._truth is None:
            return
        position, quaternion, _ = self._truth
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.x = float(quaternion[1])
        marker.pose.orientation.y = float(quaternion[2])
        marker.pose.orientation.z = float(quaternion[3])
        marker.pose.orientation.w = float(quaternion[0])
        marker.scale.x = 0.65
        marker.scale.y = 0.65
        marker.scale.z = 0.25
        marker.color.r = 0.85
        marker.color.g = 0.25
        marker.color.b = 0.25
        marker.color.a = 0.75
        markers.markers.append(marker)

        for index, item in enumerate(self.moving_obstacles):
            x, y, _, _ = self._obstacle_state(item, self.data.time)
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'opponent'
            marker.id = index + 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = item['z']
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.65
            marker.scale.y = 0.65
            marker.scale.z = 0.5
            if item['kind'] == 'l1':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.55, 0.10
            else:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.20, 0.60
            marker.color.a = 0.85
            markers.markers.append(marker)

        self.marker_publisher.publish(markers)

    def _publish_obstacles(self) -> None:
        """把仿真里的障碍真值转成导航栈唯一的障碍契约。"""
        msg = ObstacleArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for item in self.moving_obstacles:
            x, y, vx, vy = self._obstacle_state(item, self.data.time)
            obstacle = Obstacle()
            obstacle.header = msg.header
            obstacle.id = int(item['body_id'])
            obstacle.position.x = x
            obstacle.position.y = y
            obstacle.position.z = item['z']
            obstacle.velocity.x = vx
            obstacle.velocity.y = vy
            obstacle.radius = float(item['radius'])
            obstacle.height = self.obstacle_height
            obstacle.confidence = 1.0
            msg.obstacles.append(obstacle)

        for body_id, cached in zip(self.movable_bodies, self._movable_truth):
            position = cached[:3]
            velocity = cached[3:]

            obstacle = Obstacle()
            obstacle.header = msg.header
            obstacle.id = int(body_id)
            obstacle.position.x = float(position[0])
            obstacle.position.y = float(position[1])
            obstacle.position.z = float(position[2])
            obstacle.velocity.x = float(velocity[0])
            obstacle.velocity.y = float(velocity[1])
            obstacle.radius = float(self.movable_radii.get(body_id, 0.2))
            obstacle.height = float(2.0 * obstacle.radius)
            obstacle.confidence = 1.0
            msg.obstacles.append(obstacle)

        self.obstacle_publisher.publish(msg)

    def destroy_node(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

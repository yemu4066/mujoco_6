#!/usr/bin/env python3
from dataclasses import dataclass
import math
import numpy as np
from nav_msgs.msg import Odometry

def euler_from_quaternion(x, y, z, w):
    """从四元数转换到欧拉角 (roll, pitch, yaw)"""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

def quaternion_from_euler(ai, aj, ak):
    """从欧拉角 (roll, pitch, yaw) 转换到四元数 [x, y, z, w]"""
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss
    return q

@dataclass
class OdomNoiseConfig:
    # 100Hz 叠加噪声（每帧刷新）
    std_pos_100hz: float = 0.01
    std_ori_100hz: float = 0.01
    # 10Hz 叠加噪声（每 10 帧刷新一次）
    std_pos_10hz: float = 0.01
    std_ori_10hz: float = 0.01
    # 速度噪声
    std_vel: float = 0.02


class OdomNoiseGenerator:
    def __init__(self, cfg: OdomNoiseConfig | None = None):
        self.cfg = cfg or OdomNoiseConfig()
        self.tick_count = 0
        self.n_pos_10 = [0.0, 0.0, 0.0]
        self.n_yaw_10 = 0.0

    def generate_noise(self, std_pos: float, std_ori: float):
        """生成随机噪声"""
        nx = np.random.normal(0, std_pos)
        ny = np.random.normal(0, std_pos)
        nyaw = np.random.normal(0, std_ori)
        return [nx, ny, 0.0], nyaw

    def apply_to_truth(self, truth_odom: Odometry):
        """输入真值 odom，输出叠加噪声后的 (pos, quat, v_lin, v_ang)。"""
        true_pos = np.array([
            truth_odom.pose.pose.position.x,
            truth_odom.pose.pose.position.y,
            truth_odom.pose.pose.position.z,
        ])
        _, _, true_yaw = euler_from_quaternion(
            truth_odom.pose.pose.orientation.x,
            truth_odom.pose.pose.orientation.y,
            truth_odom.pose.pose.orientation.z,
            truth_odom.pose.pose.orientation.w,
        )

        n_pos_100, n_yaw_100 = self.generate_noise(self.cfg.std_pos_100hz, self.cfg.std_ori_100hz)

        # 10Hz 噪声每 10 帧更新一次，帧间保持。
        if self.tick_count % 10 == 0:
            self.n_pos_10, self.n_yaw_10 = self.generate_noise(self.cfg.std_pos_10hz, self.cfg.std_ori_10hz)

        final_pos = true_pos + np.array(n_pos_100) + np.array(self.n_pos_10)
        final_yaw = true_yaw + n_yaw_100 + self.n_yaw_10
        final_quat = quaternion_from_euler(0, 0, final_yaw)

        v_lin = np.array([
            truth_odom.twist.twist.linear.x,
            truth_odom.twist.twist.linear.y,
            0.0,
        ]) + np.random.normal(0, self.cfg.std_vel, 3)
        v_ang = np.array([
            0.0,
            0.0,
            truth_odom.twist.twist.angular.z,
        ]) + np.random.normal(0, self.cfg.std_vel, 3)

        self.tick_count += 1
        return final_pos, final_quat, v_lin, v_ang

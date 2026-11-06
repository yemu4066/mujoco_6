#!/usr/bin/env python3
import math
import numpy as np


def wrap_to_near(angle: float, center: float) -> float:
    """Wrap angle to be the nearest representation around center."""
    d = angle - center
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return center + d


def decompose_wheel_velocity(vx: float, vy: float, vyaw: float, wheel_xy: tuple[float, float]) -> tuple[float, float]:
    """速度分解：将底盘速度分解到单个舵轮接地点线速度。"""
    wx, wy = wheel_xy
    v_ix = vx - vyaw * wy
    v_iy = vy + vyaw * wx
    return v_ix, v_iy


def optimize_steer_arc(desired_steer: float, speed: float, last_steer: float) -> tuple[float, float]:
    """优劣弧优化：在(舵角+正转)与(舵角+pi+反转)中选择转角更小的一组。"""
    steer_a = desired_steer
    drive_a = speed
    steer_b = desired_steer + math.pi
    drive_b = -speed

    steer_a = wrap_to_near(steer_a, last_steer)
    steer_b = wrap_to_near(steer_b, last_steer)
    if abs(steer_a - last_steer) <= abs(steer_b - last_steer):
        return steer_a, drive_a
    return steer_b, drive_b


class SwerveSolver:
    """舵轮解算器：根据 vx/vy/yaw 计算四个舵轮目标，并处理电机一阶滞后和噪声。"""

    def __init__(self, 
                 wheels_pos: list[tuple[float, float]], 
                 wheel_radius: float,
                 steer_lag_alpha: float = 0.0,
                 drive_lag_alpha: float = 0.0,
                 steer_noise_std: float = 0.0,
                 drive_noise_std: float = 0.0,
                 steer_rate_limit: float = 0.0):
        """
        初始化舵轮解算器。
        
        Args:
            wheels_pos: 四个轮子相对底盘中心的位置 [(x1,y1), (x2,y2), ...]
            wheel_radius: 轮子半径
            steer_lag_alpha: 舵机一阶滞后系数 (0 ~ 1)
            drive_lag_alpha: 驱动电机一阶滞后系数 (0 ~ 1)
            steer_noise_std: 舵控响应噪声标准差
            drive_noise_std: 驱动控响应噪声标准差
            steer_rate_limit: 舵角速率上限 [rad/s]，<=0 表示不限制。
                默认 0 是历史行为（指令阶跃），实际底盘舵机做不到瞬时转角，
                高速时瞬间横摆会产生很大侧向冲击，建议在仿真/实车上启用。
        """
        self.wheels_pos = wheels_pos
        self.wheel_radius = wheel_radius
        self.steer_lag_alpha = steer_lag_alpha
        self.drive_lag_alpha = drive_lag_alpha
        self.steer_noise_std = steer_noise_std
        self.drive_noise_std = drive_noise_std
        self.steer_rate_limit = max(0.0, float(steer_rate_limit))
        
        # 舵轮目标角度统计（用于优劣弧选择）
        self.last_target_angles = [0.0] * len(wheels_pos)
        
        # 电机响应状态（一阶滞后）
        # 舵角用 None 表示“尚未初始化”，首次调用对齐到实际舵角，
        # 避免从 0 开始把已经在转的舵轮硬拽回去。
        self.last_steer_ctrl = [None] * len(wheels_pos)
        self.last_drive_ctrl = [0.0] * len(wheels_pos)

    def solve(self, vx: float, vy: float, vyaw: float,
              current_steer_angles: list[float] | None = None) -> list[tuple[float, float]]:
        """
        纯舵轮运动解算：根据底盘速度计算每个轮子的目标舵角和目标速度。
        
        Args:
            current_steer_angles: 当前实际舵角；传入时用它选优弧，
                避免从上一次目标角出发选到绕远的一侧。

        Returns:
            list[(target_steer_rad, target_drive_rad_s)] 每个轮子的目标
        """
        n = len(self.wheels_pos)
        speeds = [0.0] * n
        desired = [0.0] * n
        last = [0.0] * n

        for i, wheel_xy in enumerate(self.wheels_pos):
            v_ix, v_iy = decompose_wheel_velocity(vx, vy, vyaw, wheel_xy)
            speeds[i] = math.hypot(v_ix, v_iy)
            desired[i] = (math.atan2(v_iy, v_ix)
                          if speeds[i] > 0.01 or abs(vyaw) > 0.01
                          else self.last_target_angles[i])
            last[i] = (float(current_steer_angles[i])
                       if current_steer_angles is not None
                       else self.last_target_angles[i])

        # 低速平移时四轮统一选正转/反转，避免起步时四轮各选优弧、
        # 出现两轮正转两轮反转把车体扭着转。正常行驶后仍用独立优弧，
        # 保留避障时的灵活性。
        low_speed = math.hypot(vx, vy) < 0.3
        translational = (abs(vyaw) < 0.05
                         and low_speed
                         and all(s > 0.01 for s in speeds))
        if translational:
            def adiff(a, b):
                return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

            err_a = sum(adiff(desired[i], last[i]) for i in range(n))
            err_b = sum(adiff(desired[i] + math.pi, last[i]) for i in range(n))
            flip = 0.0 if err_a <= err_b else math.pi

            targets = []
            for i in range(n):
                target_steer = wrap_to_near(desired[i] + flip, last[i])
                signed_speed = speeds[i] if flip == 0.0 else -speeds[i]
                self.last_target_angles[i] = target_steer
                targets.append((target_steer, signed_speed / self.wheel_radius))
            return targets

        # 正常行驶/自转/避障：保留四轮独立优弧
        targets = []
        for i in range(n):
            target_steer = self.last_target_angles[i]
            signed_speed = speeds[i]
            if speeds[i] > 0.01 or abs(vyaw) > 0.01:
                target_steer, signed_speed = optimize_steer_arc(
                    desired[i], speeds[i], last[i])
            self.last_target_angles[i] = target_steer
            targets.append((target_steer, signed_speed / self.wheel_radius))

        return targets

    def apply_motor_dynamics(self, 
                             target_steer_list: list[float], 
                             target_drive_list: list[float],
                             current_steer_angles: list[float],
                             dt: float = 1e-3) -> list[tuple[float, float]]:
        """
        应用电机一阶滞后、舵角速率限制和响应噪声。
        
        Args:
            target_steer_list: 目标舵角 (rad)
            target_drive_list: 目标驱动速度 (rad/s)
            current_steer_angles: 当前实际舵角 (rad) - 用于 2π 归一化
            dt: 控制/仿真步长 [s]，速率限制按它换算单步最大转角

        Returns:
            list[(steer_ctrl, drive_ctrl)] 最终电机指令
        """
        controls = []
        for i in range(len(target_steer_list)):
            target_steer = target_steer_list[i]
            target_drive = target_drive_list[i]
            current_steer = current_steer_angles[i]
            
            # 将目标舵角归一化到当前舵角附近，避免 ±π 跳变
            target_steer = wrap_to_near(target_steer, current_steer)

            # 舵角还没到位时按误差降轮速：保留至少 50%，让舵轮边走边转，
            # 不会为了消小弧而停在原地等舵角。
            if self.steer_rate_limit > 0.0:
                steer_err = abs(target_steer - current_steer)
                # 死区：直线行驶时舵角误差主要是噪声，不能因此反复降速，
                # 否则会形成“降速-回正-升速”的振荡（车会一抖一抖）。
                if steer_err > 0.15:
                    target_drive *= max(0.5, 1.0 - steer_err / 1.0)

            # 首次调用（或仿真重置后）对齐到实际舵角，避免从 0 开始追
            if self.last_steer_ctrl[i] is None:
                self.last_steer_ctrl[i] = float(current_steer)
            
            # 一阶滞后（可选）：对目标角做平滑
            if self.steer_lag_alpha > 0:
                self.last_steer_ctrl[i] = wrap_to_near(self.last_steer_ctrl[i], current_steer)
                self.last_steer_ctrl[i] += self.steer_lag_alpha * (target_steer - self.last_steer_ctrl[i])
                target_steer = self.last_steer_ctrl[i]

            # 速率限制：以上一帧输出为起点，按 rate*dt 逼近目标。
            # 不能写成 "current + rate*dt"：position actuator 存在跟踪误差，
            # 这样目标角每步只超前 current 一个步长，会被阻尼拖成极慢的爬行。
            prev = self.last_steer_ctrl[i]
            # 偏差过大（仿真重置 / 手动拖动）时直接对齐，避免长时间追不上
            if abs(wrap_to_near(current_steer, prev) - prev) > 0.5:
                prev = current_steer
            delta = wrap_to_near(target_steer, prev) - prev
            if self.steer_rate_limit > 0.0 and dt > 0.0:
                max_step = self.steer_rate_limit * dt
                delta = max(-max_step, min(max_step, delta))
            steer_output = prev + delta
            self.last_steer_ctrl[i] = steer_output
            
            if self.drive_lag_alpha > 0:
                self.last_drive_ctrl[i] += self.drive_lag_alpha * (target_drive - self.last_drive_ctrl[i])
                drive_output = self.last_drive_ctrl[i]
            else:
                drive_output = target_drive
            
            # 加入响应高频噪声
            steer_ctrl = wrap_to_near(
                steer_output + np.random.normal(0.0, self.steer_noise_std),
                current_steer
            )
            drive_ctrl = drive_output + np.random.normal(0.0, self.drive_noise_std)
            
            controls.append((steer_ctrl, drive_ctrl))
            
        return controls

    def get_actuator_commands(self, 
                              vx: float, vy: float, vyaw: float, 
                              current_steer_angles: list[float],
                              dt: float = 1e-3) -> list[tuple[float, float]]:
        """
        [接口解耦实现] 一个方法完成从底盘速度到电机执行指令的全过程转换。
        封装了：运动学解算(IK)、优劣弧优化、一阶滞后、舵角速率限制、响应噪声。
        """
        # 1. 运动学解算
        target_steer_list, target_drive_list = zip(
            *self.solve(vx, vy, vyaw, current_steer_angles))
        
        # 2. 应用电机动力学与噪声
        return self.apply_motor_dynamics(
            list(target_steer_list),
            list(target_drive_list),
            current_steer_angles,
            dt=dt,
        )

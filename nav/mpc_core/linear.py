import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple, List
from typing import Union,Type
class SplinePlanner:
    def __init__(self,method: Type = CubicSpline):
        """
        极简样条规划器：仅根据控制点生成平滑路径。
        """
        self.x_path = np.array([])
        self.y_path = np.array([])
        self.yaw_path = np.array([])
        self.yaw_path_unwrapped = np.array([])
        self.t_samples = np.array([])
        self.s_samples = np.array([])
        self.method=method
    def find_nearest_point(self, x: float, y: float) -> Tuple[int, np.ndarray, float]:
        if len(self.x_path) == 0:
            raise ValueError("路径为空。")

        dx = self.x_path - x
        dy = self.y_path - y
        dist_sq = dx**2 + dy**2
        idx = int(np.argmin(dist_sq))
        
        return (
            idx,
            np.array([self.x_path[idx], self.y_path[idx], self.yaw_path[idx]]),
            float(np.sqrt(dist_sq[idx]))
        )

    def generate_path(
        self, 
        x_pts: Union[list, np.ndarray], 
        y_pts: Union[list, np.ndarray], 
        step_cm: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        输入控制点，直接生成固定间距的平滑路径。
        """
        x_pts = np.atleast_1d(x_pts)
        y_pts = np.atleast_1d(y_pts)
        
        # 1. 计算参数 t (基于弦长的参数化)
        # 这是样条拟合的标准做法，t 代表沿控制点的累积直线距离
        ds = np.sqrt(np.hstack(([0], np.diff(x_pts)**2 + np.diff(y_pts)**2)))
        t_pts = np.cumsum(ds)
        
        # 2. 拟合样条 
        # 默认不传 bc_type，CubicSpline 默认使用 'not-a-knot'
        # 这是一种不需要用户提供额外物理边界（如航向角、曲率）的通用平滑方式
        cs_x = self.method(t_pts, x_pts)
        cs_y = self.method(t_pts, y_pts)

        # 3. 均匀采样
        total_length = t_pts[-1]
        step_m = step_cm / 100.0
        num_samples = max(2, int(np.ceil(total_length / step_m)) + 1)
        
        self.t_samples = np.linspace(0, total_length, num_samples)
        
        # 4. 生成路径点
        self.x_path = cs_x(self.t_samples)
        self.y_path = cs_y(self.t_samples)
        
        # 5. 计算航向角 (通过一阶导数)
        dx = cs_x(self.t_samples, 1)        
        dy = cs_y(self.t_samples, 1)
        self.yaw_path = np.arctan2(dy, dx)
        self.yaw_path_unwrapped = np.unwrap(self.yaw_path)

        # 6. 计算实际弧长 s
        dist_segments = np.sqrt(np.diff(self.x_path)**2 + np.diff(self.y_path)**2)
        self.s_samples = np.concatenate(([0.0], np.cumsum(dist_segments)))

        return self.x_path, self.y_path, self.yaw_path

    def get_total_length(self) -> float:
        '''返回路径的真实总弧长（单位：米）'''
        if len(self.s_samples) == 0:
            raise ValueError("Path is empty. Call generate_path() first.")
        return float(self.s_samples[-1])

    def get_nearest_s(self, x: float, y: float) -> float:
        '''给定世界坐标 (x, y)，返回路径上距离最近点对应的弧长 s'''
        if len(self.s_samples) == 0:
            raise ValueError("Path is empty. Call generate_path() first.")
        # 先找最近采样点的索引，再查该点的弧长
        idx, _, _ = self.find_nearest_point(x, y)
        return float(self.s_samples[idx])

    def get_state_by_s(self, s_query: float) -> np.ndarray:
        '''给定弧长 s_query，插值返回对应路径点的 (x, y, yaw)。
        s <= 0 时钳位到起点，s >= 总长时钳位到终点（x/y 不动）。
        '''
        if len(self.s_samples) == 0:
            raise ValueError("Path is empty. Call generate_path() first.")

        # 超出起点：直接返回路径起点
        if s_query <= 0.0:
            return np.array([float(self.x_path[0]), float(self.y_path[0]), float(self.yaw_path[0])])

        total_length = float(self.s_samples[-1])
        # 超出终点：x/y 保持不动，返回路径终点
        if s_query >= total_length:
            return np.array([float(self.x_path[-1]), float(self.y_path[-1]), float(self.yaw_path[-1])])

        # 以真实弧长 s_samples 为插值轴，线性插值 x 和 y
        x_ref = float(np.interp(s_query, self.s_samples, self.x_path))
        y_ref = float(np.interp(s_query, self.s_samples, self.y_path))
        # yaw 使用解卷绕后的连续角度插值，避免 ±π 附近的跳变
        yaw_unwrapped = float(np.interp(s_query, self.s_samples, self.yaw_path_unwrapped))
        # 将插值结果重新折叠回 (-π, π]
        yaw_ref = float(np.arctan2(np.sin(yaw_unwrapped), np.cos(yaw_unwrapped)))
        return np.array([x_ref, y_ref, yaw_ref])
    def get_states_batch(self, s_queries: np.ndarray) -> np.ndarray:
        """
        一次性插值返回 N 个查询点的 (x, y, yaw)。
        s_queries: 形状为 (N,) 的弧长数组
        return: 形状为 (N, 3) 的矩阵 [[x, y, yaw], ...]
        """
        # 1. 限制范围 (Clip)，替代 if/else 判断
        total_length = self.s_samples[-1]
        s_clipped = np.clip(s_queries, 0.0, total_length)

        # 2. 向量化线性插值 (一次性算出所有点的 x, y, unwrapped_yaw)
        x_refs = np.interp(s_clipped, self.s_samples, self.x_path)
        y_refs = np.interp(s_clipped, self.s_samples, self.y_path)
        yaw_unwrapped = np.interp(s_clipped, self.s_samples, self.yaw_path_unwrapped)

        # 3. 向量化角度归一化
        # 相比于 sin/cos/arctan2，这个公式在处理数组时通常更快
        yaw_refs = (yaw_unwrapped + np.pi) % (2 * np.pi) - np.pi

        # 4. 堆叠成 (N, 3) 矩阵返回
        return np.column_stack((x_refs, y_refs, yaw_refs))
    def plot(self):
        """简单的可视化函数"""
        if len(self.x_path) == 0:
            print("No path to plot.")
            return
        
        plt.figure(figsize=(8, 8))
        plt.plot(self.x_path, self.y_path, 'b-', label="Spline Path")
        plt.quiver(self.x_path[::10], self.y_path[::10], 
                   np.cos(self.yaw_path[::10]), np.sin(self.yaw_path[::10]), 
                   color='green', scale=20, width=0.005)
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

# --- 调用示例 ---
if __name__ == "__main__":
    planner = SplinePlanner()

    # 情况 1: 指定起终点 Yaw (强制掉头)
    print("Generating path with yaw constraints...")
    x1, y1, yaw1 = planner.generate_path(
        x_pts=[0, 2, 4], 
        y_pts=[0, 1, 0], 
    )
    planner.plot()

    # 情况 2: 不指定 Yaw (自由平滑插值)
    print("Generating free path...")
    x2, y2, yaw2 = planner.generate_path(
        x_pts=[0, 1, 1, 8], 
        y_pts=[0, 2, 2.2, 5]
    )
    planner.plot()

import os
import numpy as np
from abc import ABC, abstractmethod
from casadi import SX, vertcat, cos, sin, atan2
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from .linear import SplinePlanner

class AcadosMPCBase(ABC):
    '''
    基于纯位置输入的 MPC 基类，误差跟踪依靠p参数实现
    '''
    def __init__(self, dt: float, n_horizon: int, nx: int, nu: int, np_p: int):
        self.dt = float(dt)
        self.n_horizon = n_horizon
        self.nx, self.nu, self.np_p = nx, nu, np_p
        
        self.path_planner = SplinePlanner()
        self.following = False
        
        # 初始化求解器
        self.solver = self._init_solver()

    @abstractmethod
    def _define_model(self) -> AcadosModel:
        """子类需实现：定义 AcadosModel (f_expl, x, u, p)"""
        pass

    @abstractmethod
    def _setup_cost_and_constraints(self, ocp: AcadosOcp):
        """子类需实现：设置代价函数权重、表达式及约束范围"""
        pass

    @abstractmethod
    def _process_output(self, u_0, x_1) -> np.ndarray:
        """子类需实现：将求解器结果转换为机器人底盘指令 [vx, vy, vw]"""
        pass

    def _init_solver(self) -> AcadosOcpSolver:
        ocp = AcadosOcp()
        ocp.model = self._define_model()
        ocp.solver_options.N_horizon = self.n_horizon
        ocp.solver_options.tf = self.n_horizon * self.dt
        
        # 通用求解器设置
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # 调用子类特有的权重和约束配置
        self._setup_cost_and_constraints(ocp)
        assert isinstance(ocp.model.x,SX)
        # 默认初始化参数 p
        ocp.parameter_values = np.zeros(self.np_p)
        #初始化x
        ocp.constraints.x0 = np.zeros(self.nx)
        # yref 维度必须等于 cost_y_expr 的行数。默认按 nx+nu 建，
        # 但 AugmentedSwerveMPC 的代价多了「横向偏差」一项（多 1 行），
        # 所以子类可以覆盖本函数提供的维度。
        ny = getattr(self, 'ny_cost', self.nx + self.nu)
        ny_e = getattr(self, 'ny_cost_e', self.nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)
        #初始化W
        ocp.cost.W_0 = ocp.cost.W
        # ocp.model.c
        json_name = f"mpc_ocp.json"
        return AcadosOcpSolver(ocp, json_file=json_name, verbose=False)
    def set_path(self, points: np.ndarray, target_yaw: float, ref_speed: float = 1.0):
        """设置跟踪路径"""
        self.path_planner.generate_path(points[:, 0], points[:, 1])
        self.target_yaw, self.ref_speed, self.following = target_yaw, ref_speed, True
    def set_target_point(self, target: np.ndarray):
        """设置定点停靠"""
        self.following, self.target_point = False, target.flatten()
    def update(self, x_current: np.ndarray) -> np.ndarray:
        """核心更新逻辑：参数下发与求解"""
        # 1. 准备参考轨迹 (使用 p 参数)
        if self.following:
            s_start = self.path_planner.get_nearest_s(x_current[0], x_current[1])
            s_queries = s_start + self.ref_speed * self.dt * np.arange(self.n_horizon + 1)
            ref_states_all = self.path_planner.get_states_batch(s_queries)
            
            # 由参考朝向推出路径法向量：切线 (cosθ, sinθ) → 法线 (-sinθ, cosθ)
            cos_t = np.cos(ref_states_all[:, 2])
            sin_t = np.sin(ref_states_all[:, 2])
            ref_p = np.column_stack([ref_states_all[:, 0:3], -sin_t, cos_t])  # (n+1, 5)

            [self.solver.set(k, "p", ref_p[k]) for k in range(self.n_horizon)]
            self.solver.set(self.n_horizon, "p", ref_p[-1])
        else:
            tp = getattr(self, 'target_point', x_current[:3])
            tp5 = np.array([tp[0], tp[1], tp[2], 0.0, 0.0])   # 定点停靠不需要横向约束
            [self.solver.set(k, "p", tp5) for k in range(self.n_horizon + 1)]

        # 2. 设置当前状态约束
        self.solver.set(0, "lbx", x_current)
        self.solver.set(0, "ubx", x_current)

        # 3. 求解
        status = self.solver.solve()
        if status != 0:
            print(f"Warning: {self.__class__.__name__} failed with status {status}")

        # 4. 获取结果并转化
        return self._process_output(self.solver.get(0, "u"), self.solver.get(1, "x"))
class OmniMPC(AcadosMPCBase):
    def __init__(self, dt=0.05, n_horizon=20):
        super().__init__(dt, n_horizon, nx=3, nu=3, np_p=3)
    def _define_model(self):
        model = AcadosModel()
        model.name = "omni_model"
        x, u, p = SX.sym('x', 3), SX.sym('u', 3), SX.sym('p', 3) #type: ignore
        theta = x[2]
        vx, vy, vw = u[0], u[1], u[2]
        # 全向车动力学
        f_expl = vertcat(vx * cos(theta) - vy * sin(theta), vx * sin(theta) + vy * cos(theta), vw)
        model.x, model.u, model.p = x, u, p
        model.f_expl_expr = f_expl
        model.f_impl_expr = SX.sym('x_dot', 3) - f_expl #type: ignore
        return model
    def _setup_cost_and_constraints(self, ocp):
        # 约束
        ocp.constraints.idxbu = np.arange(3)
        ocp.constraints.lbu = np.array([-3.0, -3.0, -2.0])
        ocp.constraints.ubu = np.array([3.0, 3.0, 2.0])
        ocp.constraints.x0 = np.zeros(3)
        # 代价函数表达式
        pos_err = ocp.model.x[0:2] - ocp.model.p[0:2] #type: ignore
        angle_diff = ocp.model.x[2] - ocp.model.p[2]
        wrapped_angle_diff = atan2(sin(angle_diff), cos(angle_diff))
        ocp.model.cost_y_expr = vertcat(pos_err, wrapped_angle_diff, ocp.model.u)
        ocp.model.cost_y_expr_e = vertcat(pos_err, wrapped_angle_diff)
        # 权重
        ocp.cost.W = np.diag([20.0, 20.0, 10.0, 0.5, 0.5, 0.5])
        ocp.cost.W_e = np.diag([10.0, 10.0, 20.0])

    def _process_output(self, u_0, x_1):
        return u_0 # 直接返回 [vx, vy, vw]

class SwerveMPC(AcadosMPCBase):
    def __init__(self, dt=0.05, n_horizon=20):
        super().__init__(dt, n_horizon, nx=3, nu=3, np_p=3)

    def _define_model(self):
        model = AcadosModel()
        model.name = "swerve_simple_model"
        x, u, p = SX.sym('x', 3), SX.sym('u', 3), SX.sym('p', 3) #type: ignore
        theta = x[2]
        v, alpha, vw = u[0], u[1], u[2]
        # 舵轮动力学
        f_expl = vertcat(v * cos(theta + alpha), v * sin(theta + alpha), vw)
        model.x, model.u, model.p = x, u, p
        model.f_expl_expr = f_expl
        model.f_impl_expr = SX.sym('x_dot', 3) - f_expl #type: ignore
        return model

    def _setup_cost_and_constraints(self, ocp):
        ocp.constraints.idxbu = np.arange(3)
        ocp.constraints.lbu = np.array([-4.0, -np.pi, -2.0])
        ocp.constraints.ubu = np.array([4.0, np.pi, 2.0])
        ocp.constraints.x0 = np.zeros(3)
        assert isinstance(ocp.model.x,SX)
        pos_err = ocp.model.x[0:2] - ocp.model.p[0:2]
        wrapped_angle_diff = atan2(sin(ocp.model.x[2] - ocp.model.p[2]), cos(ocp.model.x[2] - ocp.model.p[2]))
        
        ocp.model.cost_y_expr = vertcat(pos_err, wrapped_angle_diff, ocp.model.u)
        ocp.model.cost_y_expr_e = vertcat(pos_err, wrapped_angle_diff)
        
        ocp.cost.cost_type = ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W = np.diag([20.0, 20.0, 12.0, 0.35, 12.0, 12.0])
        ocp.cost.W_e = np.diag([12.0, 12.0, 24.0])

    def _process_output(self, u_0, x_1):
        # 将 [v, alpha, vw] 转为 [vx, vy, vw] 适配底盘
        return np.array([u_0[0]*cos(u_0[1]), u_0[0]*sin(u_0[1]), u_0[2]])
class AugmentedSwerveMPC(AcadosMPCBase):
    def __init__(self, dt=0.05, n_horizon=20):
        self.last_aug_state = np.zeros(3) # [v, alpha, vw]
        # 代价维度：位置(2)+姿态(1)+横向(1)+状态(3)+控制(3) = 10
        # 终端：位置(2)+姿态(1)+横向(1)+状态(3) = 7
        self.ny_cost, self.ny_cost_e = 10, 7
        super().__init__(dt, n_horizon, nx=6, nu=3, np_p=5)

    def _define_model(self):
        model = AcadosModel()
        model.name = "swerve_augmented_model"
        # x = [X, Y, theta, v, alpha, vw]
        # u = [dv, dalpha, dw]
        # p = [x_ref, y_ref, theta_ref, nx, ny]
        #     前 3 个是参考位姿，后 2 个是**路径法向量**，用来算横向偏差
        x, u, p = SX.sym('x', 6), SX.sym('u', 3), SX.sym('p', 5) #type: ignore
        theta, v, alpha, vw = x[2], x[3], x[4], x[5]
        dv, dalpha, dw = u[0], u[1], u[2]
        
        f_expl = vertcat(v * cos(theta + alpha), v * sin(theta + alpha), vw, dv, dalpha, dw)
        model.x, model.u, model.p = x, u, p
        model.f_expl_expr = f_expl
        model.f_impl_expr = SX.sym('x_dot', 6) - f_expl #type: ignore
        return model

    def _setup_cost_and_constraints(self, ocp):
        # 状态约束 [v, alpha, vw]
        ocp.constraints.idxbx = np.array([3, 4, 5])
        # vw 上下界均为 0：平移阶段完全锁死自转，靠 alpha 平移
        # v 允许为负：车头锁死后，反向移动靠倒车实现，
        # 否则要求 alpha≈±π，正好卡在约束边界，求解器会退回零解。
        ocp.constraints.lbx = np.array([-4.0, -np.pi, 0.0])
        ocp.constraints.ubx = np.array([4.0, np.pi, 0.0])
        # 控制约束 [dv, dalpha, dw]
        ocp.constraints.idxbu = np.arange(3)
        # dw 同样锁 0：控制量也不产生角速度变化
        ocp.constraints.lbu = np.array([-2.0, -2.0, 0.0])
        ocp.constraints.ubu = np.array([2.0, 2.0, 0.0])
        ocp.constraints.x0 = np.zeros(6)
        assert isinstance(ocp.model.x,SX)
        pos_err = ocp.model.x[0:2] - ocp.model.p[0:2]
        wrapped_angle_diff = atan2(sin(ocp.model.x[2] - ocp.model.p[2]), cos(ocp.model.x[2] - ocp.model.p[2]))

        # 横向偏差 = (车 - 参考点) 在**路径法向**上的投影。
        # 为什么需要它：只惩罚 pos_err 时，拐角处"切内角"能让后续参考点的误差
        # 全部变小，所以数学上切角是最优解。实测最大横向偏移 15 cm，而 1 m
        # 通道单边余量只有 17.5 cm，几乎没有容错空间。
        # 单独惩罚法向分量后，切角不再"免费"。
        dx = ocp.model.x[0] - ocp.model.p[0]
        dy = ocp.model.x[1] - ocp.model.p[1]
        lateral_err = dx * ocp.model.p[3] + dy * ocp.model.p[4]

        # Cost: [位置(2), 姿态(1), 横向(1), 状态(3), 控制率(3)] = 10
        ocp.model.cost_y_expr = vertcat(pos_err, wrapped_angle_diff, lateral_err,
                                        ocp.model.x[3:6], ocp.model.u)
        ocp.model.cost_y_expr_e = vertcat(pos_err, wrapped_angle_diff, lateral_err,
                                          ocp.model.x[3:6])
        
        ocp.cost.cost_type = ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # 位置权重下调到 0.5：反正有横向项管住"别偏"，不再需要靠位置项硬拉。
        # 横向权重 2.0：这是抑制切角的主力。
        ocp.cost.W = np.diag([0.5, 0.5, 0.2, 2.0, 0, 0, 0, 0.1, 0.5, 0.5])
        ocp.cost.W_e = np.diag([4.0, 4.0, 0.2, 2.0, 0.2, 0.2, 0.2])

    def update(self, x_current: np.ndarray) -> np.ndarray:
        """重写 update 以处理增广状态的拼接"""
        # x_pos 为外部传入的 [x, y, yaw]
        x_full = np.concatenate([x_current.flatten(), self.last_aug_state])
        return super().update(x_full)

    def _process_output(self, u_0, x_1):
        # 提取预测的下一步状态作为当前的底盘指令
        v_next, alpha_next, vw_next = x_1[3:6]
        self.last_aug_state = np.array([v_next, alpha_next, vw_next])
        return np.array([v_next * cos(alpha_next), v_next * sin(alpha_next), vw_next])
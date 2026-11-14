import os
import math

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

        # ---- 逐 stage 权重衰减（远处参考点权重更低）----
        # 现状：代价对 176 个 stage 一视同仁，而 ref_speed=4.0 时只有
        # 7.8% 的参考点落在弯道里，剩下 162 个都在弯后的直线上。
        # 于是优化器宁可切内角去迁就远处，也不肯老实过弯。
        # 这里给 W 按指数衰减，让"近处"主导。
        #
        # acados 是否支持 solver.set(k, "W", ...) 取决于编译配置，
        # 所以先探测：失败就永久退回均匀权重，并打印一次提示。
        self.stage_decay = float(getattr(self, 'stage_decay', 0.013))
        # v 上限 = |ref_speed| × margin，再夹到 floor 以上（保证折返/低速时不至于卡死）
        self.v_cap_margin = 1.20
        self.v_cap_floor = 0.60
        self._vcap_failed = False
        self._vcap_applied = False
        self._vcap_dim = None
        # 规划层可用的 |dv| 上限，用来把 v_cap 逐 step 放开
        self.dv_max_spec = 3.0
        # 曲率加权采样的参数
        # 参考点覆盖的空间长度：跟原来的 ref_speed×dt×n_horizon 一致（4.0×0.01×175=7.0m）
        self.view_range = 4.0      # 参考点视野 [m]；None = 动态按 ref_speed 算（7.0m）
        self.curv_gain = 1.0       # 曲率对密度的加权强度
        self.kappa_cap = 5.0       # 曲率上限，避免极窄拐角把密度拉爆
        # 判定"这里是弯道"的曲率阈值：κ > 此值即弯曲，密度会传过去。
        # 0.2 对应 R < 5 m；直道 κ≈0，不会被穿透。
        self.straight_kappa = 0.2
        self._last_effective_view = 0.0
        # 弯道之外「缓降」的每步保留比例（点间距约 5cm）：
        #   0.97 → 0.5m 后 0.74，1.0m 后 0.55，2.0m 后 0.30，3.0m 后 0.16
        self.density_decay = 0.97
        self.use_stage_decay = True          # 待探测
        self._stage_decay_probed = False
        self._W_base = np.array(ocp.cost.W, dtype=float, copy=True)
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
    def _apply_vcap(self, lb3: np.ndarray, ub3: np.ndarray,
                    v_now: float, dv_lim: float) -> None:
        """按 ref_speed 设置各 stage 的 v 上限。

        idxbx 只约束 v/α/ω（3 个），但部分 acados 版本的 ubx setter 期望
        **全长**（nx=6）。这里两种都试，先成功的记住。
        """
        if self._vcap_dim is None:
            for dim, lb, ub in (
                (3, lb3, ub3),
                (6, np.concatenate([[-1e9, -1e9, -1e9], lb3]),
                    np.concatenate([[1e9, 1e9, 1e9], ub3])),
            ):
                try:
                    self.solver.set(1, "lbx", lb)
                    self.solver.set(1, "ubx", ub)
                    self._vcap_dim = dim
                    print(f'[MPC] v 上限跟随 ref_speed 已启用（维度 {dim}）')
                    break
                except Exception:                        # noqa: BLE001
                    continue
            if self._vcap_dim is None:
                raise RuntimeError('3 维和 6 维都不行')

        # ★ 上限必须逐 step 放开，不能一步到位。
        #   第 0 步的 v 是当前值（固定），如果下一步就要求 v ≤ v_cap，
        #   而 v_cap 远小于当前 v，这个约束物理上不可行（dv 每步只允许变
        #   dv_lim·dt），求解器直接 infeasible → 返回 0 → 车停住。
        #   实测：v=2.0、v_cap=0.6 时需要 47 步才能降下来。
        #   正确做法：第 k 步的上限 = min(v_cap, v_now + dv_lim·dt·k)。
        target = float(ub3[0])
        for k in range(1, self.n_horizon + 1):
            # 第 k 步 v 最低只能降到 v_now − dv·dt·k（受 dv 约束），
            # 所以上限不能低于这个值，否则约束物理不可达。
            reachable_floor = v_now - dv_lim * self.dt * k
            cap = max(target, reachable_floor)
            if self._vcap_dim == 3:
                lb = np.array([-cap, -math.pi, 0.0])
                ub = np.array([cap, math.pi, 0.0])
            else:
                lb = np.array([-1e9, -1e9, -1e9, -cap, -math.pi, 0.0])
                ub = np.array([1e9, 1e9, 1e9, cap, math.pi, 0.0])
            self.solver.set(k, "lbx", lb)
            self.solver.set(k, "ubx", ub)
        self._vcap_applied = True

    def _apply_stage_decay(self) -> None:
        """按指数衰减依次设置每个 stage 的 W；不支持就退回均匀权重。"""
        if not self.use_stage_decay or self._stage_decay_probed:
            return
        self._stage_decay_probed = True
        try:
            for k in range(self.n_horizon):
                decay = math.exp(-self.stage_decay * k)
                self.solver.set(k, "W", self._W_base * decay)
            print(f'[MPC] 逐 stage 权重衰减已启用：lambda={self.stage_decay:.4f}，'
                  f'末步权重 = {math.exp(-self.stage_decay * (self.n_horizon - 1)):.3f}')
        except Exception as exc:                     # noqa: BLE001
            self.use_stage_decay = False
            print(f'[MPC] 当前 acados 不支持逐 stage 设置 W，退回均匀权重（{exc}）')

    def _view_range(self) -> float:
        """参考点覆盖的空间长度。动态跟随 ref_speed，保持原来的 7 m 视野。"""
        v = self.view_range
        if v is None or v <= 0:
            v = float(self.ref_speed) * float(self.dt) * self.n_horizon
        return max(float(v), 1e-3)

    def _curvature_weighted_queries(self, s_start: float) -> np.ndarray:
        """按局部曲率分配参考点：拐角多放点，直道少放点。

        MPC 手上只有一条参考点序列，看不到墙。它对"哪里需要转弯"的唯一
        信息来源就是路径本身的几何 —— 也就是相邻参考点的切线角变化
        （Δθ = ds/R，正比于曲率 κ）。

        均匀等弧长排列时，直角拐角只占 176 个点里的 13 个（7.4%），
        优化器为了迎合剩下的 163 个直道点，会选一条平滑过渡的曲线，
        也就是"切角"。这里把点按曲率重新分布：拐角密度高 → 点数多 →
        在 Σ||x_k − p_k||² 里占比大 → 优化器必须贴住拐角。

        总视野仍然保持 view_range 不变，只是把点数从直道挪到弯道。
        """
        pl = self.path_planner
        s_path = np.asarray(pl.s_samples, dtype=float)
        yaw_u = np.asarray(pl.yaw_path_unwrapped, dtype=float)
        if len(s_path) < 3:
            # 路径太短，退回等弧长
            step = self._view_range() / self.n_horizon
            return s_start + step * np.arange(self.n_horizon + 1)

        ds = np.maximum(np.diff(s_path), 1e-6)       # 长度 n-1
        kappa = np.abs(np.diff(yaw_u)) / ds          # 离散曲率，长度 n-1
        # 限幅，避免极窄拐角把密度拉爆（拐角处 kappa 可以到 10+）
        kappa = np.minimum(kappa, self.kappa_cap)

        # ① 原始密度：曲率越大密度越高
        density = 1.0 + self.curv_gain * kappa      # 长度 n-1

        # ② 把高密度「扩散」出去，形成约束带 —— **一直传到直道为止**。
        #   不是按固定距离衰减：那样扩散范围由 decay 决定，拐角外 1 m
        #   就回到 1 了，中间那段仍然是松的。
        #   这里用曲率当"墙"：只要邻居还是弯道（κ > straight_kappa），
        #   就把它的密度传过来（取最大值，不衰减）；碰到直道就停。
        #   于是整段弯道都拿到同一个高密度，直道保持 1。
        is_curve = kappa > self.straight_kappa            # 长度 n-1
        d = density.copy()
        for k in range(1, len(d)):                        # 后向
            if is_curve[k - 1]:
                d[k] = max(d[k], d[k - 1])
        for k in range(len(d) - 2, -1, -1):               # 前向
            if is_curve[k + 1]:
                d[k] = max(d[k], d[k + 1])

        # ③ 只在**弯道之后**往直道铺缓降（不往前铺）。
        #   高密度按 density_decay 指数衰减地蔓延到出弯后的直道上，
        #   衰减到 1 自然停止（density 最小是 1）。
        #   弯道之前不做辐射：一进弯前的直道密度就是 1。
        for k in range(1, len(d)):                        # 只保留后向
            d[k] = max(d[k], d[k - 1] * self.density_decay)
        density = d

        cum = np.concatenate(([0.0], np.cumsum(density * ds)))   # 长度 n
        if cum[-1] < 1e-9:
            step = self._view_range() / self.n_horizon
            return s_start + step * np.arange(self.n_horizon + 1)
        cum /= cum[-1]                               # 归一化到 [0, 1]

        # 关键：在**权重累积坐标**上等间隔取样，再反查弧长。
        # 如果先按弧长等分再转权重再转回来，那是恒等变换，等于没加密。
        s_lo = max(0.0, float(s_start))
        s_hi = float(s_path[-1])

        # 视野与「剩余路径长度」取小：快到终点时视野自动收短，
        # 176 个参考点全部铺在剩余路径上，不会越过终点后被 clip 成一堆
        # 重复的同一个点（那样 MPC 就失去「往前还有多远」的方向信息，
        # 越接近终点越保守、速度掉得越快）。
        remaining = max(0.0, s_hi - s_lo)
        effective_view = min(self._view_range(), remaining)
        s_end = min(s_lo + effective_view, s_hi)
        if s_end - s_lo < 1e-6:
            step = self._view_range() / self.n_horizon
            return s_start + step * np.arange(self.n_horizon + 1)
        self._last_effective_view = float(effective_view)

        w_lo = float(np.interp(s_lo, s_path, cum))
        w_hi = float(np.interp(s_end, s_path, cum))
        wq = np.linspace(w_lo, w_hi, self.n_horizon + 1)   # 权重空间等间隔
        s_q = np.interp(wq, cum, s_path)                   # 反查弧长
        return np.clip(s_q, s_lo, s_hi)

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
            s_queries = self._curvature_weighted_queries(s_start)
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
        # 首次求解前探测/应用逐 stage 权重（只做一次）
        self._apply_stage_decay()

        # ★ 让 v 的硬约束跟随 ref_speed。
        #   原来 lbx/ubx 的 v 分量写死 ±5.0，而 ref_speed 只决定参考点间距，
        #   完全不影响车速。于是弯道里 ref_speed=1.4 时，车照样能开到 4.0+
        #   去"追"参考点 —— 需要的向心加速度 v²/R 高达 45 m/s²，
        #   α 根本转不过来，车沿切线飞出弯道外侧。
        #  （这解释了"为什么是冲出外侧而不是切内角撞内侧"）
        if self.following and not self._vcap_failed:
            v_cap = abs(float(self.ref_speed)) * self.v_cap_margin
            v_cap = max(v_cap, self.v_cap_floor)
            # idxbx 只选了 v/α/ω 三个状态，所以 lbx/ubx 必须是长度 3
            lb = np.array([-v_cap, -math.pi, 0.0])
            ub = np.array([v_cap, math.pi, 0.0])
            try:
                self._apply_vcap(lb, ub, float(x_current[3]), float(self.dv_max_spec))
            except Exception as exc:                     # noqa: BLE001
                # 一旦失败就永久放弃，绝不能每帧抛异常把控制卡死
                self._vcap_failed = True
                print(f'[MPC] v 上限跟随 ref_speed 设置失败，退回固定约束（{exc}）')

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
        # （试过把 alpha 收到 ±π/2 来消掉 (v,α)/(-v,α+π) 的冗余解，效果不好，已回退）
        ocp.constraints.lbx = np.array([-5.0, -np.pi, 0.0])
        ocp.constraints.ubx = np.array([5.0, np.pi, 0.0])
        # 控制约束 [dv, dalpha, dw]
        ocp.constraints.idxbu = np.arange(3)
        # dw 同样锁 0：控制量也不产生角速度变化
        ocp.constraints.lbu = np.array([-3.0, -4.0, 0.0])
        ocp.constraints.ubu = np.array([3.0, 4.0, 0.0])
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

        # ★ 姿态项必须用「运动方向 θ+α」，不是「车头朝向 θ」。
        #   运动学是 x_dot = v·cos(θ+α)，车往哪儿走由 θ+α 决定。
        #   而 ω ≡ 0（车头锁死、转不了），所以 wrap(θ − θ_ref) 这个误差
        #   **永远消不掉**，MPC 只能靠 α 间接补偿；偏偏 W[α]=0，它又不
        #   关心 α —— 结果出发时 α 停在 0，车沿车头方向走，不沿路径走。
        #   改成 θ+α 之后，α 会通过这一项被显式优化，运动方向才对得准。
        heading_err = atan2(sin(ocp.model.x[2] + ocp.model.x[4] - ocp.model.p[2]),
                            cos(ocp.model.x[2] + ocp.model.x[4] - ocp.model.p[2]))

        # Cost: [位置(2), 运动方向(1), 横向(1), 状态(3), 控制率(3)] = 10
        ocp.model.cost_y_expr = vertcat(pos_err, heading_err, lateral_err,
                                        ocp.model.x[3:6], ocp.model.u)
        ocp.model.cost_y_expr_e = vertcat(pos_err, heading_err, lateral_err,
                                          ocp.model.x[3:6])
        
        ocp.cost.cost_type = ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # 位置权重下调到 0.5：反正有横向项管住"别偏"，不再需要靠位置项硬拉。
        # 横向权重 2.0：这是抑制切角的主力。
        ocp.cost.W = np.diag([2.0, 2.0, 0.2, 4.0, 0, 0, 0, 0.05, 0.2, 0.2])
        ocp.cost.W_e = np.diag([4.0, 4.0, 0.2, 4.0, 0.2, 0.2, 0.2])

    def update(self, x_current: np.ndarray) -> np.ndarray:
        """重写 update 以处理增广状态的拼接"""
        # x_pos 为外部传入的 [x, y, yaw]
        x_full = np.concatenate([x_current.flatten(), self.last_aug_state])
        return super().update(x_full)

    def _process_output(self, u_0, x_1):
        # 提取预测的下一步状态作为当前的底盘指令
        v_next, alpha_next, vw_next = x_1[3:6]
        self.last_aug_state = np.array([v_next, alpha_next, vw_next])

        # ★ 规范化到 |alpha| <= 90°。
        #   运动学上 (v, α) 与 (−v, α∓180°) 完全等价（速度矢量相同），
        #   但 swerve_solver 在低速平移分支（<0.3 m/s）会用
        #       err_a = |desired − last|
        #       err_b = |desired + 180° − last|
        #   选误差小的那个来决定要不要把轮子翻 180° 反向驱动。
        #   只要 |α| > 90° 它就会翻，而翻过之后 last_target_angles 变成
        #   负角、下次判据又倾向翻 —— 状态会"粘住"，表现为车头方向
        #   和路径差得多的目标点出发时路线明显偏差。
        #   这里先翻好再发出去，轮子就永远不会被 flip。
        if abs(alpha_next) > math.pi / 2.0:
            v_out = -v_next
            a_out = alpha_next - math.copysign(math.pi, alpha_next)
            a_out = math.atan2(math.sin(a_out), math.cos(a_out))
        else:
            v_out, a_out = v_next, alpha_next
        return np.array([v_out * math.cos(a_out), v_out * math.sin(a_out), vw_next])
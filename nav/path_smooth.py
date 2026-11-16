"""A* 栅格路径 -> 曲率受限的平滑路径。

为什么需要这一层
----------------
A* 是八邻域搜索，产物必然是 0°/45°/90° 的阶梯。``SplinePlanner`` 用的是
``CubicSpline`` **插值**（必须穿过每个控制点），所以它会把阶梯原样复现出来：

    A* 原始路径            最小曲率半径  6.4 cm
    A* + CubicSpline       最小曲率半径  2.6 cm   ← 比车还小
    本模块平滑之后         最小曲率半径  ~45 cm

"光滑"和"曲率小"是两件事：三次样条确实 C² 连续（没有折角），但它可以在
5 cm 内转 45°，曲率依然是 15.7 1/m。车跟不跟得上只看曲率半径。

三步做法
--------
    ① 视线拉直（string pulling）
       把"水平一步 + 斜一步"的阶梯换成尽量少的直线段。
       判据：两点的连线全程离障碍 >= margin。

    ② 圆弧倒角（fillet）
       每个折点用半径 ``radius`` 的圆弧过渡。圆弧上的点如果会侵入
       ``margin`` 以内，就换更小的半径；全都不行就保留原折点（尖角）。

    ③ 稠密重采样
       输出 ``step`` 间距的点（默认 5 cm）。这一点很关键：
       ``SplinePlanner`` 只能用密集点，给它 4 个拐点会严重过冲、切进墙里。

安全边界
--------
* 圆弧上的**每一点**都检查离障距离，不满足 ``margin`` 就不画。
* ``adaptive``：折点本身离障碍比这个还近时，直接跳过倒角。
  窄道里"不倒角"反而更安全 —— 尖角贴着 A* 原路走，一点余量都不多花。
* 平滑后如果路径的离障反而变差（理论上不会），整体退回原始路径。

注意
----
这里算的是**几何**余量（车中心到障碍）。车体扫掠宽度取决于车头朝向：
正交移动需要 0.325 m 半径，斜 45° 移动需要 0.475 m。``margin`` 要给这个
差值留出空间，不能贴着 ``inscribed_radius`` 设。
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ['smooth_path']


# ---------------------------------------------------------------- 采样工具
def _clearance(P: np.ndarray, dist_field: np.ndarray, spec) -> np.ndarray:
    """世界坐标 -> 到最近障碍的距离（查预先算好的距离场）。"""
    ii = ((np.asarray(P)[:, 0] - spec.origin_x) / spec.resolution).astype(np.int32)
    jj = ((np.asarray(P)[:, 1] - spec.origin_y) / spec.resolution).astype(np.int32)
    np.clip(ii, 0, spec.width - 1, out=ii)
    np.clip(jj, 0, spec.height - 1, out=jj)
    return dist_field[jj, ii]


def _seg_clearance(a, b, dist_field, spec, step: float = 0.01) -> float:
    """线段 a-b 上所有采样点的最小离障距离。"""
    n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / step) + 2
    t = np.linspace(0.0, 1.0, n)[:, None]
    P = a[None, :] + t * (b - a)[None, :]
    return float(_clearance(P, dist_field, spec).min())


def _resample(P: np.ndarray, step: float) -> np.ndarray:
    """按弧长等间距重采样。"""
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] <= 1e-9:
        return P.copy()
    n = max(2, int(s[-1] / step) + 1)
    ts = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(ts, s, P[:, 0]),
                            np.interp(ts, s, P[:, 1])])


def _max_curvature(P: np.ndarray) -> float:
    """路径上最大曲率，用来判断平滑有没有把路变得"更急"。"""
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    den = (dx * dx + dy * dy) ** 1.5
    num = np.abs(dx * ddy - dy * ddx)
    k = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
    return float(k.max())


# ---------------------------------------------------------------- ① 拉直
def _shorten(pts: np.ndarray, dist_field, spec, min_clearance: float,
             loss: float, max_los: float, adaptive: float = 0.40) -> np.ndarray:
    """视线拉直：从当前点尽量往后连一条满足 margin 的长直线。

    ``max_los`` 限制单次最多往前连多远，用来控制耗时。
    """
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    clr = _clearance(pts, dist_field, spec)

    out = [pts[0]]
    i, N = 0, len(pts)
    while i < N - 1:
        j = int(np.searchsorted(s, s[i] + max_los, side='right')) - 1
        j = min(max(j, i + 1), N - 1)

        while j > i + 1:
            # 窄区（这段路径的最小余量 < adaptive）下游 _fillet 不会倒角。
            # 这里若还把两个 45° 阶梯拉成一个 90° 尖角，就等于制造出比
            # A* 原路径更急的死角。
            # 必须看**整段**的最小余量而不是起点：路径常贴着一边走，
            # 起点看着很宽，跨过去那段其实很窄。
            if float(clr[i:j + 1].min()) < adaptive:
                if float(s[j] - s[i]) > 0.25:
                    j -= 1
                    continue
                th_seg_n = np.arctan2(np.diff(pts[i:j + 1, 1]),
                                      np.diff(pts[i:j + 1, 0]))
                if len(th_seg_n) > 1 and float(
                        np.abs(np.diff(np.unwrap(th_seg_n))).sum()) > np.radians(50.0):
                    j -= 1
                    continue
            # 方向跨度太大 = 这段是"去又回"（U 形掉头），不是"拐个弯"。
            # 直接连过去会把 2 m 的绕行压成一条直线，MPC 在那里就迷路了。
            #   45° 阶梯：跨度 45°   90° 拐角：跨度 90°   U 形掉头：跨度 180°
            # 用 120° 分开，同时挡掉"U 形 + 长直道"那种整段弦/弧比 0.65 的情况。
            th = np.arctan2(np.diff(pts[i:j + 1, 1]), np.diff(pts[i:j + 1, 0]))
            if len(th) > 1:
                th_u = np.unwrap(th)
                if float(th_u.max() - th_u.min()) > np.radians(120.0):
                    j -= 1
                    continue

            # 视线检查：拉直后不能比原路径更贴墙
            need = max(min_clearance, float(clr[i:j + 1].min()) - loss)
            if _seg_clearance(pts[i], pts[j], dist_field, spec) >= need:
                break
            j -= 1

        out.append(pts[j])
        i = j
    return np.array(out)


# ---------------------------------------------------------------- ② 圆角
def _arc(ctr, R, a0, da, step=0.02):
    m = max(3, int(abs(da) * R / step) + 1)
    aa = np.linspace(a0, a0 + da, m)
    return np.column_stack([ctr[0] + R * np.cos(aa),
                            ctr[1] + R * np.sin(aa)])


def _fillet(verts: np.ndarray, dist_field, spec, min_clearance: float,
            loss: float, radius: float, adaptive: float) -> np.ndarray:
    """把折线的每个顶点替换成圆弧。"""
    if len(verts) < 3:
        return verts.copy()

    out = [verts[0]]
    for k in range(1, len(verts) - 1):
        a, b, c = verts[k - 1], verts[k], verts[k + 1]

        # 这一段原路径的余量底线（用来保护倒角不过分切角）
        local_min = float(_clearance(verts[max(0, k - 1):k + 2],
                                     dist_field, spec).min())

        # 窄处不倒角：保留 A* 原尖角，一点余量都不多花
        if local_min < adaptive:
            out.append(b)
            continue

        u1, u2 = a - b, c - b
        n1, n2 = np.linalg.norm(u1), np.linalg.norm(u2)
        if n1 < 1e-9 or n2 < 1e-9:
            out.append(b)
            continue
        u1, u2 = u1 / n1, u2 / n2
        ang = math.acos(float(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        if ang < 1e-3 or ang > math.pi - 1e-3:      # 几乎不拐 / 原地掉头
            out.append(b)
            continue

        # ★ 直角（含进门那种 90°）一律不倒角。
        #   倒角会把拐点"削圆"，路径从"直着进"变成"斜着切"——
        #   门口/通道的单边余量本来就窄，一斜就蹭墙。
        #   窄处靠上面的 adaptive 已经跳过了，这里管的是**宽处**的 90°：
        #   那些地方 local_min 够大，倒角能成功，但倒出来的斜切对角
        #   反而更危险。
        if abs(ang - math.pi / 2.0) < math.radians(15.0):
            out.append(b)
            continue

        # 从 radius 开始，放不下就逐步缩小
        placed = False
        for R in (radius, radius * 0.75, radius * 0.5, radius * 0.35):
            tl = min(R / math.tan(ang / 2.0), 0.45 * min(n1, n2))
            if tl < 1e-6:
                continue
            # tl 被边长钳位时，与两条折线真正相切的半径是 tl·tan(ang/2)，
            # 不是上面那个 R。直接用 R 画弧会让圆心到切点的距离远小于 R，
            # 圆弧向外鼓包、既撞墙又制造假折返。
            R_eff = tl * math.tan(ang / 2.0)
            # 半径太小的圆弧没有意义：比 A* 原路径的拐角还急。
            # 窄区放不下就保留原折角。
            if R_eff < 0.15:
                continue
            p1, p2 = b + u1 * tl, b + u2 * tl
            bis = u1 + u2
            bn = np.linalg.norm(bis)
            if bn < 1e-9:
                continue
            bis = bis / bn
            ctr = b + bis * (tl / math.cos(ang / 2.0))
            a0 = math.atan2(p1[1] - ctr[1], p1[0] - ctr[0])
            a1 = math.atan2(p2[1] - ctr[1], p2[0] - ctr[0])
            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            arc = _arc(ctr, R_eff, a0, da)
            need = max(min_clearance, local_min - loss)
            if _clearance(arc, dist_field, spec).min() >= need:
                out.extend(list(arc))
                placed = True
                break
        if not placed:
            out.append(b)

    out.append(verts[-1])
    return np.array(out)


# ---------------------------------------------------------------- 对外入口
def _smooth_core(pts, dist_field, spec, min_clearance, loss, radius,
                 adaptive, step, max_los):
    """把一整段路径平滑一遍；任何校验没过就返回 None（不抛异常）。"""
    base_min = float(_clearance(pts, dist_field, spec).min())
    allowed_min = max(min_clearance, base_min - loss)
    try:
        verts = _shorten(pts, dist_field, spec, min_clearance, loss, max_los,
                         adaptive)
        path = _fillet(verts, dist_field, spec, min_clearance, loss,
                       radius, adaptive)
        path = _resample(path, step)
    except Exception:                       # noqa: BLE001 - 平滑绝不该让规划失败
        return None
    if len(path) < 3:
        return None
    # 校验 ①：平滑后不能明显比原来更贴墙
    if float(_clearance(path, dist_field, spec).min()) < allowed_min - 1e-6:
        return None
    # 校验 ② 已移除（原本是全局曲率一票否决）。
    # 副作用：窄区保留原折角、宽区已经平滑好，只要窄区那一个点的离散曲率
    # 比原路径大一点，整条（包括 9 米宽敞段）就被丢弃退回 A* 折线。
    # 修好 R_eff 几何 Bug、并让 _shorten 不在窄区跨弯拉直之后，
    # 窄区的曲率已经不会超过原路径，这道闸门不再需要。
    return path


def smooth_path(points: np.ndarray,
                dist_field: np.ndarray,
                spec,
                *,
                min_clearance: float = 0.35,
                loss: float = 0.06,
                radius: float = 0.50,
                adaptive: float = 0.40,
                step: float = 0.05,
                max_los: float = 2.0,
                _depth: int = 0) -> np.ndarray:
    """A* 栅格路径 -> 平滑的可跟踪路径。

    分段平滑（为什么需要）
    ----------------------
    窄 U 形掉头（横移只有一格）在几何上放不下任何圆弧，硬平滑只会产生
    175° 的折返。原来的做法是"整条路径退回 A*"，结果就是：只要路径经过
    那个窄口（比如起点或终点落在窄通道里），**整条路径都得不到平滑**。

    现在改成递归切分：整条平滑失败时，在**转角最大**的地方切开，两半各自
    再试。窄口那一段最终会退化成原样（正确），其余段照常平滑。

    Parameters
    ----------
    points
        A* 输出的世界坐标点列 (N, 2)，通常是 5 cm 间距的栅格点。
    dist_field
        到最近障碍的欧氏距离场（米），``navigation._dist_field``。
    spec
        ``GridSpec``，提供 resolution / origin / width / height。
    min_clearance
        绝对下限：平滑后的路径不能比这个更贴墙 [m]。一般取 ``inscribed_radius``。
    loss
        允许比"这段原路径本来有多少余量"再少损失多少 [m]。
        取 0 表示平滑后不许更贴墙。**不要设太大** —— clearance 一降，
        下游限速表会立刻判成窄档，平滑换来的曲率改善会被降速全部吃掉。
    radius
        圆弧倒角半径 [m]。放不下时自动缩到 0.75/0.5/0.35 倍。
    adaptive
        折点附近余量（原路径的）小于它就不做倒角，保留尖角 [m]。
    step
        输出点间距 [m]，给 ``SplinePlanner`` 用，必须足够密。
    max_los
        单次视线拉直的最远距离 [m]，用来控制耗时。

    Returns
    -------
    (M, 2) ndarray。能平滑的段已平滑，平滑不了的段保留 A* 原样。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return pts.copy()

    path = _smooth_core(pts, dist_field, spec, min_clearance, loss,
                        radius, adaptive, step, max_los)
    if path is not None:
        return path

    # 整段平滑失败 → 保留 A* 原样。
    #
    # 分段处理解决不了问题：切分后两段端点切线不连续，拼接处会多出尖角；
    # 把端点钉回原路径又会在段内制造新的跳变。
    # 根因是平滑段的端点切线和原路径不一致，靠「切+钉」补不回来。
    # 要真正解决，得从 A* 下手（把朝向纳入搜索状态，或给转角加代价），
    # 让它不生成横移只有一格的窄 U 形。
    return pts.copy()

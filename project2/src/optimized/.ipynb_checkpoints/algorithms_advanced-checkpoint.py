# src/optimized/algorithms_advanced.py
# -*- coding: utf-8 -*-

import time
import numpy as np
import config
from src.common import utils, plotting

# ------------------------------------------------------------------------------
# 可选：是否在运行时打印“使用了KDTree/linregress还是fallback”的提示
VERBOSE_IMPORT = getattr(config, 'VERBOSE_IMPORT', False)

# ------------------------------------------------------------------------------
# 1) 邻居搜索：优先用 SciPy KDTree；无 SciPy 时自动降级到“空间网格法（cell-list）”
_HAS_SCIPY_KDTREE = False
try:
    from scipy.spatial import cKDTree as _KDTree

    _HAS_SCIPY_KDTREE = True
except Exception:
    _HAS_SCIPY_KDTREE = False


def _neighbor_lists_kdtree(positions: np.ndarray, R: float):
    tree = _KDTree(positions)
    return tree.query_ball_point(positions, r=R)


def _neighbor_lists_celllist(positions: np.ndarray, R: float):
    """
    零依赖的邻域构建（cell-list，向量化半径判定），复杂度 ~ O(N * k)
    """
    n = positions.shape[0]
    invR = 1.0 / R
    # 每个点所在 cell 的整数坐标（floor，支持负坐标）
    cells_xy = np.floor(positions * invR).astype(np.int32)

    # 建桶：dict[(cx,cy)] -> List[idx]
    buckets = {}
    for idx, key in enumerate(map(tuple, cells_xy)):
        buckets.setdefault(key, []).append(idx)

    neigh = [[] for _ in range(n)]
    OFF = [(-1, -1), (-1, 0), (-1, 1),
           (0, -1), (0, 0), (0, 1),
           (1, -1), (1, 0), (1, 1)]
    R2 = R * R

    for i in range(n):
        cx, cy = cells_xy[i]
        cand = []
        for dx, dy in OFF:
            cand.extend(buckets.get((cx + dx, cy + dy), ()))
        if not cand:
            continue
        cand = np.asarray(cand, dtype=np.int32)
        d = positions[cand] - positions[i]
        mask = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) <= R2
        neigh[i] = cand[mask].tolist()
    return neigh


def _get_neighbor_lists(positions: np.ndarray, R: float):
    if _HAS_SCIPY_KDTREE:
        return _neighbor_lists_kdtree(positions, R)
    else:
        return _neighbor_lists_celllist(positions, R)


# ------------------------------------------------------------------------------
# 2) 收敛判断：优先用 SciPy linregress；无 SciPy 时降级到 numpy.polyfit
_HAS_SCIPY_LINREGRESS = False
try:
    from scipy.stats import linregress as _linregress

    _HAS_SCIPY_LINREGRESS = True
except Exception:
    _HAS_SCIPY_LINREGRESS = False


def _slope_and_std(y):
    """
    返回最近窗口的斜率与标准差
    y: 1D 序列（使用原始浮点，不要四舍五入后的值）
    """
    if _HAS_SCIPY_LINREGRESS:
        x = np.arange(len(y), dtype=np.float64)
        res = _linregress(x, np.asarray(y, dtype=np.float64))
        return float(res.slope), float(np.std(y))
    else:
        x = np.arange(len(y), dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        p = np.polyfit(x, y_arr, deg=1)
        slope = float(p[0])
        std_dev = float(np.std(y_arr))
        return slope, std_dev


# ------------------------------------------------------------------------------
# 3) 小工具
def _safe_log_pt(pt: float) -> float:
    eps = 1e-12
    return float(np.log(np.clip(pt, eps, 1.0 - eps)))



def _edges_from_neighbors(neighbor_lists: list) -> tuple[np.ndarray, np.ndarray]:
    """
    一个辅助函数，将邻居列表(list of lists)转换为 i<j 的无重复边列表。
    返回两个数组：i_indices 和 j_indices。
    """
    edges = set()
    for i, neighbors in enumerate(neighbor_lists):
        for j in neighbors:
            if i < j:
                edges.add((i, j))

    if not edges:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    edge_arr = np.array(list(edges), dtype=np.int32)
    return edge_arr[:, 0], edge_arr[:, 1]


class _ZeroMatrixProxy:
    """
    零内存占用的“全 0 矩阵”代理，支持 A[i, j] 索引，返回 0.0。
    用于在大规模时传给绘图函数，避免分配 N×N 大矩阵。
    """
    __slots__ = ()

    def __getitem__(self, idx):
        return 0.0

    def __setitem__(self, idx, val):
        # 忽略写入
        pass

    def fill(self, val: float):
        # 与 numpy API 对齐（小规模模式才会用 fill，这里保留以防万一）
        pass


# ------------------------------------------------------------------------------
# 4) 主函数
def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    进阶优化版：
      - 邻域构建：KDTree（优先）或 cell-list（fallback）
      - 半边去重 + 成对累加（每边一次，反作用力一致）
      - Jn/rn 流式统计（避免二次 O(N^2) 扫描）
      - 绘图/日志节流
      - 收敛判据：最近窗口线性回归斜率 + 标准差
      - 大规模：使用零内存的“边矩阵”代理，避免分配 N×N
    """
    # ---------------- 参数检查与派生量 ----------------
    assert 0.0 < PT < 1.0, "PT 必须在 (0, 1)"
    assert alpha > 0 and beta > 0 and v > 0 and r0 > 0, "alpha/beta/v/r0 必须为正"

    step_size = getattr(config, 'STEP_SIZE', 0.01)
    mode = getattr(config, 'MODE', 'hpc')  # "viz" 或 "hpc"
    plot_every = getattr(config, 'PLOT_EVERY', 20)
    log_every = getattr(config, 'LOG_EVERY', 20)

    # 修复：防止 plot_every/log_every 为 0 导致无限循环
    plot_every = max(1, plot_every)
    log_every = max(1, log_every)

    convergence_window = getattr(config, 'CONVERGENCE_WINDOW', 50)
    slope_threshold = getattr(config, 'CONVERGENCE_SLOPE_THRESHOLD', 1e-6)
    std_threshold = getattr(config, 'CONVERGENCE_STD_THRESHOLD', 1e-5)

    # 新增：数值稳定性检查参数
    STABILITY_THRESHOLD = getattr(config, 'STABILITY_THRESHOLD', 1e6)

    # 通信半径（由 a_ij = PT 反解）
    R = r0 * ((-_safe_log_pt(PT)) / beta) ** (1.0 / v)
    # 数值稳定：EPS 与尺度挂钩
    EPS = max(getattr(config, 'EPS', 1e-12), 1e-9 * R)

    if VERBOSE_IMPORT and (max_iter > 0):
        if _HAS_SCIPY_KDTREE:
            print("[alg-adv] 使用 SciPy KDTree 进行邻域搜索。")
        else:
            print("[alg-adv] SciPy 不可用，使用 cell-list 邻域搜索。")
        if _HAS_SCIPY_LINREGRESS:
            print("[alg-adv] 使用 SciPy linregress 进行收敛判据。")
        else:
            print("[alg-adv] SciPy 不可用，使用 numpy.polyfit 进行收敛判据。")

    # ---------------- 结构与缓存 ----------------
    swarm_control_ui = np.zeros((swarm_size, 2), dtype=swarm_position.dtype)

    # 可视化细节阈值（小规模 & 显式 viz 模式时才画边）
    VIS_THRESHOLD = getattr(config, 'VISUALIZATION_THRESHOLD', 40)
    do_viz_details = (swarm_size <= VIS_THRESHOLD and mode == "viz")

    # 小规模：分配真实矩阵；大规模：用零矩阵代理，避免内存炸裂
    if do_viz_details:
        comm_mat = np.zeros((swarm_size, swarm_size), dtype=np.float32)
    else:
        comm_mat = _ZeroMatrixProxy()

    # 指标与时间
    Jn_raw, rn_raw = [], []  # 原始序列，用于收敛判据
    Jn, rn = [], []  # 显示序列（四舍五入）
    t_elapsed = []
    start_time = time.time()

    # ---------------- 主循环 ----------------
    for it in range(max_iter):
        # 1) 邻域构建（KDTree or cell-list）
        neighbor_lists = _get_neighbor_lists(swarm_position, R)

        # 2) 流式统计计数器
        sum_phi = 0.0;
        cnt_phi = 0
        sum_r = 0.0;
        cnt_r = 0

        # 3) 小规模时清空绘制矩阵
        if do_viz_details:
            comm_mat.fill(0.0)

        # 4) 半边去重 + 成对累加（只算 i<j）
        for i in range(swarm_size):
            neigh_i = neighbor_lists[i]
            if not neigh_i:
                continue
            qi = swarm_position[i, :]

            for j in neigh_i:
                if j <= i:
                    continue
                qj = swarm_position[j, :]

                # 距离 & 通信质量
                rij = utils.calculate_distance(qi, qj)
                aij = utils.calculate_aij(alpha, config.DELTA, rij, r0, v)
                if aij < PT:
                    continue

                gij = utils.calculate_gij(rij, r0)
                rho_ij = utils.calculate_rho_ij(beta, v, rij, r0)
                eij = (qi - qj) / (rij + EPS)

                # —— 成对更新（反作用力一致）——
                force = rho_ij * eij
                swarm_control_ui[i, :] += force
                swarm_control_ui[j, :] += -force

                # —— 流式累计指标 ——
                phi_rij = gij * aij
                sum_phi += phi_rij;
                cnt_phi += 1
                sum_r += rij;
                cnt_r += 1

                # —— 小规模画边 ——
                if do_viz_details:
                    comm_mat[i, j] = phi_rij
                    comm_mat[j, i] = phi_rij

        # 5) 新增：数值稳定性检查
        # 检查控制输入是否过大，这通常是数值不稳定的迹象
        max_force_norm = np.linalg.norm(swarm_control_ui, axis=1).max()
        if max_force_norm > STABILITY_THRESHOLD:
            print(f"[error] 在 it={it} 检测到数值不稳定，最大控制输入范数: {max_force_norm:.2e}。提前终止。")
            break

        # 统一更新位置，并清零控制输入
        swarm_position += step_size * swarm_control_ui
        swarm_control_ui.fill(0.0)

        # 6) 指标入列（原始 + 显示）
        Jn_new = (sum_phi / cnt_phi) if cnt_phi > 0 else 0.0
        rn_new = (sum_r / cnt_r) if cnt_r > 0 else 0.0
        Jn_raw.append(Jn_new);
        rn_raw.append(rn_new)
        Jn.append(round(Jn_new, 4));
        rn.append(round(rn_new, 4))
        t_elapsed.append(time.time() - start_time)

        # 7) 日志节流（修复 bug：日志打印在计算后）
        if it % log_every == 0:
            print(f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f}")
            # 无边警告（忽略前几步的预热）
            if cnt_phi == 0 and it > 10:
                print("  [warn] 本步无邻接边（cnt_phi=0）。考虑降低 PT 或提高密度。")

        # 8) 绘图节流
        if it % plot_every == 0:
            plotting.plot_figures_task1(
                axs, t_elapsed, Jn, rn, swarm_position, PT,
                comm_mat, swarm_size, swarm_paths,
                node_colors, line_colors
            )

        # 9) 收敛判据：最近窗口线性回归斜率 + 标准差（支持固定步数模式）
        if not getattr(config, "FORCE_FIXED_ITERS", False):
            if len(Jn_raw) > convergence_window:
                recent = Jn_raw[-convergence_window:]
                slope, std_dev = _slope_and_std(recent)
                if abs(slope) < slope_threshold and std_dev < std_threshold:
                    print(f"[done] Jn 收敛：t={t_elapsed[-1]:.2f}s, it={it}, "
                          f"slope={slope:.2e} (<{slope_threshold:.2e}), "
                          f"std={std_dev:.2e} (<{std_threshold:.2e})")
                    break
    # 最后一帧刷新（防 max_iter=0 时 it 未定义）
    if t_elapsed and ((len(Jn) - 1) % plot_every != 0):
        plotting.plot_figures_task1(
            axs, t_elapsed, Jn, rn, swarm_position, PT,
            comm_mat, swarm_size, swarm_paths,
            node_colors, line_colors
        )

    # 为了与旧接口兼容，返回 comm_mat 和一个占位 None（neighbor 矩阵不再维护）
    return Jn, rn, swarm_position, t_elapsed, comm_mat, None
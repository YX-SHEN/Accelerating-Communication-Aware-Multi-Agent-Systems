# src/gpu/algorithms_advanced_gpu.py
# -*- coding: utf-8 -*-
"""
GPU 加速的【进阶优化版】（单文件版）
- 邻域构建：KDTree（优先）或 cell-list（fallback） -> 在 CPU
- 边列表按 i<j 去重；“边上的物理量+力”计算在 GPU（JAX/CuPy），并用 bincount 做按节点归并
- Jn/rn 流式统计；绘图/日志节流；收敛判据（斜率+方差）与 CPU 版一致
- 小规模可视化可构造 NxN 的 comm_mat；大规模使用零矩阵代理以节省内存
"""

import time
import os
import numpy as np
import config
from src.common import plotting

# 从 src.optimized 导入收敛判据和邻居搜索工具，以保持逻辑一致
from src.optimized.algorithms_advanced import (
    _get_neighbor_lists, _edges_from_neighbors, _slope_and_std,
    _safe_log_pt, _ZeroMatrixProxy,
)
# 从 GPU 模块导入后端和计算核心
from src.gpu.backend import ImprovedGPUBackend, DTYPE_DEFAULT
from src.gpu.compute import compute_forces_gpu

# ------------------------------------------------------------------------------
# 可选：是否在运行时打印“使用了KDTree/linregress还是fallback”的提示
VERBOSE_IMPORT = getattr(config, 'VERBOSE_IMPORT', False)

# 全局后端实例
_GPU_INST = ImprovedGPUBackend()


# ------------------------------------------------------------------------------
# 5) 主函数（与 CPU 版接口完全一致）
def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    进阶优化版 + GPU（单文件）：
      - 邻域构建：KDTree（优先）或 cell-list（fallback）【CPU】
      - 按边计算物理量与力，GPU 向量化 + bincount 归并
      - Jn/rn 流式统计；绘图/日志节流；收敛判据：最近窗口线性回归斜率 + 标准差
      - 大规模：使用零内存的“边矩阵”代理，避免分配 N×N
    """
    swarm_position = swarm_position.astype(np.float32, copy=False)
    AD_DEBUG = os.getenv("AD_DEBUG", "0") == "1"
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
        print(f"[alg-adv-gpu:backend] {_GPU_INST.backend}")

    # ---------------- 结构与缓存 ----------------
    # 可视化细节阈值（小规模 & 显式 viz 模式时才画边）
    VIS_THRESHOLD = getattr(config, 'VISUALIZATION_THRESHOLD', 40)
    do_viz_details = (swarm_size <= VIS_THRESHOLD and mode == "viz")

    # 小规模：用真实矩阵；大规模：零矩阵代理（传给 plot）
    if do_viz_details:
        comm_mat = np.zeros((swarm_size, swarm_size), dtype=np.float32)
    else:
        comm_mat = _ZeroMatrixProxy()

    # 指标与时间
    Jn_raw, rn_raw = [], []  # 原始序列，用于收敛判据
    Jn, rn = [], []  # 显示序列（四舍五入）
    t_elapsed = []
    start_time = time.time()

    print(f"[alg-adv-gpu] 后端: {_GPU_INST.backend}")
    if _GPU_INST.device:
        print(f"[alg-adv-gpu] 设备: {_GPU_INST.device}")
    print(f"[alg-adv-gpu] 群体规模: {swarm_size}, 最大迭代: {max_iter}")
    print("开始仿真循环...")

    # ---------------- 主循环 ----------------
    for it in range(max_iter):
        # 1) 邻域构建（CPU）
        t_neigh_start = time.time()  # 诊断计时点
        neighbor_lists = _get_neighbor_lists(swarm_position, R)
        i_idx, j_idx = _edges_from_neighbors(neighbor_lists)
        if AD_DEBUG and (it % log_every == 0):
            # i_idx 是有向边的起点索引列表，长度就是有向边数
            e_dir = int(i_idx.size)
            avg_out_deg = (e_dir / float(swarm_size)) if swarm_size > 0 else 0.0
            print(f"[dbg] edges_directed={e_dir} avg_out_degree={avg_out_deg:.2f} R={R:.3f} PT={PT:.3g}")
        t_neigh_end = time.time()  # 诊断计时点

        # 2) 若无边，直接保持静止 & 提示
        if i_idx.size == 0:
            forces_cpu = np.zeros_like(swarm_position)
            Jn_new = 0.0;
            rn_new = 0.0

            # --- 修复：指标入列，为下一轮循环做准备 ---
            Jn_raw.append(Jn_new)
            rn_raw.append(rn_new)
            Jn.append(round(Jn_new, 4))
            rn.append(round(rn_new, 4))
            t_elapsed.append(time.time() - start_time)

            # --- 修复：打印日志以显示当前状态 ---
            if it % log_every == 0:
                print(
                    f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f} "
                    f"compute=0.00ms neigh={(t_neigh_end - t_neigh_start) * 1000:.2f}ms"
                )
                if it > 10:
                    print("  [warn] 本步无邻接边（cnt=0）。考虑降低 PT 或提高密度。")

            # 绘图节流
            enable_plot = (axs is not None) and (fig is not None) and (plot_every > 0)
            if enable_plot and (it % plot_every == 0) and (it > 0):
                plotting.plot_figures_task1(
                    axs, t_elapsed, Jn, rn, swarm_position, PT,
                    _ZeroMatrixProxy(), swarm_size, swarm_paths,
                    node_colors, line_colors
                )
            continue  # 跳过本次循环的剩余部分，进入下一次迭代

        # 3) 边上的 GPU 计算（注意：传入 i_idx/j_idx，启用 ecap 固定形状）
        t_gpu_start = time.time()
        forces_dev, Jn_new, rn_new, phi_edges_dev, valid_mask_dev = compute_forces_gpu(
            gpu=_GPU_INST,
            positions=swarm_position,
            N=swarm_size,
            alpha=alpha, beta=beta, v=v, r0=r0, PT=PT, delta=config.DELTA,
            keep_full_matrix=False,
            dtype=swarm_position.dtype,
            eps=EPS,
            i_idx=i_idx,  # <<< 新增
            j_idx=j_idx,  # <<< 新增
            ecap_factor=getattr(config, "GPU_ECAP_FACTOR", 1.25),  # <<< 建议放到 config 可调
        )
        t_gpu_end = time.time()

        # 将 GPU 计算结果同步到 CPU
        forces_cpu = forces_dev
        valid_cpu = valid_mask_dev
        phi_cpu = phi_edges_dev
        if _GPU_INST.backend == "cupy":
            forces_cpu = _GPU_INST.cp.asnumpy(forces_dev)
            valid_cpu = _GPU_INST.cp.asnumpy(valid_mask_dev)
            phi_cpu = _GPU_INST.cp.asnumpy(phi_edges_dev)
        elif _GPU_INST.backend == "jax":
            forces_cpu = np.asarray(forces_dev)
            valid_cpu = np.asarray(valid_mask_dev)
            phi_cpu = np.asarray(phi_edges_dev)
        # else: NumPy 回退不需要转换

        # 4) 新增：数值稳定性检查
        max_force_norm = np.linalg.norm(forces_cpu, axis=1).max()
        if max_force_norm > STABILITY_THRESHOLD:
            print(f"[error] 在 it={it} 检测到数值不稳定，最大控制输入范数: {max_force_norm:.2e}。提前终止。")
            # --- 确保在异常终止前记录当前状态 ---
            Jn_raw.append(Jn_new)
            rn_raw.append(rn_new)
            Jn.append(round(Jn_new, 4))
            rn.append(round(rn_new, 4))
            t_elapsed.append(time.time() - start_time)
            break

        # 统一更新位置，并清零控制输入
        swarm_position += step_size * forces_cpu
        pass

        # 5) 指标入列
        Jn_raw.append(Jn_new);
        rn_raw.append(rn_new)
        Jn.append(round(Jn_new, 4));
        rn.append(round(rn_new, 4))
        t_elapsed.append(time.time() - start_time)

        # 6) 日志节流（修复 bug：日志打印在计算后）
        if it % log_every == 0:
            print(f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f} (邻居搜索: {(t_neigh_end - t_neigh_start) * 1000:.2f}ms, "
                  f"GPU计算: {(t_gpu_end - t_gpu_start) * 1000:.2f}ms)")
            print(f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f} compute={(t_gpu_end - t_gpu_start) * 1000:.2f}ms "
                  f"neigh={(t_neigh_end - t_neigh_start) * 1000:.2f}ms")

        # 7) 绘图节流
        enable_plot = (axs is not None) and (fig is not None) and (plot_every > 0)
        if enable_plot and (it % plot_every == 0) and (it > 0):
            if do_viz_details:
                comm_mat.fill(0.0)
                if i_idx.size > 0:
                    ii = i_idx[valid_cpu]
                    jj = j_idx[valid_cpu]
                    phi_valid = phi_cpu[valid_cpu]
                    comm_mat[ii, jj] = phi_valid
                    comm_mat[jj, ii] = phi_valid
            plotting.plot_figures_task1(
                axs, t_elapsed, Jn, rn, swarm_position, PT,
                comm_mat, swarm_size, swarm_paths,
                node_colors, line_colors
            )


        # 8) 收敛判据（固定步数模式下不提前退出）
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
    enable_plot = (axs is not None) and (fig is not None) and (plot_every > 0)
    if enable_plot and t_elapsed and ((len(Jn) - 1) % plot_every != 0):
        plotting.plot_figures_task1(
            axs, t_elapsed, Jn, rn, swarm_position, PT,
            comm_mat, swarm_size, swarm_paths,
            node_colors, line_colors
        )

    # 为了与旧接口兼容，返回 comm_mat 和一个占位 None（neighbor 矩阵不再维护）
    return Jn, rn, swarm_position, t_elapsed, comm_mat, None
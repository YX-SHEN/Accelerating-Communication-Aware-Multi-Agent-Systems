# src/gpu/algorithms_gpu_unified.py
# -*- coding: utf-8 -*-
"""
跨平台 GPU 加速版本（统一实现 + 与 CPU baseline 完全等价的 Gauss–Seidel 选项）

- UPDATE_MODE="gauss"：逐智能体就地更新（与 CPU baseline 的顺序一致）
- UPDATE_MODE="jacobi"：一次性并行（统一 GPU 内核）
- 读取 config.DTYPE/DT/EPS_DIV；若缺失则回退到旧名 STEP_SIZE/EPS
- 日志保持 compute=XXms 便于外部脚本解析
"""

import time
import numpy as np
import config
from src.common import plotting

from .backend import ImprovedGPUBackend, DTYPE_DEFAULT
from .compute import compute_forces_gpu

# 全局后端实例（仅初始化一次）
gpu = ImprovedGPUBackend()


def _build_comm_matrix_for_plot(positions, PT, alpha, beta, v, r0, delta, keep_full_matrix, dtype, eps_sqrt):
    """
    在需要绘图时构造 NxN 的通信质量矩阵（仅在 keep_full_matrix=True 时调用）
    支持 NumPy / CuPy / JAX（JAX 走 NumPy 回落），返回 host 上的 np.ndarray
    """
    if not keep_full_matrix:
        return None

    # 选择后端数组与 to_host
    backend = gpu.backend
    if backend == "cupy":
        xp = gpu.cp
        pos = positions  # 已是 cupy 数组
        to_host = gpu.cp.asnumpy
    elif backend == "jax":
        # 简化：JAX 先搬回主机再用 NumPy 计算（绘图频率很低）
        xp = np
        pos = np.asarray(gpu.to_host(positions), dtype=dtype)
        to_host = lambda x: x
    else:
        xp = np
        pos = positions
        to_host = lambda x: x

    N = pos.shape[0]
    # O(N^2) 构造（仅用于可视化）
    diff = pos[:, None, :] - pos[None, :, :]          # [N,N,2]
    d2   = xp.sum(diff * diff, axis=2)                # [N,N]
    rij  = xp.sqrt(xp.maximum(d2, xp.asarray(eps_sqrt, dtype=dtype)))

    eye  = xp.eye(N, dtype=bool)
    valid= xp.logical_and(rij > eps_sqrt, ~eye)

    # 通信概率/近场抑制
    aij  = xp.exp(-alpha * (2**delta - 1) * (rij / r0) ** v)
    conn = xp.logical_and(valid, aij >= PT)

    gij  = rij / xp.sqrt(rij * rij + (r0 ** 2))
    phi  = gij * aij

    comm_mat = (phi * conn).astype(dtype, copy=False)
    return to_host(comm_mat)


def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    GPU 加速的仿真主循环。
    - 若 config.UPDATE_MODE == "gauss"，逐智能体就地更新，与 CPU baseline 完全一致的顺序/口径；
    - 否则走统一 GPU 内核（一次性并行）。
    返回：
      Jn_list(四舍五入显示值), rn_list(四舍五入显示值), final_positions(np),
      t_elapsed(s), final_comm_matrix(np 或零矩阵), None
    """
    # -------- 动态读取配置（统一命名 + 兼容旧名）--------
    dtype = getattr(config, "DTYPE", DTYPE_DEFAULT)
    step_size = getattr(config, "DT", getattr(config, "STEP_SIZE", 0.01))
    eps_eij   = getattr(config, "EPS_DIV", getattr(config, "EPS", 1e-6))
    delta = getattr(config, "DELTA", 2)
    log_every = max(1, getattr(config, "LOG_EVERY", 20))
    plot_every = max(1, getattr(config, "PLOT_EVERY", 20))
    keep_full_matrix = bool(getattr(config, "KEEP_FULL_MATRIX", True))
    convergence_window = int(getattr(config, "CONVERGENCE_WINDOW", 20))
    update_mode = getattr(config, "UPDATE_MODE", "jacobi").lower()  # "gauss" 或 "jacobi"

    print(f"[GPU] 后端: {gpu.backend}")
    if gpu.device:
        print(f"[GPU] 设备: {gpu.device}")
    print(f"[GPU] dtype: {np.dtype(dtype).name}, 群体规模: {swarm_size}, 最大迭代: {max_iter}, 模式: {update_mode}")

    # ---- 设备端初始化 ----
    d_positions = gpu.to_device(swarm_position.astype(dtype, copy=False))

    # ---- 统计与计时容器 ----
    Jn_history_raw = []   # 原始浮点，统计/判停用
    rn_history_raw = []
    Jn_display = []       # 四舍五入显示值
    rn_display = []
    t_elapsed = []
    computation_times = []  # 每步 compute 耗时（秒）
    start_time = time.time()

    # 若不保留全矩阵，这个占位用于返回/绘图（保持接口一致）
    comm_mat_host_placeholder = np.zeros((swarm_size, swarm_size), dtype=dtype)
    last_comm_host = comm_mat_host_placeholder

    # sqrt 的极小量单独给一个（避免 0）
    eps_sqrt = max(1e-20 if dtype == np.float64 else 1e-12, float(eps_eij))

    print("开始仿真循环...")
    for it in range(max_iter):
        t0 = time.time()

        if (update_mode == "gauss"):
            # ========= 与 CPU baseline 完全一致的“就地更新”（Gauss–Seidel）=========
            backend = gpu.backend
            if backend == "cupy":
                xp = gpu.cp
                pos = d_positions  # cupy.ndarray [N,2]
            elif backend == "jax":
                xp = np
                pos = np.asarray(gpu.to_host(d_positions), dtype=dtype)  # 回到主机
            else:
                xp = np
                pos = d_positions  # np.ndarray

            N = swarm_size
            idx_all = xp.arange(N)

            # 流式累计（与 CPU baseline 的 “双向计数” 等价）
            tot_phi = 0.0
            tot_r = 0.0
            tot_cnt = 0

            # 将标量也转 dtype，减少隐式cast
            alpha_c = dtype(alpha)
            beta_c  = dtype(beta)
            r0_c    = dtype(r0)

            for i in range(N):
                qi   = pos[i]                              # [2]
                diff = qi - pos                            # [N,2]
                d2   = xp.sum(diff * diff, axis=1)         # [N]
                rij  = xp.sqrt(xp.maximum(d2, eps_sqrt))   # [N]

                not_self = (idx_all != i)
                aij = xp.exp(-alpha_c * (2**delta - 1) * (rij / r0_c) ** v)
                nbr = xp.logical_and(not_self, aij >= PT)  # 达阈值邻居

                gij = rij / xp.sqrt(rij * rij + (r0_c ** 2))
                # ρ_ij（与 CPU 一致）
                num = (-beta_c * v * rij**(v + 2) - beta_c * v * (r0_c**2) * (rij**v) + r0_c**(v + 2))
                den = xp.sqrt((rij * rij + r0_c**2) ** 3) + eps_eij
                rho = num * xp.exp(-beta_c * (rij / r0_c) ** v) / den

                eij = diff / (rij + eps_eij)[:, None]      # 分母加 eps，与 CPU 一致
                f_i = xp.sum((rho[:, None] * eij) * nbr[:, None], axis=0)  # 只累达阈值邻居

                # ——就地更新：与 CPU baseline 完全一致（先算 i，再改 i）——
                pos[i] = pos[i] + step_size * f_i

                # 同步累计统计（与 CPU 的“按(i,j)方向两次计数”一致）
                phi = gij * aij
                if backend == "cupy":
                    tot_phi += float(xp.sum(phi[nbr]).item())
                    tot_r   += float(xp.sum(rij[nbr]).item())
                    tot_cnt += int(xp.sum(nbr).item())
                else:
                    tot_phi += float(np.sum(phi[nbr]))
                    tot_r   += float(np.sum(rij[nbr]))
                    tot_cnt += int(np.sum(nbr))

            # 将位置写回设备（JAX/NumPy 回落时）
            if backend != "cupy":
                d_positions = gpu.to_device(pos.astype(dtype, copy=False))

            # 本步指标（与 CPU 口径一致）
            cnt = max(tot_cnt, 1)
            Jn_new = tot_phi / cnt
            rn_new = tot_r / cnt

            # 构造绘图矩阵（仅在需要时；避免每步 O(N^2)）
            if (it % plot_every == 0) and keep_full_matrix:
                last_comm_host = _build_comm_matrix_for_plot(
                    d_positions, PT, alpha_c, beta_c, v, r0_c, delta,
                    keep_full_matrix, dtype, eps_sqrt
                )

        else:
            # ========= 一次性并行（Jacobi）路径：统一 GPU 内核 =========
            forces, Jn_new, rn_new, comm_mat_dev, _ = compute_forces_gpu(
                gpu, d_positions, swarm_size, alpha, beta, v, r0, PT, delta,
                keep_full_matrix=keep_full_matrix, dtype=dtype, eps=eps_eij
            )
            # 同步/计时
            if gpu.backend == "jax":
                _ = forces.block_until_ready()
            else:
                gpu.synchronize()
            # 位置更新（Jacobi：使用同一步 forces 更新所有点）
            d_positions = d_positions + step_size * forces

            # 若本步需要绘图，取回通信矩阵
            if (it % plot_every == 0) and keep_full_matrix and (comm_mat_dev is not None):
                last_comm_host = gpu.to_host(comm_mat_dev)

        # === 计时与记录 ===
        if gpu.backend == "cupy":
            gpu.synchronize()
        step_time = time.time() - t0
        computation_times.append(step_time)

        Jn_history_raw.append(float(Jn_new))
        rn_history_raw.append(float(rn_new))
        Jn_display.append(round(float(Jn_new), 4))
        rn_display.append(round(float(rn_new), 4))
        t_elapsed.append(time.time() - start_time)

        # === 日志 ===
        if it % log_every == 0:
            k = min(log_every, len(computation_times))
            avg_ms = np.mean(computation_times[-k:]) * 1000.0
            print(f"[it={it:4d}] Jn={Jn_display[-1]:.4f} rn={rn_display[-1]:.4f} compute={avg_ms:.2f}ms")

        # === 绘图（节流）===
        if it % plot_every == 0:
            pos_cpu = gpu.to_host(d_positions)
            comm_cpu = last_comm_host if keep_full_matrix else comm_mat_host_placeholder
            plotting.plot_figures_task1(
                axs, t_elapsed, Jn_display, rn_display, pos_cpu, PT,
                comm_cpu, swarm_size, swarm_paths, node_colors, line_colors
            )

        # === 与 baseline 一致的“平台期”判停（最近 W 步显示值完全相等）===
        if len(Jn_display) >= convergence_window:
            recent = Jn_display[-convergence_window:]
            if len(set(recent)) == 1:
                plateau_start = it - convergence_window
                if getattr(config, "FORCE_FIXED_ITERS", False):
                    print(
                        f"[info] Jn would converge at t={round(t_elapsed[-1], 2)}s "
                        f"after {plateau_start} iterations, but fixed-iters mode keeps running."
                    )
                else:
                    print(
                        f"Formation completed: Jn converged in {round(t_elapsed[-1], 2)}s "
                        f"after {plateau_start} iterations."
                    )
                    break

    # ---- 汇总与返回 ----
    total_time = time.time() - start_time
    avg_ms_all = (np.mean(computation_times) * 1000.0) if computation_times else 0.0
    final_positions = gpu.to_host(d_positions)

    final_comm = last_comm_host if keep_full_matrix else comm_mat_host_placeholder

    print("\n=== 仿真完成 ===")
    print(f"总时间: {total_time:.2f}s")
    print(f"平均compute时间: {avg_ms_all:.2f}ms/iter")
    print(f"总迭代数: {len(Jn_display)}")
    if Jn_display:
        print(f"最终 Jn: {Jn_display[-1]:.4f}")
    if rn_display:
        print(f"最终 rn: {rn_display[-1]:.4f}")

    return (Jn_display, rn_display, final_positions, t_elapsed, final_comm, None)
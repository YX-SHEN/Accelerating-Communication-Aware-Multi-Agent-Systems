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
import json

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
    进阶优化版 + GPU（单文件），新增：V4 每步时间分解 JSONL 记录
      - 写入文件：环境变量 V4_TIMEBREAK_FILE（默认 ./v4_timebreak_steps.jsonl）
      - 字段：T_range/T_h2d/T_edge/T_d2h/T_update/T_overhead/T_total（单位 ms）
    """
    import time, os
    V4_FILE = os.getenv("V4_TIMEBREAK_FILE", "v4_timebreak_steps.jsonl")

    swarm_position = np.asarray(
        swarm_position,
        dtype=getattr(config, "DTYPE", swarm_position.dtype),
        copy=False
    )
    AD_DEBUG = os.getenv("AD_DEBUG", "0") == "1"
    assert 0.0 < PT < 1.0, "PT 必须在 (0, 1)"
    assert alpha > 0 and beta > 0 and v > 0 and r0 > 0, "alpha/beta/v/r0 必须为正"

    step_size = getattr(config, 'STEP_SIZE', 0.01)
    mode = getattr(config, 'MODE', 'hpc')
    plot_every = max(1, getattr(config, 'PLOT_EVERY', 20))
    log_every = max(1, getattr(config, 'LOG_EVERY', 20))

    convergence_window = getattr(config, 'CONVERGENCE_WINDOW', 50)
    slope_threshold = getattr(config, 'CONVERGENCE_SLOPE_THRESHOLD', 1e-6)
    std_threshold = getattr(config, 'CONVERGENCE_STD_THRESHOLD', 1e-5)
    STABILITY_THRESHOLD = getattr(config, 'STABILITY_THRESHOLD', 1e6)

    R = r0 * ((-_safe_log_pt(PT)) / beta) ** (1.0 / v)
    EPS = max(getattr(config, 'EPS', 1e-12), 1e-9 * R)

    if VERBOSE_IMPORT and (max_iter > 0):
        print(f"[alg-adv-gpu:backend] {_GPU_INST.backend}")

    VIS_THRESHOLD = getattr(config, 'VISUALIZATION_THRESHOLD', 40)
    do_viz_details = (swarm_size <= VIS_THRESHOLD and mode == "viz")

    comm_mat = np.zeros((swarm_size, swarm_size), dtype=np.float32) if do_viz_details else _ZeroMatrixProxy()

    Jn_raw, rn_raw, Jn, rn, t_elapsed = [], [], [], [], []
    start_time = time.time()

    print(f"[alg-adv-gpu] 后端: {_GPU_INST.backend}")
    if _GPU_INST.device:
        print(f"[alg-adv-gpu] 设备: {_GPU_INST.device}")
    print(f"[alg-adv-gpu] 群体规模: {swarm_size}, 最大迭代: {max_iter}")
    print("开始仿真循环...")

    for it in range(max_iter):
        t_step0 = time.time()

        # 1) 邻域构建（CPU）—— T_range
        t0 = time.time()
        neighbor_lists = _get_neighbor_lists(swarm_position, R)
        i_idx, j_idx = _edges_from_neighbors(neighbor_lists)
        T_range_ms = (time.time() - t0) * 1000.0

        if AD_DEBUG and (it % log_every == 0):
            e_dir = int(i_idx.size)
            avg_out_deg = (e_dir / float(swarm_size)) if swarm_size > 0 else 0.0
            print(f"[dbg] edges_directed={e_dir} avg_out_degree={avg_out_deg:.2f} R={R:.3f} PT={PT:.3g}")

        # 2) 无边时：保持静止 + 记录
        if i_idx.size == 0:
            forces_cpu = np.zeros_like(swarm_position)
            Jn_new = 0.0; rn_new = 0.0

            # 统计序列
            Jn_raw.append(Jn_new); rn_raw.append(rn_new)
            Jn.append(round(Jn_new, 4)); rn.append(round(rn_new, 4))
            t_elapsed.append(time.time() - start_time)

            # 打印
            if it % log_every == 0:
                print(f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f} compute=0.00ms neigh={T_range_ms:.2f}ms")
                if it > 10:
                    print("  [warn] 本步无邻接边（cnt=0）。考虑降低 PT 或提高密度。")

            # 绘图
            if (axs is not None) and (fig is not None) and (it % plot_every == 0) and (it > 0):
                plotting.plot_figures_task1(
                    axs, t_elapsed, Jn, rn, swarm_position, PT,
                    _ZeroMatrixProxy(), swarm_size, swarm_paths, node_colors, line_colors
                )

            # —— 记录 JSONL（无 GPU 活动，其它阶段记 0）——
            T_d2h_ms = 0.0; T_h2d_ms = 0.0; T_edge_ms = 0.0; T_update_ms = 0.0
            T_total_ms = (time.time() - t_step0) * 1000.0
            T_overhead_ms = max(0.0, T_total_ms - (T_range_ms + T_h2d_ms + T_edge_ms + T_d2h_ms + T_update_ms))
            try:
                with open(V4_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "it": it,
                        "T_range": T_range_ms, "T_h2d": T_h2d_ms, "T_edge": T_edge_ms,
                        "T_d2h": T_d2h_ms, "T_update": T_update_ms,
                        "T_overhead": T_overhead_ms, "T_total": T_total_ms
                    }) + "\n")
            except Exception:
                pass
            continue

        # 3) GPU 边计算（显式计时：H2D / Edge / D2H）
        t_gpu_h2d_ms = 0.0
        t_gpu_edge_ms = 0.0
        t_gpu_d2h_ms = 0.0

        backend = _GPU_INST.backend

        # --- H2D：把位置和边索引搬到设备，并计时 ---
        t_h2d0 = time.time()
        if backend == "cupy":
            cp = _GPU_INST.cp
            pos_dev = cp.asarray(swarm_position, dtype=swarm_position.dtype)
            ii_dev = cp.asarray(i_idx, dtype=cp.int32)
            jj_dev = cp.asarray(j_idx, dtype=cp.int32)
            _GPU_INST.synchronize()
            t_gpu_h2d_ms = (time.time() - t_h2d0) * 1000.0
        elif backend == "jax":
            jnp = _GPU_INST.jnp
            pos_dev = jnp.asarray(swarm_position, dtype=swarm_position.dtype)
            ii_dev = jnp.asarray(i_idx, dtype=jnp.int32)
            jj_dev = jnp.asarray(j_idx, dtype=jnp.int32)
            # JAX 的拷贝是 lazy，这里计时意义不大，记 0
            t_gpu_h2d_ms = 0.0
        else:
            # numpy 回退，不做搬运
            pos_dev = swarm_position
            ii_dev = i_idx
            jj_dev = j_idx
            t_gpu_h2d_ms = 0.0

        # --- Edge kernel：显式包裹计时 ---
        t_edge0 = time.time()
        forces_dev, Jn_new, rn_new, phi_edges_dev, valid_mask_dev = compute_forces_gpu(
            gpu=_GPU_INST,
            positions=pos_dev,
            N=swarm_size,
            alpha=alpha, beta=beta, v=v, r0=r0, PT=PT, delta=config.DELTA,
            keep_full_matrix=False,
            dtype=swarm_position.dtype,
            eps=EPS,
            i_idx=ii_dev, j_idx=jj_dev,
            ecap_factor=getattr(config, "GPU_ECAP_FACTOR", 1.25),
        )
        if backend == "jax":
            _ = forces_dev.block_until_ready()
        elif backend == "cupy":
            _GPU_INST.synchronize()
        t_gpu_edge_ms = (time.time() - t_edge0) * 1000.0

        # 4) D->H（回主机）并计时
        t_d2h0 = time.time()
        if backend == "cupy":
            forces_cpu = _GPU_INST.cp.asnumpy(forces_dev)
            valid_cpu = _GPU_INST.cp.asnumpy(valid_mask_dev)
            phi_cpu = _GPU_INST.cp.asnumpy(phi_edges_dev)
            t_gpu_d2h_ms = (time.time() - t_d2h0) * 1000.0
        elif backend == "jax":
            forces_cpu = np.asarray(forces_dev)
            valid_cpu = np.asarray(valid_mask_dev)
            phi_cpu = np.asarray(phi_edges_dev)
            t_gpu_d2h_ms = (time.time() - t_d2h0) * 1000.0
        else:
            forces_cpu = forces_dev
            valid_cpu = valid_mask_dev
            phi_cpu = phi_edges_dev
            t_gpu_d2h_ms = 0.0

        # 5) 稳定性检查 + 位置更新（CPU Jacobi）—— T_update
        max_force_norm = np.linalg.norm(forces_cpu, axis=1).max()
        if max_force_norm > STABILITY_THRESHOLD:
            print(f"[error] 在 it={it} 检测到数值不稳定，最大控制输入范数: {max_force_norm:.2e}。提前终止。")
            # 也记录一条 JSONL
            T_update_ms = 0.0
            T_total_ms = (time.time() - t_step0) * 1000.0
            T_overhead_ms = max(0.0, T_total_ms - (T_range_ms + t_gpu_h2d_ms + t_gpu_edge_ms + t_gpu_d2h_ms + T_update_ms))
            try:
                with open(V4_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "it": it,
                        "T_range": T_range_ms, "T_h2d": t_gpu_h2d_ms, "T_edge": t_gpu_edge_ms,
                        "T_d2h": t_gpu_d2h_ms, "T_update": T_update_ms,
                        "T_overhead": T_overhead_ms, "T_total": T_total_ms
                    }) + "\n")
            except Exception:
                pass
            # 正常的终止流程
            Jn_raw.append(Jn_new); rn_raw.append(rn_new)
            Jn.append(round(Jn_new, 4)); rn.append(round(rn_new, 4))
            t_elapsed.append(time.time() - start_time)
            break

        t3 = time.time()
        swarm_position += step_size * forces_cpu
        T_update_ms = (time.time() - t3) * 1000.0

        # 6) 指标序列
        Jn_raw.append(Jn_new); rn_raw.append(rn_new)
        Jn.append(round(Jn_new, 4)); rn.append(round(rn_new, 4))
        t_elapsed.append(time.time() - start_time)

        # 7) 日志
        if it % log_every == 0:
            print(f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f} "
                  f"compute={t_gpu_edge_ms:.2f}ms neigh={T_range_ms:.2f}ms h2d={t_gpu_h2d_ms:.2f}ms d2h={t_gpu_d2h_ms:.2f}ms")

        # 8) 绘图节流
        if (axs is not None) and (fig is not None) and (it % plot_every == 0) and (it > 0):
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
                comm_mat, swarm_size, swarm_paths, node_colors, line_colors
            )

        # 9) 收敛判据
        if not getattr(config, "FORCE_FIXED_ITERS", False):
            if len(Jn_raw) > convergence_window:
                recent = Jn_raw[-convergence_window:]
                slope, std_dev = _slope_and_std(recent)
                if abs(slope) < slope_threshold and std_dev < std_threshold:
                    print(f"[done] Jn 收敛：t={t_elapsed[-1]:.2f}s, it={it}, "
                          f"slope={slope:.2e} (<{slope_threshold:.2e}), "
                          f"std={std_dev:.2e} (<{std_threshold:.2e})")
                    # 不 break 之前先写 JSONL
                    pass

        # —— 记录 JSONL（本步）——
        T_total_ms = (time.time() - t_step0) * 1000.0
        T_overhead_ms = max(0.0, T_total_ms - (T_range_ms + t_gpu_h2d_ms + t_gpu_edge_ms + t_gpu_d2h_ms + T_update_ms))
        try:
            with open(V4_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "it": it,
                    "T_range": T_range_ms,
                    "T_h2d": t_gpu_h2d_ms,
                    "T_edge": t_gpu_edge_ms,
                    "T_d2h": t_gpu_d2h_ms,
                    "T_update": T_update_ms,
                    "T_overhead": T_overhead_ms,
                    "T_total": T_total_ms
                }) + "\n")
        except Exception:
            pass

        # 早停
        if not getattr(config, "FORCE_FIXED_ITERS", False):
            if len(Jn_raw) > convergence_window:
                recent = Jn_raw[-convergence_window:]
                slope, std_dev = _slope_and_std(recent)
                if abs(slope) < slope_threshold and std_dev < std_threshold:
                    break

    # 最后一帧
    if (axs is not None) and (fig is not None) and t_elapsed and ((len(Jn) - 1) % plot_every != 0):
        plotting.plot_figures_task1(
            axs, t_elapsed, Jn, rn, swarm_position, PT,
            comm_mat, swarm_size, swarm_paths, node_colors, line_colors
        )
    return Jn, rn, swarm_position, t_elapsed, comm_mat, None
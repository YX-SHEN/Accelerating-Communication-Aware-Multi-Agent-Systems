# src/gpu/algorithms_advanced_gpu.py
# -*- coding: utf-8 -*-
"""
GPU-accelerated [Advanced Optimized Version] (single-file version)
- Neighbor construction: KDTree (preferred) or cell-list (fallback) -> on CPU
- Edge list deduplicated for i<j; calculation of "physics+forces on edges" on GPU (JAX/CuPy), with per-node reduction via bincount
- Streaming statistics for Jn/rn; plotting/logging throttling; convergence criterion (slope+variance) consistent with CPU version
- Small-scale visualization can construct an NxN comm_mat; large-scale uses a zero-matrix proxy to save memory
"""

import time
import os
import numpy as np
import config
from src.common import plotting

# Import convergence criterion and neighbor search tools from src.optimized to maintain logical consistency
from src.optimized.algorithms_advanced import (
    _get_neighbor_lists, _edges_from_neighbors, _slope_and_std,
    _safe_log_pt, _ZeroMatrixProxy,
)
# Import backend and computation core from the GPU module
from src.gpu.backend import ImprovedGPUBackend, DTYPE_DEFAULT
from src.gpu.compute import compute_forces_gpu

# ------------------------------------------------------------------------------
# Optional: Whether to print a message at runtime indicating if KDTree/linregress or a fallback was used
VERBOSE_IMPORT = getattr(config, 'VERBOSE_IMPORT', False)

# Global backend instance
_GPU_INST = ImprovedGPUBackend()


# ------------------------------------------------------------------------------
# 5) Main function (interface identical to CPU version)
def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    Advanced optimized version + GPU (single-file):
      - Neighbor construction: KDTree (preferred) or cell-list (fallback) [CPU]
      - Edge-based calculation of physics and forces, GPU vectorized + bincount reduction
      - Streaming statistics for Jn/rn; plotting/logging throttling; convergence criterion: linear regression slope + standard deviation of recent window
      - Large-scale: Use a zero-memory "edge matrix" proxy to avoid N×N allocation
    """
    swarm_position = swarm_position.astype(np.float32, copy=False)
    AD_DEBUG = os.getenv("AD_DEBUG", "0") == "1"
    # ---------------- Parameter Checks and Derived Quantities ----------------
    assert 0.0 < PT < 1.0, "PT must be in (0, 1)"
    assert alpha > 0 and beta > 0 and v > 0 and r0 > 0, "alpha/beta/v/r0 must be positive"

    step_size = getattr(config, 'STEP_SIZE', 0.01)
    mode = getattr(config, 'MODE', 'hpc')  # "viz" or "hpc"
    plot_every = getattr(config, 'PLOT_EVERY', 20)
    log_every = getattr(config, 'LOG_EVERY', 20)

    # Fix: Prevent infinite loop if plot_every/log_every is 0
    plot_every = max(1, plot_every)
    log_every = max(1, log_every)

    convergence_window = getattr(config, 'CONVERGENCE_WINDOW', 50)
    slope_threshold = getattr(config, 'CONVERGENCE_SLOPE_THRESHOLD', 1e-6)
    std_threshold = getattr(config, 'CONVERGENCE_STD_THRESHOLD', 1e-5)

    # New: Numerical stability check parameter
    STABILITY_THRESHOLD = getattr(config, 'STABILITY_THRESHOLD', 1e6)

    # Communication radius (inversely solved from a_ij = PT)
    R = r0 * ((-_safe_log_pt(PT)) / beta) ** (1.0 / v)
    # Numerical stability: EPS is linked to the scale
    EPS = max(getattr(config, 'EPS', 1e-12), 1e-9 * R)

    if VERBOSE_IMPORT and (max_iter > 0):
        print(f"[alg-adv-gpu:backend] {_GPU_INST.backend}")

    # ---------------- Structures and Buffers ----------------
    # Visualization detail threshold (draw edges only for small scale & explicit viz mode)
    VIS_THRESHOLD = getattr(config, 'VISUALIZATION_THRESHOLD', 40)
    do_viz_details = (swarm_size <= VIS_THRESHOLD and mode == "viz")

    # Small-scale: use a real matrix; Large-scale: zero-matrix proxy (passed to plot)
    if do_viz_details:
        comm_mat = np.zeros((swarm_size, swarm_size), dtype=np.float32)
    else:
        comm_mat = _ZeroMatrixProxy()

    # Metrics and Time
    Jn_raw, rn_raw = [], []  # Raw sequences, for convergence criterion
    Jn, rn = [], []  # Display sequences (rounded)
    t_elapsed = []
    start_time = time.time()

    print(f"[alg-adv-gpu] Backend: {_GPU_INST.backend}")
    if _GPU_INST.device:
        print(f"[alg-adv-gpu] Device: {_GPU_INST.device}")
    print(f"[alg-adv-gpu] Swarm size: {swarm_size}, Max iterations: {max_iter}")
    print("Starting simulation loop...")

    # ---------------- Main Loop ----------------
    for it in range(max_iter):
        # 1) Neighbor construction (CPU)
        t_neigh_start = time.time()  # Diagnostic timing point
        neighbor_lists = _get_neighbor_lists(swarm_position, R)
        i_idx, j_idx = _edges_from_neighbors(neighbor_lists)
        if AD_DEBUG and (it % log_every == 0):
            # i_idx is the list of start indices for directed edges, its length is the number of directed edges
            e_dir = int(i_idx.size)
            avg_out_deg = (e_dir / float(swarm_size)) if swarm_size > 0 else 0.0
            print(f"[dbg] edges_directed={e_dir} avg_out_degree={avg_out_deg:.2f} R={R:.3f} PT={PT:.3g}")
        t_neigh_end = time.time()  # Diagnostic timing point

        # 2) If no edges, remain static & provide a warning
        if i_idx.size == 0:
            forces_cpu = np.zeros_like(swarm_position)
            Jn_new = 0.0;
            rn_new = 0.0

            # --- Fix: Enqueue metrics in preparation for the next loop ---
            Jn_raw.append(Jn_new)
            rn_raw.append(rn_new)
            Jn.append(round(Jn_new, 4))
            rn.append(round(rn_new, 4))
            t_elapsed.append(time.time() - start_time)

            # --- Fix: Print log to show current status ---
            if it % log_every == 0:
                print(
                    f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f} "
                    f"compute=0.00ms neigh={(t_neigh_end - t_neigh_start) * 1000:.2f}ms"
                )
                if it > 10:
                    print("  [warn] No adjacent edges in this step (cnt=0). Consider lowering PT or increasing density.")

            # Plot throttling
            enable_plot = (axs is not None) and (fig is not None) and (plot_every > 0)
            if enable_plot and (it % plot_every == 0) and (it > 0):
                plotting.plot_figures_task1(
                    axs, t_elapsed, Jn, rn, swarm_position, PT,
                    _ZeroMatrixProxy(), swarm_size, swarm_paths,
                    node_colors, line_colors
                )
            continue  # Skip the rest of this loop and proceed to the next iteration

        # 3) GPU calculation on edges (Note: pass i_idx/j_idx, enable ecap for fixed shapes)
        t_gpu_start = time.time()
        forces_dev, Jn_new, rn_new, phi_edges_dev, valid_mask_dev = compute_forces_gpu(
            gpu=_GPU_INST,
            positions=swarm_position,
            N=swarm_size,
            alpha=alpha, beta=beta, v=v, r0=r0, PT=PT, delta=config.DELTA,
            keep_full_matrix=False,
            dtype=swarm_position.dtype,
            eps=EPS,
            i_idx=i_idx,  # <<< New
            j_idx=j_idx,  # <<< New
            ecap_factor=getattr(config, "GPU_ECAP_FACTOR", 1.25),  # <<< Recommend making this adjustable in config
        )
        t_gpu_end = time.time()

        # Synchronize GPU calculation results to CPU
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
        # else: NumPy fallback requires no conversion

        # 4) New: Numerical stability check
        max_force_norm = np.linalg.norm(forces_cpu, axis=1).max()
        if max_force_norm > STABILITY_THRESHOLD:
            print(f"[error] Numerical instability detected at it={it}, max control input norm: {max_force_norm:.2e}. Terminating early.")
            # --- Ensure current state is recorded before abnormal termination ---
            Jn_raw.append(Jn_new)
            rn_raw.append(rn_new)
            Jn.append(round(Jn_new, 4))
            rn.append(round(rn_new, 4))
            t_elapsed.append(time.time() - start_time)
            break

        # Update all positions and zero out control inputs
        swarm_position += step_size * forces_cpu
        pass

        # 5) Enqueue metrics
        Jn_raw.append(Jn_new);
        rn_raw.append(rn_new)
        Jn.append(round(Jn_new, 4));
        rn.append(round(rn_new, 4))
        t_elapsed.append(time.time() - start_time)

        # 6) Log throttling (bug fix: logging is done after calculation)
        if it % log_every == 0:
            print(f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f} (Neighbor search: {(t_neigh_end - t_neigh_start) * 1000:.2f}ms, "
                  f"GPU computation: {(t_gpu_end - t_gpu_start) * 1000:.2f}ms)")
            print(f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f} compute={(t_gpu_end - t_gpu_start) * 1000:.2f}ms "
                  f"neigh={(t_neigh_end - t_neigh_start) * 1000:.2f}ms")

        # 7) Plot throttling
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


        # 8) Convergence criterion (no early exit in fixed-iteration mode)
        if not getattr(config, "FORCE_FIXED_ITERS", False):
            if len(Jn_raw) > convergence_window:
                recent = Jn_raw[-convergence_window:]
                slope, std_dev = _slope_and_std(recent)
                if abs(slope) < slope_threshold and std_dev < std_threshold:
                    print(f"[done] Jn converged: t={t_elapsed[-1]:.2f}s, it={it}, "
                          f"slope={slope:.2e} (<{slope_threshold:.2e}), "
                          f"std={std_dev:.2e} (<{std_threshold:.2e})")
                    break

    # Refresh the last frame (prevents `it` from being undefined when max_iter=0)
    enable_plot = (axs is not None) and (fig is not None) and (plot_every > 0)
    if enable_plot and t_elapsed and ((len(Jn) - 1) % plot_every != 0):
        plotting.plot_figures_task1(
            axs, t_elapsed, Jn, rn, swarm_position, PT,
            comm_mat, swarm_size, swarm_paths,
            node_colors, line_colors
        )

    # To be compatible with the old interface, return comm_mat and a placeholder None (neighbor matrix is no longer maintained)
    return Jn, rn, swarm_position, t_elapsed, comm_mat, None
# src/gpu/algorithms_gpu_unified.py
# -*- coding: utf-8 -*-
"""
Cross-platform GPU-accelerated version (unified implementation + a Gauss-Seidel option fully equivalent to the CPU baseline)

- UPDATE_MODE="gauss": Per-agent in-place updates (consistent order with CPU baseline)
- UPDATE_MODE="jacobi": One-shot parallel update (unified GPU kernel)
- All config reads are done dynamically inside run_simulation to avoid caching at import time
- Log format is kept as compute=XXms for easy parsing by external scripts
"""

import time
import numpy as np
import config
from src.common import plotting

from .backend import ImprovedGPUBackend, DTYPE_DEFAULT
from .compute import compute_forces_gpu

# Global backend instance (initialized only once)
gpu = ImprovedGPUBackend()


def _build_comm_matrix_for_plot(positions, PT, alpha, beta, v, r0, delta, keep_full_matrix, dtype):
    """
    Constructs the NxN communication quality matrix when needed for plotting (only called if keep_full_matrix=True)
    Supports NumPy / CuPy / JAX (JAX uses NumPy fallback), returns a np.ndarray on the host
    """
    if not keep_full_matrix:
        return None

    # --- Select backend array library and to_host function ---
    backend = gpu.backend
    if backend == "cupy":
        xp = gpu.cp
        pos = positions  # Already a cupy array
        to_host = gpu.cp.asnumpy
    elif backend == "jax":
        # For simplicity: JAX moves data back to host for NumPy calculation (plotting is infrequent, so the cost is acceptable)
        xp = np
        pos = np.asarray(gpu.to_host(positions), dtype=dtype)
        to_host = lambda x: x
    else:
        xp = np
        pos = positions
        to_host = lambda x: x

    N = pos.shape[0]
    # O(N^2) construction (for visualization only)
    diff = pos[:, None, :] - pos[None, :, :]          # [N,N,2]
    d2   = xp.sum(diff * diff, axis=2)                # [N,N]
    eps  = 1e-12 if dtype == np.float64 else 1e-8     # Only to avoid sqrt(0), does not affect the threshold
    rij  = xp.sqrt(xp.maximum(d2, eps))

    eye  = xp.eye(N, dtype=bool)
    valid= xp.logical_and(rij > eps, ~eye)

    aij  = xp.exp(-alpha * (2**delta - 1) * (rij / r0) ** v)
    conn = xp.logical_and(valid, aij >= PT)

    gij  = rij / xp.sqrt(rij * rij + r0 ** 2)
    phi  = gij * aij

    comm_mat = (phi * conn).astype(dtype, copy=False)
    return to_host(comm_mat)


def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    GPU-accelerated main simulation loop.
    - If config.UPDATE_MODE == "gauss", performs per-agent in-place updates, with order/metrics identical to the CPU baseline;
    - Otherwise, uses the unified GPU kernel (one-shot parallel update).
    Returns:
      Jn_list (rounded display values), rn_list (rounded display values), final_positions (np),
      t_elapsed (s), final_comm_matrix (np or zero matrix), None
    """
    # -------- Dynamically read config (critical fix) --------
    step_size = getattr(config, "STEP_SIZE", 0.01)
    delta = getattr(config, "DELTA", 2)
    log_every = max(1, getattr(config, "LOG_EVERY", 20))
    plot_every = max(1, getattr(config, "PLOT_EVERY", 20))
    keep_full_matrix = bool(getattr(config, "KEEP_FULL_MATRIX", True))
    dtype = getattr(config, "DTYPE", DTYPE_DEFAULT)
    eps_eij = getattr(config, "EPS", 1e-6)  # EPS for eij denominator, default 1e-6 is closer to CPU baseline
    convergence_window = int(getattr(config, "CONVERGENCE_WINDOW", 20))
    update_mode = getattr(config, "UPDATE_MODE", "jacobi").lower()  # "gauss" or "jacobi"

    print(f"[GPU-Baseline] Backend: {gpu.backend}")
    if gpu.device:
        print(f"[GPU-Baseline] Device: {gpu.device}")
    print(f"[GPU-Baseline] Swarm size: {swarm_size}, Max iterations: {max_iter}, Mode: {update_mode}")

    # ---- Device-side initialization ----
    # Note: In Gauss mode, we frequently read/write the position of a single i; direct in-place writing works best with CuPy
    d_positions = gpu.to_device(swarm_position.astype(dtype, copy=False))

    # ---- Statistics and timing containers ----
    Jn_history_raw = []   # Raw float values, for statistics/stopping criteria
    rn_history_raw = []
    Jn_display = []       # Rounded display values (consistent with CPU baseline output)
    rn_display = []
    t_elapsed = []
    computation_times = []  # Per-step compute time (seconds)
    start_time = time.time()

    # If not keeping the full matrix, this placeholder is used for return/plotting
    comm_mat_host_placeholder = np.zeros((swarm_size, swarm_size), dtype=dtype)
    last_comm_host = comm_mat_host_placeholder

    print("Starting simulation loop...")
    for it in range(max_iter):
        t0 = time.time()

        if (update_mode == "gauss"):
            # ========= "In-place update" (Gauss-Seidel), identical to CPU baseline =========
            # Backend selection: CuPy -> vectorized on device; JAX -> fallback to NumPy (infrequent, doesn't affect correctness)
            backend = gpu.backend
            if backend == "cupy":
                xp = gpu.cp
                pos = d_positions  # cupy.ndarray [N,2]
            elif backend == "jax":
                xp = np
                pos = np.asarray(gpu.to_host(d_positions), dtype=dtype)  # Back to host
            else:
                xp = np
                pos = d_positions  # np.ndarray

            N = swarm_size
            idx_all = xp.arange(N)

            # Streaming accumulation (equivalent to the "bidirectional counting" of the CPU baseline)
            tot_phi = 0.0
            tot_r = 0.0
            tot_cnt = 0

            for i in range(N):
                qi   = pos[i]                              # [2]
                diff = qi - pos                            # [N,2]
                d2   = xp.sum(diff * diff, axis=1)         # [N]
                rij  = xp.sqrt(xp.maximum(d2, eps_eij))    # [N]

                not_self = (idx_all != i)
                aij = xp.exp(-alpha * (2**delta - 1) * (rij / r0) ** v)
                nbr = xp.logical_and(not_self, aij >= PT)  # Neighbors that meet the threshold

                gij = rij / xp.sqrt(rij * rij + r0 ** 2)
                # ρ_ij (consistent with CPU)
                num = (-beta * v * rij**(v + 2) - beta * v * (r0**2) * (rij**v) + r0**(v + 2))
                den = xp.sqrt((rij * rij + r0**2) ** 3) + eps_eij
                rho = num * xp.exp(-beta * (rij / r0) ** v) / den

                eij = diff / (rij + eps_eij)[:, None]      # Add eps to denominator, consistent with CPU
                f_i = xp.sum((rho[:, None] * eij) * nbr[:, None], axis=0)  # Only accumulate from neighbors that meet the threshold

                # -- In-place update: identical to CPU baseline (calculate i, then update i) --
                pos[i] = pos[i] + step_size * f_i

                # Synchronously accumulate statistics (consistent with CPU's "count twice per (i,j) pair")
                phi = gij * aij
                if backend == "cupy":
                    tot_phi += float(xp.sum(phi[nbr]))
                    tot_r   += float(xp.sum(rij[nbr]))
                    tot_cnt += int(xp.sum(nbr).get())
                else:
                    tot_phi += float(np.sum(phi[nbr]))
                    tot_r   += float(np.sum(rij[nbr]))
                    tot_cnt += int(np.sum(nbr))

            # Write positions back to device (for JAX/NumPy fallback)
            if backend != "cupy":
                d_positions = gpu.to_device(pos.astype(dtype, copy=False))

            # Metrics for this step (consistent with CPU)
            cnt = max(tot_cnt, 1)
            Jn_new = tot_phi / cnt
            rn_new = tot_r / cnt

            # Construct plotting matrix (only when needed; avoids O(N^2) per step)
            comm_mat_dev = None
            if (it % plot_every == 0) and keep_full_matrix:
                # Construct only once per plot step, O(N^2)
                comm_host = _build_comm_matrix_for_plot(
                    d_positions, PT, alpha, beta, v, r0, delta, keep_full_matrix, dtype
                )
                last_comm_host = comm_host

        else:
            # ========= One-shot parallel (Jacobi) path: Unified GPU kernel =========
            forces, Jn_new, rn_new, comm_mat_dev, _ = compute_forces_gpu(
                gpu, d_positions, swarm_size, alpha, beta, v, r0, PT, delta,
                keep_full_matrix=keep_full_matrix, dtype=dtype, eps=eps_eij
            )
            # Synchronization/timing
            if gpu.backend == "jax":
                _ = forces.block_until_ready()
            else:
                gpu.synchronize()
            # Position update (Jacobi: use forces from the same step to update all points)
            d_positions = d_positions + step_size * forces

            # If plotting is needed this step, retrieve the communication matrix
            if (it % plot_every == 0) and keep_full_matrix and (comm_mat_dev is not None):
                last_comm_host = gpu.to_host(comm_mat_dev)

        # === Timing and Recording ===
        # ... after gauss or jacobi branch calculation is complete ...
        if gpu.backend == "cupy":
            gpu.synchronize()
        # Now, record the time
        step_time = time.time() - t0
        computation_times.append(step_time)

        Jn_history_raw.append(float(Jn_new))
        rn_history_raw.append(float(rn_new))
        Jn_display.append(round(float(Jn_new), 4))
        rn_display.append(round(float(rn_new), 4))
        t_elapsed.append(time.time() - start_time)

        # === Logging ===
        if it % log_every == 0:
            k = min(log_every, len(computation_times))
            avg_ms = np.mean(computation_times[-k:]) * 1000.0
            print(f"[it={it:4d}] Jn={Jn_display[-1]:.4f} rn={rn_display[-1]:.4f} compute={avg_ms:.2f}ms")

        # === Plotting (throttled) ===
        if it % plot_every == 0:
            pos_cpu = gpu.to_host(d_positions)
            comm_cpu = last_comm_host if keep_full_matrix else comm_mat_host_placeholder
            plotting.plot_figures_task1(
                axs, t_elapsed, Jn_display, rn_display, pos_cpu, PT,
                comm_cpu, swarm_size, swarm_paths, node_colors, line_colors
            )

        # === "Plateau" stopping criterion consistent with baseline (last W display values are identical) ===
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

    # ---- Summarize and Return ----
    total_time = time.time() - start_time
    avg_ms_all = (np.mean(computation_times) * 1000.0) if computation_times else 0.0
    final_positions = gpu.to_host(d_positions)

    if keep_full_matrix:
        final_comm = last_comm_host
    else:
        final_comm = comm_mat_host_placeholder

    print("\n=== Simulation Complete ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average compute time: {avg_ms_all:.2f}ms/iter")
    print(f"Total iterations: {len(Jn_display)}")
    if Jn_display:
        print(f"Final Jn: {Jn_display[-1]:.4f}")
    if rn_display:
        print(f"Final rn: {rn_display[-1]:.4f}")

    return (Jn_display, rn_display, final_positions, t_elapsed, final_comm, None)
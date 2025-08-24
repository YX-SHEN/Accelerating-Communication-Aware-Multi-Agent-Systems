# src/gpu/algorithms_advanced_gpu2.py
# -*- coding: utf-8 -*-
"""
GPU-Advanced v2: Jacobi + spatial grid (3x3 neighborhood) implementation with nearest-neighbor pruning acceleration
- Consistent with your GPU Jacobi formula/threshold/metrics, but prunes the global O(N^2) neighborhood to a 3x3 cell neighborhood.
- Step-by-step results will not be identical to the CPU Baseline's Gauss-Seidel per-point in-place update, but the final Jn/rn behavior is aligned.
- Prints compute=XXms, for scripts/run_experiments.py to parse kernel_ms_per_iter.
"""

import time
import numpy as np
import config
import os, math
import cupy as cp

from src.gpu.backend import ImprovedGPUBackend, DTYPE_DEFAULT
from src.gpu.compute_grid import build_uniform_grid, comm_radius_from_config
from src.common import plotting
import config

gpu = ImprovedGPUBackend()
# Do not calculate R_CUT/CELL_H globally to avoid undefined defaults; dynamically derive from PT in run_simulation

# ---- CUDA Kernel ----
_RAW = r'''
extern "C" __global__
void forces_grid(const float* __restrict__ pos,      // [N,2], already sorted by cell
                 const int* __restrict__ order,    // sorted_idx -> original_i
                 const int* __restrict__ cell_start,
                 const int* __restrict__ cell_end,
                 const float  ox, const float oy,
                 const int    nx, const int ny,
                 const float  cell_size,
                 const int    N,
                 const int    K,                     // <<< New: Neighborhood radius (1=3x3, 2=5x5)
                 const float  alpha, const float beta,
                 const float  v, const float r0, const float PT,
                 const float  eps,
                 float* __restrict__ outF,   // [N,2], original order
                 float* __restrict__ outJn,  // Single element, accumulate within block + atomicAdd
                 float* __restrict__ outRn,
                 int* __restrict__ outCnt)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) return;

    float xi = pos[2*tid+0];
    float yi = pos[2*tid+1];

    int gx = (int)floorf((xi - ox)/cell_size);
    int gy = (int)floorf((yi - oy)/cell_size);
    if (gx < 0) gx = 0; if (gx >= nx) gx = nx-1;
    if (gy < 0) gy = 0; if (gy >= ny) gy = ny-1;

    const float two_pow_delta_minus_1 = (powf(2.0f, %DELTA%) - 1.0f);

    float Fx = 0.0f, Fy = 0.0f;
    float phi_sum = 0.0f, r_sum = 0.0f;
    int   cnt = 0;

    // Scan (2K+1)x(2K+1) neighborhood <<< Use K here
    for (int dy=-K; dy<=K; ++dy){
        int cy = gy + dy; if (cy<0 || cy>=ny) continue;
        for (int dx=-K; dx<=K; ++dx){
            int cx = gx + dx; if (cx<0 || cx>=nx) continue;
            int cid = cx + cy*nx;
            int s = cell_start[cid];
            int e = cell_end[cid];
            if (s < 0) continue;

            for (int k=s; k<e; ++k){
                if (k == tid) continue;
                float xj = pos[2*k+0];
                float yj = pos[2*k+1];
                float dx = xi - xj;
                float dy = yi - yj;
                float d2 = dx*dx + dy*dy;
                float rij = sqrtf(fmaxf(d2, eps));

                float aij = expf(-alpha * two_pow_delta_minus_1 * powf(rij/r0, v));
                if (aij < PT) continue;

                float gij = rij / sqrtf(rij*rij + r0*r0);
                float phi = gij * aij;

                float num = (-beta * v * powf(rij, v+2) - beta * v * (r0*r0) * powf(rij, v) + powf(r0, v+2));
                float den = sqrtf(powf(rij*rij + r0*r0, 3.0f)) + eps;
                float rho = num * expf(-beta * powf(rij/r0, v)) / den;

                float ex = dx / (rij + eps);
                float ey = dy / (rij + eps);
                Fx += rho * ex;
                Fy += rho * ey;

                phi_sum += phi;
                r_sum   += rij;
                cnt     += 1;
            }
        }
    }

    int i = order[tid];
    outF[2*i+0] = Fx;
    outF[2*i+1] = Fy;

    if (cnt>0){
        atomicAdd(outJn, phi_sum);
        atomicAdd(outRn, r_sum);
        atomicAdd(outCnt, cnt);
    }
}
''';

def _kernel(delta: int) -> cp.RawKernel:
    code = _RAW.replace("%DELTA%", str(int(delta)))
    return cp.RawKernel(code, "forces_grid")


def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    Returns:
      Jn_list (rounded), rn_list (rounded), final_positions (np),
      t_elapsed (list of s), final_comm_matrix (None), None
    """
    # ---- Read/Normalize Configuration ----
    dtype = getattr(config, "DTYPE", DTYPE_DEFAULT)
    # Kernel is the float version, forcing float32 is safer
    d_pos = gpu.to_device(swarm_position.astype(cp.float32, copy=False))

    step_size  = float(getattr(config, "STEP_SIZE", 0.01))
    eps        = float(getattr(config, "EPS", 1e-6))
    delta      = int(getattr(config, "DELTA", 2))
    log_every = max(1, int(getattr(config, "LOG_EVERY", 50)))
    plot_every = max(1, int(getattr(config, "PLOT_EVERY", 10 ** 9)))  # Defaults to basically no plotting
    keep_full_matrix = False  # v2 does not construct NxN, can be changed if needed

    print("[GPU-Advanced v2] Backend:", gpu.backend)
    if gpu.device:
        print("[GPU-Advanced v2] Device:", gpu.device)
    print("[GPU-Advanced v2] Mode: jacobi")
    print(f"[GPU-Advanced v2] N={swarm_size}, max_iter={max_iter}")

    # --- Communication Radius & Cell Size (can add a multiplier if needed) ---
    R = float(comm_radius_from_config(config))
    if not np.isfinite(R) or R <= 0:
        # Fallback: estimate a scale using data span / sqrt(N)
        pos_np = gpu.to_host(d_pos)
        span = float(max(pos_np[:,0].max()-pos_np[:,0].min(),
                         pos_np[:,1].max()-pos_np[:,1].min()))
        R = max(1e-3, span / max(np.sqrt(float(swarm_size)), 1.0))


    # Adjustable: AD2_CELL_SCALE (default 1.0) -> cell_size = R / scale, neighborhood K = ceil(scale)
    scale = float(os.getenv("AD2_CELL_SCALE", "1.0"))
    if not np.isfinite(scale) or scale <= 0: scale = 1.0
    cell_size = float(R / scale)
    K = int(max(1, math.ceil(scale)))

    ker = _kernel(delta)

    # ---- Statistics Containers ----
    Jn_hist, rn_hist, t_hist = [], [], []
    comm_mat_placeholder = None  # v2 does not return NxN
    start_time = time.time()
    step_times_ms = []

    for it in range(max_iter):
        t_step0 = time.time()

        # 1) Rebuild grid at each step (simple and robust; can rebuild every few steps if faster is needed)
        grid = build_uniform_grid(d_pos, cell_size)
        order = grid["order"].astype(cp.int32)          # sorted_idx -> original_i (int32!)
        pos_sorted = d_pos[order]                        # [N,2] after sorting

        # [AD2_DEBUG] Print grid info: cell occupancy statistics
        if os.getenv("AD2_DEBUG", "0") == "1":
            occ = grid["cell_end"] - grid["cell_start"]  # Number of particles in each cell
            occ = occ[occ >= 0]  # Filter out cells with no particles
            if occ.size > 0:
                # To be compatible with different CuPy versions, move to CPU first then calculate percentiles
                occ_cpu = occ.get()
                occ_mean = float(occ_cpu.mean())
                occ_p95 = float(np.percentile(occ_cpu, 95))
                occ_max = int(occ_cpu.max())
            else:
                occ_mean = occ_p95 = occ_max = 0.0
            print(f"[grid] nx={grid['nx']} ny={grid['ny']} K={K} R={R:.3f} "
                  f"cell_size={cell_size:.3f} occ_mean={occ_mean:.2f} "
                  f"occ_p95={occ_p95:.2f} occ_max={occ_max}")

        # 2) Output buffers (fp32 / int32)
        dF   = cp.zeros_like(d_pos)                      # [N,2] fp32
        dJn  = cp.zeros((1,), dtype=cp.float32)
        dRn  = cp.zeros((1,), dtype=cp.float32)
        dCnt = cp.zeros((1,), dtype=cp.int32)

        # 3) Launch kernel
        bs = 256;
        gs = (swarm_size + bs - 1) // bs
        ker((gs,), (bs,),
            (pos_sorted.ravel(),
             order,
             grid["cell_start"], grid["cell_end"],
             cp.float32(grid["origin_x"]), cp.float32(grid["origin_y"]),
             np.int32(grid["nx"]), np.int32(grid["ny"]),
             cp.float32(grid["cell_size"]),
             np.int32(swarm_size),
             np.int32(K),  # <<< New: Neighborhood radius
             np.float32(alpha), np.float32(beta),
             np.float32(v), np.float32(r0), np.float32(PT),
             np.float32(eps),
             dF.ravel(), dJn, dRn, dCnt))

        # 4) Synchronize + Time
        gpu.synchronize()
        step_ms = (time.time() - t_step0) * 1000.0
        step_times_ms.append(step_ms)

        # 5) Jacobi update
        d_pos = d_pos + cp.float32(step_size) * dF

        # 6) Statistics
        cnt = int(dCnt.get())
        # [AD2_DEBUG] Print average out-degree (number of directed edges / N)
        if os.getenv("AD2_DEBUG", "0") == "1":
            avg_deg = (cnt / float(swarm_size)) if swarm_size > 0 else 0.0
            print(f"[edges] cnt={cnt} avg_out_degree={avg_deg:.2f}")
        Jn  = float(dJn.get() / max(cnt, 1))
        rn  = float(dRn.get() / max(cnt, 1))
        Jn_hist.append(round(Jn, 4))
        rn_hist.append(round(rn, 4))
        t_hist.append(time.time() - start_time)

        # 7) Logging
        if it % log_every == 0:
            print(f"[it={it:4d}] Jn={Jn_hist[-1]:.4f} rn={rn_hist[-1]:.4f} compute={step_ms:.2f}ms")

        # 8) Plotting (disabled by default; when needed, only positions are passed, comm_matrix is None)
        if (axs is not None) and (fig is not None) and (plot_every > 0) \
                and (it % plot_every == 0) and (it > 0):
            pos_cpu = gpu.to_host(d_pos)
            plotting.plot_figures_task1(
                axs, t_hist, Jn_hist, rn_hist, pos_cpu, PT,
                None,  # v2 does not construct the NxN matrix
                swarm_size, swarm_paths, node_colors, line_colors
            )

    total_time = time.time() - start_time
    avg_ms = float(np.mean(step_times_ms)) if step_times_ms else float("nan")

    print("\n=== v2 Simulation Complete ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average compute time: {avg_ms:.2f}ms/iter")
    print(f"compute={avg_ms:.2f}ms")  # For stable regex capture by run_experiments
    print(f"Total iterations: {len(Jn_hist)}")
    if Jn_hist:
        print(f"Final Jn: {Jn_hist[-1]:.4f}")
    if rn_hist:
        print(f"Final rn: {rn_hist[-1]:.4f}")

    final_positions = gpu.to_host(d_pos)
    return (Jn_hist, rn_hist, final_positions, t_hist, None, None)
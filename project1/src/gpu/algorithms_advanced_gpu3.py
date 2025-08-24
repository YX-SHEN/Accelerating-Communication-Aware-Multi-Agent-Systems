# src/gpu/algorithms_advanced_gpu3.py
# -*- coding: utf-8 -*-
"""
GPU-Advanced v2: Jacobi + spatial grid (3x3 equivalent neighborhood) for nearest-neighbor pruning
Adds "mathematical accelerations":
- Preconditioning: AD2_PRECOND=phi|degree (diagonal/degree scaling), with params AD2_PRECOND_GAMMA, AD2_PRECOND_EPS
- Semi-iteration: AD2_OMEGA (weighted Jacobi), AD2_MOMENTUM (heavy-ball momentum 0~0.95)
- Chebyshev semi-iteration: AD2_CHEB="lam_min,lam_max", AD2_CHEB_M (step size sequence period)
- Multi-coloring: AD2_COLORING=none|rb|c4 (red-black/four-color by cell, batch updates)
Other aspects remain consistent with the original implementation: prints compute=XXms, scripts can parse kernel ms/iter.
"""

import os, math, time
import numpy as np
import cupy as cp
import config

from src.gpu.backend import ImprovedGPUBackend, DTYPE_DEFAULT
from src.gpu.compute_grid import build_uniform_grid, comm_radius_from_config
from src.common import plotting

gpu = ImprovedGPUBackend()

# ------------------------ CUDA Kernel ------------------------
_RAW = r'''
#include <math.h>

extern "C" __global__
void forces_grid(const float* __restrict__ pos,      // [N,2], already sorted by cell
                 const int* __restrict__ order,    // sorted_idx -> original_i
                 const int* __restrict__ cell_start,
                 const int* __restrict__ cell_end,
                 const float  ox, const float oy,
                 const int    nx, const int ny,
                 const float  cell_size,
                 const int    N,
                 const int    K,                     // Neighborhood radius (1=3x3, 2=5x5, ...)
                 const int    color_mode,            // 0=none, 2=RB, 4=4-color
                 const int    active_color,          // Current color index (0..color_mode-1)
                 const float  alpha, const float beta,
                 const float  v, const float r0, const float PT,
                 const float  eps,
                 float* __restrict__ outF,        // [N,2], original order
                 float* __restrict__ outJn,       // Scalar accumulation
                 float* __restrict__ outRn,       // Scalar accumulation
                 int* __restrict__ outCnt,      // Scalar accumulation
                 float* __restrict__ outDiagNode, // [N] Per-node "diagonal approximation" (sum(phi_ij))
                 int* __restrict__ outDegNode)  // [N] Per-node degree
{
    int sid = blockDim.x * blockIdx.x + threadIdx.x; // sorted idx
    if (sid >= N) return;

    float xi = pos[2*sid+0];
    float yi = pos[2*sid+1];

    int gx = (int)floorf((xi - ox)/cell_size);
    int gy = (int)floorf((yi - oy)/cell_size);
    if (gx < 0) gx = 0; if (gx >= nx) gx = nx-1;
    if (gy < 0) gy = 0; if (gy >= ny) gy = ny-1;

    // Simple and stable cell coloring: via (gx+gy)%color_mode
    if (color_mode == 2 || color_mode == 4){
        int c = (gx + gy) % color_mode;
        if (c != active_color){
            int i_quiet = order[sid];
            outF[2*i_quiet+0] = 0.0f;
            outF[2*i_quiet+1] = 0.0f;
            outDiagNode[i_quiet] = 0.0f;
            outDegNode[i_quiet]  = 0;
            return;
        }
    }

    const float two_pow_delta_minus_1 = (powf(2.0f, %DELTA%) - 1.0f);

    float Fx = 0.0f, Fy = 0.0f;
    float phi_sum_local = 0.0f;
    float r_sum_local   = 0.0f;
    int   cnt_local     = 0;

    for (int dy=-K; dy<=K; ++dy){
        int cy = gy + dy; if (cy<0 || cy>=ny) continue;
        for (int dx=-K; dx<=K; ++dx){
            int cx = gx + dx; if (cx<0 || cx>=nx) continue;
            int cid = cx + cy*nx;
            int s = cell_start[cid];
            int e = cell_end[cid];
            if (s < 0) continue;

            for (int k=s; k<e; ++k){
                if (k == sid) continue;
                float xj = pos[2*k+0];
                float yj = pos[2*k+1];
                float dx_ = xi - xj;
                float dy_ = yi - yj;
                float d2 = dx_*dx_ + dy_*dy_;
                float rij = sqrtf(fmaxf(d2, eps));

                float aij = expf(-alpha * two_pow_delta_minus_1 * powf(rij/r0, v));
                if (aij < PT) continue;

                float gij = rij / sqrtf(rij*rij + r0*r0);
                float phi = gij * aij;

                float num = (-beta * v * powf(rij, v+2) - beta * v * (r0*r0) * powf(rij, v) + powf(r0, v+2));
                float den = sqrtf(powf(rij*rij + r0*r0, 3.0f)) + eps;
                float rho = num * expf(-beta * powf(rij/r0, v)) / den;

                float inv_r = 1.0f / (rij + eps);
                Fx += rho * dx_ * inv_r;
                Fy += rho * dy_ * inv_r;

                phi_sum_local += phi;   // Diagonal approximation
                r_sum_local   += rij;
                cnt_local     += 1;
            }
        }
    }

    int i = order[sid];
    outF[2*i+0] = Fx;
    outF[2*i+1] = Fy;
    outDiagNode[i] = phi_sum_local;
    outDegNode[i]  = cnt_local;

    if (cnt_local>0){
        atomicAdd(outJn,  phi_sum_local);
        atomicAdd(outRn,  r_sum_local);
        atomicAdd(outCnt, cnt_local);
    }
}
'''.strip()


def _kernel(delta: int) -> cp.RawKernel:
    code = _RAW.replace("%DELTA%", str(int(delta)))
    return cp.RawKernel(code, "forces_grid")


# --------------------- Main Entry: run_simulation ---------------------
def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    Returns:
      Jn_list, rn_list, final_positions(np),
      t_elapsed(list s), final_comm_matrix(None), None
    """
    # ---- Basic Configuration ----
    dtype = getattr(config, "DTYPE", DTYPE_DEFAULT)
    d_pos = gpu.to_device(swarm_position.astype(cp.float32, copy=False))

    step_size  = float(getattr(config, "STEP_SIZE", 0.01))
    eps        = float(getattr(config, "EPS", 1e-6))
    delta      = int(getattr(config, "DELTA", 2))
    log_every  = max(1, int(getattr(config, "LOG_EVERY", 50)))
    plot_every = max(1, int(getattr(config, "PLOT_EVERY", 10 ** 9)))
    keep_full_matrix = False  # v2 does not return NxN

    print("[GPU-Advanced v2] Backend:", gpu.backend)
    if gpu.device: print("[GPU-Advanced v2] Device:", gpu.device)
    print("[GPU-Advanced v2] Mode: jacobi")
    print(f"[GPU-Advanced v2] N={swarm_size}, max_iter={max_iter}")

    # ---- Communication Radius and Grid ----
    R = float(comm_radius_from_config(config))
    if not np.isfinite(R) or R <= 0:
        pos_np = gpu.to_host(d_pos)
        span = float(max(pos_np[:,0].max()-pos_np[:,0].min(),
                         pos_np[:,1].max()-pos_np[:,1].min()))
        R = max(1e-3, span / max(np.sqrt(float(swarm_size)), 1.0))

    scale = float(os.getenv("AD2_CELL_SCALE", "1.0"))
    if not np.isfinite(scale) or scale <= 0: scale = 1.0
    cell_size = float(R / scale)
    K = int(max(1, math.ceil(scale)))

    # ---- Mathematical Acceleration: Environment Variables ----
    coloring = os.getenv("AD2_COLORING", "none").lower()   # none|rb|c4
    ncolors  = 0 if coloring=="none" else (2 if coloring=="rb" else (4 if coloring=="c4" else 0))

    precond  = os.getenv("AD2_PRECOND", "none").lower()    # none|phi|degree
    pc_gamma = float(os.getenv("AD2_PRECOND_GAMMA", "1.0"))
    pc_eps   = float(os.getenv("AD2_PRECOND_EPS", "1e-6"))

    omega    = float(os.getenv("AD2_OMEGA", "1.0"))        # Weighted Jacobi coefficient
    mom      = float(os.getenv("AD2_MOMENTUM", "0.0"))     # heavy-ball momentum (recommend 0.2~0.8)

    cheb_raw = os.getenv("AD2_CHEB", "")                   # "lam_min,lam_max"
    cheb_m   = int(os.getenv("AD2_CHEB_M", "8"))
    use_cheb = False
    if cheb_raw:
        try:
            lam_min, lam_max = [float(x) for x in cheb_raw.split(",")]
            use_cheb = (lam_max > lam_min) and (lam_min > 0.0)
        except Exception:
            use_cheb = False
    cheb_k = 0

    if os.getenv("AD2_DEBUG", "0") == "1":
        print(f"[DEBUG] cell_size={cell_size:.3f} K={K} R={R:.3f}")
        print(f"[DEBUG] coloring={coloring} precond={precond} omega={omega} mom={mom} cheb={cheb_raw} m={cheb_m}")

    ker = _kernel(delta)

    # ---- Statistics Containers ----
    Jn_hist, rn_hist, t_hist = [], [], []
    start_time = time.time()
    step_times_ms = []

    # Momentum vector (N×2)
    vel = cp.zeros_like(d_pos)

    for it in range(max_iter):
        t0 = time.time()

        # 1) Rebuild uniform grid at each step (robust)
        grid = build_uniform_grid(d_pos, cell_size)
        order = grid["order"].astype(cp.int32)
        pos_sorted = d_pos[order]

        # Optional debug: cell occupancy statistics
        if os.getenv("AD2_DEBUG", "0") == "1":
            occ = grid["cell_end"] - grid["cell_start"]
            occ = occ[occ >= 0]
            if occ.size > 0:
                occ_cpu = occ.get()
                print(f"[grid] nx={grid['nx']} ny={grid['ny']} occ_mean={float(occ_cpu.mean()):.2f} "
                      f"p95={float(np.percentile(occ_cpu,95)):.2f} max={int(occ_cpu.max())}")

        # 2) Accumulation for one outer iteration (across colors)
        Jn_sum = 0.0
        rn_sum = 0.0
        cnt_sum = 0

        passes = (ncolors if ncolors>0 else 1)
        for cidx in range(passes):
            # Output buffers
            dF    = cp.zeros_like(d_pos)
            dJn   = cp.zeros((1,), dtype=cp.float32)
            dRn   = cp.zeros((1,), dtype=cp.float32)
            dCnt  = cp.zeros((1,), dtype=cp.int32)
            dDiag = cp.zeros((swarm_size,), dtype=cp.float32)
            dDeg  = cp.zeros((swarm_size,), dtype=cp.int32)

            # Chebyshev step size (if enabled)
            if use_cheb:
                theta = math.pi * (2.0*(cheb_k+1)-1.0) / (2.0*cheb_m)
                c = 0.5*(lam_max + lam_min)
                d = 0.5*(lam_max - lam_min)
                tau = 1.0 / (c - d*math.cos(theta))
            else:
                tau = omega * step_size

            # 3) Launch kernel (unified for colored/non-colored)
            bs = 256
            gs = (swarm_size + bs - 1) // bs
            ker((gs,), (bs,),
                (pos_sorted.ravel(),
                 order,
                 grid["cell_start"], grid["cell_end"],
                 cp.float32(grid["origin_x"]), cp.float32(grid["origin_y"]),
                 np.int32(grid["nx"]), np.int32(grid["ny"]),
                 cp.float32(grid["cell_size"]),
                 np.int32(swarm_size),
                 np.int32(K),
                 np.int32(ncolors), np.int32(cidx if ncolors>0 else 0),
                 np.float32(alpha), np.float32(beta),
                 np.float32(v), np.float32(r0), np.float32(PT),
                 np.float32(eps),
                 dF.ravel(), dJn, dRn, dCnt,
                 dDiag, dDeg))

            gpu.synchronize()

            # 4) Preconditioner scaling (point-wise)
            #    phi: use sum(phi_ij) as "diagonal approximation"; degree: use degree for rough scaling
            if precond == "phi":
                inv = 1.0 / (pc_eps + pc_gamma * dDiag)
            elif precond == "degree":
                inv = 1.0 / (1.0 + pc_gamma * dDeg.astype(cp.float32))
            else:
                inv = None

            step_vec = cp.float32(tau) * dF
            if inv is not None:
                step_vec = step_vec * inv[:, None]

            # For non-active colors, dF is theoretically ≈0; no extra mask here
            # For more stability, could use dDeg>0 as a mask:
            # if ncolors>0:
            #     mask = (dDeg > 0)
            #     step_vec = step_vec * mask[:, None]

            # 5) Semi-iteration: heavy-ball momentum
            if mom > 0.0:
                vel = mom * vel + step_vec
                delta_vec = vel
            else:
                delta_vec = step_vec

            d_pos = d_pos + delta_vec

            # Accumulate statistics (across colors)
            Jn_sum += float(dJn.get())
            rn_sum += float(dRn.get())
            cnt_sum += int(dCnt.get())

            if use_cheb:
                cheb_k = (cheb_k + 1) % cheb_m

        # 6) One outer iteration complete
        step_ms = (time.time() - t0) * 1000.0
        step_times_ms.append(step_ms)

        Jn = Jn_sum / max(cnt_sum, 1)
        rn = rn_sum / max(cnt_sum, 1)
        Jn_hist.append(round(Jn, 4))
        rn_hist.append(round(rn, 4))
        t_hist.append(time.time() - start_time)

        if it % log_every == 0:
            print(f"[it={it:4d}] Jn={Jn_hist[-1]:.4f} rn={rn_hist[-1]:.4f} compute={step_ms:.2f}ms")

        # 7) Optional plotting (disabled by default)
        if (axs is not None) and (fig is not None) and (plot_every > 0) and (it % plot_every == 0) and (it > 0):
            pos_cpu = gpu.to_host(d_pos)
            plotting.plot_figures_task1(
                axs, t_hist, Jn_hist, rn_hist, pos_cpu, PT,
                None,  # v2 does not construct NxN
                swarm_size, swarm_paths, node_colors, line_colors
            )

    # ---- Finalization ----
    total_time = time.time() - start_time
    avg_ms = float(np.mean(step_times_ms)) if step_times_ms else float("nan")

    print("\n=== v6 Simulation Complete ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average compute time: {avg_ms:.2f}ms/iter")
    print(f"compute={avg_ms:.2f}ms")
    print(f"Total iterations: {len(Jn_hist)}")
    if Jn_hist: print(f"Final Jn: {Jn_hist[-1]:.4f}")
    if rn_hist: print(f"Final rn: {rn_hist[-1]:.4f}")

    final_positions = gpu.to_host(d_pos)
    return (Jn_hist, rn_hist, final_positions, t_hist, None, None)
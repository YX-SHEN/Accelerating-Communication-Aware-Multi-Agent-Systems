# src/gpu/algorithms_advanced_gpu2.py
# -*- coding: utf-8 -*-
"""
GPU-Advanced v2：Jacobi + 空间网格(3×3 邻域，或 K 可调) 的近邻剪枝加速实现
- 现在完全支持可切换精度：由 config.DTYPE 决定（np.float32 / np.float64）
- 打印 compute=XXms，供 scripts/run_experiments.py 解析 kernel_ms_per_iter。
"""

import time
import math
import os
import numpy as np
import cupy as cp
import config

from src.gpu.backend import ImprovedGPUBackend, DTYPE_DEFAULT
from src.gpu.compute_grid import build_uniform_grid, comm_radius_from_config
from src.common import plotting

gpu = ImprovedGPUBackend()  # 本实现依赖 CuPy 后端

# ---- CUDA Kernel（按 dtype 动态生成）----
_RAW_TEMPLATE = r'''
// 精度选择与数学/原子加封装
#ifdef USE_DOUBLE
using real = double;
#define EXP  exp
#define SQRT sqrt
#define POW  pow
#else
using real = float;
#define EXP  expf
#define SQRT sqrtf
#define POW  powf
#endif

#ifdef USE_DOUBLE
#if __CUDA_ARCH__ < 600
__device__ inline double atomicAddDouble(double* address, double val) {
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOMIC_ADD(addr, val) atomicAddDouble((double*)(addr), (double)(val))
#else
#define ATOMIC_ADD(addr, val) atomicAdd((double*)(addr), (double)(val))
#endif
#else
#define ATOMIC_ADD(addr, val) atomicAdd((float*)(addr), (float)(val))
#endif

extern "C" __global__
void forces_grid(const real* __restrict__ pos,      // [N,2], 已按 cell 排序
                 const int*  __restrict__ order,    // sorted_idx -> original_i
                 const int*  __restrict__ cell_start,
                 const int*  __restrict__ cell_end,
                 const real  ox, const real oy,
                 const int   nx, const int ny,
                 const real  cell_size,
                 const int   N,
                 const int   K,                     // 邻域半径（1=3x3, 2=5x5）
                 const real  alpha, const real beta,
                 const real  v, const real r0, const real PT,
                 const real  eps,
                 const real  two_pow_delta_minus_1, // 预先计算好的 (2^delta - 1)
                 real* __restrict__ outF,   // [N,2], 原顺序
                 real* __restrict__ outJn,  // 单元素，atomicAdd
                 real* __restrict__ outRn,
                 int*  __restrict__ outCnt)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) return;

    real xi = pos[2*tid+0];
    real yi = pos[2*tid+1];

    int gx = (int)floor((xi - ox)/cell_size);
    int gy = (int)floor((yi - oy)/cell_size);
    if (gx < 0) gx = 0; if (gx >= nx) gx = nx-1;
    if (gy < 0) gy = 0; if (gy >= ny) gy = ny-1;

    real Fx = (real)0, Fy = (real)0;
    real phi_sum = (real)0, r_sum = (real)0;
    int  cnt = 0;

    // 扫 (2K+1)x(2K+1) 邻域
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
                real xj = pos[2*k+0];
                real yj = pos[2*k+1];
                real dx = xi - xj;
                real dy = yi - yj;
                real d2 = dx*dx + dy*dy;
                real rij = SQRT((d2 > eps) ? d2 : eps);

                real aij = EXP(-alpha * two_pow_delta_minus_1 * POW(rij/r0, v));
                if (aij < PT) continue;

                real gij = rij / SQRT(rij*rij + r0*r0);
                real phi = gij * aij;

                real num = (-beta * v * POW(rij, v+2.0) - beta * v * (r0*r0) * POW(rij, v) + POW(r0, v+2.0));
                real den = SQRT(POW(rij*rij + r0*r0, 3.0)) + eps;
                real rho = num * EXP(-beta * POW(rij/r0, v)) / den;

                real ex = dx / (rij + eps);
                real ey = dy / (rij + eps);
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
        ATOMIC_ADD(outJn, phi_sum);
        ATOMIC_ADD(outRn, r_sum);
        atomicAdd(outCnt, cnt);
    }
}
'''

def _kernel(delta: int, use_double: bool) -> cp.RawKernel:
    """按需要生成 float / double 版 kernel。"""
    code = _RAW_TEMPLATE
    # 用 NVRTC 宏控制是否使用 double
    options = ('-DUSE_DOUBLE',) if use_double else tuple()
    return cp.RawKernel(code, "forces_grid", options=options)


def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    返回：
      Jn_list(四舍五入), rn_list(四舍五入), final_positions(np),
      t_elapsed(s列表), final_comm_matrix(None), None
    """
    # ---- 读取/规范化配置（决定精度）----
    dtype_np = np.dtype(getattr(config, "DTYPE", DTYPE_DEFAULT))
    use_double = (dtype_np == np.float64)
    cp_real = cp.float64 if use_double else cp.float32

    # 位置到设备端（按 dtype）
    d_pos = gpu.to_device(swarm_position.astype(cp_real, copy=False))

    # 步长/常数（按 dtype）
    step_size = cp_real(getattr(config, "STEP_SIZE", 0.01))
    eps       = cp_real(getattr(config, "EPS", 1e-6))
    delta     = int(getattr(config, "DELTA", 2))
    two_pow_delta_minus_1 = cp_real((2.0 ** delta) - 1.0)

    log_every  = max(1, int(getattr(config, "LOG_EVERY", 50)))
    plot_every = max(1, int(getattr(config, "PLOT_EVERY", 10**9)))  # 默认基本不画

    keep_full_matrix = False  # v2 不构造 NxN，可按需改

    print("[GPU-Advanced v2] 后端:", gpu.backend)
    if gpu.device:
        print("[GPU-Advanced v2] 设备:", gpu.device)
    print("[GPU-Advanced v2] 模式: jacobi")
    print(f"[GPU-Advanced v2] N={swarm_size}, max_iter={max_iter}, dtype={'float64' if use_double else 'float32'}")

    # --- 通信半径 & cell 尺寸（可按需倍率） ---
    R = float(comm_radius_from_config(config))
    if not np.isfinite(R) or R <= 0:
        # 兜底：用数据跨度/√N 估一个尺度
        pos_np = gpu.to_host(d_pos)
        span = float(max(pos_np[:, 0].max() - pos_np[:, 0].min(),
                         pos_np[:, 1].max() - pos_np[:, 1].min()))
        R = max(1e-3, span / max(np.sqrt(float(swarm_size)), 1.0))

    # 可调：AD2_CELL_SCALE（默认 1.0）→ cell_size = R / scale, 邻域 K = ceil(scale)
    scale = float(os.getenv("AD2_CELL_SCALE", "1.0"))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cell_size = float(R / scale)
    K = int(max(1, math.ceil(scale)))

    # 编译对应精度的 kernel
    ker = _kernel(delta, use_double=use_double)

    # ---- 统计容器 ----
    Jn_hist, rn_hist, t_hist = [], [], []
    comm_mat_placeholder = None  # v2 不返回 NxN
    start_time = time.time()
    step_times_ms = []

    for it in range(max_iter):
        t_step0 = time.time()

        # 1) 每步重建网格（简单稳妥；如需更快可隔几步重建）
        grid = build_uniform_grid(d_pos.astype(cp.float32, copy=False), float(cell_size))  # 网格用 fp32 足够
        order = grid["order"].astype(cp.int32)         # sorted_idx -> original_i（int32）
        pos_sorted = d_pos[order]                      # [N,2] 排序后（与 kernel 精度一致）

        # [AD2_DEBUG] 打印网格信息：cell占用统计
        if os.getenv("AD2_DEBUG", "0") == "1":
            occ = grid["cell_end"] - grid["cell_start"]
            occ = occ[occ >= 0]
            if occ.size > 0:
                occ_cpu = occ.get()
                occ_mean = float(occ_cpu.mean())
                occ_p95  = float(np.percentile(occ_cpu, 95))
                occ_max  = int(occ_cpu.max())
            else:
                occ_mean = occ_p95 = occ_max = 0.0
            print(f"[grid] nx={grid['nx']} ny={grid['ny']} K={K} R={R:.3f} "
                  f"cell_size={cell_size:.3f} occ_mean={occ_mean:.2f} "
                  f"occ_p95={occ_p95:.2f} occ_max={occ_max}")

        # 2) 输出缓冲（与 kernel 精度一致）
        dF   = cp.zeros_like(d_pos)                   # [N,2] real
        dJn  = cp.zeros((1,), dtype=cp_real)
        dRn  = cp.zeros((1,), dtype=cp_real)
        dCnt = cp.zeros((1,), dtype=cp.int32)

        # 3) 启动 kernel（所有标量参数的 dtype 必须与 kernel 的 real 对齐）
        bs = 256
        gs = (swarm_size + bs - 1) // bs
        ker((gs,), (bs,),
            (pos_sorted.ravel(),
             order,
             grid["cell_start"], grid["cell_end"],
             cp_real(grid["origin_x"]), cp_real(grid["origin_y"]),
             np.int32(grid["nx"]), np.int32(grid["ny"]),
             cp_real(grid["cell_size"]),
             np.int32(swarm_size),
             np.int32(K),
             cp_real(alpha), cp_real(beta),
             cp_real(v), cp_real(r0), cp_real(PT),
             cp_real(eps),
             cp_real(two_pow_delta_minus_1),
             dF.ravel(), dJn, dRn, dCnt))

        # 4) 同步 + 计时
        gpu.synchronize()
        step_ms = (time.time() - t_step0) * 1000.0
        step_times_ms.append(step_ms)

        # 5) Jacobi 更新
        d_pos = d_pos + step_size * dF

        # 6) 统计
        cnt = int(dCnt.get())
        Jn  = float(dJn.get() / max(cnt, 1))
        rn  = float(dRn.get() / max(cnt, 1))
        Jn_hist.append(round(Jn, 4))
        rn_hist.append(round(rn, 4))
        t_hist.append(time.time() - start_time)

        # 7) 日志
        if it % log_every == 0:
            print(f"[it={it:4d}] Jn={Jn_hist[-1]:.4f} rn={rn_hist[-1]:.4f} compute={step_ms:.2f}ms")

        # 8) 绘图（默认关闭；需要时只取位置，通信矩阵传 None/占位）
        if (axs is not None) and (fig is not None) and (plot_every > 0) and (it % plot_every == 0) and (it > 0):
            pos_cpu = gpu.to_host(d_pos)
            plotting.plot_figures_task1(
                axs, t_hist, Jn_hist, rn_hist, pos_cpu, PT,
                None,  # v2 不构造 NxN 矩阵
                swarm_size, swarm_paths, node_colors, line_colors
            )

    total_time = time.time() - start_time
    avg_ms = float(np.mean(step_times_ms)) if step_times_ms else float("nan")

    print("\n=== v2 仿真完成 ===")
    print(f"总时间: {total_time:.2f}s")
    print(f"平均compute时间: {avg_ms:.2f}ms/iter")
    print(f"compute={avg_ms:.2f}ms")  # 供 run_experiments 的正则稳定抓取
    print(f"总迭代数: {len(Jn_hist)}")
    if Jn_hist:
        print(f"最终 Jn: {Jn_hist[-1]:.4f}")
    if rn_hist:
        print(f"最终 rn: {rn_hist[-1]:.4f}")

    final_positions = gpu.to_host(d_pos)
    return (Jn_hist, rn_hist, final_positions, t_hist, None, None)
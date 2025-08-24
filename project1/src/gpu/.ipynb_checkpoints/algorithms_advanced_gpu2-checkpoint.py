# src/gpu/algorithms_advanced_gpu2.py
# -*- coding: utf-8 -*-
"""
GPU-Advanced v2：Jacobi + 空间网格(3×3 邻域) 的近邻剪枝加速实现
- 与你的 GPU Jacobi 公式/阈值/口径一致，只是把全局 O(N^2) 邻域裁剪为本 cell 的 3×3。
- 与 CPU Baseline 的 Gauss–Seidel 逐点就地更新不会逐步完全相同，但最终 Jn/rn 行为对齐。
- 打印 compute=XXms，供 scripts/run_experiments.py 解析 kernel_ms_per_iter。
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
# 不在全局计算 R_CUT/CELL_H，避免默认值未定义；在 run_simulation 里按 PT 动态推导

# ---- CUDA Kernel ----
_RAW = r'''
extern "C" __global__
void forces_grid(const float* __restrict__ pos,      // [N,2], 已按 cell 排序
                 const int*   __restrict__ order,    // sorted_idx -> original_i
                 const int*   __restrict__ cell_start,
                 const int*   __restrict__ cell_end,
                 const float  ox, const float oy,
                 const int    nx, const int ny,
                 const float  cell_size,
                 const int    N,
                 const int    K,                     // <<< 新增：邻域半径（1=3x3, 2=5x5）
                 const float  alpha, const float beta,
                 const float  v, const float r0, const float PT,
                 const float  eps,
                 float* __restrict__ outF,   // [N,2], 原顺序
                 float* __restrict__ outJn,  // 单元素，block 内累加 + atomicAdd
                 float* __restrict__ outRn,
                 int*   __restrict__ outCnt)
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

    // 扫 (2K+1)x(2K+1) 邻域   <<< 这里用 K
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
    返回：
      Jn_list(四舍五入), rn_list(四舍五入), final_positions(np),
      t_elapsed(s列表), final_comm_matrix(None), None
    """
    # ---- 读取/规范化配置 ----
    dtype = getattr(config, "DTYPE", DTYPE_DEFAULT)
    # kernel 为 float 版，强制使用 float32 更稳妥
    d_pos = gpu.to_device(swarm_position.astype(cp.float32, copy=False))

    step_size  = float(getattr(config, "STEP_SIZE", 0.01))
    eps        = float(getattr(config, "EPS", 1e-6))
    delta      = int(getattr(config, "DELTA", 2))
    log_every = max(1, int(getattr(config, "LOG_EVERY", 50)))
    plot_every = max(1, int(getattr(config, "PLOT_EVERY", 10 ** 9)))  # 默认基本不画
    keep_full_matrix = False  # v2 不构造 NxN，可按需改

    print("[GPU-Advanced v2] 后端:", gpu.backend)
    if gpu.device:
        print("[GPU-Advanced v2] 设备:", gpu.device)
    print("[GPU-Advanced v2] 模式: jacobi")
    print(f"[GPU-Advanced v2] N={swarm_size}, max_iter={max_iter}")

    # --- 通信半径 & cell 尺寸（可按需加个倍率） ---
    R = float(comm_radius_from_config(config))
    if not np.isfinite(R) or R <= 0:
        # 兜底：用数据跨度/√N 估一个尺度
        pos_np = gpu.to_host(d_pos)
        span = float(max(pos_np[:,0].max()-pos_np[:,0].min(),
                         pos_np[:,1].max()-pos_np[:,1].min()))
        R = max(1e-3, span / max(np.sqrt(float(swarm_size)), 1.0))


    # 可调：AD2_CELL_SCALE（默认 1.0）→ cell_size = R / scale, 邻域 K = ceil(scale)
    scale = float(os.getenv("AD2_CELL_SCALE", "1.0"))
    if not np.isfinite(scale) or scale <= 0: scale = 1.0
    cell_size = float(R / scale)
    K = int(max(1, math.ceil(scale)))

    ker = _kernel(delta)

    # ---- 统计容器 ----
    Jn_hist, rn_hist, t_hist = [], [], []
    comm_mat_placeholder = None  # v2 不返回 NxN
    start_time = time.time()
    step_times_ms = []

    for it in range(max_iter):
        t_step0 = time.time()

        # 1) 每步重建网格（简单稳妥；如需更快可隔几步重建）
        grid = build_uniform_grid(d_pos, cell_size)
        order = grid["order"].astype(cp.int32)          # sorted_idx -> original_i（int32！）
        pos_sorted = d_pos[order]                        # [N,2] 排序后

        # [AD2_DEBUG] 打印网格信息：cell占用统计
        if os.getenv("AD2_DEBUG", "0") == "1":
            occ = grid["cell_end"] - grid["cell_start"]  # 每个 cell 的粒子数
            occ = occ[occ >= 0]  # 过滤没有粒子的 cell
            if occ.size > 0:
                # 为了兼容不同 CuPy 版本，先搬回 CPU 再算分位数
                occ_cpu = occ.get()
                occ_mean = float(occ_cpu.mean())
                occ_p95 = float(np.percentile(occ_cpu, 95))
                occ_max = int(occ_cpu.max())
            else:
                occ_mean = occ_p95 = occ_max = 0.0
            print(f"[grid] nx={grid['nx']} ny={grid['ny']} K={K} R={R:.3f} "
                  f"cell_size={cell_size:.3f} occ_mean={occ_mean:.2f} "
                  f"occ_p95={occ_p95:.2f} occ_max={occ_max}")

        # 2) 输出缓冲（fp32 / int32）
        dF   = cp.zeros_like(d_pos)                      # [N,2] fp32
        dJn  = cp.zeros((1,), dtype=cp.float32)
        dRn  = cp.zeros((1,), dtype=cp.float32)
        dCnt = cp.zeros((1,), dtype=cp.int32)

        # 3) 启动 kernel
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
             np.int32(K),  # <<< 新增：邻域半径
             np.float32(alpha), np.float32(beta),
             np.float32(v), np.float32(r0), np.float32(PT),
             np.float32(eps),
             dF.ravel(), dJn, dRn, dCnt))

        # 4) 同步 + 计时
        gpu.synchronize()
        step_ms = (time.time() - t_step0) * 1000.0
        step_times_ms.append(step_ms)

        # 5) Jacobi 更新
        d_pos = d_pos + cp.float32(step_size) * dF

        # 6) 统计
        cnt = int(dCnt.get())
        # [AD2_DEBUG] 打印平均出度（有向边数 / N）
        if os.getenv("AD2_DEBUG", "0") == "1":
            avg_deg = (cnt / float(swarm_size)) if swarm_size > 0 else 0.0
            print(f"[edges] cnt={cnt} avg_out_degree={avg_deg:.2f}")
        Jn  = float(dJn.get() / max(cnt, 1))
        rn  = float(dRn.get() / max(cnt, 1))
        Jn_hist.append(round(Jn, 4))
        rn_hist.append(round(rn, 4))
        t_hist.append(time.time() - start_time)

        # 7) 日志
        if it % log_every == 0:
            print(f"[it={it:4d}] Jn={Jn_hist[-1]:.4f} rn={rn_hist[-1]:.4f} compute={step_ms:.2f}ms")

        # 8) 绘图（默认关闭；需要时只取位置，通信矩阵传 None）
        if (axs is not None) and (fig is not None) and (plot_every > 0) \
                and (it % plot_every == 0) and (it > 0):
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
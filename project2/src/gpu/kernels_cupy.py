# src/gpu/kernels_cupy.py
# -*- coding: utf-8 -*-
import numpy as np

def compute_forces_cupy_optimized(cp, pos, N, alpha, beta, v, r0, PT, delta,
                                  keep_full_matrix=True, dtype=np.float32, eps=1e-8):
    """
    CuPy 版向量化核（全矩阵路径）。
    返回：
      forces (cp.ndarray [N,2], dtype)
      Jn     (float)
      rn     (float)
      phi_mat(cp.ndarray [N,N], dtype) 或 0.0（当 keep_full_matrix=False）
      conn   (cp.ndarray [N,N], bool)
    """
    # —— 统一精度：把输入与所有标量常量都变成目标 dtype ——
    pos   = cp.asarray(pos, dtype=dtype)
    alpha = cp.asarray(alpha, dtype=dtype)
    beta  = cp.asarray(beta,  dtype=dtype)
    v     = cp.asarray(v,     dtype=dtype)
    r0    = cp.asarray(r0,    dtype=dtype)
    PT    = cp.asarray(PT,    dtype=dtype)
    delta = cp.asarray(delta, dtype=dtype)
    eps_f = cp.asarray(eps,   dtype=dtype)

    # 预计算 (2**delta - 1) 以减少重复开销
    exp2d_1 = cp.power(cp.asarray(2.0, dtype=dtype), delta) - cp.asarray(1.0, dtype=dtype)

    # —— 全对全差分与距离 ——
    diff = pos[:, None, :] - pos[None, :, :]           # [N,N,2]
    d2   = cp.sum(diff * diff, axis=2)                 # [N,N], dtype
    rij  = cp.sqrt(cp.maximum(d2, eps_f))              # 避免 sqrt(0)

    # 自环无效，并排除 0 距离
    eye   = cp.eye(N, dtype=cp.bool_)
    valid = cp.logical_and(rij > eps_f, ~eye)

    # —— a_ij、连边判定、g_ij ——
    rrat = rij / r0
    aij  = cp.exp(-alpha * exp2d_1 * cp.power(rrat, v))
    conn = cp.logical_and(valid, aij >= PT)

    gij  = rij / cp.sqrt(rij * rij + r0 * r0)

    # —— ρ_ij（与 CPU 一致的表达式）——
    # num = (-β v r^{v+2} - β v r0^2 r^v + r0^{v+2})
    # den = sqrt( (r^2 + r0^2)^3 ) + eps
    rv_v   = cp.power(rij, v)
    rv_v2  = rv_v * rij * rij
    r0_2   = r0 * r0
    r0_v2  = cp.power(r0, v + cp.asarray(2.0, dtype=dtype))

    num = (-beta * v * rv_v2) - (beta * v * r0_2 * rv_v) + r0_v2
    den = cp.sqrt(cp.power(rij * rij + r0_2, cp.asarray(3.0, dtype=dtype))) + eps_f
    rho = num * cp.exp(-beta * cp.power(rij / r0, v)) / den

    # —— 单位方向 e_ij 与力累计 ——
    eij = diff / (rij[:, :, None] + eps_f)             # [N,N,2]
    fcontrib = (conn[:, :, None].astype(dtype) * rho[:, :, None]) * eij
    forces = cp.sum(fcontrib, axis=1).astype(dtype, copy=False)  # [N,2]

    # —— 统计量与可视化矩阵 ——
    phi_all = (gij * aij).astype(dtype, copy=False)
    if keep_full_matrix:
        phi_mat = (phi_all * conn.astype(dtype)).astype(dtype, copy=False)
    else:
        phi_mat = cp.array(0.0, dtype=dtype)

    # 邻边计数（避免除 0）
    denom_cnt = cp.maximum(conn.sum(), cp.asarray(1, dtype=conn.dtype))

    # Jn, rn 转 Python float（与 CPU 版口径一致）
    Jn = float((phi_all * conn).sum() / denom_cnt)
    rn = float((rij * conn).sum() / denom_cnt)

    return forces, Jn, rn, phi_mat, conn
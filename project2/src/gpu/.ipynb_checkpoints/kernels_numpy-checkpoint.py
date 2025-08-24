# src/gpu/kernels_numpy.py
# -*- coding: utf-8 -*-
import numpy as np

def compute_forces_numpy_optimized(pos, N, alpha, beta, v, r0, PT, delta,
                                   keep_full_matrix=True, dtype=np.float32, eps=1e-8):
    """NumPy 回退核：保持 baseline O(N^2) 逻辑（向量化），与 config.DTYPE 对齐。"""
    # —— 与 dtype 对齐的常量/参数 ——
    pos   = np.asarray(pos, dtype=dtype, order="C")
    alpha = np.asarray(alpha, dtype=dtype)
    beta  = np.asarray(beta,  dtype=dtype)
    v     = np.asarray(v,     dtype=dtype)
    r0    = np.asarray(r0,    dtype=dtype)
    PT    = np.asarray(PT,    dtype=dtype)
    delta = np.asarray(delta, dtype=dtype)
    eps   = np.asarray(eps,   dtype=dtype)
    one   = np.asarray(1.0,   dtype=dtype)
    two   = np.asarray(2.0,   dtype=dtype)

    forces = np.zeros((N, 2), dtype=dtype)

    # 距离与掩码
    diff = pos[:, None, :] - pos[None, :, :]          # [N,N,2]
    d2   = np.sum(diff * diff, axis=2, dtype=dtype)   # [N,N]
    rij  = np.sqrt(np.maximum(d2, eps))               # [N,N]

    mask = (rij > eps)
    np.fill_diagonal(mask, False)

    # a_ij 与连边
    exp2d_1 = np.power(two, delta) - one
    aij  = np.exp(-alpha * exp2d_1 * np.power(rij / r0, v))
    conn = np.logical_and(mask, aij >= PT)

    # 没有任何连边：直接返回零力与占位通信矩阵
    if not np.any(conn):
        comm_out = (np.zeros((N, N), dtype=dtype) if keep_full_matrix
                    else np.array(0.0, dtype=dtype))
        return forces, 0.0, 0.0, comm_out, conn

    # g_ij
    r0_2 = r0 * r0
    gij  = rij / np.sqrt(rij * rij + r0_2)

    # ρ_ij
    rv_v   = np.power(rij, v)
    rv_v2  = rv_v * rij * rij                   # rij**(v+2)
    r0_v2  = np.power(r0, v + np.asarray(2.0, dtype=dtype))
    num    = (-beta * v * rv_v2) - (beta * v * r0_2 * rv_v) + r0_v2
    den    = np.sqrt(np.power(rij * rij + r0_2, np.asarray(3.0, dtype=dtype))) + eps
    rho    = num * np.exp(-beta * np.power(rij / r0, v)) / den

    # 力累计
    eij      = diff / (rij[:, :, None] + eps)
    fcontrib = (conn[:, :, None].astype(dtype) * rho[:, :, None]) * eij
    forces   = np.sum(fcontrib, axis=1, dtype=dtype)

    # 统计量
    phi_all = (gij * aij).astype(dtype, copy=False)
    if keep_full_matrix:
        phi_mat = (phi_all * conn.astype(dtype, copy=False)).astype(dtype, copy=False)
    else:
        phi_mat = np.array(0.0, dtype=dtype)

    cnt = max(int(np.sum(conn)), 1)
    Jn  = float(np.sum(phi_all * conn) / cnt)
    rn  = float(np.sum(rij * conn)    / cnt)

    return forces, Jn, rn, phi_mat, conn
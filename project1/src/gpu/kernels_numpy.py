# src/gpu/kernels_numpy.py
# -*- coding: utf-8 -*-
import numpy as np

def compute_forces_numpy_optimized(pos, N, alpha, beta, v, r0, PT, delta,
                                   keep_full_matrix=True, dtype=np.float32, eps=1e-8):
    """NumPy fallback kernel, maintaining baseline O(N^2) logical (vectorized)."""
    pos = pos.astype(dtype, copy=False)
    forces = np.zeros((N, 2), dtype=dtype)

    diff = pos[:, None, :] - pos[None, :, :]        # [N,N,2]
    d2   = np.sum(diff*diff, axis=2)                # [N,N]
    rij  = np.sqrt(np.maximum(d2, eps))

    mask = (rij > eps)
    np.fill_diagonal(mask, False)

    aij  = np.exp(-alpha * (2**delta - 1) * (rij / r0)**v)
    conn = np.logical_and(mask, aij >= PT)

    if not np.any(conn):
        comm_out = np.zeros((N, N), dtype=dtype) if keep_full_matrix else 0.0
        return forces, 0.0, 0.0, comm_out, conn

    gij  = rij / np.sqrt(rij*rij + r0*r0)

    num  = (-beta * v * rij**(v+2) - beta * v * (r0**2) * (rij**v) + r0**(v+2))
    den  = np.sqrt((rij*rij + r0*r0)**3) + eps
    rho  = num * np.exp(-beta * (rij / r0)**v) / den

    eij  = diff / (rij[:, :, None] + eps)
    fcontrib = (conn[:, :, None] * rho[:, :, None]) * eij
    forces = np.sum(fcontrib, axis=1)

    phi_all = gij * aij
    phi_mat = (phi_all * conn.astype(phi_all.dtype)) if keep_full_matrix else 0.0

    Jn = float(np.sum(phi_all * conn) / np.sum(conn))
    rn = float(np.sum(rij * conn) / np.sum(conn))

    return forces, Jn, rn, phi_mat, conn
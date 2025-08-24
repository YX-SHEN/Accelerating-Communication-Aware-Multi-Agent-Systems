# src/gpu/kernels_cupy.py
# -*- coding: utf-8 -*-
import numpy as np

def compute_forces_cupy_optimized(cp, pos, N, alpha, beta, v, r0, PT, delta,
                                  keep_full_matrix=True, dtype=np.float32, eps=1e-8):
    """CuPy version of the vectorized kernel. Returns: forces(cp array), Jn(float), rn(float), phi_mat(cp array or 0.0)"""
    pos = cp.asarray(pos, dtype=dtype)

    diff = pos[:, None, :] - pos[None, :, :]
    d2   = cp.sum(diff*diff, axis=2)
    rij  = cp.sqrt(cp.maximum(d2, eps))

    eye  = cp.eye(N, dtype=cp.bool_)
    valid= cp.logical_and(rij > eps, ~eye)

    rrat = rij / r0
    aij  = cp.exp(-alpha * (2**delta - 1) * rrat**v)
    conn = cp.logical_and(valid, aij >= PT)

    gij  = rij / cp.sqrt(rij*rij + r0**2)

    num  = (-beta * v * rij**(v+2) - beta * v * (r0**2) * (rij**v) + r0**(v+2))
    den  = cp.sqrt((rij*rij + r0**2)**3) + eps
    rho  = num * cp.exp(-beta * (rij / r0)**v) / den

    eij  = diff / (rij[:, :, None] + eps)
    fcontrib = (conn[:, :, None] * rho[:, :, None]) * eij
    forces = cp.sum(fcontrib, axis=1)

    phi_all = gij * aij
    phi_mat = (phi_all * conn) if keep_full_matrix else cp.array(0.0, dtype=dtype)

    denom = cp.maximum(cp.sum(conn), 1)
    Jn = float(cp.sum(phi_all * conn) / denom)
    rn = float(cp.sum(rij * conn) / denom)

    return forces, Jn, rn, phi_mat, conn
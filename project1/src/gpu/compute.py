# src/gpu/compute.py
# -*- coding: utf-8 -*-
import math
import numpy as np

from .kernels_numpy import compute_forces_numpy_optimized
from .kernels_jax import create_jax_kernel
from .kernels_cupy import compute_forces_cupy_optimized

# Old: JAX full-matrix kernel cache (for backward compatibility)
_JAX_KERNEL_CACHE_FULL = {}

# New: JAX edge-based kernel cache -> keyed by (E_cap, dtype, alpha, beta, v, r0, PT, delta, N, eps)
_JAX_KERNEL_CACHE_EDGE = {}

def _ensure_tuple(arr):
    """Converts a possible list/np.ndarray to a 1D np.ndarray[int32]"""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.int32).ravel()
    return a

def _pad_edges_for_jax(i_idx, j_idx, ecap_factor=1.25):
    """
    JAX requires fixed shapes to avoid frequent JIT recompilation.
    Returns: ii_cap, jj_cap, mask_init(0/1 float32), E_used, E_cap
    """
    E_used = int(i_idx.size)
    if E_used == 0:
        # Placeholder: at least 1, to avoid empty shapes
        return (np.zeros((1,), np.int32),
                np.zeros((1,), np.int32),
                np.zeros((1,), np.float32),
                0, 1)
    E_cap = max(1, int(math.ceil(E_used * float(ecap_factor))))
    if E_cap == E_used:
        mask_init = np.ones((E_cap,), np.float32)
        return i_idx, j_idx, mask_init, E_used, E_cap

    pad = E_cap - E_used
    last_i = int(i_idx[-1])
    last_j = int(j_idx[-1])
    ii_cap = np.pad(i_idx, (0, pad), mode="constant", constant_values=last_i).astype(np.int32, copy=False)
    jj_cap = np.pad(j_idx, (0, pad), mode="constant", constant_values=last_j).astype(np.int32, copy=False)
    mask_init = np.concatenate([np.ones((E_used,), np.float32),
                                np.zeros((pad,),   np.float32)], axis=0)
    return ii_cap, jj_cap, mask_init, E_used, E_cap


# =========================
# Edge-based implementations
# =========================
def _edge_kernel_jax(gpu, E_cap, dtype, alpha, beta, v, r0, PT, delta, N, eps):
    """
    Constructs/caches a JAX edge-based kernel (with a fixed E_cap).
    Returns a callable: kernel(positions, ii_cap, jj_cap, mask_init) ->
        forces[N,2], Jn(float), rn(float), phi[E_cap], valid_bool[E_cap]
    """
    key = (int(E_cap), str(np.dtype(dtype)), float(alpha), float(beta), float(v),
           float(r0), float(PT), float(delta), int(N), float(eps))
    ker = _JAX_KERNEL_CACHE_EDGE.get(key)
    if ker is not None:
        return ker

    jnp, jax = gpu.jnp, gpu.jax

    exp2d_1 = (2.0 ** delta - 1.0)  # Hoist constant

    @jax.jit
    def kernel(positions, ii, jj, mask_init):
        # positions: [N,2], ii/jj/mask_init: [E_cap]
        qi = positions[ii]
        qj = positions[jj]
        diff = qi - qj                       # [E,2]
        d2 = jnp.sum(diff * diff, axis=1)    # [E]
        rij = jnp.sqrt(jnp.maximum(d2, eps))

        aij = jnp.exp(-alpha * exp2d_1 * (rij / r0) ** v)  # [E]
        valid_pt  = (aij >= PT).astype(positions.dtype)    # 0/1
        valid     = valid_pt * mask_init                   # Will be 1 only for real edges

        inv_r = 1.0 / (rij + eps)
        eijx = diff[:, 0] * inv_r
        eijy = diff[:, 1] * inv_r

        gij = rij / jnp.sqrt(rij * rij + r0 ** 2)

        num = (-beta * v * rij ** (v + 2) - beta * v * (r0 ** 2) * (rij ** v) + r0 ** (v + 2))
        den = jnp.sqrt((rij * rij + r0 ** 2) ** 3) + eps
        rho = num * jnp.exp(-beta * (rij / r0) ** v) / den

        fx_e = rho * eijx * valid
        fy_e = rho * eijy * valid

        fx = jnp.bincount(ii, weights=fx_e, length=N) - jnp.bincount(jj, weights=fx_e, length=N)
        fy = jnp.bincount(ii, weights=fy_e, length=N) - jnp.bincount(jj, weights=fy_e, length=N)
        forces = jnp.stack([fx, fy], axis=1).astype(positions.dtype)

        phi = gij * aij * valid
        cnt = jnp.maximum(jnp.sum(valid), 1.0)
        Jn  = jnp.sum(phi) / cnt
        rn  = jnp.sum(rij * valid) / cnt

        return forces, Jn, rn, phi, (valid > 0.5)

    _JAX_KERNEL_CACHE_EDGE[key] = kernel
    return kernel


def _edge_path_jax(gpu, positions, i_idx, j_idx, N,
                   alpha, beta, v, r0, PT, delta, dtype, eps, ecap_factor):
    # Pad to a fixed shape
    ii_cap, jj_cap, mask_init, E_used, E_cap = _pad_edges_for_jax(i_idx, j_idx, ecap_factor)
    # To device
    jnp = gpu.jnp
    pos_d  = jnp.asarray(positions, dtype=dtype)
    ii_d   = jnp.asarray(ii_cap, dtype=jnp.int32)
    jj_d   = jnp.asarray(jj_cap, dtype=jnp.int32)
    mask_d = jnp.asarray(mask_init, dtype=pos_d.dtype)

    ker = _edge_kernel_jax(gpu, E_cap, dtype, alpha, beta, v, r0, PT, delta, N, eps)
    forces, Jn, rn, phi_all, valid_all = ker(pos_d, ii_d, jj_d, mask_d)
    _ = forces.block_until_ready()  # Synchronize

    # Return only the part corresponding to real edges (aligned with original i_idx/j_idx)
    return (forces,
            float(Jn), float(rn),
            phi_all[:E_used], valid_all[:E_used])


def _edge_path_cupy(gpu, positions, i_idx, j_idx, N,
                    alpha, beta, v, r0, PT, delta, dtype, eps):
    cp = gpu.cp
    pos = cp.asarray(positions, dtype=dtype)
    ii  = cp.asarray(i_idx, dtype=cp.int32)
    jj  = cp.asarray(j_idx, dtype=cp.int32)

    qi = pos[ii]; qj = pos[jj]
    diff = qi - qj
    rij  = cp.sqrt(cp.maximum(cp.sum(diff*diff, axis=1), eps))

    aij = cp.exp(-alpha * (2**delta - 1) * (rij / r0)**v)
    valid = (aij >= PT)

    if not bool(valid.any()):
        zero_forces = cp.zeros((N,2), dtype=dtype)
        empty = cp.asarray([], dtype=dtype)
        return zero_forces, 0.0, 0.0, empty, valid

    dv = diff[valid]
    rv = rij[valid]
    aiv = aij[valid]
    gij = rv / cp.sqrt(rv*rv + r0**2)

    num = (-beta * v * rv**(v+2) - beta * v * (r0**2) * (rv**v) + r0**(v+2))
    den = cp.sqrt((rv*rv + r0**2)**3) + eps
    rho = num * cp.exp(-beta * (rv / r0)**v) / den

    eij = dv / rv[:, None]
    f   = rho[:, None] * eij

    ii_v = ii[valid]; jj_v = jj[valid]
    fx = cp.bincount(ii_v, weights=f[:,0], minlength=N) - cp.bincount(jj_v, weights=f[:,0], minlength=N)
    fy = cp.bincount(ii_v, weights=f[:,1], minlength=N) - cp.bincount(jj_v, weights=f[:,1], minlength=N)
    forces = cp.stack([fx, fy], axis=1).astype(pos.dtype)

    phi = gij * aiv
    cnt = max(int(valid.sum().get()), 1)
    Jn  = float(cp.sum(phi) / cnt)
    rn  = float(cp.sum(rv)  / cnt)

    cp.cuda.Stream.null.synchronize()
    return forces, Jn, rn, phi, valid


def _edge_path_numpy(positions, i_idx, j_idx, N,
                     alpha, beta, v, r0, PT, delta, dtype, eps):
    pos = np.asarray(positions, dtype=dtype)
    ii  = np.asarray(i_idx, dtype=np.int32)
    jj  = np.asarray(j_idx, dtype=np.int32)

    qi = pos[ii]; qj = pos[jj]
    diff = qi - qj
    rij  = np.sqrt((diff*diff).sum(axis=1))
    rij  = np.maximum(rij, eps)

    aij = np.exp(-alpha * (2**delta - 1) * (rij / r0)**v)
    valid = (aij >= PT)

    if not np.any(valid):
        return np.zeros((N,2), dtype=dtype), 0.0, 0.0, np.asarray([], dtype=dtype), valid

    dv = diff[valid]
    rv = rij[valid]
    aiv = aij[valid]
    gij = rv / np.sqrt(rv*rv + r0**2)

    num = (-beta * v * rv**(v+2) - beta * v * (r0**2) * (rv**v) + r0**(v+2))
    den = np.sqrt((rv*rv + r0**2)**3) + eps
    rho = num * np.exp(-beta * (rv / r0)**v) / den

    eij = dv / rv[:, None]
    f   = rho[:, None] * eij

    ii_v = ii[valid]; jj_v = jj[valid]
    fx = np.bincount(ii_v, weights=f[:,0], minlength=N) - np.bincount(jj_v, weights=f[:,0], minlength=N)
    fy = np.bincount(ii_v, weights=f[:,1], minlength=N) - np.bincount(jj_v, weights=f[:,1], minlength=N)
    forces = np.stack([fx, fy], axis=1).astype(pos.dtype, copy=False)

    phi = gij * aiv
    cnt = max(phi.size, 1)
    Jn  = float(phi.sum() / cnt)
    rn  = float(rv.sum()  / cnt)

    return forces, Jn, rn, phi, valid


# =========================
# Unified external interface
# =========================
def compute_forces_gpu(
    gpu, positions, N, alpha, beta, v, r0, PT, delta,
    keep_full_matrix=True, dtype=np.float32, eps=1e-8,
    i_idx=None, j_idx=None, ecap_factor=1.25
):
    """
    Unified entry point.
    - When i_idx/j_idx are provided, use "edge-based mode" (recommended, JAX will use E-cap padding to stabilize JIT).
      Returns: forces(dev/np), Jn(float), rn(float), phi_edges(dev/np), valid_mask(dev/np)
      Here, the length of phi/valid == the original number of edges E_used (i.e., aligned with i_idx/j_idx).
    - Otherwise, use the old "full-matrix mode" for compatibility, directly calling your original kernels_*.
    """
    # ---- Edge-based mode ----
    if (i_idx is not None) and (j_idx is not None):
        i_idx = _ensure_tuple(i_idx)
        j_idx = _ensure_tuple(j_idx)
        if gpu.backend == "jax":
            return _edge_path_jax(gpu, positions, i_idx, j_idx, N,
                                  alpha, beta, v, r0, PT, delta, dtype, eps, ecap_factor)
        elif gpu.backend == "cupy":
            return _edge_path_cupy(gpu, positions, i_idx, j_idx, N,
                                   alpha, beta, v, r0, PT, delta, dtype, eps)
        else:
            return _edge_path_numpy(positions, i_idx, j_idx, N,
                                    alpha, beta, v, r0, PT, delta, dtype, eps)

    # ---- Full-matrix fallback (old interface) ----
    if gpu.backend == "jax":
        key = (alpha, beta, v, r0, PT, delta, bool(keep_full_matrix), float(eps))
        ker = _JAX_KERNEL_CACHE_FULL.get(key)
        if ker is None:
            ker = create_jax_kernel(gpu.jnp, gpu.jax, alpha, beta, v, r0, PT, delta,
                                    keep_full_matrix=keep_full_matrix, eps=eps)
            _JAX_KERNEL_CACHE_FULL[key] = ker
        return ker(positions)

    if gpu.backend == "cupy":
        return compute_forces_cupy_optimized(
            gpu.cp, positions, N, alpha, beta, v, r0, PT, delta,
            keep_full_matrix=keep_full_matrix, dtype=dtype, eps=eps
        )

    # NumPy fallback
    return compute_forces_numpy_optimized(
        gpu.to_host(positions), N, alpha, beta, v, r0, PT, delta,
        keep_full_matrix=keep_full_matrix, dtype=dtype, eps=eps
    )
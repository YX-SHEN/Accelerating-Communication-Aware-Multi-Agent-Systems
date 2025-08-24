# src/gpu/kernels_jax.py
# -*- coding: utf-8 -*-
def create_jax_kernel(jnp, jax, alpha, beta, v, r0, PT, delta,
                      keep_full_matrix=True, eps=1e-8):
    """Returns a @jit JAX kernel (input positions -> output forces, Jn, rn, comm)."""
    @jax.jit
    def kernel(positions):
        N = positions.shape[0]
        diff = positions[:, None, :] - positions[None, :, :]   # [N,N,2]
        d2   = jnp.sum(diff*diff, axis=2)
        rij  = jnp.sqrt(jnp.maximum(d2, eps))

        eye  = jnp.eye(N, dtype=bool)
        valid= jnp.logical_and(rij > eps, ~eye)

        rrat = rij / r0
        aij  = jnp.exp(-alpha * (2**delta - 1) * rrat**v)
        conn = jnp.logical_and(valid, aij >= PT)

        gij  = rij / jnp.sqrt(rij*rij + r0**2)

        num  = (-beta * v * rij**(v+2) - beta * v * (r0**2) * (rij**v) + r0**(v+2))
        den  = jnp.sqrt((rij*rij + r0**2)**3) + eps
        rho  = num * jnp.exp(-beta * (rij / r0)**v) / den

        eij  = diff / (rij[:, :, None] + eps)
        fcontrib = (conn[:, :, None] * rho[:, :, None]) * eij
        forces = jnp.sum(fcontrib, axis=1)

        phi_all = gij * aij
        phi_mat = (phi_all * conn) if keep_full_matrix else jnp.array(0.0, dtype=positions.dtype)

        denom = jnp.maximum(jnp.sum(conn), 1)
        Jn = jnp.sum(phi_all * conn) / denom
        rn = jnp.sum(rij * conn) / denom

        return forces, Jn, rn, phi_mat, conn

    return kernel
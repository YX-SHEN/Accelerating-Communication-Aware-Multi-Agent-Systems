# src/gpu/kernels_jax.py
# -*- coding: utf-8 -*-
def create_jax_kernel(jnp, jax, alpha, beta, v, r0, PT, delta,
                      keep_full_matrix=True, eps=1e-8):
    """返回一个 @jit 的 JAX 核（输入 positions -> 输出 forces, Jn, rn, comm）。"""
    @jax.jit
    def kernel(positions):
        # —— 与 positions 保持同一精度（f32 / f64）——
        dtype  = positions.dtype
        alpha_ = jnp.asarray(alpha, dtype=dtype)
        beta_  = jnp.asarray(beta,  dtype=dtype)
        v_     = jnp.asarray(v,     dtype=dtype)
        r0_    = jnp.asarray(r0,    dtype=dtype)
        PT_    = jnp.asarray(PT,    dtype=dtype)
        delta_ = jnp.asarray(delta, dtype=dtype)
        eps_   = jnp.asarray(eps,   dtype=dtype)

        two   = jnp.asarray(2.0, dtype=dtype)
        one   = jnp.asarray(1.0, dtype=dtype)
        three = jnp.asarray(3.0, dtype=dtype)

        N    = positions.shape[0]
        diff = positions[:, None, :] - positions[None, :, :]   # [N,N,2]
        d2   = jnp.sum(diff * diff, axis=2)                    # [N,N]
        rij  = jnp.sqrt(jnp.maximum(d2, eps_))                 # 避免 sqrt(0)

        eye   = jnp.eye(N, dtype=jnp.bool_)
        valid = jnp.logical_and(rij > eps_, ~eye)

        # a_ij 与连边
        exp2d_1 = jnp.power(two, delta_) - one
        rrat    = rij / r0_
        aij     = jnp.exp(-alpha_ * exp2d_1 * jnp.power(rrat, v_))
        conn    = jnp.logical_and(valid, aij >= PT_)

        # g_ij
        gij = rij / jnp.sqrt(rij * rij + r0_ * r0_)

        # ρ_ij（与 CPU 一致）
        rv_v  = jnp.power(rij, v_)
        rv_v2 = rv_v * rij * rij
        r0_2  = r0_ * r0_
        r0_v2 = jnp.power(r0_, v_ + jnp.asarray(2.0, dtype=dtype))

        num = (-beta_ * v_ * rv_v2) - (beta_ * v_ * r0_2 * rv_v) + r0_v2
        den = jnp.sqrt(jnp.power(rij * rij + r0_2, three)) + eps_
        rho = num * jnp.exp(-beta_ * jnp.power(rij / r0_, v_)) / den

        # e_ij 与力累计
        eij      = diff / (rij[:, :, None] + eps_)
        fcontrib = (conn[:, :, None].astype(dtype) * rho[:, :, None]) * eij
        forces   = jnp.sum(fcontrib, axis=1).astype(dtype)

        # 统计量与（可选）全矩阵
        phi_all = (gij * aij).astype(dtype)
        if keep_full_matrix:
            phi_mat = (phi_all * conn.astype(dtype)).astype(dtype)
        else:
            phi_mat = jnp.array(0.0, dtype=dtype)

        denom = jnp.maximum(jnp.sum(conn), jnp.asarray(1, dtype=conn.dtype))
        Jn = jnp.sum(phi_all * conn) / denom
        rn = jnp.sum(rij * conn) / denom
        return forces, Jn, rn, phi_mat, conn

    return kernel
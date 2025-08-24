# src/common/utils.py
import numpy as np
import config  # ← 关键：从 config 里拿 Real / 常量

# 统一 dtype/常量
Real = config.Real
EPS_DIV = config.EPS_DIV  # 这里备用；e_ij 的分母保护一般在 algorithms 里用

def calculate_distance(agent_i, agent_j):
    """
    Calculate the Euclidean distance between two agents.
    Returns a scalar with dtype == config.Real.
    """
    ai = np.asarray(agent_i, dtype=Real)
    aj = np.asarray(agent_j, dtype=Real)
    dx = ai[0] - aj[0]
    dy = ai[1] - aj[1]
    # 输入是 Real，np.sqrt 输出也会保持 Real
    return np.sqrt(dx * dx + dy * dy)

def calculate_aij(alpha, delta, rij, r0, v):
    """
    a_ij = exp( -alpha * (2^delta - 1) * (rij / r0)^v )
    所有参与计算的量请传入 config.Real 类型。
    """
    two = Real(2)
    one = Real(1)
    return np.exp(-alpha * (two ** delta - one) * (rij / r0) ** v)

def calculate_gij(rij, r0):
    """
    g_ij = rij / sqrt(rij^2 + r0^2)
    """
    s = rij * rij + r0 * r0
    return rij / np.sqrt(s)

def calculate_rho_ij(beta, v, rij, r0):
    """
    计算 ρ_ij（φ 的径向导数的缩放版本），保持数值稳定与 dtype 一致。
    原式中的 (rij^2 + r0^2)^(3/2) 用 s * sqrt(s) 实现，避免隐式升精度。
    """
    # 预计算
    s = rij * rij + r0 * r0             # rij^2 + r0^2
    sqrt_s = np.sqrt(s)                  # sqrt(rij^2 + r0^2)
    den = s * sqrt_s                     # (rij^2 + r0^2)^(3/2)

    rij_v   = rij ** v                   # rij^v
    rij_vp2 = rij ** (v + Real(2))       # rij^(v+2)
    r0_sq   = r0 * r0
    r0_vp2  = r0 ** (v + Real(2))

    exp_term = np.exp(-beta * (rij / r0) ** v)

    num = (-beta * v * rij_vp2
           - beta * v * r0_sq * rij_v
           + r0_vp2) * exp_term

    return num / den

def find_closest_agent(swarm_position, swarm_centroid):
    """
    Find index of the agent closest to the centroid.
    返回 int 索引；中间计算保持 Real。
    """
    P = np.asarray(swarm_position, dtype=Real)
    c = np.asarray(swarm_centroid, dtype=Real)
    diffs = P - c
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    return int(np.argmin(dists))
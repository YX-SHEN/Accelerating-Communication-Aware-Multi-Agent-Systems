# src/common/algorithms.py
import numpy as np
import matplotlib.pyplot as plt
import time
from src.common import utils
from src.common import plotting
import config

# === 精度 & 数值开关（来自 config） ===
Real    = getattr(config, "DTYPE", np.float32)  # np.float32 或 np.float64
DT      = getattr(config, "DT", 1e-2)           # 统一步长
EPS_DIV = getattr(config, "EPS_DIV", 1e-6)      # 除零保护

def calculate_Jn(communication_qualities_matrix, neighbor_agent_matrix, PT):
    """
    计算 Jn（仅统计满足阈值的有向边）
    """
    PT_local = Real(PT)
    total_communication_quality = Real(0)
    total_neighbors = 0
    swarm_size = communication_qualities_matrix.shape[0]

    for i in range(swarm_size):
        for j in [x for x in range(swarm_size) if x != i]:
            if neighbor_agent_matrix[i, j] >= PT_local:
                total_communication_quality += communication_qualities_matrix[i, j]
                total_neighbors += 1

    return float(total_communication_quality / total_neighbors) if total_neighbors > 0 else 0.0


def calculate_rn(distances_matrix, neighbor_agent_matrix, PT):
    """
    计算 rn（仅统计满足阈值的有向边）
    """
    PT_local = Real(PT)
    total_distance = Real(0)
    total_neighbors = 0
    swarm_size = distances_matrix.shape[0]

    for i in range(swarm_size):
        for j in [x for x in range(swarm_size) if x != i]:
            if neighbor_agent_matrix[i, j] >= PT_local:
                total_distance += distances_matrix[i, j]
                total_neighbors += 1

    return float(total_distance / total_neighbors) if total_neighbors > 0 else 0.0


def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    基线 O(N^2) 实现（Gauss-Seidel 更新），已对齐全局精度开关 Real/DT/EPS_DIV
    """
    LOG_EVERY = getattr(config, "LOG_EVERY", 20)

    # --- 统一精度：位置 + 工作矩阵都转成 Real ---
    swarm_position = np.asarray(swarm_position, dtype=Real)

    swarm_control_ui = np.zeros((swarm_size, 2), dtype=Real)
    communication_qualities_matrix = np.zeros((swarm_size, swarm_size), dtype=Real)
    distances_matrix = np.zeros((swarm_size, swarm_size), dtype=Real)
    neighbor_agent_matrix = np.zeros((swarm_size, swarm_size), dtype=Real)

    # 性能指标
    Jn = []
    rn = []

    # 计时与收敛标记
    start_time = time.time()
    t_elapsed = []
    Jn_converged = False

    # 常量本地化到 Real，避免混精度触发隐式 upcast
    deltaR = Real(getattr(config, "DELTA", 2))
    alphaR = Real(alpha)
    betaR  = Real(beta)
    vR     = Real(v)
    r0R    = Real(r0)
    PTR    = Real(PT)
    epsR   = Real(EPS_DIV)
    dtR    = Real(DT)

    # 主循环
    for iter in range(max_iter):
        for i in range(swarm_size):
            # 累加来自所有 j 的作用
            for j in [x for x in range(swarm_size) if x != i]:
                rij = utils.calculate_distance(swarm_position[i], swarm_position[j])   # dtype 跟随输入 Real
                aij = utils.calculate_aij(alphaR, deltaR, rij, r0R, vR)
                gij = utils.calculate_gij(rij, r0R)

                if aij >= PTR:
                    rho_ij = utils.calculate_rho_ij(betaR, vR, rij, r0R)
                else:
                    rho_ij = Real(0)

                qi = swarm_position[i, :]
                qj = swarm_position[j, :]
                eij = (qi - qj) / (rij + epsR)

                swarm_control_ui[i, :] += rho_ij * eij

                # 统计矩阵（给 Jn/rn 用）
                distances_matrix[i, j] = rij
                distances_matrix[j, i] = rij
                neighbor_agent_matrix[i, j] = aij
                neighbor_agent_matrix[j, i] = aij

                if aij >= PTR:
                    phi_rij = gij * aij
                    communication_qualities_matrix[i, j] = phi_rij
                    communication_qualities_matrix[j, i] = phi_rij
                else:
                    communication_qualities_matrix[i, j] = Real(0)
                    communication_qualities_matrix[j, i] = Real(0)

            # —— Gauss-Seidel：i 的内层完成后，立即更新位置（用统一 dt）——
            swarm_position[i, :] += dtR * swarm_control_ui[i, :]
            swarm_control_ui[i, :] = Real(0)

        # 性能指标与判停
        Jn_new = calculate_Jn(communication_qualities_matrix, neighbor_agent_matrix, PTR)
        rn_new = calculate_rn(distances_matrix, neighbor_agent_matrix, PTR)

        Jn.append(round(float(Jn_new), 4))
        rn.append(round(float(rn_new), 4))
        t_elapsed.append(time.time() - start_time)

        # 动画 / 可视化
        plotting.plot_figures_task1(
            axs, t_elapsed, Jn, rn, swarm_position, float(PTR),
            communication_qualities_matrix, swarm_size, swarm_paths,
            node_colors, line_colors
        )

        if iter % LOG_EVERY == 0:
            print(f"[it={iter}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f}")

        # 原有收敛判据（保持不变）
        if len(Jn) > 19 and len(set(Jn[-20:])) == 1:
            if not Jn_converged:
                if getattr(config, "FORCE_FIXED_ITERS", False):
                    print(f"[info] Jn would converge at t={round(t_elapsed[-1], 2)}s "
                          f"after {iter - 20} iterations, but fixed-iters mode keeps running.")
                else:
                    print(f"Formation completed: Jn converged in {round(t_elapsed[-1], 2)}s "
                          f"after {iter - 20} iterations.")
                Jn_converged = True
            if not getattr(config, "FORCE_FIXED_ITERS", False):
                break

    return Jn, rn, swarm_position, t_elapsed, communication_qualities_matrix, neighbor_agent_matrix
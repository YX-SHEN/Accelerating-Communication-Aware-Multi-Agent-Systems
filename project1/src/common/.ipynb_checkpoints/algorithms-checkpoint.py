# src/common/algorithms.py
import numpy as np
import matplotlib.pyplot as plt
import time
from src.common import utils
from src.common import plotting
import config

def calculate_Jn(communication_qualities_matrix, neighbor_agent_matrix, PT):
    '''
         Calculate the Jn (average communication performance indicator) value

        Parameters:
            communication_qualities_matrix (numpy.ndarray): The communication qualities matrix among agents
            neighbor_agent_matrix (numpy.ndarray): The neighbor_agent matrix which is adjacency matrix of aij value
            PT (float): The reception probability threshold

        Returns:
            float: The calculated Jn value
    '''
    # --- ADD THESE LINES ---
    total_communication_quality = 0
    total_neighbors = 0
    swarm_size = communication_qualities_matrix.shape[0]
    # -----------------------

    for i in range(swarm_size):
        for j in [x for x in range(swarm_size) if x != i]:
            if neighbor_agent_matrix[i, j] >= PT:
                total_communication_quality += communication_qualities_matrix[i, j]
                total_neighbors += 1
    return total_communication_quality / total_neighbors if total_neighbors > 0 else 0

def calculate_rn(distances_matrix, neighbor_agent_matrix, PT):
    '''
        Calculate the rn (average neighboring distance performance indicator) value

        Parameters:
            distances_matrix (numpy.ndarray): The distances matrix among agents
            neighbor_agent_matrix (numpy.ndarray): The neighbor_agent matrix which is adjacency matrix of aij value
            PT (float): The reception probability threshold

        Returns:
            float: The calculated rn value
    '''
    total_distance = 0
    total_neighbors = 0
    swarm_size = distances_matrix.shape[0]
    for i in range(swarm_size):
        for j in [x for x in range(swarm_size) if x != i]:
            if neighbor_agent_matrix[i, j] >= PT:
                total_distance += distances_matrix[i, j]
                total_neighbors += 1
    return total_distance / total_neighbors if total_neighbors > 0 else 0


def run_simulation(axs, fig, swarm_position, max_iter, swarm_size, alpha, beta, v, r0, PT, swarm_paths, node_colors,
                   line_colors):
    '''
        Runs the formation control simulation for a given number of iterations.
        ...
    '''

    LOG_EVERY = getattr(config, "LOG_EVERY", 20)

    # Initialize data structures for the simulation
    swarm_control_ui = np.zeros((swarm_size, 2))
    communication_qualities_matrix = np.zeros((swarm_size, swarm_size))
    distances_matrix = np.zeros((swarm_size, swarm_size))
    neighbor_agent_matrix = np.zeros((swarm_size, swarm_size))

    # Initialize performance indicators
    Jn = []
    rn = []

    # Initialize timer and convergence flag
    start_time = time.time()
    t_elapsed = []
    Jn_converged = False

    # 获取 delta 参数
    delta = config.DELTA

    # Main simulation loop
    for iter in range(max_iter):

        # --- Your original O(N^2) algorithm core loop ---
        # --- Corrected O(N^2) algorithm core loop ---
        for i in range(swarm_size):
            # First, the inner loop calculates the total force from ALL neighbors
            for j in [x for x in range(swarm_size) if x != i]:
                rij = utils.calculate_distance(swarm_position[i], swarm_position[j])
                aij = utils.calculate_aij(alpha, delta, rij, r0, v)
                gij = utils.calculate_gij(rij, r0)

                if aij >= PT:
                    rho_ij = utils.calculate_rho_ij(beta, v, rij, r0)
                else:
                    rho_ij = 0

                qi = swarm_position[i, :]
                qj = swarm_position[j, :]
                epsilon = 1e-6
                eij = (qi - qj) / (rij + epsilon)

                # Accumulate the force from neighbor j
                swarm_control_ui[i, :] += rho_ij * eij

                # --- The matrix updates can stay inside the j loop ---
                # 始终记录距离 & aij（给 rn/Jn 统计）
                distances_matrix[i, j] = rij
                distances_matrix[j, i] = rij
                neighbor_agent_matrix[i, j] = aij
                neighbor_agent_matrix[j, i] = aij

                # 只有“达阈值”的边才在可视化矩阵里留值，否则置 0
                if aij >= PT:
                    phi_rij = gij * aij
                    communication_qualities_matrix[i, j] = phi_rij
                    communication_qualities_matrix[j, i] = phi_rij
                else:
                    communication_qualities_matrix[i, j] = 0.0
                    communication_qualities_matrix[j, i] = 0.0

            # !!! CORRECT LOCATION !!!
            # Now that the j-loop is finished, update the position ONCE for agent i
            # using the SUM of all forces.
            step_size = 0.01
            swarm_position[i, :] += step_size * swarm_control_ui[i, :]

            # Reset the control input for the next agent i in the next iteration
            swarm_control_ui[i, :] = 0

        # --- Performance indicators and convergence check ---
        Jn_new = calculate_Jn(communication_qualities_matrix, neighbor_agent_matrix, PT)
        rn_new = calculate_rn(distances_matrix, neighbor_agent_matrix, PT)

        Jn.append(round(Jn_new, 4))
        rn.append(round(rn_new, 4))
        t_elapsed.append(time.time() - start_time)

        # 核心动画逻辑：在每次迭代后调用绘图函数
        plotting.plot_figures_task1(
            axs, t_elapsed, Jn, rn, swarm_position, PT,
            communication_qualities_matrix, swarm_size, swarm_paths,
            node_colors, line_colors
        )
        # --- 新增：统一风格的迭代日志（不影响计算） ---
        if iter % LOG_EVERY == 0:
            print(f"[it={iter}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f}")

        # --- 原判停逻辑 + fixed-iters 兼容 ---
        if len(Jn) > 19 and len(set(Jn[-20:])) == 1:
            if not Jn_converged:
                if getattr(config, "FORCE_FIXED_ITERS", False):
                    print(f"[info] Jn would converge at t={round(t_elapsed[-1], 2)}s "
                          f"after {iter - 20} iterations, but fixed-iters mode keeps running.")
                else:
                    print(f"Formation completed: Jn converged in {round(t_elapsed[-1], 2)}s "
                          f"after {iter - 20} iterations.")
                Jn_converged = True  # 防止重复打印
            if not getattr(config, "FORCE_FIXED_ITERS", False):
                break

    return Jn, rn, swarm_position, t_elapsed, communication_qualities_matrix, neighbor_agent_matrix
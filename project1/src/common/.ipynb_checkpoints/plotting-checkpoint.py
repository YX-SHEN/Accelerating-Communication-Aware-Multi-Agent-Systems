# src/common/plotting.py

import matplotlib.pyplot as plt
import numpy as np

def plot_figures_task1(axs, t_elapsed, Jn, rn, swarm_position, PT,
                       communication_qualities_matrix, swarm_size, swarm_paths,
                       node_colors, line_colors):
    """

        Plot 4 figures (Formation Scene, Swarm Trajectories, Jn Performance, rn Performance)

        Parameters:
            axs (numpy.ndarray): The axes of the figure
            t_elapsed (list): The elapsed time
            Jn (list): The Jn values
            rn (list): The rn values
            swarm_position (numpy.ndarray): The positions of the swarm
            PT (float): The reception probability threshold
            communication_qualities_matrix (numpy.ndarray): The communication qualities matrix among agents
            swarm_size (int): The number of agents in the swarm
            swarm_paths (list): The paths of the swarm
            node_colors (list): The colors of the nodes
            line_colors (list): The colors of the lines

        Returns:
            None

    """
    # 清除旧的图像内容
    for ax in axs.flatten():
        ax.clear()

    # --- 1. 绘制“编队场景图”(axs[0, 0]) - 核心修改在这里 ---
    axs[0, 0].set_title('Formation Scene')
    axs[0, 0].set_xlabel('$x$')
    axs[0, 0].set_ylabel('$y$', rotation=0)

    # 定义一个阈值来区分大小规模
    VISUALIZATION_THRESHOLD = 40

    if swarm_size < VISUALIZATION_THRESHOLD:
        # --- 小规模绘图逻辑：显示所有细节 ---
        # a. 绘制连接线
        for i in range(swarm_size):
            for j in range(i + 1, swarm_size):
                if communication_qualities_matrix[i, j] > 0:
                    axs[0, 0].plot(*zip(swarm_position[i], swarm_position[j]),
                                   color=line_colors[i, j], linestyle='--', alpha=0.8)

        # b. 绘制智能体（点可以画大一点）和它们的ID
        axs[0, 0].scatter(swarm_position[:, 0], swarm_position[:, 1],
                          color=node_colors, s=50, zorder=3)  # zorder确保点在最上层
        for i in range(swarm_size):
            axs[0, 0].text(swarm_position[i, 0] + 0.5, swarm_position[i, 1] + 0.5, f'{i + 1}', fontsize=9)

    else:
        # --- 大规模绘图逻辑：只显示轮廓 ---
        # 只绘制智能体的当前位置，点画得小一些，并带透明度
        axs[0, 0].scatter(swarm_position[:, 0], swarm_position[:, 1],
                          color=node_colors, s=15, alpha=0.7)

    axs[0, 0].axis('equal')

    # --- 2. 绘制“轨迹图”(axs[0, 1]) ---
    # 大规模时建议只画当前位置，或抽样画轨迹
    axs[0, 1].set_title('Swarm Trajectories')
    axs[0, 1].set_xlabel('$x$')
    axs[0, 1].set_ylabel('$y$', rotation=0)
    # 为了保持清晰，这里我们只画出当前位置
    axs[0, 1].scatter(swarm_position[:, 0], swarm_position[:, 1],
                      color=node_colors, s=15)
    axs[0, 1].axis('equal')

    # --- 3. 绘制性能指标图 (axs[1, 0] 和 axs[1, 1]) ---
    # 这两张图不受规模影响，保持原样即可
    # Jn 性能图
    axs[1, 0].set_title('Average Communication Performance Indicator')
    axs[1, 0].plot(t_elapsed, Jn)
    axs[1, 0].set_xlabel('$t(s)$')
    axs[1, 0].set_ylabel('$J_n$', rotation=0, labelpad=20)
    if Jn:  # 避免列表为空时出错
        axs[1, 0].text(t_elapsed[-1], Jn[-1], f'Jn={Jn[-1]:.4f}', ha='right', va='top')

    # rn 性能图
    axs[1, 1].set_title('Average Distance Performance Indicator')
    axs[1, 1].plot(t_elapsed, rn)
    axs[1, 1].set_xlabel('$t(s)$')
    axs[1, 1].set_ylabel('$r_n$', rotation=0, labelpad=20)
    if rn:  # 避免列表为空时出错
        axs[1, 1].text(t_elapsed[-1], rn[-1], f'$r_n$={rn[-1]:.4f}', ha='right', va='top')

    # 统一布局和显示
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)


def plot_figures_task2(axs, t_elapsed, Jn, rn, swarm_position, swarm_destination, PT, communication_qualities_matrix,
                       swarm_size, swarm_paths, node_colors, line_colors):
    '''
        Plot 4 figures (Formation Scene, Swarm Trajectories, Jn Performance, rn Performance)

        Parameters:
            axs (numpy.ndarray): The axes of the figure
            t_elapsed (list): The elapsed time
            Jn (list): The Jn values
            rn (list): The rn values
            swarm_position (numpy.ndarray): The positions of the swarm
            swarm_destination (list): The destination of the swarm
            PT (float): The reception probability threshold
            communication_qualities_matrix (numpy.ndarray): The communication qualities matrix among agents
            swarm_size (int): The number of agents in the swarm
            swarm_paths (list): The paths of the swarm
            node_colors (list): The colors of the nodes
            line_colors (list): The colors of the lines

        Returns:
            None
    '''
    for ax in axs.flatten():
        ax.clear()

    ########################
    # Plot formation scene #
    ########################
    axs[0, 0].set_title('Formation Scene')
    axs[0, 0].set_xlabel('$x$')
    axs[0, 0].set_ylabel('$y$', rotation=0)

    # Plot the nodes
    for i in range(swarm_position.shape[0]):
        axs[0, 0].scatter(*swarm_position[i], color=node_colors[i])

    # Plot the destination
    axs[0, 0].plot(*swarm_destination, marker='s', markersize=10, color='none', mec='black')
    axs[0, 0].text(swarm_destination[0], swarm_destination[1] + 3, 'Destination', ha='center', va='bottom')

    # Plot the edges
    for i in range(swarm_position.shape[0]):
        for j in range(i + 1, swarm_position.shape[0]):
            if communication_qualities_matrix[i, j] > 0:
                axs[0, 0].plot(*zip(swarm_position[i], swarm_position[j]), color=line_colors[i, j], linestyle='--')

    axs[0, 0].axis('equal')

    ###########################
    # Plot swarm trajectories #
    ###########################
    axs[0, 1].set_title('Swarm Trajectories')
    axs[0, 1].set_xlabel('$x$')
    axs[0, 1].set_ylabel('$y$', rotation=0)

    # Store the current swarm positions
    swarm_paths.append(swarm_position.copy())

    # Convert the list of positions to a numpy array
    trajectory_array = np.array(swarm_paths)

    # Plot the trajectories
    for i in range(swarm_position.shape[0]):
        axs[0, 1].plot(trajectory_array[:, i, 0], trajectory_array[:, i, 1], color=node_colors[i])

        # Calculate the differences between consecutive points
        dx = np.diff(trajectory_array[::swarm_size, i, 0])
        dy = np.diff(trajectory_array[::swarm_size, i, 1])

        # Normalize the vectors where dx and dy are not both zero
        dx_norm = np.zeros_like(dx)
        dy_norm = np.zeros_like(dy)
        for j in range(len(dx)):
            if dx[j] != 0 or dy[j] != 0:
                norm = np.sqrt(dx[j] ** 2 + dy[j] ** 2)
                dx_norm[j] = dx[j] / norm
                dy_norm[j] = dy[j] / norm

        # Scale the vectors by a constant factor
        scale_factor = 2
        dx_scaled = dx_norm * scale_factor
        dy_scaled = dy_norm * scale_factor

        # Plot the trajectory with larger arrows
        axs[0, 1].quiver(trajectory_array[::swarm_size, i, 0][:-1], trajectory_array[::swarm_size, i, 1][:-1],
                         dx_scaled, dy_scaled, color=node_colors[i], scale_units='xy', angles='xy', scale=1,
                         headlength=10, headaxislength=9, headwidth=8)

    # Plot the initial positions
    axs[0, 1].scatter(trajectory_array[0, :, 0], trajectory_array[0, :, 1], color=node_colors)

    # Plot the destination
    axs[0, 1].plot(*swarm_destination, marker='s', markersize=10, color='none', mec='black')
    axs[0, 1].text(swarm_destination[0], swarm_destination[1] + 3, 'Destination', ha='center', va='bottom')

    #######################
    # Plot Jn performance #
    #######################
    axs[1, 0].set_title('Average Communication Performance Indicator')
    axs[1, 0].plot(t_elapsed, Jn)
    axs[1, 0].set_xlabel('$t(s)$')
    axs[1, 0].set_ylabel('$J_n$', rotation=0, labelpad=20)
    axs[1, 0].text(t_elapsed[-1], Jn[-1], 'Jn={:.4f}'.format(Jn[-1]), ha='right', va='top')

    #######################
    # Plot rn performance #
    #######################
    axs[1, 1].set_title('Average Distance Performance Indicator')
    axs[1, 1].plot(t_elapsed, rn)
    axs[1, 1].set_xlabel('$t(s)$')
    axs[1, 1].set_ylabel('$r_n$', rotation=0, labelpad=20)
    axs[1, 1].text(t_elapsed[-1], rn[-1], '$r_n$={:.4f}'.format(rn[-1]), ha='right', va='top')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
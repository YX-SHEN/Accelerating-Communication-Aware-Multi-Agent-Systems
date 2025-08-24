# src/common/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import config  # 读取可视化阈值等

def _is_dense_comm(mat) -> bool:
    """只有在 comm 矩阵真的是 2D ndarray 时才认为可用于画边。"""
    return isinstance(mat, np.ndarray) and mat.ndim == 2 and mat.size > 0

def plot_figures_task1(axs, t_elapsed, Jn, rn, swarm_position, PT,
                       communication_qualities_matrix, swarm_size, swarm_paths,
                       node_colors, line_colors):
    # 清空
    for ax in axs.flatten():
        ax.clear()

    axs[0, 0].set_title('Formation Scene')
    axs[0, 0].set_xlabel('$x$')
    axs[0, 0].set_ylabel('$y$', rotation=0)

    # 可被 config 覆盖
    VISUALIZATION_THRESHOLD = getattr(config, "VISUALIZATION_THRESHOLD", 40)

    # 位置转 float 仅用于绘图（不影响数值计算）
    P = np.asarray(swarm_position, dtype=float)
    comm = communication_qualities_matrix
    dense_comm = _is_dense_comm(comm)

    # --- Formation Scene ---
    if swarm_size < VISUALIZATION_THRESHOLD:
        # 只有在 comm 是真实矩阵时才画边
        if dense_comm:
            for i in range(swarm_size):
                for j in range(i + 1, swarm_size):
                    if comm[i, j] > 0:
                        axs[0, 0].plot(*zip(P[i], P[j]),
                                       color=line_colors[i, j], linestyle='--', alpha=0.8)
        axs[0, 0].scatter(P[:, 0], P[:, 1], color=node_colors, s=50, zorder=3)
        for i in range(swarm_size):
            axs[0, 0].text(P[i, 0] + 0.5, P[i, 1] + 0.5, f'{i + 1}', fontsize=9)
    else:
        axs[0, 0].scatter(P[:, 0], P[:, 1], color=node_colors, s=15, alpha=0.7)

    axs[0, 0].axis('equal')

    # --- Swarm Trajectories（此处仅画当前位置） ---
    axs[0, 1].set_title('Swarm Trajectories')
    axs[0, 1].set_xlabel('$x$')
    axs[0, 1].set_ylabel('$y$', rotation=0)
    axs[0, 1].scatter(P[:, 0], P[:, 1], color=node_colors, s=15)
    axs[0, 1].axis('equal')

    # --- 指标 Jn ---
    axs[1, 0].set_title('Average Communication Performance Indicator')
    axs[1, 0].set_xlabel('$t(s)$')
    axs[1, 0].set_ylabel('$J_n$', rotation=0, labelpad=20)
    axs[1, 0].plot(t_elapsed, Jn)
    if len(Jn) > 0 and len(t_elapsed) > 0:
        axs[1, 0].text(float(t_elapsed[-1]), float(Jn[-1]),
                       f'Jn={float(Jn[-1]):.4f}', ha='right', va='top')

    # --- 指标 rn ---
    axs[1, 1].set_title('Average Distance Performance Indicator')
    axs[1, 1].set_xlabel('$t(s)$')
    axs[1, 1].set_ylabel('$r_n$', rotation=0, labelpad=20)
    axs[1, 1].plot(t_elapsed, rn)
    if len(rn) > 0 and len(t_elapsed) > 0:
        axs[1, 1].text(float(t_elapsed[-1]), float(rn[-1]),
                       f'$r_n$={float(rn[-1]):.4f}', ha='right', va='top')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)


def plot_figures_task2(axs, t_elapsed, Jn, rn, swarm_position, swarm_destination, PT,
                       communication_qualities_matrix, swarm_size, swarm_paths,
                       node_colors, line_colors):
    for ax in axs.flatten():
        ax.clear()

    axs[0, 0].set_title('Formation Scene')
    axs[0, 0].set_xlabel('$x$')
    axs[0, 0].set_ylabel('$y$', rotation=0)

    P = np.asarray(swarm_position, dtype=float)
    comm = communication_qualities_matrix
    dense_comm = _is_dense_comm(comm)

    # 节点
    for i in range(P.shape[0]):
        axs[0, 0].scatter(*P[i], color=node_colors[i])

    # 目标
    axs[0, 0].plot(*swarm_destination, marker='s', markersize=10, color='none', mec='black')
    axs[0, 0].text(swarm_destination[0], swarm_destination[1] + 3,
                   'Destination', ha='center', va='bottom')

    # 边（仅当 comm 是真实矩阵时）
    if dense_comm:
        for i in range(P.shape[0]):
            for j in range(i + 1, P.shape[0]):
                if comm[i, j] > 0:
                    axs[0, 0].plot(*zip(P[i], P[j]), color=line_colors[i, j], linestyle='--')

    axs[0, 0].axis('equal')

    # 轨迹
    axs[0, 1].set_title('Swarm Trajectories')
    axs[0, 1].set_xlabel('$x$')
    axs[0, 1].set_ylabel('$y$', rotation=0)

    swarm_paths.append(P.copy())
    traj = np.array(swarm_paths, dtype=float)

    for i in range(P.shape[0]):
        axs[0, 1].plot(traj[:, i, 0], traj[:, i, 1], color=node_colors[i])

        dx = np.diff(traj[::swarm_size, i, 0])
        dy = np.diff(traj[::swarm_size, i, 1])

        dx_norm = np.zeros_like(dx)
        dy_norm = np.zeros_like(dy)
        for j in range(len(dx)):
            if dx[j] != 0 or dy[j] != 0:
                norm = np.hypot(dx[j], dy[j])
                dx_norm[j] = dx[j] / norm
                dy_norm[j] = dy[j] / norm

        scale_factor = 2.0
        dx_scaled = dx_norm * scale_factor
        dy_scaled = dy_norm * scale_factor

        axs[0, 1].quiver(traj[::swarm_size, i, 0][:-1], traj[::swarm_size, i, 1][:-1],
                         dx_scaled, dy_scaled, color=node_colors[i],
                         scale_units='xy', angles='xy', scale=1,
                         headlength=10, headaxislength=9, headwidth=8)

    axs[0, 1].scatter(traj[0, :, 0], traj[0, :, 1], color=node_colors)
    axs[0, 1].plot(*swarm_destination, marker='s', markersize=10, color='none', mec='black')
    axs[0, 1].text(swarm_destination[0], swarm_destination[1] + 3, 'Destination',
                   ha='center', va='bottom')

    # 指标
    axs[1, 0].set_title('Average Communication Performance Indicator')
    axs[1, 0].set_xlabel('$t(s)$')
    axs[1, 0].set_ylabel('$J_n$', rotation=0, labelpad=20)
    axs[1, 0].plot(t_elapsed, Jn)
    if len(Jn) > 0 and len(t_elapsed) > 0:
        axs[1, 0].text(float(t_elapsed[-1]), float(Jn[-1]),
                       f'Jn={float(Jn[-1]):.4f}', ha='right', va='top')

    axs[1, 1].set_title('Average Distance Performance Indicator')
    axs[1, 1].set_xlabel('$t(s)$')
    axs[1, 1].set_ylabel('$r_n$', rotation=0, labelpad=20)
    axs[1, 1].plot(t_elapsed, rn)
    if len(rn) > 0 and len(t_elapsed) > 0:
        axs[1, 1].text(float(t_elapsed[-1]), float(rn[-1]),
                       f'$r_n$={float(rn[-1]):.4f}', ha='right', va='top')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
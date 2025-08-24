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
    # Clear the content of the old figures
    for ax in axs.flatten():
        ax.clear()

    # --- 1. Plot "Formation Scene" (axs[0, 0]) - core modifications are here ---
    axs[0, 0].set_title('Formation Scene')
    axs[0, 0].set_xlabel('$x$')
    axs[0, 0].set_ylabel('$y$', rotation=0)

    # Define a threshold to distinguish between small and large scales
    VISUALIZATION_THRESHOLD = 40

    if swarm_size < VISUALIZATION_THRESHOLD:
        # --- Small-scale plotting logic: show all details ---
        # a. Plot connection lines
        for i in range(swarm_size):
            for j in range(i + 1, swarm_size):
                if communication_qualities_matrix[i, j] > 0:
                    axs[0, 0].plot(*zip(swarm_position[i], swarm_position[j]),
                                   color=line_colors[i, j], linestyle='--', alpha=0.8)

        # b. Plot agents (points can be drawn larger) and their IDs
        axs[0, 0].scatter(swarm_position[:, 0], swarm_position[:, 1],
                          color=node_colors, s=50, zorder=3)  # zorder ensures the points are on the top layer
        for i in range(swarm_size):
            axs[0, 0].text(swarm_position[i, 0] + 0.5, swarm_position[i, 1] + 0.5, f'{i + 1}', fontsize=9)

    else:
        # --- Large-scale plotting logic: show only the outline ---
        # Only plot the current positions of the agents, with smaller points and transparency
        axs[0, 0].scatter(swarm_position[:, 0], swarm_position[:, 1],
                          color=node_colors, s=15, alpha=0.7)

    axs[0, 0].axis('equal')

    # --- 2. Plot "Trajectories" (axs[0, 1]) ---
    # For large scales, it is recommended to plot only the current positions or sample the trajectories
    axs[0, 1].set_title('Swarm Trajectories')
    axs[0, 1].set_xlabel('$x$')
    axs[0, 1].set_ylabel('$y$', rotation=0)
    # To maintain clarity, we only plot the current positions here
    axs[0, 1].scatter(swarm_position[:, 0], swarm_position[:, 1],
                      color=node_colors, s=15)
    axs[0, 1].axis('equal')

    # --- 3. Plot performance indicator graphs (axs[1, 0] and axs[1, 1]) ---
    # These two graphs are not affected by scale and can be kept as they are
    # Jn performance graph
    axs[1, 0].set_title('Average Communication Performance Indicator')
    axs[1, 0].plot(t_elapsed, Jn)
    axs[1, 0].set_xlabel('$t(s)$')
    axs[1, 0].set_ylabel('$J_n$', rotation=0, labelpad=20)
    if Jn:  # Avoid errors when the list is empty
        axs[1, 0].text(t_elapsed[-1], Jn[-1], f'Jn={Jn[-1]:.4f}', ha='right', va='top')

    # rn performance graph
    axs[1, 1].set_title('Average Distance Performance Indicator')
    axs[1, 1].plot(t_elapsed, rn)
    axs[1, 1].set_xlabel('$t(s)$')
    axs[1, 1].set_ylabel('$r_n$', rotation=0, labelpad=20)
    if rn:  # Avoid errors when the list is empty
        axs[1, 1].text(t_elapsed[-1], rn[-1], f'$r_n$={rn[-1]:.4f}', ha='right', va='top')

    # Unify layout and display
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
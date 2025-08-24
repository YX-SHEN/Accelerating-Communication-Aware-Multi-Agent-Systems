# src/common/utils.py
import numpy as np
import matplotlib.pyplot as plt


def calculate_distance(agent_i, agent_j):
    '''
        Calculate the distance between two agents

        Parameters:
            agent_i (list): The position of agent i
            agent_j (list): The position of agent j

        Returns:
            float: The distance between agent i and agent j
    '''
    return np.sqrt((agent_i[0] - agent_j[0]) ** 2 + (agent_i[1] - agent_j[1]) ** 2)


def calculate_aij(alpha, delta, rij, r0, v):
    '''
        Calculate the aij value

        Parameters:
            alpha (float): System parameter about antenna characteristics
            delta (float): The required application data rate
            rij (float): The distance between two agents
            r0 (float): Reference distance value
            v (float): Path loss exponent

        Returns:
            float: The calculated aij (communication quality in antenna far-field) value
    '''
    return np.exp(-alpha * (2 ** delta - 1) * (rij / r0) ** v)


def calculate_gij(rij, r0):
    '''
        Calculate the gij value

        Parameters:
            rij (float): The distance between two agents
            r0 (float): Reference distance value

        Returns:
            float: The calculated gij (communication quality in antenna near-field) value
    '''
    return rij / np.sqrt(rij ** 2 + r0 ** 2)


def calculate_rho_ij(beta, v, rij, r0):
    '''
        Calculate the rho_ij (the derivative of phi_ij) value

        Parameters:
            beta (float): alpha * (2**delta - 1)
            v (float): Path loss exponent
            rij (float): The distance between two agents
            r0 (float): Reference distance value

        Returns:
            float: The calculated rho_ij value
    '''
    return (-beta * v * rij ** (v + 2) - beta * v * (r0 ** 2) * (rij ** v) + r0 ** (v + 2)) * np.exp(
        -beta * (rij / r0) ** v) / np.sqrt((rij ** 2 + r0 ** 2) ** 3)


def find_closest_agent(swarm_position, swarm_centroid):
    '''
    Find the index of the agent with the minimum distance to the destination

    Parameters:
        swarm_position (numpy.ndarray): The positions of the swarm
        swarm_centroid (numpy.ndarray): The centroid of the swarm

    Returns:
        int: The index of the agent with the minimum distance to the destination
    '''
    # Calculate the Euclidean distance from each agent to the destination
    distances_matrix = np.sqrt(np.sum((swarm_position - swarm_centroid) ** 2, axis=1))

    # Find the index of the agent with the minimum distance
    closest_agent_index = np.argmin(distances_matrix)

    return closest_agent_index




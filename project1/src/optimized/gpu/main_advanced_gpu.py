# src/optimized/gpu/main_advanced_gpu.py
# -*- coding: utf-8 -*-
"""
Standalone entry point: Advanced algorithm + GPU version
Maintains the same output and save format as other main scripts
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import config

# Key: Only import the "Advanced Algorithm + GPU" thin wrapper
from src.gpu.algorithms_advanced_gpu import run_simulation


def main():
    """
    Main program to run the "Advanced + GPU" algorithm and automatically save the results.
    Note: The algorithm logic is consistent with the advanced CPU version, only the per-edge calculations are moved to the GPU.
    """
    print("\n--- Loading [V4: Advanced Algorithm + GPU] engine ---\n")

    # --- 1) Create a dedicated results folder with a timestamp (naming style consistent with other main scripts) ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_advanced_gpu_grid{config.GRID_SIZE}_noise{getattr(config, 'noise_magnitude', 'N/A')}"
    run_folder = os.path.join("results", run_name)
    os.makedirs(run_folder, exist_ok=True)
    print(f"Results for this run will be saved in: {run_folder}")

    # --- 2) Initialize data required for the simulation (identical to other main scripts) ---
    swarm_position = config.SWARM_INITIAL_POSITIONS.copy()
    swarm_size = swarm_position.shape[0]
    line_colors = np.random.rand(swarm_size, swarm_size, 3)
    swarm_paths = []

    # Standard 2x2 visualization layout + interactive refresh
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.ion()

    # --- 3) Call the advanced GPU algorithm (interface consistent with other versions) ---
    final_Jn, final_rn, final_positions, t_elapsed, final_comm_matrix, _ = run_simulation(
        axs=axs,
        fig=fig,
        swarm_position=swarm_position,
        max_iter=config.MAX_ITER,
        swarm_size=swarm_size,
        alpha=config.ALPHA,
        beta=config.BETA,
        v=config.V,
        r0=config.R0,
        PT=config.PT,
        swarm_paths=swarm_paths,
        node_colors=config.NODE_COLORS,
        line_colors=line_colors
    )

    # --- 4) Save results (filenames and fields aligned with baseline/optimized versions) ---
    print("Simulation finished, saving results...")

    plot_filepath      = os.path.join(run_folder, "final_plot.png")
    positions_filepath = os.path.join(run_folder, "final_positions.csv")
    jn_filepath        = os.path.join(run_folder, "jn_history.txt")
    rn_filepath        = os.path.join(run_folder, "rn_history.txt")
    time_filepath      = os.path.join(run_folder, "time_elapsed.txt")
    config_filepath    = os.path.join(run_folder, "config_summary.txt")

    # Plot and data
    fig.savefig(plot_filepath, dpi=300)
    np.savetxt(positions_filepath, final_positions, delimiter=",", header="x,y")
    np.savetxt(jn_filepath,  np.array(final_Jn))
    np.savetxt(rn_filepath,  np.array(final_rn))
    np.savetxt(time_filepath, np.array(t_elapsed))

    # Configuration summary and convergence/completion information (distinguishing between fixed steps vs. early convergence)
    with open(config_filepath, "w") as f:
        f.write(f"--- Simulation Run Summary ---\n")
        f.write(f"Algorithm Version: advanced_gpu\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"GRID_SIZE: {config.GRID_SIZE}\n")
        f.write(f"SWARM_SIZE: {config.SWARM_SIZE}\n")
        f.write(f"NOISE_MAGNITUDE: {getattr(config, 'noise_magnitude', 'N/A')}\n")
        f.write(f"STEP_SIZE: {getattr(config, 'STEP_SIZE', 'N/A')}\n")

        final_time = t_elapsed[-1] if t_elapsed else 0.0
        iters_done = len(final_Jn)
        max_iter = getattr(config, "MAX_ITER", iters_done)
        fixed_mode = bool(getattr(config, "FORCE_FIXED_ITERS", False))

        f.write("\n--- Run Result ---\n")
        if fixed_mode and iters_done >= max_iter:
            # Fixed steps: Do not claim "convergence", but rather "completed fixed steps"
            f.write(f"Completed fixed iterations: {iters_done} steps in {round(final_time, 2)} seconds\n")
        else:
            # Non-fixed steps: Most likely converged early (or terminated early due to numerical stability protection, depending on algorithm output)
            f.write(f"Converged in: {round(final_time, 2)} seconds\n")
            f.write(f"Converged after: {iters_done} iterations\n")
    print(f"All results have been successfully saved to {run_folder}")

    # --- 5) Display the final plot (keep window open) ---
    plt.ioff()
    plt.show()


import argparse
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixed-iters', action='store_true', help='Run exactly MAX_ITER steps (no early stop)')
    args = ap.parse_args()
    if args.fixed_iters:
        setattr(config, 'FORCE_FIXED_ITERS', True)
    main()
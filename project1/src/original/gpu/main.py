# src/original/gpu/main.py
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Server/headless environment, no popup windows
import matplotlib.pyplot as plt
import config

# Key: import from src/gpu/algorithms_gpu_unified.py
from src.gpu.algorithms_gpu_unified import run_simulation


def main():
    print("\n--- Loading [GPU: Baseline Equivalent Version] ---\n")

    # Result directory (consistent naming with other entry points, includes grid and noise)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(
        "results",
        f"{timestamp}_gpu_baseline_grid{config.GRID_SIZE}_noise{getattr(config, 'noise_magnitude', 'N/A')}"
    )
    os.makedirs(run_folder, exist_ok=True)
    print(f"Results from this run will be saved to: {run_folder}")

    # Initialization
    swarm_position = config.SWARM_INITIAL_POSITIONS.copy()
    swarm_size = swarm_position.shape[0]
    line_colors = np.random.rand(swarm_size, swarm_size, 3)
    swarm_paths = []

    # 2x2 canvas (no popup with Agg backend)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Run (maintain baseline convergence and logic)
    Jn, rn, final_pos, t_elapsed, comm_mat, _ = run_simulation(
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

    # Save (consistent with other entry points)
    print("Simulation completed, saving results...")
    plot_filepath      = os.path.join(run_folder, "final_plot.png")
    positions_filepath = os.path.join(run_folder, "final_positions.csv")
    jn_filepath        = os.path.join(run_folder, "jn_history.txt")
    rn_filepath        = os.path.join(run_folder, "rn_history.txt")
    time_filepath      = os.path.join(run_folder, "time_elapsed.txt")
    config_filepath    = os.path.join(run_folder, "config_summary.txt")

    fig.savefig(plot_filepath, dpi=300)
    np.savetxt(positions_filepath, final_pos, delimiter=",", header="x,y")
    np.savetxt(jn_filepath,  np.array(Jn))
    np.savetxt(rn_filepath,  np.array(rn))
    np.savetxt(time_filepath, np.array(t_elapsed))

    # Unified statement: write "Run Result" based on whether fixed steps mode
    final_time = t_elapsed[-1] if t_elapsed else 0.0
    iters_done = len(Jn)
    max_iter   = getattr(config, "MAX_ITER", iters_done)
    fixed_mode = bool(getattr(config, "FORCE_FIXED_ITERS", False))

    with open(config_filepath, "w") as f:
        f.write(f"--- Simulation Run Summary ---\n")
        f.write(f"Algorithm Version: gpu_baseline\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"GRID_SIZE: {config.GRID_SIZE}\n")
        f.write(f"SWARM_SIZE: {config.SWARM_SIZE}\n")
        f.write(f"NOISE_MAGNITUDE: {getattr(config, 'noise_magnitude', 'N/A')}\n")
        f.write(f"STEP_SIZE: {getattr(config, 'STEP_SIZE', 'baseline default or config')}\n")
        f.write(f"ALPHA: {config.ALPHA}\n")
        f.write(f"DELTA: {config.DELTA}\n")
        f.write(f"V: {config.V}\n")
        f.write(f"R0: {config.R0}\n")
        f.write(f"PT: {config.PT}\n")

        f.write("\n--- Run Result ---\n")
        if fixed_mode and iters_done >= max_iter:
            f.write(f"Completed fixed iterations: {iters_done} steps in {round(final_time, 2)} seconds\n")
        else:
            f.write(f"Converged in: {round(final_time, 2)} seconds\n")
            f.write(f"Converged after: {iters_done} iterations\n")

        if len(Jn):
            f.write(f"Final Jn: {Jn[-1]:.6f}\n")
        if len(rn):
            f.write(f"Final rn: {rn[-1]:.6f}\n")

    print(f"All results successfully saved to {run_folder}")


if __name__ == "__main__":
    # Optional: this entry also supports --fixed-iters for running fixed steps for fair timing
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixed-iters', action='store_true',
                    help='Run exactly MAX_ITER steps (no early stop)')
    args = ap.parse_args()
    if args.fixed_iters:
        setattr(config, 'FORCE_FIXED_ITERS', True)
    main()
# src/optimized/cpu/main_advanced.py
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import config

# 关键：相对导入同目录上一级里的 algorithms_advanced.py
from .. import algorithms_advanced as algorithms

def main():
    print("\n--- 正在加载 [V3: 进阶健壮版] 算法引擎 ---\n")

    # 结果目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_advanced_grid{config.GRID_SIZE}_noise{getattr(config, 'noise_magnitude', 'N/A')}"
    run_folder = os.path.join("results", run_name)
    os.makedirs(run_folder, exist_ok=True)
    print(f"本次运行的结果将保存在：{run_folder}")

    # 初始化
    swarm_position = config.SWARM_INITIAL_POSITIONS.copy()
    swarm_size = swarm_position.shape[0]
    line_colors = np.random.rand(swarm_size, swarm_size, 3)
    swarm_paths = []
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.ion()

    # 运行
    final_Jn, final_rn, final_positions, t_elapsed, final_comm_matrix, _ = algorithms.run_simulation(
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

    # 保存
    print("仿真结束，正在保存结果...")
    plot_filepath = os.path.join(run_folder, "final_plot.png")
    positions_filepath = os.path.join(run_folder, "final_positions.csv")
    jn_filepath = os.path.join(run_folder, "jn_history.txt")
    rn_filepath = os.path.join(run_folder, "rn_history.txt")
    time_filepath = os.path.join(run_folder, "time_elapsed.txt")
    config_filepath = os.path.join(run_folder, "config_summary.txt")

    fig.savefig(plot_filepath, dpi=300)
    np.savetxt(positions_filepath, final_positions, delimiter=",", header="x,y")
    np.savetxt(jn_filepath, np.array(final_Jn))
    np.savetxt(rn_filepath, np.array(final_rn))
    np.savetxt(time_filepath, np.array(t_elapsed))
    with open(config_filepath, "w") as f:
        f.write(f"--- Simulation Run Summary ---\n")
        f.write(f"Algorithm Version: advanced\n")
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
            f.write(f"Completed fixed iterations: {iters_done} steps in {round(final_time, 2)} seconds\n")
        else:
            f.write(f"Converged in: {round(final_time, 2)} seconds\n")
            f.write(f"Converged after: {iters_done} iterations\n")

    print(f"所有结果已成功保存至 {run_folder}")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixed-iters', action='store_true',
                    help='Run exactly MAX_ITER steps (no early stop)')
    args = ap.parse_args()
    if args.fixed_iters:
        setattr(config, 'FORCE_FIXED_ITERS', True)
    main()
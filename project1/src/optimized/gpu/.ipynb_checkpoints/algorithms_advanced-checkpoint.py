# src/optimized/gpu/main_advanced_gpu.py
# -*- coding: utf-8 -*-
"""
独立入口：进阶算法 + GPU 版
保持与其他 main 相同的输出与保存格式，不改算法逻辑
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import config

# 关键：只导入“进阶算法 + GPU”薄壳
from src.gpu.algorithms_advanced_gpu import run_simulation


def main():
    """
    运行“进阶 + GPU”算法的主程序，并自动保存结果。
    注：算法逻辑与 CPU 版进阶一致，只是把按边计算搬到 GPU。
    """
    print("\n--- 正在加载 [V4: 进阶算法 + GPU] 算法引擎 ---\n")

    # --- 1) 创建带时间戳的专属结果文件夹（命名风格与其他 main 保持一致） ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_advanced_gpu_grid{config.GRID_SIZE}_noise{getattr(config, 'noise_magnitude', 'N/A')}"
    run_folder = os.path.join("results", run_name)
    os.makedirs(run_folder, exist_ok=True)
    print(f"本次运行的结果将保存在：{run_folder}")

    # --- 2) 初始化仿真所需数据（与其他 main 完全一致） ---
    swarm_position = config.SWARM_INITIAL_POSITIONS.copy()
    swarm_size = swarm_position.shape[0]
    line_colors = np.random.rand(swarm_size, swarm_size, 3)
    swarm_paths = []

    # 统一的 2x2 可视化布局 + 交互式刷新
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.ion()

    # --- 3) 调用 GPU 进阶算法（接口与其他版本一致） ---
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

    # --- 4) 保存结果（文件名和字段对齐基线/优化版） ---
    print("仿真结束，正在保存结果...")

    plot_filepath      = os.path.join(run_folder, "final_plot.png")
    positions_filepath = os.path.join(run_folder, "final_positions.csv")
    jn_filepath        = os.path.join(run_folder, "jn_history.txt")
    rn_filepath        = os.path.join(run_folder, "rn_history.txt")
    time_filepath      = os.path.join(run_folder, "time_elapsed.txt")
    config_filepath    = os.path.join(run_folder, "config_summary.txt")

    # 图与数据
    fig.savefig(plot_filepath, dpi=300)
    np.savetxt(positions_filepath, final_positions, delimiter=",", header="x,y")
    np.savetxt(jn_filepath,  np.array(final_Jn))
    np.savetxt(rn_filepath,  np.array(final_rn))
    np.savetxt(time_filepath, np.array(t_elapsed))

    # 配置概要与收敛信息
    # 配置概要与收敛/完成信息（区分固定步数 vs 提前收敛）
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
            # 固定步数：不宣称“收敛”，而是“完成固定步数”
            f.write(f"Completed fixed iterations: {iters_done} steps in {round(final_time, 2)} seconds\n")
        else:
            # 非固定步数：大概率是提前收敛（也可能是数值稳定性保护提前终止，视算法打印而定）
            f.write(f"Converged in: {round(final_time, 2)} seconds\n")
            f.write(f"Converged after: {iters_done} iterations\n")
    print(f"所有结果已成功保存至 {run_folder}")

    # --- 5) 显示最终图像（保持窗口） ---
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
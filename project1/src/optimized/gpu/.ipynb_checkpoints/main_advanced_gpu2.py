# src/optimized/gpu/main_advanced_gpu2.py
# -*- coding: utf-8 -*-
"""
独立入口：V5 - GPU-Advanced v2（纯GPU Jacobi + 网格近邻剪枝）
与其他 main 保持一致的输出与保存格式，供 scripts.run_experiments 调度。
"""

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import config

# 只导入 V5 算法：GPU-Advanced v2
from src.gpu.algorithms_advanced_gpu2 import run_simulation


def main():
    """
    运行 V5（GPU-Advanced v2）并保存结果。
    """
    print("\n--- 正在加载 [V5: GPU-Advanced v2] 算法引擎 ---\n")

    # === 1) 结果目录（含 dtype 标签，与其他 main 保持一致） ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _dtype_tag = getattr(config, "DTYPE", np.float32)
    try:
        _dtype_name = np.dtype(_dtype_tag).name  # 'float32' / 'float64' / ...
    except Exception:
        _dtype_name = str(_dtype_tag)

    run_name = (
        f"{timestamp}_advanced_gpu2_grid{getattr(config, 'GRID_SIZE', 'N/A')}"
        f"_dtype{_dtype_name}_noise{getattr(config, 'noise_magnitude', 'N/A')}"
    )
    run_folder = os.path.join("results", run_name)
    os.makedirs(run_folder, exist_ok=True)
    print(f"本次运行的结果将保存在：{run_folder}")

    # （可选）若需要逐步时间分解文件，可在算法内部读取该环境变量
    os.environ.setdefault("V5_TIMEBREAK_FILE", os.path.join(run_folder, "v5_timebreak_steps.jsonl"))

    # === 2) 准备初始数据（统一精度、与其他 main 口径一致） ===
    swarm_position = np.asarray(getattr(config, "SWARM_INITIAL_POSITIONS").copy(), dtype=_dtype_tag)
    swarm_size = swarm_position.shape[0]
    line_colors = np.random.rand(swarm_size, swarm_size, 3)
    swarm_paths = []

    # 可视化容器（脚本模式下通常不画；保持一致即可）
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.ion()

    # === 3) 调用 V5 算法 ===
    Jn_hist, rn_hist, final_positions, t_hist, final_comm_matrix, _ = run_simulation(
        axs=axs,
        fig=fig,
        swarm_position=swarm_position,
        max_iter=getattr(config, "MAX_ITER", 50000),
        swarm_size=swarm_size,
        alpha=getattr(config, "ALPHA", 1e-5),
        beta=getattr(config, "BETA", getattr(config, "ALPHA", 1e-5) * (2 ** getattr(config, "DELTA", 2) - 1)),
        v=getattr(config, "V", 3),
        r0=getattr(config, "R0", 5),
        PT=getattr(config, "PT", 0.94),
        swarm_paths=swarm_paths,
        node_colors=getattr(config, "NODE_COLORS", np.random.rand(swarm_size, 3)),
        line_colors=line_colors
    )

    # === 4) 保存结果（文件名/字段对齐） ===
    print("仿真结束，正在保存结果...")

    plot_filepath      = os.path.join(run_folder, "final_plot.png")
    positions_filepath = os.path.join(run_folder, "final_positions.csv")
    jn_filepath        = os.path.join(run_folder, "jn_history.txt")
    rn_filepath        = os.path.join(run_folder, "rn_history.txt")
    time_filepath      = os.path.join(run_folder, "time_elapsed.txt")
    config_filepath    = os.path.join(run_folder, "config_summary.txt")

    fig.savefig(plot_filepath, dpi=300)
    np.savetxt(positions_filepath, final_positions, delimiter=",", header="x,y")
    np.savetxt(jn_filepath,  np.array(Jn_hist))
    np.savetxt(rn_filepath,  np.array(rn_hist))
    np.savetxt(time_filepath, np.array(t_hist))

    # 运行概要（支持 fixed-iters 与提前收敛两种文案）
    with open(config_filepath, "w", encoding="utf-8") as f:
        f.write(f"--- Simulation Run Summary ---\n")
        f.write(f"Algorithm Version: advanced_gpu2\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"GRID_SIZE: {getattr(config, 'GRID_SIZE', 'N/A')}\n")
        f.write(f"SWARM_SIZE: {getattr(config, 'SWARM_SIZE', final_positions.shape[0])}\n")
        f.write(f"NOISE_MAGNITUDE: {getattr(config, 'noise_magnitude', 'N/A')}\n")
        f.write(f"DTYPE: {_dtype_name}\n")
        f.write(f"DT: {getattr(config, 'DT', 'N/A')}\n")
        f.write(f"STEP_SIZE: {getattr(config, 'STEP_SIZE', 'N/A')}  # legacy key, may be N/A\n")
        f.write(f"ALPHA: {getattr(config, 'ALPHA', 'N/A')}\n")
        f.write(f"DELTA: {getattr(config, 'DELTA', 'N/A')}\n")
        f.write(f"V: {getattr(config, 'V', 'N/A')}\n")
        f.write(f"R0: {getattr(config, 'R0', 'N/A')}\n")
        f.write(f"PT: {getattr(config, 'PT', 'N/A')}\n")

        final_time = t_hist[-1] if t_hist else 0.0
        iters_done = len(Jn_hist)
        max_iter = getattr(config, "MAX_ITER", iters_done)
        fixed_mode = bool(getattr(config, "FORCE_FIXED_ITERS", False))

        f.write("\n--- Run Result ---\n")
        if fixed_mode and iters_done >= max_iter:
            f.write(f"Completed fixed iterations: {iters_done} steps in {round(final_time, 2)} seconds\n")
        else:
            f.write(f"Converged in: {round(final_time, 2)} seconds\n")
            f.write(f"Converged after: {iters_done} iterations\n")

    print(f"所有结果已成功保存至 {run_folder}")

    # === 5) 关闭交互绘图窗口 ===
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixed-iters', action='store_true',
                    help='Run exactly MAX_ITER steps (no early stop)')
    args = ap.parse_args()
    if args.fixed_iters:
        setattr(config, 'FORCE_FIXED_ITERS', True)
    main()
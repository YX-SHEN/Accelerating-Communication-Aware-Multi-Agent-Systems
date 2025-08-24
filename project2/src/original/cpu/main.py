# src/original/cpu/main.py
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import config

# 基线：仍然调用原 algorithms（不改逻辑）
from src.common import algorithms, plotting, utils

# 精度标签：用于区分结果目录
DTYPE_TAG = "f64" if np.dtype(getattr(config, "DTYPE", np.float32)) == np.float64 else "f32"

def main():
    """
    运行基准版(O(N^2))算法：仅改输出/记录格式，不改算法逻辑。
    """
    # --- 0. 横幅（统一风格） ---
    print("\n--- 正在加载 [V1: 基准版] 算法引擎 ---\n")

    # --- 1. 结果目录（统一命名风格） ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (f"{timestamp}_baseline_{DTYPE_TAG}"
                f"_grid{config.GRID_SIZE}_noise{getattr(config, 'noise_magnitude', 'N/A')}")
    run_folder = os.path.join("results", run_name)
    os.makedirs(run_folder, exist_ok=True)
    print(f"本次运行的结果将保存在：{run_folder}")

    # --- 2. 初始化 ---
    swarm_position = config.SWARM_INITIAL_POSITIONS.astype(config.DTYPE, copy=True)
    swarm_size = swarm_position.shape[0]
    line_colors = np.random.rand(swarm_size, swarm_size, 3)
    swarm_paths = []
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.ion()

    # --- 3. 调用基线算法 ---
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

    # --- 4. 保存结果（统一文件名/提示） ---
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
        f.write(f"--- Simulation Run Summary (Baseline Version) ---\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"GRID_SIZE: {config.GRID_SIZE}\n")
        f.write(f"SWARM_SIZE: {config.SWARM_SIZE}\n")
        f.write(f"NOISE_MAGNITUDE: {getattr(config, 'noise_magnitude', 'N/A')}\n")
        f.write(f"DTYPE: {np.dtype(config.DTYPE).name}\n")
        f.write(f"STEP_SIZE (DT): {getattr(config, 'DT', 1e-2)}\n")
        f.write(f"EPS_DIV: {getattr(config, 'EPS_DIV', 1e-6)}\n")
        f.write(f"ALPHA: {config.ALPHA}\n")
        f.write(f"DELTA: {config.DELTA}\n")
        f.write(f"V: {config.V}\n")
        f.write(f"R0: {config.R0}\n")
        f.write(f"PT: {config.PT}\n")

        # 统一口径：根据是否固定步数写“Run Result”
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

        # 可选：记录最终指标
        if len(final_Jn):
            f.write(f"Final Jn: {final_Jn[-1]:.6f}\n")
        if len(final_rn):
            f.write(f"Final rn: {final_rn[-1]:.6f}\n")

    print(f"所有结果已成功保存至 {run_folder}")

    # --- 5. 展示图像 ---
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
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

    # --- 1) 创建带时间戳的专属结果文件夹（命名风格与其他 main 保持一致），加入精度标签 ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _dtype_tag = getattr(config, "DTYPE", np.float32)
    try:
        _dtype_name = np.dtype(_dtype_tag).name  # 'float32' / 'float64' / ...
    except Exception:
        _dtype_name = str(_dtype_tag)

    run_name = (
        f"{timestamp}_advanced_gpu_grid{config.GRID_SIZE}"
        f"_dtype{_dtype_name}_noise{getattr(config, 'noise_magnitude', 'N/A')}"
    )
    run_folder = os.path.join("results", run_name)
    os.makedirs(run_folder, exist_ok=True)
    print(f"本次运行的结果将保存在：{run_folder}")

    # 只在 run_folder 可用后设置：供算法记录逐步时间到 JSONL
    os.environ.setdefault("V4_TIMEBREAK_FILE", os.path.join(run_folder, "v4_timebreak_steps.jsonl"))

    # --- 2) 初始化仿真所需数据（与其他 main 完全一致），但强制为统一精度 ---
    swarm_position = np.asarray(config.SWARM_INITIAL_POSITIONS.copy(), dtype=_dtype_tag)
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

    # --- 4) 保存结果（文件名和字段对齐基线/优化版） ---
    print("仿真结束，正在保存结果...")

    plot_filepath      = os.path.join(run_folder, "final_plot.png")
    positions_filepath = os.path.join(run_folder, "final_positions.csv")
    jn_filepath        = os.path.join(run_folder, "jn_history.txt")
    rn_filepath        = os.path.join(run_folder, "rn_history.txt")
    time_filepath      = os.path.join(run_folder, "time_elapsed.txt")
    config_filepath    = os.path.join(run_folder, "config_summary.txt")

    fig.savefig(plot_filepath, dpi=300)
    np.savetxt(positions_filepath, final_positions, delimiter=",", header="x,y")
    np.savetxt(jn_filepath,  np.array(final_Jn))
    np.savetxt(rn_filepath,  np.array(final_rn))
    np.savetxt(time_filepath, np.array(t_elapsed))

    # 配置概要与收敛/完成信息（区分固定步数 vs 提前收敛）
    with open(config_filepath, "w") as f:
        f.write(f"--- Simulation Run Summary ---\n")
        f.write(f"Algorithm Version: advanced_gpu\n")
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

    # --- 5) 显示最终图像（保持窗口） ---
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
# src/optimized/gpu/main_advanced_gpu3.py
# -*- coding: utf-8 -*-
"""
独立入口：GPU-Advanced v3 (V6) —— Jacobi + 统一网格 + 预处理/动量（可选）
- 仅改“入口”和保存逻辑；算法在 src/gpu/algorithms_advanced_gpu3.py
- 输出文件结构、字段与其他 main 保持一致，便于脚本汇总
"""

import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import config

# 关键：导入 V6 算法薄壳（你已放好的）
from src.gpu.algorithms_advanced_gpu3 import run_simulation


def _load_positions_or_make(N_default=256):
    """优先从 SWARM_POSITIONS_FILE 加载；否则：
       1) 若 config.SWARM_INITIAL_POSITIONS 存在，用之；
       2) 否则生成一个高斯云。
    """
    pos_file = os.getenv("SWARM_POSITIONS_FILE", "").strip()
    if pos_file and os.path.isfile(pos_file):
        arr = np.load(pos_file)
        arr = np.asarray(arr, dtype=np.float32)
        return arr

    if hasattr(config, "SWARM_INITIAL_POSITIONS"):
        arr = np.asarray(config.SWARM_INITIAL_POSITIONS, dtype=np.float32)
        return arr

    rng = np.random.default_rng(2025)
    arr = (rng.standard_normal((N_default, 2)) * 20.0).astype(np.float32)
    return arr


def _get_env_float(name: str, default=None):
    v = os.getenv(name, "")
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def main():
    print("\n--- 正在加载 [V6: GPU-Advanced v3] 算法引擎 ---\n")

    # === 1) 结果目录命名（沿用你的风格） ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _dtype_name = "float32"   # V6 内核用 fp32，更快更稳
    grid_sz = getattr(config, "GRID_SIZE", "N/A")
    noise_tag = getattr(config, "noise_magnitude", "N/A")
    run_name = f"{timestamp}_advanced_gpu3_grid{grid_sz}_dtype{_dtype_name}_noise{noise_tag}"
    run_folder = os.path.join("results", run_name)
    os.makedirs(run_folder, exist_ok=True)
    print(f"本次运行的结果将保存在：{run_folder}")

    # 可选：给 V6 预留逐步时间分解 JSONL（算法里若实现会写入）
    os.environ.setdefault("V6_TIMEBREAK_FILE", os.path.join(run_folder, "v6_timebreak_steps.jsonl"))

    # === 2) 装载初始位置（支持 SWARM_POSITIONS_FILE） ===
    swarm_position = _load_positions_or_make()
    swarm_size = int(swarm_position.shape[0])

    # 统一的可视化句柄（V6 默认不画；需要可设 PLOT_EVERY）
    fig = None
    axs = None

    # === 3) 调用 V6 算法 ===
    max_iter = int(os.getenv("MAX_ITER_OVERRIDE", getattr(config, "MAX_ITER", 300)))
    alpha = float(getattr(config, "ALPHA", 1e-5))
    beta = float(getattr(config, "BETA", alpha * (2 ** getattr(config, "DELTA", 2) - 1)))
    v = float(getattr(config, "V", 3))
    r0 = float(getattr(config, "R0", 5))
    PT = float(getattr(config, "PT", 0.94))

    t0 = time.time()
    Jn_hist, rn_hist, final_positions, t_hist, _, _ = run_simulation(
        axs=axs, fig=fig,
        swarm_position=swarm_position.astype(np.float32, copy=False),
        max_iter=max_iter,
        swarm_size=swarm_size,
        alpha=alpha, beta=beta, v=v, r0=r0, PT=PT,
        swarm_paths=None, node_colors=None, line_colors=None
    )
    t1 = time.time()

    # === 4) 保存产物（对齐字段/文件名） ===
    print("仿真结束，正在保存结果...")

    plot_filepath      = os.path.join(run_folder, "final_plot.png")
    positions_filepath = os.path.join(run_folder, "final_positions.csv")
    jn_filepath        = os.path.join(run_folder, "jn_history.txt")
    rn_filepath        = os.path.join(run_folder, "rn_history.txt")
    time_filepath      = os.path.join(run_folder, "time_elapsed.txt")
    config_filepath    = os.path.join(run_folder, "config_summary.txt")

    # 简单出一张散点终态图，避免空图
    try:
        plt.figure(figsize=(5, 5))
        fp = np.asarray(final_positions, dtype=np.float32)
        plt.scatter(fp[:, 0], fp[:, 1], s=6, alpha=0.6)
        plt.title("V6 final positions")
        plt.tight_layout()
        plt.savefig(plot_filepath, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[warn] 绘图失败：{e}")

    np.savetxt(positions_filepath, final_positions, delimiter=",", header="x,y")
    np.savetxt(jn_filepath,  np.asarray(Jn_hist, dtype=np.float32))
    np.savetxt(rn_filepath,  np.asarray(rn_hist, dtype=np.float32))
    np.savetxt(time_filepath, np.asarray(t_hist, dtype=np.float32))

    # 记录关键配置与完成信息（兼容 fixed-iters 的解析）
    with open(config_filepath, "w", encoding="utf-8") as f:
        print("--- Simulation Run Summary ---", file=f)
        print("Algorithm Version: advanced_gpu3", file=f)
        print(f"Timestamp: {timestamp}", file=f)
        print(f"GRID_SIZE: {grid_sz}", file=f)
        print(f"SWARM_SIZE: {swarm_size}", file=f)
        print(f"NOISE_MAGNITUDE: {noise_tag}", file=f)
        print(f"DTYPE: {_dtype_name}", file=f)
        print(f"DT: {getattr(config, 'DT', 'N/A')}", file=f)
        print(f"STEP_SIZE: {getattr(config, 'STEP_SIZE', 'N/A')}  # legacy key, may be N/A", file=f)
        print(f"ALPHA: {alpha}", file=f)
        print(f"DELTA: {getattr(config, 'DELTA', 'N/A')}", file=f)
        print(f"V: {v}", file=f)
        print(f"R0: {r0}", file=f)
        print(f"PT: {PT}", file=f)
        # 记录 V6 的关键可调项（若用到了）
        print(f"AD6_MOMENTUM: {_get_env_float('AD6_MOMENTUM', 'N/A')}", file=f)
        print(f"AD6_PRECOND: {os.getenv('AD6_PRECOND', 'N/A')}", file=f)
        print(f"AD6_PRECOND_GAMMA: {_get_env_float('AD6_PRECOND_GAMMA', 'N/A')}", file=f)
        print(f"AD6_REGRID_EVERY: {int(os.getenv('AD6_REGRID_EVERY', '0') or 0)}", file=f)

        total = float(t1 - t0)
        iters_done = len(Jn_hist)
        print("\n--- Run Result ---", file=f)
        # V6 一般跑固定步；与现有脚本口径对齐：
        print(f"Completed fixed iterations: {iters_done} steps in {total:.2f} seconds", file=f)

    print(f"所有结果已成功保存至 {run_folder}\n")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixed-iters', action='store_true',
                    help='Run exactly MAX_ITER steps (no early stop)')
    args = ap.parse_args()
    if args.fixed_iters:
        setattr(config, 'FORCE_FIXED_ITERS', True)  # 仅用于在 summary 里区分口径
    main()
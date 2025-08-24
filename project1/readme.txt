# Simulation of Crowd Formation Communication (V1~V6)

	• If you are using CuPy, you can unannotate cupy-cuda11x or cupy-cuda12x according to the local CUDA version.

This project realizes the simulation of formation evolution of multi-agent/UAV under the constraints of communication thresholds, supports multiple versions of CPU and GPU comparative evaluation, and generates publishable statistical charts and reports.
> Version Comparison Table (by your name)
> - **V1** = CPU Baseline
> - **V2** = CPU Advanced
> - **V3** = GPU Baseline
> - **V4** = GPU Advanced
> - **V5** = GPU Advanced 2 (Raster Neighborhood Clipping CUDA Implementation)

> - **V6** = GPUAdvanced 3 (add preconditioning/momentum/shading/Chebyshev etc. prototypes on top of V5)

---

## 1. Installation
Virtual environments are recommended:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:

source .venv/bin/activate
pip install -U pip

pip install -r requirements.txt
GPU backend instructions
	• requirements.txt JAX (CPU) installed by default; If you need JAX+CUDA, please install jaxlib that matches the CUDA/driver strictly according to the official JAX guidelines.

⸻

2. Run each version quickly (root directory execution)

V1 — CPU Baseline

Entrance: src/original/cpu/main.py

python -m src.original.cpu.main
# Fixed number of steps (no pre-convergence):
# python -m src.original.cpu.main --fixed-iters

V2 — CPU Advanture (Optimized)

Entrance: src/optimized/cpu/main_advanced.py

python -m src.optimized.cpu.main_advanced
# Support fixed number of steps:
# python -m src.optimized.cpu.main_advanced --fixed-iters

V3 — GPU Baseline (Unified Implementation, Optional Gauss/Jacobi)

Entrance: src/original/gpu/main.py

python -m src.original.gpu.main

Toggle update policies on config.py:

UPDATE_MODE = "gauss" # Update in place sequentially (exactly in line with CPU baseline)
# or
UPDATE_MODE = "jacobi" # Update once in parallel (default)

V4 — GPU Advanture (Advanced Optimization + GPU Edge List Parallel)

Entrance: src/optimized/gpu/main_advanced_gpu.py

python -m src.optimized.gpu.main_advanced_gpu
# Support fixed number of steps:
# python -m src.optimized.gpu.main_advanced_gpu --fixed-iters

V5 — GPU Advanture 2 (CUDA Grid 3×3 Neighborhood Clipping)

This release provides algorithm files src/gpu/algorithms_advanced_gpu2.py with run_simulation that can be run via a batch experiment script or a temporary launcher.

Method A: Batch Experiment Script (Recommended)

python -m scripts.run_experiments --full --sizes 50,100,200 --iters 300 --repeats 3 --only GPU-Advanced-v2
# Equivalent alias: --only ad2 / ga2 / adv2
# Optional: Environment variable AD2_CELL_SCALE=1.5 affects cell size with K (neighborhood radius)

Method B: One-time temporary start (Python one-line command)

python - <<'PY'
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, config
from src.gpu.algorithms_advanced_gpu2 import run_simulation
pos = config. SWARM_INITIAL_POSITIONS.copy()
fig,axs=plt.subplots(2,2,figsize=(8,8))
Jn,rn,final_pos,t,_,_ = run_simulation(axs=axs, fig=fig,
    swarm_position=pos, max_iter=config. MAX_ITER, swarm_size=pos.shape[0],
    alpha=config. ALPHA, beta=config. BETA, v=config. V, r0=config. R0, PT=config.PT,
    swarm_paths=[], node_colors=config. NODE_COLORS, line_colors=np.random.rand(pos.shape[0],pos.shape[0],3))
fig.savefig("results/v5_final.png",dpi=200)
print("V5 done, steps:", len(Jn))
PY

Common environment variables for V5: AD2_CELL_SCALE (default 1.0; larger → thinner cells, larger K=ceil(scale) larger), AD2_DEBUG=1 prints grid/edge statistics.

V6 — GPU Advanture 3 (V5 + Precondition/Momentum/Shading/Chebyshev Prototype)

Algorithm file: src/gpu/algorithms_advanced_gpu3.py (including run_simulation). If you don't have a batch script that you can't access, you can start it temporarily.

One-time temporary start

python - <<'PY'
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, config, os
from src.gpu.algorithms_advanced_gpu3 import run_simulation
pos = config. SWARM_INITIAL_POSITIONS.copy()
fig,axs=plt.subplots(2,2,figsize=(8,8))
Jn,rn,final_pos,t,_,_ = run_simulation(axs=axs, fig=fig,
    swarm_position=pos, max_iter=config. MAX_ITER, swarm_size=pos.shape[0],
    alpha=config. ALPHA, beta=config. BETA, v=config. V, r0=config. R0, PT=config.PT,
    swarm_paths=[], node_colors=config. NODE_COLORS, line_colors=np.random.rand(pos.shape[0],pos.shape[0],3))
fig.savefig("results/v6_final.png",dpi=200)
print("V6 done, steps:", len(Jn))
PY

V6 optional environment variable
	• Color Blocking: AD2_COLORING=none|rb|c4 (red/black/quad)
	• Precondition: AD2_PRECOND=none|phi|degree, AD2_PRECOND_GAMMA, AD2_PRECOND_EPS
	• Semi-iteration: AD2_OMEGA (weighted Jacobi coefficient), AD2_MOMENTUM (heavy-ball 0~0.95)
	• Chebyshev:AD2_CHEB="lam_min,lam_max",AD2_CHEB_M(period)
	• Grid scale: AD2_CELL_SCALE; Debug output: AD2_DEBUG=1

⸻

3. Batch evaluation and charts (paper-level one-click reproduction experiment)

Script: scripts/run_experiments.py
Functions: Multi-scale N × multi-repeat × (optional) GPUs, counting runtime/iterations/Jn/rn/convergence rate; Generate CSV, metadata.json, report.txt, and multiple graphs (runtime, speedup, ms/iter, complexity slope, throughput, GPU memory, etc.).

Example:

# Full set (including available GPU routes are automatically included)
python -m scripts.run_experiments --full --sizes 50,100,200 --iters 300 --repeats 5

# Only a few routes are tested (aliases and case-insensitive/hyphens are supported)
python -m scripts.run_experiments --full --only baseline,cpu,gpu-advanced-v2

ONLY filter supported names/aliases (case and hyphen insensitive):

baseline|base|b
cpu-advanced|cpu|adcpu
gpu-baseline|gpubase|gb
gpu-advanced|ga|ad1|adv1
gpu-advanced-v2|ga2|ad2|adv2

Product path: results/<时间戳>_paper_benchmark/

⸻

4. Result output (common to all versions)

The end of the run is generated in results/<时间戳>_<variant><GRID_SIZE>_grid_noise<N/A>/ or the bulk directory:
	• final_plot.png: Quadruple (Scene/Trajectory/Jn/rn)
	• final_positions.csv: Final coordinates (column name x,y)
	•	jn_history.txt、rn_history.txt、time_elapsed.txt
	• config_summary.txt: Parameter summary + convergence/fixed step information
	• Batch :runs_long.csv, summary.csv, complexity_slopes.csv, several .png diagrams and report.txt

⸻

5. Configuration (config.py)
	• Basic parameters:
MAX_ITER，ALPHA，DELTA，BETA=ALPHA*(2**DELTA-1)，V，R0，PT∈(0,1)
Communication radius: COMM_RADIUS = R0*((-ln PT)/BETA)^(1/V)
	• Initial formation:
GRID_SIZE co-generates SWARM_INITIAL_POSITIONS / SWARM_SIZE / NODE_COLORS with RANDOM_POINT_RATIO
	• Numeric/Visualization:
KEEP_FULL_MATRIX (False Suggestion at Scale), PLOT_EVERY, LOG_EVERY, EPS, DTYPE, VISUALIZATION_THRESHOLD
	• Convergence (V2/V4 route):
CONVERGENCE_WINDOW、CONVERGENCE_SLOPE_THRESHOLD、CONVERGENCE_STD_THRESHOLD
	• GPU Baseline (V3):
UPDATE_MODE="gauss"|" jacobi"，GPU_ECAP_FACTOR
	• Stability Protection:
STABILITY_THRESHOLD (excessive control norm → early termination)

Environment variable override (no code changes):

KEEP_FULL_MATRIX_OVERRIDE=0|1
MODE_OVERRIDE=viz|hpc
PLOT_EVERY_OVERRIDE=<int>
LOG_EVERY_OVERRIDE=<int>
SWARM_POSITIONS_FILE=<.npy> # External initial formation
MAX_ITER_OVERRIDE=<int>
# V5/V6:
AD2_CELL_SCALE=<float>
AD2_DEBUG=1
AD2_COLORING=none|rb|c4
AD2_PRECOND=none|phi|degree
AD2_PRECOND_GAMMA=<float>
AD2_PRECOND_EPS=<float>
AD2_OMEGA=<float>
AD2_MOMENTUM=<float 0..0.95>
AD2_CHEB="lam_min,lam_max"  AD2_CHEB_M=<int>


⸻

6. Indicators and formulas
	• Far-field communication: a_ij = exp(-ALPHA*(2**DELTA-1)*(r_ij/R0)^V)
	• Near-field coupling: g_ij = r_ij / sqrt(r_ij^2 + R0^2)
	• Control Factor: rho_ij (beta, V, rij, R0) (see src/common/utils.py)
	• Only if a_ij >= PT is counted in the edge and control
	• Average communication performance: Jn = mean(g_ij * a_ij over edges)
	• Average neighbor distance: rn = mean(r_ij over edges)

⸻

7. FAQs
	• No adjacent edges (cnt=0): Decrease PT, increase density, or increase formation coverage.
	• Memory/Memory Pressure: Increase KEEP_FULL_MATRIX=False by PLOT_EVERY.
	• Unstable values: Decrease the STEP_SIZE (can be added in config.py such as STEP_SIZE=0.005), or increase DTYPE=np.float64 (backend support required), or increase the STABILITY_THRESHOLD (use with caution).
	• Exactly the same as CPU update order: UPDATE_MODE="gauss" in V3, or just use V1.
# config.py

import numpy as np

np.random.seed(42)

#---------------------------#
# Initialize all parameters #
#---------------------------#
MAX_ITER = 50000     # Maximum number of iterations
ALPHA = 10**(-5)   # System parameter about antenna characteristics
DELTA = 2          # Required application data rate
BETA = ALPHA * (2**DELTA - 1) # Derived parameter
V = 3              # Path loss exponent
R0 = 5             # Reference distance
PT = 0.94          # The threshold value for communication quality
FORCE_FIXED_ITERS = False

# --- Method to adjust drone count and layout ---

# 1. Define grid size and random point ratio separately
GRID_SIZE = 3
RANDOM_POINT_RATIO = 0.1  # <--- Set to 10%, you can freely adjust this ratio

# --- 2. Calculate specific counts for grid points and random points ---
num_grid_points = GRID_SIZE * GRID_SIZE
# Calculate random points based on ratio and grid points, convert to integer with int()
num_random_points = int(num_grid_points * RANDOM_POINT_RATIO)

# --- 3. Generate grid points ---
spacing = 10
x_coords, y_coords = np.meshgrid(np.arange(GRID_SIZE) * spacing, np.arange(GRID_SIZE) * spacing)
grid_positions = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)

# --- 4. Generate calculated number of random points ---
random_range_min = -10
random_range_max = (GRID_SIZE * spacing)
random_positions = np.random.uniform(random_range_min, random_range_max, size=(num_random_points, 2))

# --- 5. Merge both sets of positions ---
SWARM_INITIAL_POSITIONS = np.concatenate([grid_positions, random_positions], axis=0).astype(float)

# --- 6. Update total agent count ---
SWARM_SIZE = num_grid_points + num_random_points

# --- 7. Automatically generate colors ---
NODE_COLORS = np.random.rand(SWARM_SIZE, 3)

# === Defaults used by advanced/GPU paths ===
KEEP_FULL_MATRIX = True          # Keep NxN matrix for small-scale plotting; recommend setting to False for large scale
VERBOSE_IMPORT = False           # Print prompts about using KDTree/linregress or fallback
STABILITY_THRESHOLD = 1e6        # Numerical stability protection threshold (terminate early if control force norm is too large)
GPU_ECAP_FACTOR = 1.25           # JAX edge list padding factor for stable JIT shapes

# === Communication radius derived from thresholds ===
# R = r0 * ((-ln PT) / BETA) ** (1.0 / V)   # because BETA = ALPHA * (2**DELTA - 1)
EPS = 1e-12
COMM_RADIUS = R0 * ((-np.log(np.clip(PT, 1e-12, 1 - 1e-12))) / BETA) ** (1.0 / V)


# === Advanced Simulation Control ===
PLOT_EVERY = 20  # Refresh plot every 20 iterations to reduce plotting overhead
CONVERGENCE_WINDOW = 50 # Sliding window size for convergence judgment
CONVERGENCE_SLOPE_THRESHOLD = 1e-6 # Threshold for slope of Jn curve at convergence
CONVERGENCE_STD_THRESHOLD = 1e-5   # Threshold for standard deviation of Jn curve at convergence
LOG_EVERY = 20 # Print log every 20 iterations

UPDATE_MODE = "jacobi"  # "jacobi" (parallel update) or "gauss" (sequential update, completely equivalent to CPU baseline)

# === Optional overrides for automation (do NOT break existing behavior) ===
import os

# Allow overriding some parameters with environment variables (keep original value if not set)
_keep = os.getenv("KEEP_FULL_MATRIX_OVERRIDE")
if _keep is not None:
    try:
        KEEP_FULL_MATRIX = bool(int(_keep))  # "0/1"
    except Exception:
        pass

_mode = os.getenv("MODE_OVERRIDE")
if _mode:
    MODE = _mode

_plot_every = os.getenv("PLOT_EVERY_OVERRIDE")
if _plot_every:
    try:
        PLOT_EVERY = int(_plot_every)
    except Exception:
        pass

_log_every = os.getenv("LOG_EVERY_OVERRIDE")
if _log_every:
    try:
        LOG_EVERY = int(_log_every)
    except Exception:
        pass

# Override initial formation with external position file (keep original logic if not provided)
_positions_file = os.getenv("SWARM_POSITIONS_FILE")
if _positions_file and os.path.exists(_positions_file):
    _arr = np.load(_positions_file)
    SWARM_INITIAL_POSITIONS = _arr.astype(float)
    SWARM_SIZE = SWARM_INITIAL_POSITIONS.shape[0]
# ===== Optional: more overrides for automation =====
_max_iter = os.getenv("MAX_ITER_OVERRIDE")
if _max_iter:
    try:
        MAX_ITER = int(_max_iter)
    except Exception:
        pass

# If external position file overrides, complete color length to avoid out-of-bounds when main directly reads config.NODE_COLORS
if _positions_file and os.path.exists(_positions_file):
    NODE_COLORS = np.random.rand(SWARM_SIZE, 3)

# ===== Optional defaults (optional but recommended to write explicitly) =====
DTYPE = np.float32
VISUALIZATION_THRESHOLD = globals().get("VISUALIZATION_THRESHOLD", 40)
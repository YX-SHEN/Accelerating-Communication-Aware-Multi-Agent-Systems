# config.py  —— 带精度总开关的版本
import os
import numpy as np

np.random.seed(42)

# =========================
#  Precision (global toggle)
# =========================
# 用法：
#   代码里默认单精度；如需双精度：
#     1) 直接改 DTYPE = np.float64
#     2) 或运行时设环境变量：DTYPE_OVERRIDE=64 / float64 / double / 32 / float32 / single
DTYPE = np.float32
_prec = os.getenv("DTYPE_OVERRIDE", "").strip().lower()
if _prec in ("64", "float64", "double"):
    DTYPE = np.float64
elif _prec in ("32", "float32", "single"):
    DTYPE = np.float32

IS_FP64 = (np.dtype(DTYPE) == np.float64)
Real = DTYPE  # 统一别名，CPU/GPU 都用它来创建数组/常数

# 时间步与数值保护项随精度自适应
DT = Real(0.01)                      # Euler 步长
EPS_DIV = Real(1e-6) if not IS_FP64 else Real(1e-12)   # eij 分母用
EPS_LOG = Real(1e-12)                                   # log 裁剪用（两种精度都够小）
# 供老代码兼容（以前有用 EPS 的地方）
EPS = EPS_LOG

# ================
# Core parameters
# ================
MAX_ITER = int(50000)
ALPHA = Real(1e-5)                 # 天线/链路常数
DELTA = Real(2)                    # 目标速率
BETA  = Real(ALPHA * (2**DELTA - 1))
V     = Real(3)                    # 路径损耗指数
R0    = Real(5)                    # 参考距离
PT    = Real(0.94)                 # 接收概率阈值
FORCE_FIXED_ITERS = False


# --- 布局参数 ---
GRID_SIZE = 3
RANDOM_POINT_RATIO = 0.1

# 生成初始位置（按 dtype）
num_grid_points = GRID_SIZE * GRID_SIZE
num_random_points = int(num_grid_points * RANDOM_POINT_RATIO)

spacing = 10
x_coords, y_coords = np.meshgrid(np.arange(GRID_SIZE) * spacing,
                                 np.arange(GRID_SIZE) * spacing)
grid_positions = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)

random_range_min = -10
random_range_max = (GRID_SIZE * spacing)
random_positions = np.random.uniform(random_range_min, random_range_max,
                                     size=(num_random_points, 2))

SWARM_INITIAL_POSITIONS = np.concatenate(
    [grid_positions, random_positions], axis=0
).astype(Real)

SWARM_SIZE = SWARM_INITIAL_POSITIONS.shape[0]
NODE_COLORS = np.random.rand(SWARM_SIZE, 3)  # 颜色不必跟随 dtype

# =========== Advanced / GPU defaults ===========
KEEP_FULL_MATRIX = True
VERBOSE_IMPORT = False
STABILITY_THRESHOLD = Real(1e6)
GPU_ECAP_FACTOR = 1.25

# 通信半径（注意 log 裁剪用 EPS_LOG）
COMM_RADIUS = Real(R0) * (
    (-np.log(np.clip(PT, EPS_LOG, Real(1) - EPS_LOG))) / BETA
) ** (Real(1) / V)

# ============ Simulation control ============
PLOT_EVERY = 20
CONVERGENCE_WINDOW = 50
CONVERGENCE_SLOPE_THRESHOLD = Real(1e-6)
CONVERGENCE_STD_THRESHOLD   = Real(1e-5)
LOG_EVERY = 20
UPDATE_MODE = "jacobi"  # "jacobi" or "gauss"

# ============ Optional overrides ============
# 允许外部文件覆盖初始位置
_positions_file = os.getenv("SWARM_POSITIONS_FILE")
if _positions_file and os.path.exists(_positions_file):
    _arr = np.load(_positions_file)
    SWARM_INITIAL_POSITIONS = _arr.astype(Real)
    SWARM_SIZE = SWARM_INITIAL_POSITIONS.shape[0]
    NODE_COLORS = np.random.rand(SWARM_SIZE, 3)

_max_iter = os.getenv("MAX_ITER_OVERRIDE")
if _max_iter:
    try:
        MAX_ITER = int(_max_iter)
    except Exception:
        pass

_keep = os.getenv("KEEP_FULL_MATRIX_OVERRIDE")
if _keep is not None:
    try:
        KEEP_FULL_MATRIX = bool(int(_keep))
    except Exception:
        pass

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

# 画图阈值
VISUALIZATION_THRESHOLD = globals().get("VISUALIZATION_THRESHOLD", 40)

# ============ 供 GPU 侧读取的小旗子 ============
# 若你用 JAX，读取这个来决定 jax_enable_x64；用 CuPy/RawKernel 则决定 typedef real_t
JAX_ENABLE_X64 = bool(IS_FP64)

# src/optimized/algorithms_advanced.py
# -*- coding: utf-8 -*-

import time
import numpy as np
import config
from src.common import utils, plotting

# ------------------------------------------------------------------------------
# Optional: Whether to print a message at runtime indicating whether KDTree/linregress or the fallback was used
VERBOSE_IMPORT = getattr(config, 'VERBOSE_IMPORT', False)

# ------------------------------------------------------------------------------
# 1) Neighbor Search: Prioritize SciPy KDTree; automatically fall back to the "cell-list method" when SciPy is unavailable
_HAS_SCIPY_KDTREE = False
try:
    from scipy.spatial import cKDTree as _KDTree

    _HAS_SCIPY_KDTREE = True
except Exception:
    _HAS_SCIPY_KDTREE = False


def _neighbor_lists_kdtree(positions: np.ndarray, R: float):
    tree = _KDTree(positions)
    return tree.query_ball_point(positions, r=R)


def _neighbor_lists_celllist(positions: np.ndarray, R: float):
    """
    Dependency-free neighbor list construction (cell-list, vectorized radius check), complexity ~ O(N * k)
    """
    n = positions.shape[0]
    invR = 1.0 / R
    # Integer coordinates of the cell each point belongs to (floor, supports negative coordinates)
    cells_xy = np.floor(positions * invR).astype(np.int32)

    # Create buckets: dict[(cx,cy)] -> List[idx]
    buckets = {}
    for idx, key in enumerate(map(tuple, cells_xy)):
        buckets.setdefault(key, []).append(idx)

    neigh = [[] for _ in range(n)]
    OFF = [(-1, -1), (-1, 0), (-1, 1),
           (0, -1), (0, 0), (0, 1),
           (1, -1), (1, 0), (1, 1)]
    R2 = R * R

    for i in range(n):
        cx, cy = cells_xy[i]
        cand = []
        for dx, dy in OFF:
            cand.extend(buckets.get((cx + dx, cy + dy), ()))
        if not cand:
            continue
        cand = np.asarray(cand, dtype=np.int32)
        d = positions[cand] - positions[i]
        mask = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) <= R2
        neigh[i] = cand[mask].tolist()
    return neigh


def _get_neighbor_lists(positions: np.ndarray, R: float):
    if _HAS_SCIPY_KDTREE:
        return _neighbor_lists_kdtree(positions, R)
    else:
        return _neighbor_lists_celllist(positions, R)


# ------------------------------------------------------------------------------
# 2) Convergence Criterion: Prioritize SciPy linregress; fall back to numpy.polyfit when SciPy is unavailable
_HAS_SCIPY_LINREGRESS = False
try:
    from scipy.stats import linregress as _linregress

    _HAS_SCIPY_LINREGRESS = True
except Exception:
    _HAS_SCIPY_LINREGRESS = False


def _slope_and_std(y):
    """
    Returns the slope and standard deviation of the most recent window.
    y: 1D sequence (use raw float values, not rounded ones).
    """
    if _HAS_SCIPY_LINREGRESS:
        x = np.arange(len(y), dtype=np.float64)
        res = _linregress(x, np.asarray(y, dtype=np.float64))
        return float(res.slope), float(np.std(y))
    else:
        x = np.arange(len(y), dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        p = np.polyfit(x, y_arr, deg=1)
        slope = float(p[0])
        std_dev = float(np.std(y_arr))
        return slope, std_dev


# ------------------------------------------------------------------------------
# 3) Utilities
def _safe_log_pt(pt: float) -> float:
    eps = 1e-12
    return float(np.log(np.clip(pt, eps, 1.0 - eps)))



def _edges_from_neighbors(neighbor_lists: list) -> tuple[np.ndarray, np.ndarray]:
    """
    A helper function to convert neighbor lists (list of lists) into a unique edge list where i < j.
    Returns two arrays: i_indices and j_indices.
    """
    edges = set()
    for i, neighbors in enumerate(neighbor_lists):
        for j in neighbors:
            if i < j:
                edges.add((i, j))

    if not edges:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    edge_arr = np.array(list(edges), dtype=np.int32)
    return edge_arr[:, 0], edge_arr[:, 1]


class _ZeroMatrixProxy:
    """
    A "zero matrix" proxy with zero memory footprint, supporting A[i, j] indexing and returning 0.0.
    Used for large-scale simulations to avoid allocating a large N×N matrix for plotting functions.
    """
    __slots__ = ()

    def __getitem__(self, idx):
        return 0.0

    def __setitem__(self, idx, val):
        # Ignore writes
        pass

    def fill(self, val: float):
        # Align with numpy API (fill is only used in small-scale mode, kept here just in case)
        pass


# ------------------------------------------------------------------------------
# 4) Main Function
def run_simulation(axs, fig, swarm_position, max_iter, swarm_size,
                   alpha, beta, v, r0, PT, swarm_paths, node_colors, line_colors):
    """
    Advanced optimized version:
      - Neighbor construction: KDTree (preferred) or cell-list (fallback)
      - Half-edge deduplication + pairwise accumulation (once per edge, consistent reaction force)
      - Jn/rn streaming statistics (avoids a second O(N^2) scan)
      - Plotting/logging throttling
      - Convergence criterion: Linear regression slope + standard deviation of the recent window
      - Large-scale: Use a zero-memory "edge matrix" proxy to avoid N×N allocation
    """
    # ---------------- Parameter Checks and Derived Quantities ----------------
    assert 0.0 < PT < 1.0, "PT must be in (0, 1)"
    assert alpha > 0 and beta > 0 and v > 0 and r0 > 0, "alpha/beta/v/r0 must be positive"

    step_size = getattr(config, 'STEP_SIZE', 0.01)
    mode = getattr(config, 'MODE', 'hpc')  # "viz" or "hpc"
    plot_every = getattr(config, 'PLOT_EVERY', 20)
    log_every = getattr(config, 'LOG_EVERY', 20)

    # Fix: Prevent infinite loop if plot_every/log_every is 0
    plot_every = max(1, plot_every)
    log_every = max(1, log_every)

    convergence_window = getattr(config, 'CONVERGENCE_WINDOW', 50)
    slope_threshold = getattr(config, 'CONVERGENCE_SLOPE_THRESHOLD', 1e-6)
    std_threshold = getattr(config, 'CONVERGENCE_STD_THRESHOLD', 1e-5)

    # New: Numerical stability check parameter
    STABILITY_THRESHOLD = getattr(config, 'STABILITY_THRESHOLD', 1e6)

    # Communication radius (inversely solved from a_ij = PT)
    R = r0 * ((-_safe_log_pt(PT)) / beta) ** (1.0 / v)
    # Numerical stability: EPS is linked to the scale
    EPS = max(getattr(config, 'EPS', 1e-12), 1e-9 * R)

    if VERBOSE_IMPORT and (max_iter > 0):
        if _HAS_SCIPY_KDTREE:
            print("[alg-adv] Using SciPy KDTree for neighbor search.")
        else:
            print("[alg-adv] SciPy not available, using cell-list for neighbor search.")
        if _HAS_SCIPY_LINREGRESS:
            print("[alg-adv] Using SciPy linregress for convergence criterion.")
        else:
            print("[alg-adv] SciPy not available, using numpy.polyfit for convergence criterion.")

    # ---------------- Structures and Buffers ----------------
    swarm_control_ui = np.zeros((swarm_size, 2), dtype=swarm_position.dtype)

    # Visualization detail threshold (draw edges only for small scale & explicit viz mode)
    VIS_THRESHOLD = getattr(config, 'VISUALIZATION_THRESHOLD', 40)
    do_viz_details = (swarm_size <= VIS_THRESHOLD and mode == "viz")

    # Small-scale: allocate a real matrix; Large-scale: use the zero matrix proxy to avoid memory explosion
    if do_viz_details:
        comm_mat = np.zeros((swarm_size, swarm_size), dtype=np.float32)
    else:
        comm_mat = _ZeroMatrixProxy()

    # Metrics and Time
    Jn_raw, rn_raw = [], []  # Raw sequences, for convergence criterion
    Jn, rn = [], []  # Display sequences (rounded)
    t_elapsed = []
    start_time = time.time()

    # ---------------- Main Loop ----------------
    for it in range(max_iter):
        # 1) Neighbor construction (KDTree or cell-list)
        neighbor_lists = _get_neighbor_lists(swarm_position, R)

        # 2) Streaming statistics counters
        sum_phi = 0.0;
        cnt_phi = 0
        sum_r = 0.0;
        cnt_r = 0

        # 3) Clear the plotting matrix for small-scale cases
        if do_viz_details:
            comm_mat.fill(0.0)

        # 4) Half-edge deduplication + pairwise accumulation (only calculate for i < j)
        for i in range(swarm_size):
            neigh_i = neighbor_lists[i]
            if not neigh_i:
                continue
            qi = swarm_position[i, :]

            for j in neigh_i:
                if j <= i:
                    continue
                qj = swarm_position[j, :]

                # Distance & communication quality
                rij = utils.calculate_distance(qi, qj)
                aij = utils.calculate_aij(alpha, config.DELTA, rij, r0, v)
                if aij < PT:
                    continue

                gij = utils.calculate_gij(rij, r0)
                rho_ij = utils.calculate_rho_ij(beta, v, rij, r0)
                eij = (qi - qj) / (rij + EPS)

                # -- Pairwise update (consistent reaction force) --
                force = rho_ij * eij
                swarm_control_ui[i, :] += force
                swarm_control_ui[j, :] += -force

                # -- Stream-accumulate metrics --
                phi_rij = gij * aij
                sum_phi += phi_rij;
                cnt_phi += 1
                sum_r += rij;
                cnt_r += 1

                # -- Draw edges for small scale --
                if do_viz_details:
                    comm_mat[i, j] = phi_rij
                    comm_mat[j, i] = phi_rij

        # 5) New: Numerical stability check
        # Check if control inputs are excessively large, which is often a sign of numerical instability
        max_force_norm = np.linalg.norm(swarm_control_ui, axis=1).max()
        if max_force_norm > STABILITY_THRESHOLD:
            print(f"[error] Numerical instability detected at it={it}, max control input norm: {max_force_norm:.2e}. Terminating early.")
            break

        # Update all positions and zero out control inputs
        swarm_position += step_size * swarm_control_ui
        swarm_control_ui.fill(0.0)

        # 6) Enqueue metrics (raw + display)
        Jn_new = (sum_phi / cnt_phi) if cnt_phi > 0 else 0.0
        rn_new = (sum_r / cnt_r) if cnt_r > 0 else 0.0
        Jn_raw.append(Jn_new);
        rn_raw.append(rn_new)
        Jn.append(round(Jn_new, 4));
        rn.append(round(rn_new, 4))
        t_elapsed.append(time.time() - start_time)

        # 7) Log throttling (bug fix: logging is done after calculation)
        if it % log_every == 0:
            print(f"[it={it}] Jn={Jn[-1]:.4f} rn={rn[-1]:.4f}")
            # No-edge warning (ignore the first few warm-up steps)
            if cnt_phi == 0 and it > 10:
                print("   [warn] No adjacent edges in this step (cnt_phi=0). Consider lowering PT or increasing density.")

        # 8) Plot throttling
        if it % plot_every == 0:
            plotting.plot_figures_task1(
                axs, t_elapsed, Jn, rn, swarm_position, PT,
                comm_mat, swarm_size, swarm_paths,
                node_colors, line_colors
            )

        # 9) Convergence criterion: Linear regression slope + standard deviation of recent window (supports fixed iteration mode)
        if not getattr(config, "FORCE_FIXED_ITERS", False):
            if len(Jn_raw) > convergence_window:
                recent = Jn_raw[-convergence_window:]
                slope, std_dev = _slope_and_std(recent)
                if abs(slope) < slope_threshold and std_dev < std_threshold:
                    print(f"[done] Jn converged: t={t_elapsed[-1]:.2f}s, it={it}, "
                          f"slope={slope:.2e} (<{slope_threshold:.2e}), "
                          f"std={std_dev:.2e} (<{std_threshold:.2e})")
                    break
    # Refresh the last frame (prevents `it` from being undefined when max_iter=0)
    if t_elapsed and ((len(Jn) - 1) % plot_every != 0):
        plotting.plot_figures_task1(
            axs, t_elapsed, Jn, rn, swarm_position, PT,
            comm_mat, swarm_size, swarm_paths,
            node_colors, line_colors
        )

    # To be compatible with the old interface, return comm_mat and a placeholder None (neighbor matrix is no longer maintained)
    return Jn, rn, swarm_position, t_elapsed, comm_mat, None
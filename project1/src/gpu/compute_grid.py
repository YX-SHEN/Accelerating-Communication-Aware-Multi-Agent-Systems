# src/gpu/compute_grid.py
import cupy as cp
import numpy as np

def comm_radius_from_config(cfg):
    # R = r0 * ((-ln PT)/BETA)^(1/v)
    PT = float(cfg.PT); r0 = float(cfg.R0)
    beta = float(cfg.BETA); v = float(cfg.V)
    PT = np.clip(PT, 1e-12, 1 - 1e-12)
    return r0 * ((-np.log(PT)) / beta) ** (1.0 / v)

def build_uniform_grid(pos_dev: cp.ndarray, cell_size: float):
    """pos_dev: [N,2] (cp.float32)
    return:
      dict(order, cell_start, cell_end, nx, ny, origin_x, origin_y, cell_size)
      - order:        sorted_idx -> original_i (cp.int32)
      - cell_start/end: Each cell in the [start, end) range after sorting (cp.int32)
      - nx, ny:       Python int
      - origin_x/y:   Python float
    """
    assert isinstance(pos_dev, cp.ndarray) and pos_dev.ndim == 2 and pos_dev.shape[1] == 2

    N = pos_dev.shape[0]
    xmin = cp.min(pos_dev[:, 0])
    xmax = cp.max(pos_dev[:, 0])
    ymin = cp.min(pos_dev[:, 1])
    ymax = cp.max(pos_dev[:, 1])

    # Boundary padding
    cell_size_f = cp.asarray(cell_size, dtype=cp.float32)
    pad = cp.float32(1e-3) * cell_size_f
    ox = xmin - pad
    oy = ymin - pad

    # Calculate grid dimensions (in CuPy, avoid casting like cp.int32(array))
    nx_arr = cp.ceil((xmax - ox) / cell_size_f).astype(cp.int32)
    ny_arr = cp.ceil((ymax - oy) / cell_size_f).astype(cp.int32)
    nx_arr = cp.maximum(nx_arr, cp.asarray(1, dtype=cp.int32))
    ny_arr = cp.maximum(ny_arr, cp.asarray(1, dtype=cp.int32))
    # Python scalars (more stable for subsequent operations)
    nx = int(nx_arr.item())
    ny = int(ny_arr.item())
    n_cells = int(nx * ny)

    # Cell for each point (calculate as float, then floor, then clip to [0, nx-1] / [0, ny-1])
    gx = cp.floor((pos_dev[:, 0] - ox) / cell_size_f).astype(cp.int32)
    gy = cp.floor((pos_dev[:, 1] - oy) / cell_size_f).astype(cp.int32)
    gx = cp.clip(gx, 0, nx - 1)
    gy = cp.clip(gy, 0, ny - 1)

    # Linear cell ID, and ensure it is int32
    nx_i32 = cp.asarray(nx, dtype=cp.int32)
    cell_id = (gx + gy * nx_i32).astype(cp.int32)

    # Sort by cell; order: sorted_idx -> original_i
    order = cp.argsort(cell_id, kind="stable").astype(cp.int32)
    cell_sorted = cell_id[order]

    # The [start, end) range for each cell in the sorted result
    uniq, idx_start, counts = cp.unique(
        cell_sorted, return_index=True, return_counts=True
    )
    uniq = uniq.astype(cp.int32, copy=False)
    idx_start = idx_start.astype(cp.int32, copy=False)
    counts = counts.astype(cp.int32, copy=False)

    cell_start = cp.full((n_cells,), -1, dtype=cp.int32)
    cell_end   = cp.full((n_cells,), -1, dtype=cp.int32)
    cell_start[uniq] = idx_start
    cell_end[uniq]   = idx_start + counts

    return dict(
        order=order,
        cell_start=cell_start,
        cell_end=cell_end,
        nx=nx, ny=ny,
        origin_x=float(ox.item()), origin_y=float(oy.item()),
        cell_size=float(cell_size),
    )
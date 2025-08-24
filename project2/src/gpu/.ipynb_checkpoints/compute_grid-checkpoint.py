# src/gpu/compute_grid.py
# -*- coding: utf-8 -*-
import numpy as np
try:
    import cupy as cp
except Exception as e:
    cp = None
    _CUPY_ERR = e


def comm_radius_from_config(cfg):
    """
    R = r0 * ((-ln PT)/BETA)^(1/v)
    返回 Python float（上层可按需转成目标 dtype）
    """
    PT = float(cfg.PT)
    r0 = float(cfg.R0)
    beta = float(cfg.BETA)
    v = float(cfg.V)
    PT = np.clip(PT, 1e-12, 1 - 1e-12)
    return r0 * ((-np.log(PT)) / beta) ** (1.0 / v)


def build_uniform_grid(pos_dev, cell_size: float):
    """
    使用统一网格为 2D 点集构建桶索引（GPU/CuPy 实现）。

    参数
    ----
    pos_dev : cp.ndarray, shape [N,2], dtype 任意浮点（建议跟随 config.DTYPE）
        设备端坐标数组（CuPy）。
    cell_size : float
        网格单元边长（建议与通信半径同阶，如 R 或 R/√2）。

    返回
    ----
    dict(
        order=cp.ndarray[int32],          # sorted_idx -> original_i
        cell_start=cp.ndarray[int32],     # 每个 cell 的起始下标（在“按 cell 排序”后的序列里）
        cell_end=cp.ndarray[int32],       # 每个 cell 的结束下标（半开区间）
        nx=int, ny=int,                   # 网格在 x/y 方向的格数
        origin_x=float, origin_y=float,   # 网格原点（Python float）
        cell_size=float,                  # Python float（原样保存）
    )
    """
    if cp is None:
        raise RuntimeError(
            "CuPy 未可用，无法在 GPU 上构建 uniform grid。原始错误：{}".format(_CUPY_ERR)
        )

    if not isinstance(pos_dev, cp.ndarray) or pos_dev.ndim != 2 or pos_dev.shape[1] != 2:
        raise ValueError("pos_dev 必须是 cp.ndarray，形状 [N,2]。")

    # —— 统一精度：跟随输入数据的 dtype（支持 float32 / float64） ——
    dt = pos_dev.dtype

    N = pos_dev.shape[0]
    xmin = cp.min(pos_dev[:, 0])
    xmax = cp.max(pos_dev[:, 0])
    ymin = cp.min(pos_dev[:, 1])
    ymax = cp.max(pos_dev[:, 1])

    # 边界冗余（使用相同 dtype，避免隐式升/降精度）
    cell_size_f = cp.asarray(cell_size, dtype=dt)
    pad = cp.asarray(1e-3, dtype=dt) * cell_size_f
    ox = xmin - pad
    oy = ymin - pad

    # 计算格数（避免 cp.int32(array) 这种不稳写法）
    nx_arr = cp.ceil((xmax - ox) / cell_size_f).astype(cp.int32)
    ny_arr = cp.ceil((ymax - oy) / cell_size_f).astype(cp.int32)
    nx_arr = cp.maximum(nx_arr, cp.asarray(1, dtype=cp.int32))
    ny_arr = cp.maximum(ny_arr, cp.asarray(1, dtype=cp.int32))

    # 转 Python 标量（某些索引/分配路径更稳）
    nx = int(nx_arr.item())
    ny = int(ny_arr.item())
    n_cells = int(nx * ny)

    # 每点所在 cell（floor 后裁剪到合法范围）
    gx = cp.floor((pos_dev[:, 0] - ox) / cell_size_f).astype(cp.int32)
    gy = cp.floor((pos_dev[:, 1] - oy) / cell_size_f).astype(cp.int32)
    gx = cp.clip(gx, 0, nx - 1)
    gy = cp.clip(gy, 0, ny - 1)

    # 线性 cell id：id = gx + gy * nx
    nx_i32 = cp.asarray(nx, dtype=cp.int32)
    cell_id = (gx + gy * nx_i32).astype(cp.int32)

    # 按 cell 排序；order: sorted_idx -> original_i
    order = cp.argsort(cell_id, kind="stable").astype(cp.int32)
    cell_sorted = cell_id[order]

    # 每个 cell 在排序结果中的 [start, end)
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
        nx=nx,
        ny=ny,
        origin_x=float(ox.item()),
        origin_y=float(oy.item()),
        cell_size=float(cell_size),
    )
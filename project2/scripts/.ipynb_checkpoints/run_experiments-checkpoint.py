# scripts/run_experiments.py
# -*- coding: utf-8 -*-
"""
论文级一键实验脚本（多规模 N、多次重复、可选 GPU-Baseline）：
- Baseline:              src.common.algorithms
- CPU-Advanced:          src.optimized.algorithms_advanced
- GPU-Baseline (可选):   自动探测 src.gpu.algorithms_gpu_unified / algorithms_gpu_improved
- GPU-Advanced:          自动探测 src.gpu.algorithms_advanced_gpu / src.gpu.algorithms_advanced_gpu

统计与产物
- 每次 run：runtime、iters、final Jn/rn、是否收敛、(可选) GPU backend、峰值显存、(可选) kernel ms
- 聚合：mean/std/CI、相对 Baseline 的 speedup、log-log 复杂度斜率
- 图：runtime、speedup、ms/iter、kernel ms/iter、complexity、edges/sec、GPU mem、runtime/N^2
- 文本报告 + metadata.json + 两张 CSV（长表 & 汇总）
"""

import os, sys, io, re, json, math, time, glob, platform, threading
from datetime import datetime
from contextlib import redirect_stdout
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # 服务器/无界面环境
import matplotlib.pyplot as plt

# 以模块方式运行：python -m scripts.run_experiments
# 项目根就是 sys.path[0]；确保能 import config/src.*
import config
try:
    import psutil  # 用来拿可用 RAM
except Exception:
    psutil = None
# =============================
# 全局 & 标签
# =============================

GLOBAL_SEED = 20250101
np.random.seed(GLOBAL_SEED)

METHOD_BASE = "Baseline"
METHOD_CPU_ADV = "CPU-Advanced"
METHOD_GPU_BASE = "GPU-Baseline"
METHOD_GPU_ADV = "GPU-Advanced"
METHOD_GPU_ADV2 = "GPU-Advanced v2"
ALL_METHODS = [METHOD_BASE, METHOD_CPU_ADV, METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]

# --- ONLY 白名单解析（环境变量/命令行皆可） ---
import re

def _parse_only_filter() -> set:
    """
    解析 ONLY 白名单，返回 canonical 名称集合（与 METHOD_* 一致）。
    支持的写法（大小写/连字符不敏感）：baseline/cpu/gpu-advanced/gpu-advanced-v2/ad1/ad2/ga/ga2 等。
    来源优先级：命令行 --only > 环境变量 ONLY。
    """
    # 命令行通过 __main__ 保存（见改动⑤）
    raw_cli = getattr(sys.modules.get("__main__"), "_ONLY_CLI", "") or ""
    raw_env = (os.getenv("ONLY", "") or "").strip()
    raw = raw_cli or raw_env
    if not raw:
        return set()

    tokens = [t.strip().lower() for t in re.split(r"[,\s/;]+", raw) if t.strip()]

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s)

    alias = {
        "baseline": METHOD_BASE, "base": METHOD_BASE, "b": METHOD_BASE,
        "cpuadvanced": METHOD_CPU_ADV, "cpu": METHOD_CPU_ADV, "adcpu": METHOD_CPU_ADV,
        "gpubaseline": METHOD_GPU_BASE, "gpubase": METHOD_GPU_BASE, "gb": METHOD_GPU_BASE,
        "gpuadvanced": METHOD_GPU_ADV, "ga": METHOD_GPU_ADV, "ad1": METHOD_GPU_ADV, "adv1": METHOD_GPU_ADV,
        "gpuadvancedv2": METHOD_GPU_ADV2, "ga2": METHOD_GPU_ADV2, "ad2": METHOD_GPU_ADV2, "adv2": METHOD_GPU_ADV2,
    }

    out = set()
    for t in tokens:
        key = norm(t)
        if key in alias:
            out.add(alias[key])
        else:
            # 兜底：例如 "gpu-advanced v2" 这样的空格写法
            if key.replace("v", "") in ("gpuadvanced2",):
                out.add(METHOD_GPU_ADV2)
    return out

def _derive_cut_radius_from_PT() -> float:
    # R_cut = R0 * ( ln(1/PT) / (alpha * (2^delta - 1)) )^(1/v)
    import math, config
    A = float(getattr(config, "ALPHA", 1.0))
    V = float(getattr(config, "V", 2.0))
    R0 = float(getattr(config, "R0", 1.0))
    D = float(getattr(config, "DELTA", 1.0))
    PT = float(getattr(config, "PT", 0.01))
    PT = min(max(PT, 1e-12), 1 - 1e-12)   # clamp (0,1)
    C = A * (2.0**D - 1.0)
    return R0 * ((math.log(1.0 / PT) / C) ** (1.0 / V))

# =============================
# 简易 GPU 显存峰值监视（NVML/CuPy）
# =============================
class GPUMemoryMonitor:
    def __init__(self, device_index: int = 0, interval_sec: float = 0.05):
        self.device_index = device_index
        self.interval = interval_sec
        self.enabled = False
        self._stop = threading.Event()
        self._thr = None
        self.peak_bytes = None
        self._nvml = None
        self._handle = None
        self._cupy = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.enabled = True
        except Exception:
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() > 0:
                    self._cupy = cp
                    self.enabled = True
            except Exception:
                self.enabled = False

    def _used(self) -> int:
        if self._nvml:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            return int(info.used)
        if self._cupy:
            free_b, total_b = self._cupy.cuda.runtime.memGetInfo()
            return int(total_b - free_b)
        return 0

    def start(self):
        if not self.enabled: return
        self.peak_bytes = 0
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                u = self._used()
                if u > (self.peak_bytes or 0):
                    self.peak_bytes = u
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        if not self.enabled: return
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)


# =============================
# 工具：初始位置
# =============================
def make_positions(N: int, seed: int, mode: str = "gaussian") -> np.ndarray:
    g = np.random.default_rng(seed)
    dtype = getattr(config, "DTYPE", np.float32)
    if mode == "gaussian":
        x = g.standard_normal((N, 2)).astype(dtype)
        return x * np.asarray(20.0, dtype=dtype)
    # grid + noise
    side = int(np.ceil(np.sqrt(N)))
    spacing = np.asarray(10.0, dtype=dtype)
    xs, ys = np.meshgrid(np.arange(side, dtype=dtype) * spacing,
                         np.arange(side, dtype=dtype) * spacing)
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(dtype)[:N]
    noise = g.normal(0.0, 5.0, size=grid.shape).astype(dtype)
    return grid + noise


# =============================
# 运行单次：Baseline / CPU-Advanced / GPU(B/L)
# 捕获 stdout 以解析 GPU 'compute=XXms'
# =============================
def _prepare_colors(N):
    rs = np.random.RandomState(GLOBAL_SEED + 1337 + N)
    return rs.rand(N, N, 3)


def run_baseline_once(N, max_iter, init_positions):
    from src.common import algorithms as alg
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.ioff()
    try:
        line_colors = _prepare_colors(N)
        swarm_paths = []
        t0 = time.time()
        Jn, rn, final_pos, *_ = alg.run_simulation(
            axs=axs, fig=fig, swarm_position=init_positions.astype(getattr(config, "DTYPE", np.float32), copy=True),
            max_iter=max_iter, swarm_size=N,
            alpha=config.ALPHA, beta=config.BETA, v=config.V, r0=config.R0, PT=config.PT,
            swarm_paths=swarm_paths,
            node_colors=(
                (getattr(config, "NODE_COLORS", [])[:N])
                if len(getattr(config, "NODE_COLORS", [])) >= N
                else ["blue"] * N
            ),
            line_colors=line_colors
        )
        rt = time.time() - t0
        plt.close(fig)
        return dict(runtime=rt, iters=len(Jn), final_Jn=float(Jn[-1]) if Jn else np.nan,
                    final_rn=float(rn[-1]) if rn else np.nan, converged=(len(Jn) < max_iter),
                    final_positions=final_pos)
    finally:
        plt.close(fig)


def run_cpu_adv_once(N, max_iter, init_positions):
    from src.optimized import algorithms_advanced as adv
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.ioff()
    try:
        line_colors = _prepare_colors(N)
        swarm_paths = []
        t0 = time.time()
        Jn, rn, final_pos, *_ = adv.run_simulation(
            axs=axs, fig=fig, swarm_position=init_positions.astype(getattr(config, "DTYPE", np.float32), copy=True),
            max_iter=max_iter, swarm_size=N,
            alpha=config.ALPHA, beta=config.BETA, v=config.V, r0=config.R0, PT=config.PT,
            swarm_paths=swarm_paths,
            node_colors=(
                (getattr(config, "NODE_COLORS", [])[:N])
                if len(getattr(config, "NODE_COLORS", [])) >= N
                else ["blue"] * N
            ),
            line_colors=line_colors
        )
        rt = time.time() - t0
        return dict(runtime=rt, iters=len(Jn), final_Jn=float(Jn[-1]) if Jn else np.nan,
                    final_rn=float(rn[-1]) if rn else np.nan, converged=(len(Jn) < max_iter),final_positions=final_pos)
    finally:
        plt.close(fig)


def _import_first(mod_names: List[str]) -> Optional[object]:
    for name in mod_names:
        try:
            mod = __import__(name, fromlist=["run_simulation"])
            if hasattr(mod, "run_simulation"):
                return mod
        except Exception:
            continue
    return None


def _guess_gpu_backend(mod) -> str:
    for attr in ["_GPU_INST", "gpu", "backend"]:
        try:
            obj = getattr(mod, attr)
            if isinstance(obj, str):
                return obj
            if hasattr(obj, "backend"):
                return str(obj.backend)
        except Exception:
            pass
    return "unknown"


def _run_gpu_once_generic(N, max_iter, init_positions, module_names: List[str], monitor_gpu: bool = True):
    mod = _import_first(module_names)
    if mod is None:
        raise RuntimeError(f"Module does not exist：{module_names}")
    gpu_backend = _guess_gpu_backend(mod)

    # 取消 CuPy 默认 1GB 内存池限额（你的 backend.py 会设 1GB；这里覆盖为“不限”）
    try:
        import cupy as _cp
        _cp.get_default_memory_pool().set_limit(size=0)  # 0 = 不限额
    except Exception:
        pass

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.ioff()

    # 大规模避免 NxN 矩阵
    orig_keep = getattr(config, "KEEP_FULL_MATRIX", True)
    # 当你想测显存上限/随 N 的 O(N^2) 扩张时，导出 GPU_BASE_FORCE_DENSE=1
    force_dense = os.getenv("GPU_BASE_FORCE_DENSE", "0") == "1"
    config.KEEP_FULL_MATRIX = True if force_dense else False

    mon = GPUMemoryMonitor() if monitor_gpu else None
    if mon and mon.enabled:
        mon.start()

    buf = io.StringIO()
    try:
        line_colors = _prepare_colors(N)
        swarm_paths = []
        t0 = time.time()
        # 捕获 stdout，解析 compute=XXms
        with redirect_stdout(buf):
            Jn, rn, final_pos, *_ = mod.run_simulation(
                axs=axs, fig=fig, swarm_position=init_positions.copy(),
                max_iter=max_iter, swarm_size=N,
                alpha=config.ALPHA, beta=config.BETA, v=config.V, r0=config.R0, PT=config.PT,
                swarm_paths=swarm_paths,
                node_colors=(
                    (getattr(config, "NODE_COLORS", [])[:N])
                    if len(getattr(config, "NODE_COLORS", [])) >= N
                    else ["blue"] * N
                ),
                line_colors=line_colors
            )
        rt = time.time() - t0
        # 解析 kernel 平均时延
        log = buf.getvalue()
        if os.getenv("PASSTHRU_LOG", "0") == "1":
            print(log, end="")
        ms_list = []
        ms_list += [float(x) for x in re.findall(r"(?:compute\s*=\s*|GPU calculate:\s*)([0-9]+(?:\.[0-9]+)?)ms", log)]
        ms_list += [float(x) for x in re.findall(r"平均compute时间:\s*([0-9]+(?:\.[0-9]+)?)ms", log)]
        kernel_ms_mean = float(np.mean(ms_list)) if ms_list else np.nan
        # NEW: parse current update mode from captured stdout (algorithms_gpu_unified prints “模式: xxx”)
        mode = None
        m = re.search(r"模式:\s*([A-Za-z]+)", log)
        if m:
            mode = m.group(1)
        peak = mon.peak_bytes if (mon and mon.enabled) else np.nan
        return dict(runtime=rt, iters=len(Jn), final_Jn=float(Jn[-1]) if Jn else np.nan,
                    final_rn=float(rn[-1]) if rn else np.nan, converged=(len(Jn) < max_iter),
                    gpu_backend=gpu_backend, gpu_mem_peak_bytes=peak,
                    kernel_ms_mean=kernel_ms_mean,final_positions=final_pos,gpu_mode=mode,)
    finally:
        if mon and mon.enabled:
            mon.stop()
        config.KEEP_FULL_MATRIX = orig_keep
        plt.close(fig)


def run_gpu_base_once(N, max_iter, init_positions, monitor_gpu=True):
    return _run_gpu_once_generic(
        N, max_iter, init_positions,
        module_names=[
            "src.gpu.algorithms_gpu_unified",
            "src.gpu.algorithms_gpu_improved",
        ],
        monitor_gpu=monitor_gpu,
    )

def run_gpu_adv_once(N, max_iter, init_positions, monitor_gpu=True):
    return _run_gpu_once_generic(
        N, max_iter, init_positions,
        module_names=[
            "src.gpu.algorithms_advanced_gpu",
        ],
        monitor_gpu=monitor_gpu,
    )

def run_gpu_adv2_once(N, max_iter, init_positions, monitor_gpu=True):
    return _run_gpu_once_generic(
        N, max_iter, init_positions,
        module_names=[
            "src.gpu.algorithms_advanced_gpu2",  # 你的 v2 文件
        ],
        monitor_gpu=monitor_gpu,
    )


# =============================
# 聚合/绘图/报告
# =============================
def ci95(std: float, n: int) -> float:
    if not np.isfinite(std) or n <= 1: return np.nan
    return 1.96 * std / np.sqrt(n)


def summarize(df_long: pd.DataFrame) -> pd.DataFrame:
    agg = df_long.groupby(["N", "method"], as_index=False).agg(
        runtime_mean=("runtime", "mean"), runtime_std=("runtime", "std"), repeats=("runtime", "count"),
        iters_mean=("iters", "mean"), iters_std=("iters", "std"),
        ms_per_iter_mean=("ms_per_iter", "mean"), ms_per_iter_std=("ms_per_iter", "std"),
        kernel_ms_per_iter_mean=("kernel_ms_per_iter", "mean"), kernel_ms_per_iter_std=("kernel_ms_per_iter", "std"),
        converged_rate=("converged", "mean"),
        Jn_mean=("final_Jn", "mean"), rn_mean=("final_rn", "mean"),
        edges_per_sec_mean=("edges_per_sec", "mean"),
        gpu_mem_peak_bytes_mean=("gpu_mem_peak_bytes", "mean")
    )
    # speedup 相对 Baseline
    base = agg[agg["method"] == METHOD_BASE][["N", "runtime_mean"]].set_index("N")["runtime_mean"].to_dict()
    agg["speedup_vs_baseline"] = agg.apply(
        lambda r: (base.get(r["N"], np.nan) / r["runtime_mean"]) if (
                    r["runtime_mean"] > 0 and np.isfinite(base.get(r["N"], np.nan))) else np.nan,
        axis=1
    )
    return agg.sort_values(["method", "N"])


def fit_loglog_slopes(df_sum: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in df_sum["method"].unique():
        sub = df_sum[(df_sum["method"] == m) & np.isfinite(df_sum["runtime_mean"])].sort_values("N")
        if len(sub) >= 2:
            x = np.log(sub["N"].values.astype(float))
            y = np.log(sub["runtime_mean"].values.astype(float))
            A = np.vstack([np.ones_like(x), x]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            rows.append({"method": m, "slope": float(coef[1])})
    return pd.DataFrame(rows)


def _human_bytes(x):
    if not np.isfinite(x): return "NA"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(x);
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024;
        i += 1
    return f"{v:.2f}{units[i]}"


def plot_all(df_sum: pd.DataFrame, df_long: pd.DataFrame, out_dir: str):
    colors = {
        METHOD_BASE: "C0",
        METHOD_CPU_ADV: "C1",
        METHOD_GPU_BASE: "C2",
        METHOD_GPU_ADV: "C3",
        METHOD_GPU_ADV2: "C4",
    }

    # 1) Runtime（log-log）
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    for m in [METHOD_BASE, METHOD_CPU_ADV, METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]:
        sub = df_sum[df_sum["method"] == m].sort_values("N")
        if len(sub) == 0: continue
        ax.errorbar(sub["N"], sub["runtime_mean"], yerr=sub["runtime_std"],
                    fmt="o-", linewidth=2, markersize=6, capsize=3, label=m, color=colors.get(m))
    ax.set_xscale("log");
    ax.set_yscale("log")
    ax.set_xlabel("Swarm Size (N)");
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime (mean ± std, log-log)")
    ax.grid(True, alpha=0.3);
    ax.legend()
    plt.tight_layout();
    plt.savefig(os.path.join(out_dir, "runtime_mean_std.png"), dpi=300);
    plt.close()

    # 2) Speedup vs Baseline
    # 2) Speedup vs Baseline
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    for m in [METHOD_CPU_ADV, METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]:
        sub = df_sum[df_sum["method"] == m].sort_values("N")
        if len(sub) == 0: continue
        ax.plot(sub["N"], sub["speedup_vs_baseline"], "o-", linewidth=2, markersize=6, label=m, color=colors.get(m))
    ax.set_xscale("log");
    ax.set_xlabel("Swarm Size (N)");
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Speedup relative to Baseline");
    ax.grid(True, alpha=0.3);
    ax.legend()
    plt.tight_layout();
    plt.savefig(os.path.join(out_dir, "speedup_vs_baseline.png"), dpi=300);
    plt.close()

    # 3) Per-iteration time（log-log）
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    for m in [METHOD_BASE, METHOD_CPU_ADV, METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]:
        sub = df_sum[df_sum["method"] == m].sort_values("N")
        if len(sub) == 0: continue
        ax.errorbar(sub["N"], sub["ms_per_iter_mean"], yerr=sub["ms_per_iter_std"],
                    fmt="s-", linewidth=2, markersize=6, capsize=3, label=m, color=colors.get(m))
    ax.set_xscale("log");
    ax.set_yscale("log")
    ax.set_xlabel("Swarm Size (N)");
    ax.set_ylabel("Iteration time (ms)")
    ax.set_title("Per-iteration time (mean ± std, log-log)")
    ax.grid(True, alpha=0.3);
    ax.legend()
    plt.tight_layout();
    plt.savefig(os.path.join(out_dir, "iter_time_mean_std.png"), dpi=300);
    plt.close()

    # 4) Kernel per-iteration（若可解析）
    has_kernel = df_sum["kernel_ms_per_iter_mean"].notna().any()
    if has_kernel:
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        for m in [METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]:
            sub = df_sum[(df_sum["method"] == m) & df_sum["kernel_ms_per_iter_mean"].notna()].sort_values("N")
            if len(sub) == 0: continue
            ax.errorbar(sub["N"], sub["kernel_ms_per_iter_mean"], yerr=sub["kernel_ms_per_iter_std"],
                        fmt="^-", linewidth=2, markersize=6, capsize=3, label=m, color=colors.get(m))
        ax.set_xscale("log");
        ax.set_yscale("log")
        ax.set_xlabel("Swarm Size (N)");
        ax.set_ylabel("Kernel time per iter (ms)")
        ax.set_title("GPU kernel time per iteration");
        ax.grid(True, alpha=0.3);
        ax.legend()
        plt.tight_layout();
        plt.savefig(os.path.join(out_dir, "kernel_iter_time.png"), dpi=300);
        plt.close()

    # 5) Complexity fit：log t vs log N + 直线拟合
    plt.figure(figsize=(9, 6))
    for m in [METHOD_BASE, METHOD_CPU_ADV, METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]:
        sub = df_sum[df_sum["method"] == m].sort_values("N")
        if len(sub) < 2: continue
        x = np.log(sub["N"].values.astype(float))
        y = np.log(sub["runtime_mean"].values.astype(float))
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        yfit = A @ coef
        plt.scatter(x, y, label=f"{m} (slope≈{coef[1]:.2f})", s=30)
        plt.plot(x, yfit, "--")
    plt.xlabel("log N");
    plt.ylabel("log Runtime (s)")
    plt.title("Complexity fit: log t = a + b log N");
    plt.grid(True, alpha=0.3);
    plt.legend()
    plt.tight_layout();
    plt.savefig(os.path.join(out_dir, "complexity_fit.png"), dpi=300);
    plt.close()

    # 6) 吞吐（edges/sec）
    plt.figure(figsize=(9, 6))
    for m in [METHOD_BASE, METHOD_CPU_ADV, METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]:
        sub = df_sum[df_sum["method"] == m].sort_values("N")
        if len(sub) == 0: continue
        plt.plot(sub["N"], sub["edges_per_sec_mean"], "o-", label=m, linewidth=2, markersize=6)
    plt.xscale("log");
    plt.xlabel("Swarm Size (N)");
    plt.ylabel("Directed edges / sec")
    plt.title("Throughput on final graph");
    plt.grid(True, alpha=0.3);
    plt.legend()
    plt.tight_layout();
    plt.savefig(os.path.join(out_dir, "throughput.png"), dpi=300);
    plt.close()

    # 7) GPU 内存峰值
    gpu_df = df_sum[(df_sum["method"].isin([METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2])) &
                    df_sum["gpu_mem_peak_bytes_mean"].notna()]
    if len(gpu_df) > 0:
        plt.figure(figsize=(9, 6))
        for m in [METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]:
            sub = gpu_df[gpu_df["method"] == m].sort_values("N")
            if len(sub) == 0: continue
            plt.plot(sub["N"], sub["gpu_mem_peak_bytes_mean"] / 1024 / 1024, "^-", linewidth=2, markersize=6, label=m)
        plt.xscale("log");
        plt.xlabel("Swarm Size (N)");
        plt.ylabel("GPU Memory Peak (MB, mean)")
        plt.title("GPU memory peak vs N");
        plt.grid(True, alpha=0.3);
        plt.legend()
        plt.tight_layout();
        plt.savefig(os.path.join(out_dir, "gpu_memory_peak.png"), dpi=300);
        plt.close()

    # 8) 归一化时间 t/N^2（观察 O(N^2) 瓶颈）
    plt.figure(figsize=(9, 6))
    for m in [METHOD_BASE, METHOD_CPU_ADV, METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]:
        sub = df_sum[df_sum["method"] == m].sort_values("N")
        if len(sub) == 0: continue
        norm = sub["runtime_mean"].values / (sub["N"].values.astype(float) ** 2)
        plt.plot(sub["N"], norm, "o-", label=m, linewidth=2, markersize=6)
    plt.xscale("log");
    plt.yscale("log")
    plt.xlabel("Swarm Size (N)");
    plt.ylabel("Runtime / N^2 (s)")
    plt.title("Runtime normalized by N^2");
    plt.grid(True, alpha=0.3);
    plt.legend()
    plt.tight_layout();
    plt.savefig(os.path.join(out_dir, "runtime_over_N2.png"), dpi=300);
    plt.close()


def count_edges_final(positions: Optional[np.ndarray]) -> Tuple[int, int]:
    """返回 (E_directed, E_undirected) 按 PT 阈值筛选。"""
    if positions is None: return 0, 0
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    rij = np.sqrt((diff ** 2).sum(axis=2))
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    aij = np.exp(-config.ALPHA * (2 ** config.DELTA - 1) * (rij / float(config.R0)) ** config.V)
    conn = (aij >= config.PT) & mask
    e_dir = int(conn.sum())
    e_und = e_dir // 2
    return e_dir, e_und


def print_and_save_report(df_sum: pd.DataFrame, slopes: pd.DataFrame, out_dir: str):
    lines = []
    lines.append("🎯 === Comprehensive performance test report ===")
    lines.append(f"Output Directory: {out_dir}")
    lines.append("Key charts：")
    lines.append("  - runtime_mean_std.png（Overall runtime，log-log）")
    lines.append("  - speedup_vs_baseline.png（Speedup relative to Baseline）")
    lines.append("  - iter_time_mean_std.png（Time per step ms/iter，log-log）")
    lines.append("  - kernel_iter_time.png（GPU kernel ms/iter，If it can be resolved）")
    lines.append("  - complexity_fit.png（log-log Fitting slope）")
    lines.append("  - throughput.png（The final state is hesitant and spitting out）")
    lines.append("  - gpu_memory_peak.png（GPU memory peak vs N）")
    lines.append("  - runtime_over_N2.png（Normalizing time to reveal the O(N²) bottleneck）")
    lines.append("")

    # 复杂度斜率
    if len(slopes) > 0:
        lines.append("  - throughput.png (Final graph throughput, edges/sec)")
        for _, r in slopes.iterrows():
            lines.append(f"  {r['method']}: b ≈ {r['slope']:.2f}")
        lines.append("")

    # 汇总表（简要）
    piv = df_sum.pivot(index="N", columns="method", values="runtime_mean").sort_index()
    lines.append("Mean runtime (seconds)：")
    lines.append(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    lines.append("")
    # 最佳加速（只有在存在 Baseline 且该方法有有效 speedup 时才打印）
    for m in [METHOD_CPU_ADV, METHOD_GPU_BASE, METHOD_GPU_ADV, METHOD_GPU_ADV2]:
        sub = df_sum[(df_sum["method"] == m) & df_sum["speedup_vs_baseline"].notna()]
        if sub.empty:
            continue  # 无 Baseline 或该方法没有有效 speedup，跳过
        idx = sub["speedup_vs_baseline"].idxmax()
        row = sub.loc[idx]
        lines.append(f"🏆 {m} Maximum acceleration：{row['speedup_vs_baseline']:.2f}× @ N={int(row['N'])}")
    txt = "\n".join(lines)
    print("\n" + txt + "\n")
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


def env_fingerprint() -> dict:
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import numpy as _np;
        info["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import scipy as _sp;
        info["scipy"] = _sp.__version__
    except Exception:
        pass
    try:
        import jax;
        info["jax"] = jax.__version__;
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    try:
        import cupy as cp
        info["cupy"] = cp.__version__
        try:
            cnt = cp.cuda.runtime.getDeviceCount()
            info["cuda_device_count"] = int(cnt)
            if cnt > 0: info["cuda_device"] = str(cp.cuda.Device())
        except Exception:
            pass
    except Exception:
        pass
    return info


# =============================
# 主流程
# =============================
def _auto_limits(max_iterations: int):
    """
    估算每条路线可承受的最大 N（保守值）。
    - CPU Baseline/Advanced：按 O(N^2) 且有 3 个 N×N 矩阵（float64）估算，并留 50% 安全余量
    - GPU Baseline：脚本里会强制 KEEP_FULL_MATRIX=False → 近似 O(N)，给个很大的上限
      若你显式 KEEP_FULL_MATRIX=True，则按显存粗略估一次 O(N^2) 的上限
    """
    # 可用 RAM（B）
    if psutil:
        ram_avail = int(psutil.virtual_memory().available)
    else:
        # 没有 psutil 就用一个保守的 4GB 可用内存估算
        ram_avail = 4 * 1024**3

    # CPU：3 个 N×N 矩阵，float64（8B），再 ×1.5 安全系数
    bytes_per_elem_cpu = 8
    k_cpu = 3.0 * 1.5
    usable_cpu = 0.5 * ram_avail  # 只用 50% 可用内存
    N_cpu_max = int((usable_cpu / (k_cpu * bytes_per_elem_cpu)) ** 0.5)

    # GPU：默认我们在 _run_gpu_once_generic 里把 KEEP_FULL_MATRIX=False
    # → O(N) 用量很小，这里给一个很大的上限
    gpu_base_max_default = 200_000

    # 如果显式 KEEP_FULL_MATRIX=True，尝试按显存估一把 O(N^2) 上限
    gpu_base_max_dense = gpu_base_max_default
    try:
        import cupy as _cp
        free_b, total_b = _cp.cuda.runtime.memGetInfo()
        # float32（4B），临时张量多，给 16 * N^2 * 4B 的保守上限
        bytes_per_elem_gpu = 4
        k_gpu = 16.0
        usable_gpu = 0.7 * free_b
        gpu_base_max_dense = int((usable_gpu / (k_gpu * bytes_per_elem_gpu)) ** 0.5)
    except Exception:
        pass

    return {
        "CPU_BASE_MAX_N":  max(N_cpu_max, 512),           # 不至于太小
        "CPU_ADV_MAX_N":   max(int(N_cpu_max * 1.5), 512),
        "GPU_BASE_MAX_N":  gpu_base_max_default,          # KEEP_FULL_MATRIX=False 时使用
        "GPU_BASE_MAX_N_DENSE": gpu_base_max_dense,       # KEEP_FULL_MATRIX=True 时用
        "CPU_ITERS_CAP":   max_iterations                  # 默认不降步；可按需改
    }

def run_comprehensive_experiments(
        sizes: Optional[List[int]] = None,
        max_iterations: int = 200,
        repeats: int = 3,
        pos_mode: str = "gaussian",
        poll_gpu_mem: bool = True,
        fixed_iters: bool = False,   # <<< 已有
        out_name: Optional[str] = None,   # <<< 新增
):

    # 透传 AD2_CELL_SCALE（若 algorithms_advanced_gpu2 使用 config.AD2_CELL_SCALE）
    try:
        if "AD2_CELL_SCALE" in os.environ:
            setattr(config, "AD2_CELL_SCALE", float(os.environ["AD2_CELL_SCALE"]))
    except Exception:
        pass
    # 透传 PT（密度扫描时非常关键）
    if "PT" in os.environ:
        try:
            setattr(config, "PT", float(os.environ["PT"]))
            print(f"[cfg] PT from env -> {getattr(config, 'PT')}")
        except Exception as e:
            print("[cfg] PT env parse failed:", e)

    sizes = sizes or [50, 100, 200]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 仅用于 metadata
    tag = out_name or f"{ts}_paper_benchmark"  # 目录名：优先用 --name，否则用时间戳
    out_dir = os.path.join("results", tag)
    # 保存环境指纹……
    with open(os.path.join(out_dir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env_fingerprint(), f, indent=2, ensure_ascii=False)

    # ✨ 新增：把 V4 时间分解 JSONL 固定到当前实验目录
    import os
    os.environ.setdefault("V4_TIMEBREAK_FILE", os.path.join(out_dir, "v4_timebreak_steps.jsonl"))
    print(f"[v4] timebreak jsonl -> {os.environ['V4_TIMEBREAK_FILE']}")
    os.makedirs(out_dir, exist_ok=True)

    # 保存环境指纹
    with open(os.path.join(out_dir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env_fingerprint(), f, indent=2, ensure_ascii=False)

    # 降低绘图/日志开销（公平计时）
    orig = {
        "MODE": getattr(config, "MODE", "hpc"),
        "PLOT_EVERY": getattr(config, "PLOT_EVERY", 20),
        "LOG_EVERY": getattr(config, "LOG_EVERY", 20),
        "KEEP_FULL_MATRIX": getattr(config, "KEEP_FULL_MATRIX", True),
        "SWARM_INITIAL_POSITIONS": config.SWARM_INITIAL_POSITIONS.copy(),
        "SWARM_SIZE": getattr(config, "SWARM_SIZE", config.SWARM_INITIAL_POSITIONS.shape[0]),

    }

    # 备份收敛阈值（fixed-iters 模式需要恢复）
    orig_conv_win   = getattr(config, "CONVERGENCE_WINDOW", 50)
    orig_conv_slope = getattr(config, "CONVERGENCE_SLOPE_THRESHOLD", 1e-6)
    orig_conv_std   = getattr(config, "CONVERGENCE_STD_THRESHOLD", 1e-5)

    # 检查哪些方法可用（GPU-Baseline/Advanced 可能缺）
    have_gpu_base = _import_first(["src.gpu.algorithms_gpu_unified", "src.gpu.algorithms_gpu_improved"]) is not None
    have_gpu_adv = _import_first(["src.gpu.algorithms_advanced_gpu"]) is not None
    have_gpu_adv2 = _import_first(["src.gpu.algorithms_advanced_gpu2"]) is not None

    methods = [METHOD_BASE, METHOD_CPU_ADV]
    if have_gpu_base: methods.append(METHOD_GPU_BASE)
    if have_gpu_adv:  methods.append(METHOD_GPU_ADV)
    if have_gpu_adv2: methods.append(METHOD_GPU_ADV2)

    # === 自动阈值（可用 RAM/显存推断）+ 环境变量可覆盖 ===
    auto = _auto_limits(max_iterations)
    CPU_BASE_MAX_N = int(os.getenv("CPU_BASE_MAX_N",  str(auto["CPU_BASE_MAX_N"])))
    CPU_ADV_MAX_N  = int(os.getenv("CPU_ADV_MAX_N",   str(auto["CPU_ADV_MAX_N"])))
    CPU_ITERS_CAP  = int(os.getenv("CPU_ITERS_CAP",   str(auto["CPU_ITERS_CAP"])))

    # GPU Baseline：如果你在别处强行 KEEP_FULL_MATRIX=True，就用更保守的 dense 上限
    want_dense_gpu_base = bool(int(os.getenv("GPU_BASE_FORCE_DENSE", "0")))
    GPU_BASE_MAX_N = int(os.getenv(
        "GPU_BASE_MAX_N",
        str(auto["GPU_BASE_MAX_N_DENSE"] if want_dense_gpu_base else auto["GPU_BASE_MAX_N"])
    ))

    print(f"[auto-limits] CPU_BASE_MAX_N={CPU_BASE_MAX_N}, CPU_ADV_MAX_N={CPU_ADV_MAX_N}, "
          f"GPU_BASE_MAX_N={GPU_BASE_MAX_N}, CPU_ITERS_CAP={CPU_ITERS_CAP}")

    # === 解析 ONLY 白名单（命令行 --only 优先，环境变量 ONLY 其次） ===
    allowed = _parse_only_filter()
    def _want(name: str) -> bool:
        # 白名单为空 → 跑全部；否则只跑白名单内的方法
        return (len(allowed) == 0) or (name in allowed)

    # 打印 ONLY 白名单和模块可用性，便于定位问题
    print("[only] allow =", sorted(list(allowed)) if allowed else "(all methods)")
    print("[mods] have_gpu_base=", have_gpu_base,
          "have_gpu_adv=", have_gpu_adv,
          "have_gpu_adv2=", have_gpu_adv2)

    rows = []
    try:
        # 统一高性能模式
        config.MODE = "hpc"
        config.PLOT_EVERY = max_iterations + 1
        config.LOG_EVERY = max_iterations + 1
        # 固定步数：禁用提前收敛
        if fixed_iters:
            config.CONVERGENCE_WINDOW = int(1e12)
            config.CONVERGENCE_SLOPE_THRESHOLD = -1.0
            config.CONVERGENCE_STD_THRESHOLD   = -1.0
            setattr(config, "FORCE_FIXED_ITERS", True)  # 供 Baseline 读取

            # === 解析 ONLY 白名单 ===
            allowed = _parse_only_filter()

            def _want(name: str) -> bool:
                # 白名单为空 → 跑全部；否则只跑白名单内的方法
                return (len(allowed) == 0) or (name in allowed)

            # —— 把 PT 映射为 AD 系列真正使用的邻域半径/网格尺寸 ——
            try:
                rcut = _derive_cut_radius_from_PT()
                setattr(config, "R_CUT", rcut)  # 供模块从 config 读取
                os.environ["R_CUT"] = f"{rcut:.6f}"  # 也从环境变量兜底
                cell_scale = float(os.getenv("AD2_CELL_SCALE", "1.5"))
                setattr(config, "AD2_CELL_H", rcut / cell_scale)
                print(f"[dens] PT={getattr(config, 'PT', None)} → R_CUT={rcut:.4f}, "
                      f"AD2_CELL_H={rcut / cell_scale:.4f} (DELTA={getattr(config, 'DELTA', None)})")
            except Exception as _e:
                print("[dens] PT→R_CUT mapping failed, fallback to module defaults:", _e)

        for N in sizes:
            config.SWARM_SIZE = N
            print(f"\n===== N = {N} =====")
            for rep in range(repeats):
                seed = GLOBAL_SEED + N * 1000 + rep
                init_pos = make_positions(N, seed, pos_mode)

                # Baseline（CPU）
                if _want(METHOD_BASE):
                    if N > CPU_BASE_MAX_N:
                        print(f"  (r={rep + 1}/{repeats}) 🐢 {METHOD_BASE} ... SKIP (N>{CPU_BASE_MAX_N})")
                        res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                   final_positions=init_pos)
                    else:
                        print(f"  (r={rep + 1}/{repeats}) 🐢 {METHOD_BASE} ... ", end="", flush=True)
                        try:
                            max_iter_eff = min(max_iterations, CPU_ITERS_CAP)  # 可选：大 N 时限步数
                            res = run_baseline_once(N, max_iter_eff, init_pos)
                            print(f"OK {res['runtime']:.2f}s")
                        except MemoryError:
                            print("SKIP (RAM)")
                            res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                       final_positions=init_pos)
                        except Exception as e:
                            print("SKIP (error)")
                            res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                       final_positions=init_pos)
                    rows.append(_pack_row(N, METHOD_BASE, rep, res, init_pos))
                else:
                    print(f"  (r={rep + 1}/{repeats}) 🐢 {METHOD_BASE} ... SKIP (ONLY)")

                # CPU-Advanced

                # CPU-Advanced
                if _want(METHOD_CPU_ADV):
                    if N > CPU_ADV_MAX_N:
                        print(f"  (r={rep + 1}/{repeats}) ⚡ {METHOD_CPU_ADV} ... SKIP (N>{CPU_ADV_MAX_N})")
                        res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                   final_positions=init_pos)
                    else:
                        print(f"  (r={rep + 1}/{repeats}) ⚡ {METHOD_CPU_ADV} ... ", end="", flush=True)
                        try:
                            max_iter_eff = min(max_iterations, CPU_ITERS_CAP)
                            res = run_cpu_adv_once(N, max_iter_eff, init_pos)
                            print(f"OK {res['runtime']:.2f}s")
                        except MemoryError:
                            print("SKIP (RAM)")
                            res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                       final_positions=init_pos)
                        except Exception as e:
                            print("SKIP (error)")
                            res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                       final_positions=init_pos)
                    rows.append(_pack_row(N, METHOD_CPU_ADV, rep, res, init_pos))
                else:
                    print(f"  (r={rep + 1}/{repeats}) ⚡ {METHOD_CPU_ADV} ... SKIP (ONLY)")

                # GPU-Baseline（可选）
                if have_gpu_base:
                    if _want(METHOD_GPU_BASE):
                        if N > GPU_BASE_MAX_N:
                            print(f"  (r={rep + 1}/{repeats}) 🚀 {METHOD_GPU_BASE} ... SKIP (N>{GPU_BASE_MAX_N})")
                            res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                       gpu_backend="skipped", gpu_mem_peak_bytes=np.nan, kernel_ms_mean=np.nan,
                                       final_positions=init_pos)
                        else:
                            print(f"  (r={rep + 1}/{repeats}) 🚀 {METHOD_GPU_BASE} ... ", end="", flush=True)
                            try:
                                res = run_gpu_base_once(N, max_iterations, init_pos, monitor_gpu=poll_gpu_mem)
                                print(
                                    f"OK {res['runtime']:.2f}s backend={res.get('gpu_backend', '?')} mode={res.get('gpu_mode', '?')}")
                            except Exception as e:
                                msg = str(e)
                                oom = ("OutOfMemory" in msg) or ("out of memory" in msg.lower())
                                print(f"SKIP {'(OOM)' if oom else '(error)'}")
                                res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                           gpu_backend="error", gpu_mem_peak_bytes=np.nan, kernel_ms_mean=np.nan,
                                           final_positions=init_pos)
                        rows.append(_pack_row(N, METHOD_GPU_BASE, rep, res, init_pos))
                    else:
                        print(f"  (r={rep + 1}/{repeats}) 🚀 {METHOD_GPU_BASE} ... SKIP (ONLY)")

                # GPU-Advanced（AD1，可选）
                if have_gpu_adv:
                    if _want(METHOD_GPU_ADV):
                        print(f"  (r={rep + 1}/{repeats}) 🏎  {METHOD_GPU_ADV} ... ", end="", flush=True)
                        try:
                            res = run_gpu_adv_once(N, max_iterations, init_pos, monitor_gpu=poll_gpu_mem)
                            print(
                                f"OK {res['runtime']:.2f}s backend={res.get('gpu_backend', '?')} mode={res.get('gpu_mode', '?')}")
                        except Exception as e:
                            msg = str(e)
                            oom = ("OutOfMemory" in msg) or ("out of memory" in msg.lower())
                            print(f"SKIP {'(OOM)' if oom else '(error)'}")
                            res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                       gpu_backend="error", gpu_mem_peak_bytes=np.nan, kernel_ms_mean=np.nan,
                                       final_positions=init_pos)
                        rows.append(_pack_row(N, METHOD_GPU_ADV, rep, res, init_pos))
                    else:
                        print(f"  (r={rep + 1}/{repeats}) 🏎  {METHOD_GPU_ADV} ... SKIP (ONLY)")

                # GPU-Advanced v2（AD2，可选）—— 注意：与 AD1 平级，不要嵌在 have_gpu_adv 里面
                if have_gpu_adv2:
                    if _want(METHOD_GPU_ADV2):
                        print(f"  (r={rep + 1}/{repeats}) 🏎🧩  {METHOD_GPU_ADV2} ... ", end="", flush=True)
                        try:
                            res = run_gpu_adv2_once(N, max_iterations, init_pos, monitor_gpu=poll_gpu_mem)
                            print(
                                f"OK {res['runtime']:.2f}s backend={res.get('gpu_backend', '?')} mode={res.get('gpu_mode', '?')}")
                        except Exception as e:
                            msg = str(e)
                            oom = ("OutOfMemory" in msg) or ("out of memory" in msg.lower())
                            print(f"SKIP {'(OOM)' if oom else '(error)'}")
                            res = dict(runtime=np.nan, iters=0, final_Jn=np.nan, final_rn=np.nan, converged=False,
                                       gpu_backend="error", gpu_mem_peak_bytes=np.nan, kernel_ms_mean=np.nan,
                                       final_positions=init_pos)
                        rows.append(_pack_row(N, METHOD_GPU_ADV2, rep, res, init_pos))
                    else:
                        print(f"  (r={rep + 1}/{repeats}) 🏎🧩  {METHOD_GPU_ADV2} ... SKIP (ONLY)")
                else:
                    # 这行方便你发现 have_gpu_adv2=False 的根因（比如模块名拼错/没在 sys.path）
                    print(f"  (r={rep + 1}/{repeats}) 🏎🧩  {METHOD_GPU_ADV2} ... SKIP (module not found)")

        # 长表
        df_long = pd.DataFrame(rows)
        df_long.to_csv(os.path.join(out_dir, "runs_long.csv"), index=False)

        # 汇总
        if df_long.empty:
            print("[fatal] No runs recorded. Check ONLY filter and module availability above.")
            raise SystemExit(2)
        df_sum = summarize(df_long)
        df_sum.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

        # 复杂度斜率
        slopes = fit_loglog_slopes(df_sum)
        slopes.to_csv(os.path.join(out_dir, "complexity_slopes.csv"), index=False)

        # 图
        plot_all(df_sum, df_long, out_dir)

        # 报告
        print_and_save_report(df_sum, slopes, out_dir)

        # 元数据
        meta = dict(
            timestamp=ts, sizes=sizes, repeats=repeats, max_iterations=max_iterations,
            methods=methods, pos_mode=pos_mode,
            config_snapshot=_config_snapshot(),
        )
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n🎉 Experiment complete! Results Table of Contents：{out_dir}")
        return out_dir, df_sum
    finally:
        # 恢复 config
        for k, v in orig.items():
            setattr(config, k, v)
        # 恢复收敛阈值 & 固定步控制
        config.CONVERGENCE_WINDOW = orig_conv_win
        config.CONVERGENCE_SLOPE_THRESHOLD = orig_conv_slope
        config.CONVERGENCE_STD_THRESHOLD = orig_conv_std
        if hasattr(config, "FORCE_FIXED_ITERS"):
            delattr(config, "FORCE_FIXED_ITERS")


def _pack_row(N, method, rep, res, init_pos: np.ndarray):
    iters = int(res.get("iters", 0) or 0)
    runtime = float(res.get("runtime", np.nan))
    ms_per_iter = (runtime / iters * 1000.0) if (iters and np.isfinite(runtime)) else np.nan

    # 终态连边吞吐（按最终位置近似衡量）
    # 这里只能近似：我们没有把最终位置接回来，所以用初始位置衡量“起始负载”
    # 若你更希望用“最终位置”，可在算法里写文件回传；这里保持无侵入。
    pos_for_edges = res.get("final_positions", init_pos)  # 优先用最终位置，兜底用初始
    e_dir, _ = count_edges_final(pos_for_edges)
    edges_per_sec = (e_dir / runtime) if (np.isfinite(runtime) and runtime > 0) else np.nan

    row = dict(
        N=N, method=method, repeat=rep,
        runtime=runtime, iters=iters, ms_per_iter=ms_per_iter,
        final_Jn=float(res.get("final_Jn", np.nan)),
        final_rn=float(res.get("final_rn", np.nan)),
        converged=bool(res.get("converged", False)),
        edges_per_sec=edges_per_sec,
        gpu_mem_peak_bytes=float(res.get("gpu_mem_peak_bytes", np.nan))
    )
    # kernel ms：算法日志中若打印 compute=XXms，我们已在 GPU 路径解析为均值
    kmean = res.get("kernel_ms_mean", np.nan)
    if np.isfinite(kmean):
        # 有的日志是“每若干步平均”，这里直接视为“kernel per iter”
        row["kernel_ms_per_iter"] = float(kmean)
    else:
        row["kernel_ms_per_iter"] = np.nan
    return row


def _config_snapshot() -> dict:
    keys = ["ALPHA", "BETA", "V", "R0", "PT", "DELTA", "STEP_SIZE",
            "MODE", "PLOT_EVERY", "LOG_EVERY", "KEEP_FULL_MATRIX",
            "GRID_SIZE", "SWARM_SIZE"]
    out = {}
    for k in keys:
        out[k] = getattr(config, k, None)
    return out


# =============================
# CLI
# =============================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("All-in-one experiment runner")
    p.add_argument("--full", action="store_true", help="Run the full benchmark suite")
    p.add_argument("--sizes", type=str, default="50,100,200", help="Comma-separated list of N values, e.g.,50,100,200")
    p.add_argument("--iters", type=int, default=300, help="Maximum iterations per run")
    p.add_argument("--repeats", type=int, default=5, help="Number of repeats per N")
    p.add_argument("--pos", type=str, default="gaussian", choices=["gaussian", "grid"],
                   help="Initial position generator")
    p.add_argument("--fixed-iters", action="store_true",
                   help="Disable early stopping so each run executes exactly --iters steps")
    p.add_argument("--no-gpu-mem", action="store_true",
                   help="Disable GPU memory peak monitoring")
    # NEW: choose gauss / jacobi for GPU routes
    p.add_argument("--update-mode", choices=["gauss", "jacobi"], default=None,
                   help="GPU update rule: gauss (in-place, baseline-equivalent) or jacobi (parallel)")
    p.add_argument("--name", type=str, default=None,
                   help="results/ 下的输出文件夹名（不填则用时间戳）")
    p.add_argument("--dtype", choices=["fp16", "fp32", "fp64"], default=None,
                   help="数值精度（只对 GPU 路径生效）：fp16/fp32/fp64")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated methods/aliases: e.g. 'ad2' or 'GPU-Advanced-v2' or 'GPU-Advanced,GPU-Advanced-v2'")
    args = p.parse_args()
    # 让 _parse_only_filter() 能读到命令行 --only 的值（优先于环境变量）
    setattr(sys.modules[__name__], "_ONLY_CLI", args.only or "")

    if not args.full:
        print("Example：python -m scripts.run_experiments --full --sizes 50,100,200 --iters 300 --repeats 5")
        sys.exit(0)

    try:
        sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    except Exception as e:
        print(f"[WARN] sizes parsing failed: {e}, using default 50,100,200")
        sizes = [50, 100, 200]
    # NEW: forward CLI flag to config so algorithms_gpu_unified can read it
    if args.update_mode is not None:
        setattr(config, "UPDATE_MODE", args.update_mode)

    if args.dtype is not None:
        import numpy as _np

        _map = {"fp16": _np.float16, "fp32": _np.float32, "fp64": _np.float64}
        setattr(config, "DTYPE", _map[args.dtype])

    run_comprehensive_experiments(
        sizes=sizes,
        max_iterations=args.iters,
        repeats=args.repeats,
        pos_mode=args.pos,
        poll_gpu_mem=not args.no_gpu_mem,
        fixed_iters=args.fixed_iters,
        out_name=args.name,
    )
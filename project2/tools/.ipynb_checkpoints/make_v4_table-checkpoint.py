# tools/make_v4_table.py
# -*- coding: utf-8 -*-
"""
从 v4_timebreak_steps.jsonl 统计每步耗时，输出两张 LaTeX 表格（booktabs 风格）。
用法：
  python tools/make_v4_table.py [路径或文件] --warmup 10 --n 1024 --pt 0.9995

说明：
- 位置参数既可以是目录（将从其中读取 v4_timebreak_steps.jsonl），也可以直接是该 .jsonl 文件路径。
- --warmup 用于丢弃前若干步的热身数据。
- 表头里会把 N 与 PT 写进 caption，默认为 N=1024, PT=0.9995。
"""
import argparse, json, numpy as np, sys, pathlib, math

def load_rows(path_like: str):
    p = pathlib.Path(path_like)
    if p.is_dir():
        p = p / "v4_timebreak_steps.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(json.loads(s))
    if not rows:
        raise RuntimeError("no data")
    return rows, p

def main():
    ap = argparse.ArgumentParser("Make LaTeX tables for V4 time breakdown")
    ap.add_argument("path", nargs="?", default=".", help="目录或 .jsonl 文件路径（默认：当前目录）")
    ap.add_argument("--warmup", type=int, default=10, help="丢弃前 WARMUP 步（默认 10）")
    ap.add_argument("--n", type=int, default=1024, help="caption 中显示的 N（默认 1024）")
    ap.add_argument("--pt", type=float, default=0.9995, help="caption 中显示的 PT（默认 0.9995）")
    args = ap.parse_args()

    rows, used_path = load_rows(args.path)
    if args.warmup >= len(rows):
        raise ValueError(f"warmup({args.warmup}) >= rows({len(rows)}).")

    data = rows[args.warmup:]

    def stat(key: str):
        xs = np.array([r[key] for r in data if key in r], dtype=float)
        if xs.size == 0:
            return float("nan"), float("nan")
        return float(xs.mean()), float(xs.std(ddof=0))

    keys = ["T_range","T_h2d","T_edge","T_d2h","T_update","T_overhead","T_total"]
    stats = {k: stat(k) for k in keys}
    total_ms_mean, total_ms_std = stats["T_total"]

    def pct(mean: float) -> float:
        if not (isinstance(total_ms_mean, float) and math.isfinite(total_ms_mean) and total_ms_mean > 0):
            return float("nan")
        return 100.0 * mean / total_ms_mean

    # --- 三段式聚合 ---
    t_range_m, t_range_s = stats["T_range"]
    t_h2d_m,  t_h2d_s  = stats["T_h2d"]
    t_d2h_m,  t_d2h_s  = stats["T_d2h"]
    t_edge_m, t_edge_s = stats["T_edge"]

    t_transfer_m = t_h2d_m + t_d2h_m
    # 方差近似相加（独立假设）
    t_transfer_s = (t_h2d_s**2 + t_d2h_s**2)**0.5

    print(rf"""\begin{table}[t]
\centering
\caption{{V4 每步时间分解（$N=\,$\textbf{{{args.n}}}, $PT=\,$\textbf{{{args.pt}}}, fixed-iters）。来源：\texttt{{{used_path.name}}}，热身丢弃 {args.warmup} 步。}}
\label{{tab:v4-timebreak-3}}
\begin{tabular}{lrrrr}
\toprule
阶段 & 平均(ms) & 标准差(ms) & 占比(\%) & 说明 \\
\midrule
CPU 邻域构建 $T_{{\text{{range}}}}$ & {t_range_m:.2f} & {t_range_s:.2f} & {pct(t_range_m):.1f} & KDTree/Cell-list+去重 \\
CPU$\leftrightarrow$GPU 传输 $T_{{\text{{transfer}}}}$ & {t_transfer_m:.2f} & {t_transfer_s:.2f} & {pct(t_transfer_m):.1f} & H$\to$D + D$\to$H \\
GPU 内核（按边） $T_{{\text{{edge}}}}$ & {t_edge_m:.2f} & {t_edge_s:.2f} & {pct(t_edge_m):.1f} & \texttt{{compute\_forces}} \\
\midrule
合计 & {total_ms_mean:.2f} & {total_ms_std:.2f} & 100.0 &  \\
\bottomrule
\end{tabular}
\end{table}
""")

    # --- 细分版 ---
    t_update_m, t_update_s     = stats["T_update"]
    t_overhead_m, t_overhead_s = stats["T_overhead"]

    print(rf"""\begin{table}[t]
\centering
\caption{{V4 每步时间细分（$N=\,$\textbf{{{args.n}}}, $PT=\,$\textbf{{{args.pt}}}, fixed-iters）。来源：\texttt{{{used_path.name}}}，热身丢弃 {args.warmup} 步。}}
\label{{tab:v4-timebreak-5}}
\begin{tabular}{lrrrr}
\toprule
阶段 & 平均(ms) & 标准差(ms) & 占比(\%) & 说明 \\
\midrule
CPU 邻域构建（含去重） & {t_range_m:.2f} & {t_range_s:.2f} & {pct(t_range_m):.1f} & $T_{{\text{{range}}}}$ \\
H$\to$D 传输（$Q,E$） & {t_h2d_m:.2f} & {t_h2d_s:.2f} & {pct(t_h2d_m):.1f} & PCIe \\
GPU 内核（按边力与指标） & {t_edge_m:.2f} & {t_edge_s:.2f} & {pct(t_edge_m):.1f} & $T_{{\text{{edge}}}}$ \\
D$\to$H 传输（$U$与标量） & {t_d2h_m:.2f} & {t_d2h_s:.2f} & {pct(t_d2h_m):.1f} & PCIe \\
CPU Jacobi 更新 & {t_update_m:.2f} & {t_update_s:.2f} & {pct(t_update_m):.1f} & $Q\leftarrow Q+\Delta t\,U$ \\
启动/同步等开销 & {t_overhead_m:.2f} & {t_overhead_s:.2f} & {pct(t_overhead_m):.1f} & launch \& sync \\
\midrule
合计 & {total_ms_mean:.2f} & {total_ms_std:.2f} & 100.0 & \\
\bottomrule
\end{tabular}
\end{table}
""")

if __name__ == "__main__":
    main()
#scripts/collect_ad2_ablation.py
# -*- coding: utf-8 -*-
import os, re, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def infer_scale_and_K(name: str):
    # Supports ad2_cells_1_0 / ad2_cells_1.5 / ad2_cells_K1 / ad2_cells_K2
    s = name.lower()
    m = re.search(r"(?:cells[_-]k)(\d+)", s)
    if m:
        K = int(m.group(1))
        scale = float(K)  # Rough: K=ceil(scale); only used to distinguish K
        return scale, K
    m = re.search(r"cells[_-](\d+)[._-](\d+)", s)
    if m:
        scale = float(f"{int(m.group(1))}.{int(m.group(2))}")
        return scale, int(np.ceil(scale))
    m = re.search(r"cells[_-](\d+(?:\.\d+)?)", s)
    if m:
        scale = float(m.group(1))
        return scale, int(np.ceil(scale))
    return np.nan, np.nan

def load_one_dir(d):
    meta_path = os.path.join(d, "metadata.json")
    sum_path  = os.path.join(d, "summary.csv")
    if not (os.path.exists(meta_path) and os.path.exists(sum_path)):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta.get("config_snapshot", {}) or {}
    PT  = cfg.get("PT", np.nan)
    # Take the GPU-Advanced v2 row
    df = pd.read_csv(sum_path)
    df = df[df["method"] == "GPU-Advanced v2"].copy()
    if df.empty:
        return None
    scale, K = infer_scale_and_K(os.path.basename(d))
    df["dir"]   = os.path.basename(d)
    df["PT"]    = float(PT) if pd.notna(PT) else np.nan
    df["scale"] = scale
    df["K"]     = K
    return df

def collect(roots, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    dirs = []
    for pat in roots:
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d):
                dirs.append(d)
    for d in dirs:
        one = load_one_dir(d)
        if one is not None:
            rows.append(one)
    if not rows:
        raise SystemExit("No valid results found under: " + ", ".join(roots))
    df = pd.concat(rows, ignore_index=True)
    # Select a few of the most commonly used columns
    keep = [
        "dir","PT","K","scale","N",
        "runtime_mean","runtime_std","ms_per_iter_mean","ms_per_iter_std",
        "kernel_ms_per_iter_mean","kernel_ms_per_iter_std",
        "edges_per_sec_mean","repeats"
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].sort_values(["PT","N","K","dir"]).reset_index(drop=True)
    long_csv = os.path.join(outdir, "ad2_ablation_long.csv")
    df.to_csv(long_csv, index=False)

    # Generate a pivot table: for each (PT,N), compare K=1 / K=2 + ratio
    piv = df.pivot_table(index=["PT","N"], columns="K",
                         values="ms_per_iter_mean", aggfunc="mean")
    # Column name convention: K=1 -> ms_iter_K1, K=2 -> ms_iter_K2
    piv = piv.rename(columns={1:"ms_iter_K1", 2:"ms_iter_K2"})
    piv["K2_over_K1"] = piv["ms_iter_K2"] / piv["ms_iter_K1"]
    piv_csv = os.path.join(outdir, "ad2_ablation_pivot.csv")
    piv.to_csv(piv_csv)

    # Plotting: one figure per N (ms/iter vs PT, with two lines for K=1/2)
    for N, sub in df.groupby("N"):
        plt.figure(figsize=(7,5))
        for K in sorted(sub["K"].dropna().unique()):
            subk = sub[sub["K"]==K].sort_values("PT")
            if len(subk)==0: continue
            plt.plot(subk["PT"], subk["ms_per_iter_mean"], "o-", label=f"K={int(K)}")
        plt.xlabel("PT"); plt.ylabel("ms / iter (mean)")
        plt.title(f"AD2 ablation: ms/iter vs PT @ N={int(N)}")
        plt.grid(True, alpha=.3); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"ms_per_iter_vs_PT_N{int(N)}.png"), dpi=200)
        plt.close()

        # If the kernel column has values, draw another kernel ms/iter plot
        if sub["kernel_ms_per_iter_mean"].notna().any():
            plt.figure(figsize=(7,5))
            for K in sorted(sub["K"].dropna().unique()):
                subk = sub[(sub["K"]==K) & sub["kernel_ms_per_iter_mean"].notna()].sort_values("PT")
                if len(subk)==0: continue
                plt.plot(subk["PT"], subk["kernel_ms_per_iter_mean"], "s--", label=f"K={int(K)}")
            plt.xlabel("PT"); plt.ylabel("kernel ms / iter (mean)")
            plt.title(f"AD2 ablation: kernel ms/iter vs PT @ N={int(N)}")
            plt.grid(True, alpha=.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"kernel_ms_per_iter_vs_PT_N{int(N)}.png"), dpi=200)
            plt.close()

        # Plot K2/K1 ratio (>1 means K=2 is slower)
        pivN = piv.reset_index()
        pivN = pivN[pivN["N"]==N].sort_values("PT")
        if not pivN.empty and "K2_over_K1" in pivN.columns:
            plt.figure(figsize=(7,5))
            plt.plot(pivN["PT"], pivN["K2_over_K1"], "d-")
            plt.axhline(1.0, linestyle="--", linewidth=1)
            plt.xlabel("PT"); plt.ylabel("K2 / K1 (ms/iter ratio)")
            plt.title(f"AD2 ablation: K2/K1 ratio @ N={int(N)}")
            plt.grid(True, alpha=.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"K2_over_K1_vs_PT_N{int(N)}.png"), dpi=200)
            plt.close()

    print(f"[OK] wrote:\n  {long_csv}\n  {piv_csv}\n  {outdir}/*.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["results/ad2_*"],
                    help="glob patterns to result folders")
    ap.add_argument("--out", type=str, default="results/ad2_ablation_summary",
                    help="output dir")
    args = ap.parse_args()
    collect(args.roots, args.out)
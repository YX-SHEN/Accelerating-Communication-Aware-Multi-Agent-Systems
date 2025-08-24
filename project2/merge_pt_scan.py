# merge_pt_scan.py
import pandas as pd, glob, re, os, math
# 你的三个结果目录名
runs = [
    ("1e-3","results/dens_pt1e-3"),
    ("1e-2","results/dens_pt1e-2"),
    ("5e-2","results/dens_pt5e-2"),
]
rows = []
for pt, path in runs:
    # 猜测几种常见统计文件名，取到就用
    cand = []
    for name in ["summary.csv","per_method_stats.csv","metrics.csv","stats.csv"]:
        f = os.path.join(path, name)
        if os.path.exists(f):
            cand.append(f)
    if not cand:
        raise FileNotFoundError(f"No stats csv found in {path}")
    df = pd.read_csv(cand[0])
    # 只取 N=1024 的行；列名按你的导出调整
    # 常见列名示例：["N","method","iter_ms_mean","iter_ms_std","kernel_ms_mean","kernel_ms_std","speedup_vs_baseline"]
    df = df[df["N"]==1024].copy()
    df["PT"] = pt
    rows.append(df)
all_df = pd.concat(rows, ignore_index=True)
# 保存汇总
os.makedirs("results/pt_scan_1024", exist_ok=True)
all_df.to_csv("results/pt_scan_1024/pt_scan_1024.csv", index=False)

# 画图：iter_ms vs PT（对数 x 轴）
import matplotlib.pyplot as plt
methods = ["GPU-Baseline","GPU-Advanced","GPU-Advanced v2"]
plt.figure()
for m in methods:
    sub = all_df[all_df["method"]==m].sort_values("PT", key=lambda s: s.map({"1e-3":1e-3,"1e-2":1e-2,"5e-2":5e-2}))
    plt.errorbar([float(x) for x in sub["PT"]], sub["iter_ms_mean"], yerr=sub.get("iter_ms_std", None), label=m, marker="o")
plt.xscale("log")
plt.xlabel("PT (log scale)"); plt.ylabel("Time per iter (ms)"); plt.title("N=1024 • Fixed iters • PT scan")
plt.legend(); plt.tight_layout()
plt.savefig("results/pt_scan_1024/iter_ms_vs_PT.png", dpi=200)

# 可选：kernel_ms vs PT（GPU 三条）
plt.figure()
for m in methods:
    sub = all_df[all_df["method"]==m].sort_values("PT", key=lambda s: s.map({"1e-3":1e-3,"1e-2":1e-2,"5e-2":5e-2}))
    if "kernel_ms_mean" in sub:
        plt.errorbar([float(x) for x in sub["PT"]], sub["kernel_ms_mean"], yerr=sub.get("kernel_ms_std", None), label=m, marker="o")
plt.xscale("log")
plt.xlabel("PT (log scale)"); plt.ylabel("Kernel time per iter (ms)"); plt.title("N=1024 • GPU kernel • PT scan")
plt.legend(); plt.tight_layout()
plt.savefig("results/pt_scan_1024/kernel_ms_vs_PT.png", dpi=200)

# 可选：AD2 相对 Baseline-GPU 的加速比
base = all_df[all_df["method"]=="GPU-Baseline"][["PT","iter_ms_mean"]].rename(columns={"iter_ms_mean":"base_iter"})
ad2  = all_df[all_df["method"]=="GPU-Advanced v2"][["PT","iter_ms_mean"]].rename(columns={"iter_ms_mean":"ad2_iter"})
sp = pd.merge(base, ad2, on="PT")
sp["speedup_ad2_vs_gpubase"] = sp["base_iter"] / sp["ad2_iter"]
sp.to_csv("results/pt_scan_1024/ad2_vs_base_speedup.csv", index=False)

plt.figure()
plt.bar([float(x) for x in sp["PT"]], sp["speedup_ad2_vs_gpubase"])
plt.xscale("log")
plt.xlabel("PT (log scale)"); plt.ylabel("Speedup (AD2 vs GPU-Baseline)"); plt.title("N=1024 • AD2 speedup vs PT")
plt.tight_layout()
plt.savefig("results/pt_scan_1024/ad2_speedup_vs_PT.png", dpi=200)
print("Saved merged CSV and figures under results/pt_scan_1024/")
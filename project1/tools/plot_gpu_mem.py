# tools/plot_gpu_mem.py
import json, csv, glob, pathlib
import numpy as np
import matplotlib.pyplot as plt

def find_peaks():
    rows = []
    # Compatible with two storage formats: jsonl or csv
    for p in glob.glob("results/**/gpu_memory_peak.*", recursive=True):
        path = pathlib.Path(p)
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append({"N": int(row["N"]), "peak_mb": float(row["peak_mb"]), "method": row.get("method","unknown")})
        elif path.suffix in (".json", ".jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                obj = json.loads(line)
                rows.append({"N": int(obj["N"]), "peak_mb": float(obj["peak_mb"]), "method": obj.get("method","unknown")})
    return rows

rows = find_peaks()
# Only plot V5: GPU-Advanced v2
rows = [r for r in rows if "Advanced v2" in r["method"] or "V5" in r["method"]]
rows.sort(key=lambda r: r["N"])

Ns = np.array([r["N"] for r in rows], dtype=float)
M  = np.array([r["peak_mb"] for r in rows], dtype=float)

# Linear fit: M ≈ a + b * N
A = np.vstack([np.ones_like(Ns), Ns]).T
coef, *_ = np.linalg.lstsq(A, M, rcond=None)  # a, b
a, b = coef

plt.figure(figsize=(6.4,4.2))
plt.plot(Ns, M, marker="o", linewidth=1.5, label="V5 peak memory")
plt.plot(Ns, a + b*Ns, linestyle="--", linewidth=1.2, label=f"linear fit: M≈{a:.1f}+{b:.3f}·N MB")
plt.xlabel("N (agents)")
plt.ylabel("Peak GPU memory (MB)")
plt.title("Peak GPU memory vs N (V5)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("gpu_memory_peak.png", dpi=200)
print("Saved: gpu_memory_peak.png")
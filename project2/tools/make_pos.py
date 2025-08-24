import sys, numpy as np
N = int(sys.argv[1])
out = sys.argv[2]
rng = np.random.default_rng(2025)
# 高斯云（和你脚本里的 gaussian 生成器一致口径）
pos = rng.standard_normal((N,2)).astype(np.float32) * 20.0
np.save(out, pos)
print("saved", out, pos.shape)

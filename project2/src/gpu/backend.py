# src/gpu/backend.py
# -*- coding: utf-8 -*-
import numpy as np

# 尝试读取 config.DTYPE 作为默认精度；若失败则回退到 float32
try:
    import config
    DTYPE_DEFAULT = getattr(config, "DTYPE", np.float32)
except Exception:
    DTYPE_DEFAULT = np.float32


class ImprovedGPUBackend:
    def __init__(self):
        self.backend = "numpy"
        self.device = None
        self.jnp = None
        self.cp = None
        self.jax = None
        self.default_dtype = DTYPE_DEFAULT
        self._setup()

    def _setup(self):
        # 先尝试 JAX（Apple Metal 也会出现在 "gpu" 设备列表）
        try:
            import jax
            # 若用户选择了双精度，必须在导入 jax.numpy 之前打开 x64
            if self.default_dtype == np.float64:
                jax.config.update("jax_enable_x64", True)
            # 先设平台，再查设备
            jax.config.update("jax_platform_name", "gpu")
            import jax.numpy as jnp

            gpus = jax.devices("gpu")
            if len(gpus) > 0:
                self.backend = "jax"
                self.jnp, self.jax = jnp, jax
                self.device = gpus[0]
                print(f"[GPU] 使用 JAX 后端：{self.device.device_kind} (dtype={np.dtype(self.default_dtype).name})")
                return
            # fallback: JAX CPU 也可用（同样保留 x64 设置）
            cpus = jax.devices("cpu")
            if len(cpus) > 0:
                self.backend = "jax"
                self.jnp, self.jax = jnp, jax
                self.device = cpus[0]
                print(f"[CPU] 使用 JAX CPU 后端 (dtype={np.dtype(self.default_dtype).name})")
                return
        except Exception as e:
            print(f"[INFO] JAX 不可用：{e}")

        # 再尝试 CuPy（CUDA）
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                self.backend = "cupy"
                self.cp = cp
                self.device = cp.cuda.Device()
                # 简单设置一个内存池上限（按需调整或移除）
                mempool = cp.get_default_memory_pool()
                try:
                    mempool.set_limit(size=2**30)  # 1GB
                except Exception:
                    pass
                print(f"[GPU] 使用 CuPy 后端：设备 {self.device.id} (dtype={np.dtype(self.default_dtype).name})")
                return
        except Exception as e:
            print(f"[INFO] CuPy 不可用：{e}")

        print(f"[INFO] 使用 NumPy CPU 回退 (dtype={np.dtype(self.default_dtype).name})")

    # 统一把数组放到设备，确保 dtype 一致
    def to_device(self, array, dtype=None):
        if dtype is None:
            dtype = self.default_dtype
        if self.backend == "jax":
            return self.jnp.asarray(array, dtype=dtype)
        if self.backend == "cupy":
            return self.cp.asarray(array, dtype=dtype)
        return np.asarray(array, dtype=dtype)

    # 从设备取回 host 上的 numpy 数组
    def to_host(self, array):
        if self.backend == "jax":
            import numpy as _np
            return _np.asarray(array)
        if self.backend == "cupy":
            return self.cp.asnumpy(array)
        return array

    # 设备端 zeros
    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self.default_dtype
        if self.backend == "jax":
            return self.jnp.zeros(shape, dtype=dtype)
        if self.backend == "cupy":
            return self.cp.zeros(shape, dtype=dtype)
        import numpy as _np
        return _np.zeros(shape, dtype=dtype)

    # 同步（CuPy 需要显式同步；JAX 在外部 .block_until_ready 或这里无需额外处理）
    def synchronize(self):
        if self.backend == "cupy":
            self.cp.cuda.Stream.null.synchronize()
        # JAX 用 .block_until_ready()；这里无需额外操作
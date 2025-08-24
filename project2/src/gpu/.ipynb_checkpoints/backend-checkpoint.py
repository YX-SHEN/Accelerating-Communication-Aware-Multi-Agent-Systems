# src/gpu/backend.py
# -*- coding: utf-8 -*-
import numpy as np

DTYPE_DEFAULT = np.float32

class ImprovedGPUBackend:
    def __init__(self):
        self.backend = "numpy"
        self.device = None
        self.jnp = None
        self.cp = None
        self.jax = None
        self._setup()

    def _setup(self):
        # 先尝试 JAX（Apple Metal 也会出现在 "gpu" 设备列表）
        try:
            import jax
            jax.config.update("jax_platform_name", "gpu")  # 先设平台，再查设备
            import jax.numpy as jnp
            gpus = jax.devices("gpu")
            if len(gpus) > 0:
                self.backend = "jax"
                self.jnp, self.jax = jnp, jax
                self.device = gpus[0]
                print(f"[GPU] 使用 JAX 后端：{self.device.device_kind}")
                return
            # fallback: JAX CPU 也可用
            cpus = jax.devices("cpu")
            if len(cpus) > 0:
                self.backend = "jax"
                self.jnp, self.jax = jnp, jax
                self.device = cpus[0]
                print("[CPU] 使用 JAX CPU 后端")
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
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=2**30)  # 1GB 简单限额，按需调整
                print(f"[GPU] 使用 CuPy 后端：设备 {self.device.id}")
                return
        except Exception as e:
            print(f"[INFO] CuPy 不可用：{e}")

        print("[INFO] 使用 NumPy CPU 回退")

    def to_device(self, array, dtype=DTYPE_DEFAULT):
        if self.backend == "jax":
            return self.jnp.asarray(array, dtype=dtype)
        if self.backend == "cupy":
            return self.cp.asarray(array, dtype=dtype)
        return np.asarray(array, dtype=dtype)

    def to_host(self, array):
        if self.backend == "jax":
            import numpy as _np
            return _np.asarray(array)
        if self.backend == "cupy":
            return self.cp.asnumpy(array)
        return array

    def zeros(self, shape, dtype=DTYPE_DEFAULT):
        if self.backend == "jax":
            return self.jnp.zeros(shape, dtype=dtype)
        if self.backend == "cupy":
            return self.cp.zeros(shape, dtype=dtype)
        import numpy as _np
        return _np.zeros(shape, dtype=dtype)

    def synchronize(self):
        if self.backend == "cupy":
            self.cp.cuda.Stream.null.synchronize()
        # JAX 用 .block_until_ready() 同步；这里无需额外操作
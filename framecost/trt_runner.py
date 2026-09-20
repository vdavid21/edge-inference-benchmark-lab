"""TensorRT inference wrapper.

The runtime half of the harness: deserialise an engine, allocate device buffers,
bind tensors by name, execute, copy back. Used by both the accuracy evaluation
and (later) the latency benchmark, so that both measure the same code path.

TensorRT 10 API notes
---------------------
TRT 10 replaced the old integer-indexed `bindings` list with a name-based API:
    engine.num_io_tensors / get_tensor_name / get_tensor_mode
    context.set_input_shape / set_tensor_address / execute_async_v3
Most tutorials online still use `execute_async_v2` and a bindings list. That is
the TRT 8 API and it is gone. If you copy code that uses it, it will not run.

Memory
------
Host buffers here are ordinary pageable numpy arrays, which makes
cudaMemcpyAsync effectively synchronous. That is correct but not fast. Pinned
(page-locked) host memory is a later optimisation and one worth measuring —
on Jetson's unified memory it is also a candidate for elimination entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

# cuda-python moved the runtime module between major versions.
try:
    from cuda.bindings import runtime as cudart          # cuda-python >= 13
except ImportError:                                       # pragma: no cover
    from cuda import cudart                               # cuda-python 12.x


def _check(call):
    """Unwrap a cuda-python return tuple, raising on a non-success status.

    Every cudart function returns (error, *values). This turns a silent bad
    status into an exception, which matters because a failed cudaMalloc that
    goes unnoticed produces garbage inference results rather than a crash.
    """
    err, *rest = call
    if err != cudart.cudaError_t.cudaSuccess:
        name = cudart.cudaGetErrorString(err)[1].decode()
        raise RuntimeError(f"CUDA error {int(err)}: {name}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


class TRTRunner:
    """Load a serialised TensorRT engine and run inference on numpy arrays."""

    def __init__(self, engine_path, verbose: bool = False):
        self.engine_path = Path(engine_path)
        self.engine_bytes = self.engine_path.read_bytes()

        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialise {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = _check(cudart.cudaStreamCreate())

        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self._allocate()

    # -- setup ---------------------------------------------------------------

    def _allocate(self) -> None:
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))

            # A -1 means a dynamic dimension. Our engines are built with
            # min == opt == max so nothing should be dynamic; if it is, we pin
            # it to the profile's shape rather than guessing.
            if any(d < 0 for d in shape):
                if mode == trt.TensorIOMode.INPUT:
                    shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[1])
                    self.context.set_input_shape(name, shape)
                else:
                    shape = tuple(self.context.get_tensor_shape(name))

            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            device_ptr = _check(cudart.cudaMalloc(nbytes))
            host = np.empty(shape, dtype=dtype)

            self.context.set_tensor_address(name, int(device_ptr))

            entry = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "nbytes": nbytes,
                "device": device_ptr,
                "host": host,
            }
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(entry)

        if not self.inputs or not self.outputs:
            raise RuntimeError("engine has no inputs or no outputs")

    # -- inference -----------------------------------------------------------

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Run one inference. Single-input, single-output engines only.

        Args:
            x: array matching the engine input shape, with or without the
               leading batch dimension.

        Returns:
            The output array (a view into the reusable host buffer — copy it
            if you need to keep it past the next call).
        """
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise RuntimeError("use infer_many() for multi-tensor engines")

        inp, out = self.inputs[0], self.outputs[0]

        if x.shape != inp["shape"]:
            x = x.reshape(inp["shape"])
        # Cast rather than fail: an FP16 engine may still take an FP32 input,
        # but an unnoticed dtype mismatch would corrupt the results silently.
        if x.dtype != inp["dtype"]:
            x = x.astype(inp["dtype"])

        np.copyto(inp["host"], x)

        _check(cudart.cudaMemcpyAsync(
            inp["device"], inp["host"].ctypes.data, inp["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))

        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 failed")

        _check(cudart.cudaMemcpyAsync(
            out["host"].ctypes.data, out["device"], out["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))

        _check(cudart.cudaStreamSynchronize(self.stream))
        return out["host"]

    # -- metadata ------------------------------------------------------------

    def describe(self) -> dict:
        """Engine facts worth recording in a result row."""
        try:
            dev_mem = self.engine.device_memory_size_v2
        except AttributeError:
            dev_mem = self.engine.device_memory_size

        import hashlib
        return {
            "engine_path": str(self.engine_path),
            "engine_sha256_16": hashlib.sha256(self.engine_bytes).hexdigest()[:16],
            "engine_size_bytes": len(self.engine_bytes),
            "device_memory_bytes": int(dev_mem),
            "num_layers": self.engine.num_layers,
            "trt_version": trt.__version__,
            "inputs": [
                {"name": t["name"], "shape": t["shape"], "dtype": np.dtype(t["dtype"]).name}
                for t in self.inputs
            ],
            "outputs": [
                {"name": t["name"], "shape": t["shape"], "dtype": np.dtype(t["dtype"]).name}
                for t in self.outputs
            ],
        }

    # -- teardown ------------------------------------------------------------

    def close(self) -> None:
        for t in self.inputs + self.outputs:
            if t.get("device") is not None:
                cudart.cudaFree(t["device"])
                t["device"] = None
        if getattr(self, "stream", None) is not None:
            cudart.cudaStreamDestroy(self.stream)
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "models/mobilenetv2_fp32.plan"
    with TRTRunner(path) as r:
        print(json.dumps(r.describe(), indent=2))
        shape = r.inputs[0]["shape"]
        y = r.infer(np.random.randn(*shape).astype(np.float32))
        print(f"\noutput shape {y.shape}  argmax {int(y.reshape(-1).argmax())}")
        print(f"logit range  {y.min():.3f} .. {y.max():.3f}")
        print("\nsmoke test passed")

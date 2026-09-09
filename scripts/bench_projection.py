"""Benchmark serial vs Numba/CuPy batch projection."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadmeshtesser import PipelineParams, TreeQuadPipeline
from quadmeshtesser.appr_parallel import available_backends, gpu_device_name, resolve_backend


def bench(name: str, backend: str) -> float:
    swc = ROOT / "data" / "cell021.CNG.swc"
    t0 = time.perf_counter()
    TreeQuadPipeline(
        PipelineParams(subdiv_levels=1, project=True, project_backend=backend, restore_offset=False)
    ).run(swc)
    elapsed = time.perf_counter() - t0
    print(f"  {name:12s} ({backend:6s}): {elapsed:.2f}s")
    return elapsed


if __name__ == "__main__":
    print("available backends:", available_backends())
    gpu = gpu_device_name()
    if gpu:
        print("GPU:", gpu)
    print("auto resolves to:", resolve_backend("auto", n_vertices=1801))
    print("cell021.CNG.swc subdiv=1 project only:")
    bench("serial", "serial")
    if "numba" in available_backends():
        bench("numba", "numba")
    if "cupy" in available_backends():
        bench("cupy", "cupy")
    bench("auto", "auto")

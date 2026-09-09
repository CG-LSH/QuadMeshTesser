"""Quick pipeline smoke test."""
from pathlib import Path

from quadmeshtesser import PipelineParams, TreeQuadPipeline
from quadmeshtesser.export_obj import export_obj

swc = Path(__file__).resolve().parent / "data" / "test_linear_0.swc"
out = Path(__file__).resolve().parent / "data" / "_test_out.obj"

result = TreeQuadPipeline(PipelineParams(subdiv_levels=2, project=True)).run(swc)
print(f"vertices={result.mesh.n_vertices} quads={result.mesh.n_quads}")
export_obj(result.mesh, out)
print(f"exported -> {out}")

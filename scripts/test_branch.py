"""Pipeline and branch convex tests (C++ port)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadmeshtesser import PipelineParams, TreeQuadPipeline
from quadmeshtesser.meshgen import build_base_quad_mesh

ROOT = Path(__file__).resolve().parent.parent


def test_branch_hull_on_cell():
    swc = ROOT / "data" / "cell021.CNG.swc"
    base = build_base_quad_mesh(str(swc))
    assert base.n_quads > 0
    assert base.n_triangles >= 0, "branch hull may add triangles for cell021"
    print(f"cell021 base: v={base.n_vertices} q={base.n_quads} t={base.n_triangles}")


def test_full_pipeline():
    for name in ("test_linear_0.swc", "cell021.CNG.swc"):
        swc = ROOT / "data" / name
        if not swc.is_file():
            continue
        r = TreeQuadPipeline(PipelineParams(subdiv_levels=1, project=False)).run(swc)
        print(f"{name}: v={r.mesh.n_vertices} q={r.mesh.n_quads} t={r.mesh.n_triangles}")


if __name__ == "__main__":
    test_branch_hull_on_cell()
    test_full_pipeline()

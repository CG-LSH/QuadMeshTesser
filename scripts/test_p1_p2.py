"""P1/P2 alignment smoke tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadmeshtesser import PipelineParams, TreeQuadPipeline
from quadmeshtesser.obj_io import load_obj_mesh

DATA = ROOT / "data"


def _run(label: str, params: PipelineParams, swc: Path) -> None:
    r = TreeQuadPipeline(params).run(swc)
    m = r.mesh
    print(f"{label}: v={m.n_vertices} q={m.n_quads} t={m.n_triangles}")


def test_quartic_numba():
    swc = DATA / "cell021.CNG.swc"
    if not swc.is_file():
        return
    p = PipelineParams(
        subdiv_levels=1,
        kernel="quartic",
        project_backend="numba",
        project=True,
    )
    _run("quartic+numba", p, swc)


def test_weight_global():
    swc = DATA / "test_linear_0.swc"
    if not swc.is_file():
        return
    p = PipelineParams(subdiv_levels=1, weight_mode="global", smooth_weights=True)
    _run("global+smooth", p, swc)


def test_limit_modes():
    swc = DATA / "test_linear_0.swc"
    if not swc.is_file():
        return
    _run("limit", PipelineParams(subdiv_levels=2, appr_style="limit"), swc)
    _run(
        "limit_lmt+local",
        PipelineParams(subdiv_levels=2, use_lmt_cnt=True, weight_mode="local"),
        swc,
    )
    _run(
        "limit_lmt+global",
        PipelineParams(subdiv_levels=2, use_lmt_cnt=True, weight_mode="global"),
        swc,
    )


def test_metaball_line():
    swc = DATA / "test_linear_0.swc"
    if not swc.is_file():
        return
    p = PipelineParams(subdiv_levels=1, appr_style="metaball_line", project_backend="numba")
    _run("metaball_line", p, swc)


def test_restore_offset():
    swc = DATA / "test_linear_0.swc"
    if not swc.is_file():
        return
    base = TreeQuadPipeline(PipelineParams(subdiv_levels=1, project=False)).run(swc)
    obj_path = ROOT / "output" / "_test_ref.obj"
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    from quadmeshtesser.export_obj import export_obj

    export_obj(base.mesh, obj_path)
    v, t = load_obj_mesh(obj_path)
    assert len(v) > 0 and len(t) > 0

    p = PipelineParams(
        subdiv_levels=1,
        restore_offset=True,
        ref_obj_path=str(obj_path),
        project_backend="numba",
    )
    _run("ref_obj_offset", p, swc)


if __name__ == "__main__":
    test_quartic_numba()
    test_weight_global()
    test_limit_modes()
    test_metaball_line()
    test_restore_offset()
    print("P1/P2 tests OK")

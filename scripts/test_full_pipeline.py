"""End-to-end: watertight manifold mesh + convolution surface projection."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from quadmeshtesser import PipelineParams, TreeQuadPipeline
from quadmeshtesser.cpp_constants import projection_iso_for_kernel
from quadmeshtesser.mesh_topology import is_edge_manifold, is_watertight, topology_summary
from quadmeshtesser.tree_skel import collect_tree_segments, conv_field_tree

DATA = ROOT / "data"


def _degen_quad_count(mesh) -> int:
    v = mesh.vertices
    n = 0
    for q in mesh.quads:
        d = [float(np.linalg.norm(v[q[(i + 1) % 4]] - v[q[i]])) for i in range(4)]
        if min(d) < 1e-8:
            n += 1
    return n


def test_cell021_full_pipeline():
    swc = DATA / "cell021.CNG.swc"
    if not swc.is_file():
        return

    iso = None
    p = PipelineParams(
        sweep_only=False,
        subdiv_levels=2,
        insert_assist=True,
        bound_tet_scaled=True,
        project=True,
        kernel="cauchy",
        appr_style="local",
        project_backend="auto",
        weight_mode="local",
    )
    r = TreeQuadPipeline(p).run(swc)
    m = r.mesh
    topo = topology_summary(m)

    assert topo["non_manifold"] == 0, topo
    assert topo["boundary"] == 0, topo
    assert is_edge_manifold(m)
    assert is_watertight(m)
    assert m.n_vertices > 5000, m.n_vertices
    assert m.n_quads > 5000, m.n_quads
    assert _degen_quad_count(m) == 0, f"degenerate quads {_degen_quad_count(m)}"

    segs = collect_tree_segments(r.root)
    iso = p.iso_value if p.iso_value is not None else projection_iso_for_kernel(p.kernel)
    sample_idx = np.linspace(0, m.n_vertices - 1, min(256, m.n_vertices), dtype=int)
    vals = [conv_field_tree(m.vertices[i], segs, p.kernel) for i in sample_idx]
    med = float(np.median(vals))
    assert abs(med - iso) / max(iso, 1e-9) < 0.05, f"field median {med} vs iso {iso}"

    print(
        f"cell021 full: v={m.n_vertices} q={m.n_quads} t={m.n_triangles} "
        f"topo={topo} iso={iso:.4f} field_med={med:.4f}"
    )


def test_linear_full_pipeline():
    swc = DATA / "test_linear_0.swc"
    if not swc.is_file():
        return
    p = PipelineParams(
        sweep_only=False,
        subdiv_levels=1,
        project=True,
        kernel="cauchy",
        appr_style="local",
        project_backend="numba",
    )
    r = TreeQuadPipeline(p).run(swc)
    m = r.mesh
    topo = topology_summary(m)
    assert topo["non_manifold"] == 0
    assert topo["boundary"] == 0
    assert is_watertight(m)
    assert _degen_quad_count(m) == 0
    print(f"linear full: v={m.n_vertices} q={m.n_quads} topo={topo}")


def test_cell021_morphtesser_pipeline():
    swc = DATA / "cell021.CNG.swc"
    if not swc.is_file():
        return
    from quadmeshtesser.morphtesser_field import DEFAULT_MORPH_ISO, collect_morph_segments, conv_field_morph

    p = PipelineParams(
        sweep_only=False,
        subdiv_levels=1,
        insert_assist=True,
        bound_tet_scaled=True,
        project=True,
        appr_style="morphtesser",
        iso_value=DEFAULT_MORPH_ISO,
        project_iters=30,
    )
    r = TreeQuadPipeline(p).run(swc)
    m = r.mesh
    topo = topology_summary(m)
    assert topo["non_manifold"] == 0, topo
    assert topo["boundary"] == 0, topo
    assert is_watertight(m)
    assert r.iso_surface_points is not None and len(r.iso_surface_points) > 100

    segs = collect_morph_segments(r.root)
    sample_idx = np.linspace(0, m.n_vertices - 1, min(128, m.n_vertices), dtype=int)
    vals = np.array([conv_field_morph(m.vertices[i], segs) for i in sample_idx])
    active = vals[vals > 1e-3]
    assert len(active) >= max(32, int(0.65 * len(vals))), f"too few active verts {len(active)}/{len(vals)}"
    med = float(np.median(active))
    assert abs(med - DEFAULT_MORPH_ISO) / DEFAULT_MORPH_ISO < 0.35, f"field median {med}"
    print(
        f"cell021 morphtesser: v={m.n_vertices} q={m.n_quads} iso_pts={len(r.iso_surface_points)} "
        f"field_med={med:.4f}"
    )


if __name__ == "__main__":
    test_linear_full_pipeline()
    test_cell021_full_pipeline()
    test_cell021_morphtesser_pipeline()
    print("test_full_pipeline OK")

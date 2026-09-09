"""Sweep-only base quad pipe (segment-by-segment along SWC edges)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadmeshtesser import PipelineParams, TreeQuadPipeline
from quadmeshtesser.mesh_topology import is_edge_manifold, is_watertight, topology_summary


def _branch_count(root) -> int:
    return sum(1 for j in root.iter_all() if len(j.children) > 1)


def test_linear_sweep():
    swc = ROOT / "data" / "test_linear_0.swc"
    r = TreeQuadPipeline(PipelineParams(sweep_only=True)).run(swc)
    m = r.mesh
    assert m.n_quads > 0
    assert m.n_vertices > 0
    assert all(j.radius >= 0.01 for j in r.root.iter_all())
    print(f"linear: v={m.n_vertices} q={m.n_quads} t={m.n_triangles}")


def test_cell021_sweep():
    swc = ROOT / "data" / "cell021.CNG.swc"
    r = TreeQuadPipeline(
        PipelineParams(
            sweep_only=True,
            insert_assist=True,
            bound_tet_scaled=True,
            branch_junction="rmf_loft",
            connect_branch_junction=True,
        )
    ).run(swc)
    root = r.root
    skel = r.skeleton
    n_branch = _branch_count(root)
    n_skel_branch = sum(1 for _, ch in skel.children.items() if len(ch) > 1)
    assert n_branch == n_skel_branch, f"joint/skeleton branch mismatch {n_branch} vs {n_skel_branch}"
    assert n_branch >= 5, f"expected multiple branches, got {n_branch}"

    m = r.mesh
    topo = topology_summary(m)
    assert topo["non_manifold"] == 0, f"non-manifold edges: {topo['non_manifold']}"
    assert is_edge_manifold(m)
    assert topo["boundary"] == 0, f"boundary edges (holes): {topo['boundary']}"
    assert is_watertight(m)
    assert m.n_quads > 200
    assert len(skel.edges) == len(skel.nodes) - 1, f"SWC tree edges {len(skel.edges)}"
    pp = r.preprocess_stats
    assert pp is not None and pp.max_overlap_after < 1e-3
    print(
        f"cell021 sweep [rmf_loft]: v={m.n_vertices} q={m.n_quads} t={m.n_triangles} "
        f"branches={n_branch} nodes={len(skel.nodes)} topo={topo} "
        f"preprocess del={pp.deleted} ins={pp.inserted}"
    )


def test_convex_hull_alias_rmf_loft():
    """Deprecated convex_hull param still builds watertight RMF loft mesh."""
    swc = ROOT / "data" / "cell021.CNG.swc"
    r = TreeQuadPipeline(
        PipelineParams(
            sweep_only=True,
            insert_assist=True,
            bound_tet_scaled=True,
            branch_junction="convex_hull",
            connect_branch_junction=True,
        )
    ).run(swc)
    topo = topology_summary(r.mesh)
    assert topo["non_manifold"] == 0, topo
    assert topo["boundary"] == 0, topo
    assert is_watertight(r.mesh)


def test_vjp_all_quad_sweep():
    """RMF loft sweep should be quad-dominant (few branch tris only)."""
    swc = ROOT / "data" / "cell021.CNG.swc"
    r = TreeQuadPipeline(
        PipelineParams(sweep_only=True, insert_assist=True, bound_tet_scaled=True, connect_branch_junction=True)
    ).run(swc)
    m = r.mesh
    assert m.n_quads >= 440
    assert m.n_triangles < 280, f"too many junction tris {m.n_triangles}"


def test_vjp_no_origin_spikes():
    """Loft bridge centers must not collapse to world origin (regression)."""
    import numpy as np

    swc = ROOT / "data" / "cell021.CNG.swc"
    r = TreeQuadPipeline(
        PipelineParams(sweep_only=True, insert_assist=True, bound_tet_scaled=True, connect_branch_junction=True)
    ).run(swc)
    v = r.mesh.vertices
    assert not np.any(np.linalg.norm(v, axis=1) < 0.5), "vertices stuck at origin"
    from quadmeshtesser.mesh_topology import edge_face_counts

    lens = [
        float(np.linalg.norm(v[a] - v[b]))
        for a, b in edge_face_counts(r.mesh)
    ]
    assert max(lens) < 20.0, f"degenerate long edge {max(lens)}"


def test_y_fork_binary_closure():
    """Single Y-fork: AP→A→B trunk; one A↔B face open; C closes hole via RMF."""
    swc = ROOT / "data" / "test_y_fork.swc"
    r = TreeQuadPipeline(
        PipelineParams(
            sweep_only=True,
            swc_preprocess=False,
            connect_branch_junction=True,
        )
    ).run(swc)
    m = r.mesh
    n_branch = _branch_count(r.root)
    assert n_branch == 1, n_branch
    assert m.n_quads >= 45, f"expected closed fork hull, got {m.n_quads} quads"
    print(
        f"y_fork RMF trunk: v={m.n_vertices} q={m.n_quads} t={m.n_triangles}"
    )


def test_class_bc1_nested_fork_connect():
    """Regression: nested continuous-B fork 3046→14420 must loft without gaps."""
    import numpy as np
    import quadmeshtesser.hole_close as hc
    from quadmeshtesser.polyhedron_builder import build_polyhedron_surface
    from quadmeshtesser.swc_preprocess import load_swc_nodes
    from quadmeshtesser.joint import build_joint_tree_from_nodes
    from quadmeshtesser.mesh_topology import boundary_edges

    swc = ROOT / "data" / "class-BC1.CNG.swc"
    if not swc.is_file():
        print("skip class-BC1 nested fork test")
        return
    nodes, _ = load_swc_nodes(str(swc))
    root = build_joint_tree_from_nodes(nodes)
    real_close = hc.close_mesh_holes
    hc.close_mesh_holes = lambda *a, **k: {"boundary_after": 0}
    try:
        mesh = build_polyhedron_surface(
            root,
            insert_assist=True,
            bound_tet_scaled=True,
            connect_branch_junction=True,
        ).mesh
    finally:
        hc.close_mesh_holes = real_close

    be = boundary_edges(mesh)
    for fid in (3046, 14420):
        j = next(x for x in root.iter_all() if x.node_id == fid)
        fp = np.asarray(j.pos)
        r = max(j.radius, j.parent.radius if j.parent else j.radius) * 4.0
        near = {i for i, v in enumerate(mesh.vertices) if np.linalg.norm(v - fp) < r}
        local = [e for e in be if e[0] in near or e[1] in near]
        assert not local, f"fork {fid} has {len(local)} boundary edges"
    print("class-BC1 nested fork 3046/14420: ok")


if __name__ == "__main__":
    test_linear_sweep()
    test_y_fork_binary_closure()
    test_class_bc1_nested_fork_connect()
    test_cell021_sweep()
    test_convex_hull_alias_rmf_loft()
    test_vjp_all_quad_sweep()
    test_vjp_no_origin_spikes()
    print("test_sweep_base OK")

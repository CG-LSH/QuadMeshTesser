"""Binary fork hull rings: A/B/C outward placement and mutual clearance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadmeshtesser.branch_ring_layout import compute_all_branch_hull_layouts
from quadmeshtesser.fork_ring_layout_opt import _required_ring_gap, _vertex_min_gap
from quadmeshtesser.joint import apply_branch_radius_limits, build_joint_tree_from_nodes
from quadmeshtesser.rmf import apply_rmf_frames
from quadmeshtesser.swc_preprocess import load_swc_nodes


def _ring_mean_radius(ring: list[np.ndarray]) -> float:
    cen = np.mean(ring, axis=0)
    return float(np.mean(np.linalg.norm(np.asarray(ring) - cen, axis=1)))


def _fork_separation_violations(
    root,
    layouts,
    *,
    fork_ids: set[int] | None = None,
    min_ratio: float = 1.0,
) -> list[tuple[int, str, float, float]]:
    bad: list[tuple[int, str, float, float]] = []
    for j in root.iter_all():
        if fork_ids is not None and j.node_id not in fork_ids:
            continue
        lay = layouts.get(j.node_id)
        if lay is None or len(lay.child_order) != 2:
            continue
        id_b, id_c = lay.child_order[0], lay.child_order[1]
        if id_b not in lay.downstream or id_c not in lay.downstream:
            continue
        a_ring = lay.upstream
        b_ring = lay.downstream[id_b]
        c_ring = lay.downstream[id_c]
        if not a_ring or not b_ring or not c_ring:
            continue
        r_a = _ring_mean_radius(a_ring)
        r_b = _ring_mean_radius(b_ring)
        r_c = _ring_mean_radius(c_ring)
        pairs = (
            ("A-B", a_ring, b_ring, _required_ring_gap(r_a, r_b)),
            ("A-C", a_ring, c_ring, _required_ring_gap(r_a, r_c)),
            ("B-C", b_ring, c_ring, _required_ring_gap(r_b, r_c)),
        )
        for label, ra, rb, need in pairs:
            gap = _vertex_min_gap(ra, rb)
            if gap + 1e-9 < need * min_ratio:
                bad.append((j.node_id, label, gap, need))
    return bad


def _build_root(swc: Path):
    nodes, _ = load_swc_nodes(str(swc))
    root = build_joint_tree_from_nodes(nodes)
    apply_rmf_frames(root, 0.0)
    apply_branch_radius_limits(root, f_scale=1.0)
    root.create_bound_sweep(1.0, False, 0, 0, bound_tet_scaled=True, use_rmf=True)
    valid = [0]
    root.create_bound_sweep_id(valid, [0], [0], [0])
    return root


def main() -> None:
    y_path = ROOT / "data" / "test_y_fork.swc"
    bc_path = ROOT / "data" / "class-BC1.CNG.swc"

    if y_path.is_file():
        root = _build_root(y_path)
        layouts = compute_all_branch_hull_layouts(root)
        bad = _fork_separation_violations(root, layouts)
        assert not bad, (
            f"test_y_fork: {len(bad)} ring pairs too close "
            f"(worst {bad[0][0]} {bad[0][1]})"
        )
        print(f"test_y_fork A/B/C outward: ok")

    if bc_path.is_file():
        root = _build_root(bc_path)
        layouts = compute_all_branch_hull_layouts(root)
        nested = {3046, 14420}
        bad = _fork_separation_violations(
            root, layouts, fork_ids=nested, min_ratio=0.85,
        )
        assert not bad, (
            f"class-BC1 nested forks: {len(bad)} ring pairs too close "
            f"(worst {bad[0][0]} {bad[0][1]})"
        )
        print(f"class-BC1 nested fork 3046/14420 A/B/C outward: ok")


if __name__ == "__main__":
    main()

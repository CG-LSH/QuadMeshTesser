"""Test SWC preprocessing: non-overlapping spheres + soma / binary fork fixes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadmeshtesser.swc_io import read_swc
from quadmeshtesser.swc_preprocess import (
    PreprocessConfig,
    compute_adaptive_radius_floor,
    preprocess_swc,
)
import numpy as np


def _max_overlap(nodes) -> float:
    by_id = {n.id: n for n in nodes}
    m = 0.0
    for n in nodes:
        if n.parent in by_id:
            p = by_id[n.parent]
            d = ((n.x - p.x) ** 2 + (n.y - p.y) ** 2 + (n.z - p.z) ** 2) ** 0.5
            m = max(m, p.r + n.r - d)
    children: dict[int, list[int]] = {nid: [] for nid in by_id}
    for n in nodes:
        if n.parent in by_id:
            children[n.parent].append(n.id)
    for ch in children.values():
        sibs = [by_id[cid] for cid in ch]
        for i in range(len(sibs)):
            for j in range(i + 1, len(sibs)):
                a, b = sibs[i], sibs[j]
                d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
                m = max(m, a.r + b.r - d)
    return max(0.0, m)


def _run_case(name: str, path: Path, *, tol: float = 1e-3) -> None:
    if not path.is_file():
        print(f"skip: {path} not found")
        return
    raw = read_swc(path)
    ov0 = _max_overlap(raw)
    result = preprocess_swc(raw, config=PreprocessConfig())
    ov1 = _max_overlap(result.nodes)
    st = result.stats
    print(f"{name}: nodes {len(raw)} -> {len(result.nodes)}")
    print(
        f"  deleted={st.deleted} inserted={st.inserted} moved={st.moved_nodes} "
        f"soma_pruned={st.soma_pruned} soma_pushed={st.soma_pushed} "
        f"binary_fork_fixes={st.binary_fork_fixes}"
    )
    print(
        f"  max overlap: {ov0:.4f} -> {ov1:.4f} "
        f"(stats {st.max_overlap_before:.4f}->{st.max_overlap_after:.4f})"
    )
    assert ov1 < tol, f"{name}: 仍有球体相交 max overlap={ov1}"
    print(f"{name}: ok")


def _test_radius_floor() -> None:
    path = ROOT / "data" / "class-BC1.CNG.swc"
    if not path.is_file():
        print("skip radius floor: class-BC1 not found")
        return
    raw = read_swc(path)
    branch = np.array([n.r for n in raw if n.type != 1 and n.r > 0], dtype=np.float64)
    floor = compute_adaptive_radius_floor(branch, min_absolute=0.01)
    assert floor > 0.05, f"floor too small: {floor}"
    assert floor < float(np.median(branch)), f"floor too large: {floor}"

    result = preprocess_swc(raw, config=PreprocessConfig())
    st = result.stats
    assert st.radius_floor > 0.0
    out_branch = np.array([n.r for n in result.nodes if n.type != 1], dtype=np.float64)
    assert float(out_branch.min()) >= st.radius_floor - 1e-9, (
        f"min radius {out_branch.min()} < floor {st.radius_floor}"
    )
    print(
        f"radius_floor: {st.radius_floor:.4f} "
        f"(raw min {st.radius_min_before:.4f} mean {st.radius_mean_before:.4f}, "
        f"clamped {st.radius_clamped})"
    )


def main() -> None:
    _test_radius_floor()
    _run_case("test_y_fork", ROOT / "data" / "test_y_fork.swc")
    _run_case("cell021", ROOT / "data" / "cell021.CNG.swc")
    _run_case("class-BC1", ROOT / "data" / "class-BC1.CNG.swc", tol=1e-4)
    _run_case("Gol", ROOT / "data" / "Gol.swc", tol=1e-2)
    print("all preprocess tests passed")


if __name__ == "__main__":
    main()

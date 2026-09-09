"""CalcLineSkelWeights / NNLS_Global port."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from quadmeshtesser.cpp_constants import DEFAULT_CAUCHY_ISO, DEFAULT_INFLU_S, DEFAULT_QUARTIC_ISO
from quadmeshtesser.formula_finite import conv_field_quartic_segment
from quadmeshtesser.formula_infinite import integ_line_cauchy
from quadmeshtesser.joint import Joint

Vec3 = np.ndarray


def collect_control_points(root: Joint) -> np.ndarray:
    """Default control points: all joint positions (SWC nodes)."""
    return np.array([j.pos for j in root.iter_all()], dtype=np.float64)


def _spine_field_at(
    p: Vec3,
    begin: Vec3,
    end: Vec3,
    edge_radius: float,
    kernel: str,
    base_influ: float,
) -> float:
    """CSkelSpine::Field at control point (NNLS matrix, field_wt=1)."""
    influ_s = base_influ * 0.6
    support_r = edge_radius * 2.0
    if kernel == "cauchy":
        return integ_line_cauchy(p, begin, end, influ_s)
    return conv_field_quartic_segment(p, begin, end, support_r, 1.0)


def calc_line_skel_weights_global(
    root: Joint,
    *,
    kernel: str = "cauchy",
    iso: float = 0.5,
    base_influ: float = DEFAULT_INFLU_S,
    control_points: np.ndarray | None = None,
    smooth_weights: bool = False,
) -> None:
    """Port of CBlt::NNLS_Global + SetWts."""
    ctr = control_points if control_points is not None else collect_control_points(root)
    if len(ctr) == 0:
        return

    edges: list[tuple[Joint, float]] = []
    for j in root.iter_all():
        if j.parent is not None:
            edges.append((j, j.radius))

    n_edges = len(edges)
    n_ctr = len(ctr)
    if n_edges == 0:
        return

    a_mat = np.zeros((n_ctr, n_edges), dtype=np.float64)
    for col, (joint, edge_r) in enumerate(edges):
        begin = joint.parent.pos  # type: ignore[union-attr]
        end = joint.pos
        for row, cp in enumerate(ctr):
            a_mat[row, col] = _spine_field_at(cp, begin, end, edge_r, kernel, base_influ)

    b_vec = np.full(n_ctr, iso, dtype=np.float64)
    weights, _ = nnls(a_mat, b_vec)

    for col, (joint, edge_r) in enumerate(edges):
        influ_s = base_influ * 0.6
        support_r = edge_r * 2.0
        joint.spine_influ_s = influ_s
        joint.spine_support_r = support_r
        joint.spine_field_wt = float(weights[col])
        joint.spine_kernel = kernel

    if smooth_weights:
        _smooth_weights(root)


def _smooth_weights(root: Joint) -> None:
    """Port of CRecursiveJointFun::SmoothWts."""

    def walk(j: Joint) -> None:
        if j.parent is not None:
            total = 0.0
            cnt = len(j.children) + 1
            if j.parent.spine_field_wt > 0:
                total += j.parent.spine_field_wt
            for c in j.children:
                total += c.spine_field_wt
            j.spine_field_wt = total / max(cnt, 1)
        for c in j.children:
            walk(c)

    walk(root)


def calc_line_skel_weights(
    root: Joint,
    *,
    mode: str = "local",
    kernel: str = "cauchy",
    cauchy_iso: float = DEFAULT_CAUCHY_ISO,
    quartic_iso: float = DEFAULT_QUARTIC_ISO,
    base_influ: float = DEFAULT_INFLU_S,
    bound_tet_scaled: bool = False,
    smooth_weights: bool = False,
    control_points: np.ndarray | None = None,
) -> None:
    """Port of CBlt::CalcLineSkelWeights."""
    if mode == "global":
        iso = cauchy_iso * 5.0 if kernel == "cauchy" else quartic_iso
        calc_line_skel_weights_global(
            root,
            kernel=kernel,
            iso=iso,
            base_influ=base_influ,
            control_points=control_points,
            smooth_weights=smooth_weights,
        )
        return

    from quadmeshtesser.tree_skel import create_conv_line_skel_local

    create_conv_line_skel_local(
        root,
        kernel=kernel,
        cauchy_iso=cauchy_iso,
        quartic_iso=quartic_iso,
        base_influ=base_influ,
        bound_tet_scaled=bound_tet_scaled,
    )

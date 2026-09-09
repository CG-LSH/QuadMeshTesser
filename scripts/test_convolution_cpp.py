"""Convolution field alignment tests (C++ BLT vs formula ports)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from quadmeshtesser.cpp_constants import (
    DEFAULT_CAUCHY_PROJ_ISO,
    DEFAULT_INFLU_S,
    DEFAULT_QUARTIC_ISO,
    projection_iso_for_kernel,
)
from quadmeshtesser.formula_finite import (
    conv_field_quartic_segment,
    field_finite_unbounded,
    integ_line_plyn,
    seg_in_sphere,
)
from quadmeshtesser.formula_infinite import field_infinite_unbounded, integ_line_cauchy
from quadmeshtesser.joint import build_joint_tree_from_swc
from quadmeshtesser.nnls_weights import calc_line_skel_weights
from quadmeshtesser.tree_skel import collect_tree_segments, conv_field_tree, create_conv_line_skel_local


def test_projection_iso_defaults():
    assert abs(projection_iso_for_kernel("cauchy") - DEFAULT_CAUCHY_PROJ_ISO) < 1e-9
    assert abs(projection_iso_for_kernel("quartic") - DEFAULT_QUARTIC_ISO) < 1e-6


def test_quartic_support_r_not_halved():
    swc = ROOT / "data" / "test_linear_0.swc"
    root = build_joint_tree_from_swc(swc)
    create_conv_line_skel_local(root, kernel="quartic")
    segs = collect_tree_segments(root)
    assert len(segs) > 0
    p = segs[0].begin + (segs[0].end - segs[0].begin) * 0.5
    f_tree = conv_field_quartic_segment(
        p, segs[0].begin, segs[0].end, segs[0].support_r, segs[0].field_wt
    )
    ok, b, e = seg_in_sphere(p, segs[0].begin, segs[0].end, segs[0].support_r)
    assert ok
    f_direct = integ_line_plyn(p, b, e, segs[0].support_r) * segs[0].field_wt
    assert abs(f_tree - f_direct) < 1e-9


def test_finite_unbounded_calibration():
    r = 1.0
    support = 2.0 * r
    influ = DEFAULT_INFLU_S * 0.6 * 2.0 / r
    c_inf = field_infinite_unbounded(r, influ)
    c_fin = field_finite_unbounded(r, support)
    assert c_inf > 0 and c_fin > 0


def test_global_nnls_influ_s():
    swc = ROOT / "data" / "test_linear_0.swc"
    root = build_joint_tree_from_swc(swc)
    calc_line_skel_weights(root, mode="global", kernel="cauchy")
    for j in root.iter_all():
        if j.parent is not None:
            assert abs(j.spine_influ_s - DEFAULT_INFLU_S * 0.6) < 1e-9
            assert abs(j.spine_support_r - j.radius * 2.0) < 1e-9


if __name__ == "__main__":
    test_projection_iso_defaults()
    test_quartic_support_r_not_halved()
    test_finite_unbounded_calibration()
    test_global_nnls_influ_s()
    print("convolution alignment tests OK")

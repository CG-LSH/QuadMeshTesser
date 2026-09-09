"""Tests for FieldOffset_Branch, Metaball, offset modes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from quadmeshtesser import PipelineParams, TreeQuadPipeline
from quadmeshtesser.joint import build_joint_tree_from_swc
from quadmeshtesser.metaball import build_metaball_field, collect_tree_skel_metaball
from quadmeshtesser.offset_surf import field_offset_branch, field_offset_combined, field_offset_segment
from quadmeshtesser.tree_skel import create_conv_line_skel_local


def test_metaball_collect():
    swc = ROOT / "data" / "test_linear_0.swc"
    root = build_joint_tree_from_swc(str(swc))
    samples = collect_tree_skel_metaball(root)
    assert len(samples) >= 2
    mb = build_metaball_field(root, mode="point")
    v = mb.conv_field(root.pos)
    assert v >= 0.0
    print(f"metaball samples={len(samples)} field@root={v:.4f}")


def test_field_offset_branch():
    swc = ROOT / "data" / "cell021.CNG.swc"
    root = build_joint_tree_from_swc(str(swc))
    create_conv_line_skel_local(root)
    root.create_half_angle()
    pos = root.pos + np.array([0.05, 0.0, 0.0])
    _, f_seg = field_offset_segment(pos, root, 0.0, 0.0)
    _, f_br = field_offset_branch(pos, root, 0.0, 0.0)
    _, f_cb = field_offset_combined(pos, root)
    print(f"offset field: seg={f_seg:.6f} branch={f_br:.6f} combined={f_cb:.6f}")
    assert f_cb >= f_seg or f_br > 0


def test_pipeline_metaball():
    swc = ROOT / "data" / "test_linear_0.swc"
    r = TreeQuadPipeline(
        PipelineParams(subdiv_levels=1, project=True, appr_style="metaball", restore_offset=False)
    ).run(swc)
    print(f"metaball pipeline: v={r.mesh.n_vertices} q={r.mesh.n_quads}")


if __name__ == "__main__":
    test_metaball_collect()
    test_field_offset_branch()
    test_pipeline_metaball()

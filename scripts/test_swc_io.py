"""SWC loader: arbitrary format variants."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadmeshtesser.pipeline import PipelineParams, TreeQuadPipeline
from quadmeshtesser.swc_io import read_swc, read_swc_file


def _write(tmp: Path, name: str, content: str, *, encoding: str = "utf-8") -> Path:
    p = tmp / name
    p.write_text(content, encoding=encoding)
    return p


def test_header_without_hash(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "hdr.swc",
        "id type x y z radius parent\n1 1 0 0 0 1 -1\n2 1 1 0 0 1 1\n",
    )
    nodes = read_swc(p)
    assert len(nodes) == 2


def test_extra_columns_and_blank_lines(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "extra.swc",
        "# comment\n1 1 0 0 0 1 -1 255\n\n2 1 1 0 0 1 1 128\n",
    )
    nodes = read_swc(p)
    assert len(nodes) == 2
    r = TreeQuadPipeline(PipelineParams(sweep_only=True, swc_preprocess=False)).run(p)
    assert r.mesh.n_quads > 0


def test_parent_zero_and_tabs(tmp_path: Path) -> None:
    p = _write(tmp_path, "pz.swc", "1 1 0 0 0 1 0\n2 1 1 0 0 1\t1\n")
    loaded = read_swc_file(p)
    assert len(loaded.nodes) == 2
    assert loaded.nodes[0].parent < 0


def test_multi_root_keeps_largest_component(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "multi.swc",
        "1 1 0 0 0 1 -1\n2 1 1 0 0 1 1\n3 1 10 0 0 1 -1\n",
    )
    loaded = read_swc_file(p)
    assert len(loaded.nodes) == 2
    assert any("根节点" in w for w in loaded.warnings)
    r = TreeQuadPipeline(PipelineParams(sweep_only=True, swc_preprocess=False)).run(p)
    assert r.mesh.n_quads > 0


def test_morphtesser_style_header(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "neu.swc",
        "#name\n#comment\n##n,type,x,y,z,radius,parent\n"
        "1 1 0 0 0 0.2 -1\n2 3 1 0 0 0.2 1\n",
    )
    from quadmeshtesser.swc_io import read_swc_dataframe, read_swc_with_header

    header, df = read_swc_with_header(p)
    assert len(header) >= 3
    assert len(df) == 2
    assert read_swc_dataframe(p)["parent"].iloc[0] == -1


def test_morphtesser_sample_file_if_present() -> None:
    candidates = [
        Path(r"D:\morphtesser\data\17302_00054_dendrite.swc"),
        ROOT.parent.parent / "morphtesser" / "data" / "17302_00054_dendrite.swc",
    ]
    p = next((c for c in candidates if c.is_file()), None)
    if p is None:
        return
    from quadmeshtesser.swc_io import read_swc_dataframe

    df = read_swc_dataframe(p)
    loaded = read_swc_file(p)
    assert len(df) == len(loaded.nodes) > 1000
    assert loaded.nodes[0].parent == -1


def test_gol_collapse_auxiliary_soma() -> None:
    p = ROOT / "data" / "Gol.swc"
    if not p.is_file():
        return
    from quadmeshtesser.swc_preprocess import load_swc_nodes

    nodes, stats = load_swc_nodes(str(p), preprocess=False)
    soma = [n for n in nodes if n.type == 1]
    assert len(soma) == 1, f"expected 1 soma, got {len(soma)} ids {[n.id for n in soma]}"
    assert soma[0].id == 1
    assert stats is not None
    assert any("辅助 soma" in w for w in stats.warnings)


def test_bundled_samples_still_load() -> None:
    for name in ("test_linear_0.swc", "test_y_fork.swc"):
        p = ROOT / "data" / name
        if not p.is_file():
            continue
        r = TreeQuadPipeline(PipelineParams(sweep_only=True)).run(p)
        assert r.mesh.n_quads > 0, name
    cell = ROOT / "data" / "cell021.CNG.swc"
    if cell.is_file():
        loaded = read_swc_file(cell)
        assert len(loaded.nodes) > 10
        r = TreeQuadPipeline(PipelineParams(sweep_only=True)).run(cell)
        assert r.mesh.n_quads > 100


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-q"])

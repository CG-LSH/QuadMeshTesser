"""SWC file I/O (aligned with MorphTesser read_swc / _read_swc_dataframe)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "SWC_COLS",
    "SwcNode",
    "SwcLoadResult",
    "dataframe_to_nodes",
    "normalize_swc_tree",
    "read_swc",
    "read_swc_dataframe",
    "read_swc_file",
    "read_swc_with_header",
    "write_swc",
]

SWC_COLS = ["id", "type", "x", "y", "z", "r", "parent"]


@dataclass
class SwcNode:
    id: int
    type: int
    x: float
    y: float
    z: float
    r: float
    parent: int


@dataclass
class SwcLoadResult:
    nodes: list[SwcNode]
    warnings: list[str] = field(default_factory=list)


def _decode_swc_bytes(raw: bytes) -> str:
    """Encoding fallback (MorphTesser mirror_swc_coords / convolution_field)."""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_swc_with_header(path: str | Path) -> tuple[list[str], pd.DataFrame]:
    """
    Read SWC like MorphTesser ``mirror_swc_coords.read_swc``:
    preserve ``#`` / blank header lines; parse first 7 whitespace columns per data row.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SWC 文件不存在: {path}")

    text = _decode_swc_bytes(path.read_bytes())
    lines = text.splitlines()
    header = [ln for ln in lines if ln.strip().startswith("#") or len(ln.strip()) == 0]
    data = [ln for ln in lines if not (ln.strip().startswith("#") or len(ln.strip()) == 0)]
    if not data:
        raise ValueError(f"SWC 文件没有有效数据行: {path}")

    df = pd.read_csv(
        io.StringIO("\n".join(data)),
        sep=r"\s+",
        header=None,
        names=SWC_COLS,
        usecols=list(range(7)),
        engine="python",
    )
    df = _coerce_swc_dataframe(df)
    return header, df


def read_swc_dataframe(path: str | Path) -> pd.DataFrame:
    """
    Read SWC as DataFrame.

    Primary path matches MorphTesser utility scripts (header split + 7 columns).
    Falls back to ``convolution_field._read_swc_dataframe`` (``comment='#'``) when needed.
    """
    path = Path(path)
    try:
        _, df = read_swc_with_header(path)
        return df
    except (ValueError, pd.errors.ParserError):
        pass

    kwargs = dict(
        sep=r"\s+",
        comment="#",
        header=None,
        names=SWC_COLS,
        dtype={"id": int, "type": int, "parent": int},
    )
    last_error: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            return _coerce_swc_dataframe(df)
        except UnicodeDecodeError as exc:
            last_error = exc
        except (ValueError, pd.errors.ParserError):
            raw = _decode_swc_bytes(path.read_bytes())
            df = pd.read_csv(io.StringIO(raw), **kwargs)
            return _coerce_swc_dataframe(df)
    if last_error is not None:
        raise last_error
    df = pd.read_csv(path, **kwargs)
    return _coerce_swc_dataframe(df)


def _coerce_swc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop header / junk rows; normalize dtypes (MorphTesser scripts)."""
    out = df.copy()
    id_num = pd.to_numeric(out["id"], errors="coerce")
    mask = id_num.notna()
    if not mask.all():
        out = out.loc[mask].copy()
    if out.empty:
        raise ValueError("SWC 没有可解析的数据行")

    out["id"] = out["id"].astype(int)
    out["type"] = out["type"].astype(int)
    out["parent"] = out["parent"].astype(int)
    out.loc[out["parent"] == 0, "parent"] = -1
    for col in ("x", "y", "z", "r"):
        out[col] = out[col].astype(float)
    return out.reset_index(drop=True)


def dataframe_to_nodes(df: pd.DataFrame) -> list[SwcNode]:
    nodes: list[SwcNode] = []
    for row in df.itertuples(index=False):
        nodes.append(
            SwcNode(
                id=int(row.id),
                type=int(row.type),
                x=float(row.x),
                y=float(row.y),
                z=float(row.z),
                r=float(row.r),
                parent=int(row.parent),
            )
        )
    return nodes


def read_swc(path: str | Path) -> list[SwcNode]:
    """Read SWC nodes (MorphTesser-compatible parsing)."""
    return dataframe_to_nodes(read_swc_dataframe(path))


def _build_children(by_id: dict[int, SwcNode]) -> tuple[dict[int, list[int]], list[int]]:
    """Same root rule as MorphTesser ``build_graph``: parent=-1 or missing → root."""
    children: dict[int, list[int]] = {nid: [] for nid in by_id}
    roots: list[int] = []
    for nid, n in by_id.items():
        if n.parent < 0 or n.parent not in by_id or n.parent == nid:
            roots.append(nid)
        else:
            children[n.parent].append(nid)
    for ch in children.values():
        ch.sort()
    return children, roots


def _subtree_size(root_id: int, children: dict[int, list[int]]) -> int:
    stack = [root_id]
    seen: set[int] = set()
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        stack.extend(children.get(nid, ()))
    return len(seen)


def _collect_reachable(root_id: int, children: dict[int, list[int]]) -> set[int]:
    out: set[int] = set()
    stack = [root_id]
    while stack:
        nid = stack.pop()
        if nid in out:
            continue
        out.add(nid)
        stack.extend(children.get(nid, ()))
    return out


def _pick_primary_root(
    roots: list[int],
    by_id: dict[int, SwcNode],
    children: dict[int, list[int]],
) -> int:
    scored: list[tuple[int, int, int]] = []
    for rid in roots:
        size = _subtree_size(rid, children)
        soma = 1 if by_id[rid].type == 1 else 0
        scored.append((size, soma, rid))
    scored.sort(reverse=True)
    return scored[0][2]


def normalize_swc_tree(nodes: list[SwcNode]) -> SwcLoadResult:
    """
    Repair common SWC issues before joint-tree build.

    Quad mesh pipeline requires a single tree; when MorphTesser would model
    multiple roots separately, we keep the largest connected component and warn.
    """
    warnings: list[str] = []
    by_id: dict[int, SwcNode] = {}
    for n in nodes:
        if n.id in by_id:
            raise ValueError(f"SWC 节点 id 重复: {n.id}")
        by_id[n.id] = SwcNode(
            id=n.id,
            type=n.type,
            x=n.x,
            y=n.y,
            z=n.z,
            r=n.r,
            parent=n.parent,
        )

    for nid, n in list(by_id.items()):
        if n.parent == nid:
            n.parent = -1
            warnings.append(f"节点 {nid} 的 parent 指向自身，已视为根节点")
        elif n.parent >= 0 and n.parent not in by_id:
            warnings.append(f"节点 {nid} 的 parent={n.parent} 不存在，已视为根节点")
            n.parent = -1

    children, roots = _build_children(by_id)
    if not roots:
        raise ValueError("SWC 文件没有根节点 (parent = -1、0 或缺失)")

    if len(roots) > 1:
        primary = _pick_primary_root(roots, by_id, children)
        keep = _collect_reachable(primary, children)
        dropped = sorted(set(by_id) - keep)
        warnings.append(
            f"检测到 {len(roots)} 个根节点，已保留最大连通分量 "
            f"(根 id={primary}，{len(keep)} 节点)，忽略 {len(dropped)} 个游离节点"
        )
        by_id = {nid: by_id[nid] for nid in keep}

    out = [by_id[nid] for nid in sorted(by_id)]
    return SwcLoadResult(nodes=out, warnings=warnings)


def read_swc_file(path: str | Path) -> SwcLoadResult:
    """Parse SWC (MorphTesser I/O) and normalize to a single-root tree."""
    return normalize_swc_tree(read_swc(path))


def write_swc(
    path: str | Path,
    nodes: list[SwcNode],
    note: str = "",
    *,
    header: list[str] | None = None,
) -> None:
    """Write SWC (MorphTesser-style header + data rows)."""
    lines: list[str] = list(header) if header else ["# SWC exported by QuadMeshTesser"]
    if note:
        lines.append(f"# {note}")
    if not header:
        lines.append("# id type x y z radius parent")
    for n in nodes:
        parent = -1 if n.parent < 0 else n.parent
        lines.append(
            f"{n.id} {n.type} {n.x:.6f} {n.y:.6f} {n.z:.6f} {n.r:.6f} {parent}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def nodes_to_arrays(
    nodes: list[SwcNode],
) -> tuple[dict[int, SwcNode], np.ndarray, np.ndarray, list[int]]:
    """Return id map, positions (N,3), radii (N,), ids."""
    id_map = {n.id: n for n in nodes}
    ids = [n.id for n in nodes]
    pos = np.array([[n.x, n.y, n.z] for n in nodes], dtype=np.float64)
    rad = np.array([n.r for n in nodes], dtype=np.float64)
    return id_map, pos, rad, ids

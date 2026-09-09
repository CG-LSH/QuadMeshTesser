"""Tree skeleton built from SWC."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quadmeshtesser.swc_io import SwcNode, read_swc

__all__ = ["TreeEdge", "TreeSkeleton"]


@dataclass
class TreeEdge:
    parent_id: int
    child_id: int


@dataclass
class TreeSkeleton:
    nodes: dict[int, SwcNode]
    children: dict[int, list[int]] = field(default_factory=dict)
    roots: list[int] = field(default_factory=list)
    edges: list[TreeEdge] = field(default_factory=list)

    @classmethod
    def from_nodes(cls, nodes: list[SwcNode]) -> TreeSkeleton:
        by_id = {n.id: n for n in nodes}
        children: dict[int, list[int]] = {nid: [] for nid in by_id}
        roots: list[int] = []
        edges: list[TreeEdge] = []

        for n in by_id.values():
            if n.parent < 0 or n.parent not in by_id:
                roots.append(n.id)
            else:
                children[n.parent].append(n.id)
                edges.append(TreeEdge(n.parent, n.id))

        for ch in children.values():
            ch.sort()

        if not roots:
            raise ValueError("SWC 文件没有根节点 (parent = -1 或 0)")

        return cls(nodes=by_id, children=children, roots=roots, edges=edges)

    @classmethod
    def from_swc(
        cls,
        path: str,
        *,
        radius_scale: float = 1.0,
        min_radius: float = 0.01,
        preprocess: bool = True,
        preprocess_config=None,
    ) -> TreeSkeleton:
        from quadmeshtesser.swc_preprocess import load_swc_nodes

        nodes, _ = load_swc_nodes(
            path,
            radius_scale=radius_scale,
            min_radius=min_radius,
            preprocess=preprocess,
            config=preprocess_config,
        )
        return cls.from_nodes(nodes)

    def position(self, node_id: int) -> np.ndarray:
        n = self.nodes[node_id]
        return np.array([n.x, n.y, n.z], dtype=np.float64)

    def radius(self, node_id: int) -> float:
        return float(self.nodes[node_id].r)

    def parent(self, node_id: int) -> int | None:
        p = self.nodes[node_id].parent
        if p < 0 or p not in self.nodes:
            return None
        return p

    def node_axis(self, node_id: int) -> np.ndarray:
        """Average incoming/outgoing direction at a node (BLT-style)."""
        vecs: list[np.ndarray] = []
        p = self.parent(node_id)
        if p is not None:
            vecs.append(self.position(node_id) - self.position(p))
        for c in self.children.get(node_id, []):
            vecs.append(self.position(c) - self.position(node_id))
        if not vecs:
            return np.array([0.0, 0.0, 1.0])
        axis = np.mean(np.stack(vecs, axis=0), axis=0)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return axis / norm

    def skeleton_polylines(self) -> list[np.ndarray]:
        """Return list of (M,3) polylines for visualization."""
        lines: list[np.ndarray] = []

        def walk(node_id: int, path: list[np.ndarray]) -> None:
            path = path + [self.position(node_id)]
            kids = self.children.get(node_id, [])
            if not kids:
                if len(path) >= 2:
                    lines.append(np.stack(path, axis=0))
                return
            for i, c in enumerate(kids):
                if i == 0:
                    walk(c, path)
                else:
                    walk(c, [self.position(node_id)])

        for r in self.roots:
            walk(r, [])
        return lines

    def segments(self) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
        """(pa, pb, ra, rb) for each SWC edge."""
        segs: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        for e in self.edges:
            pa = self.position(e.parent_id)
            pb = self.position(e.child_id)
            ra = self.radius(e.parent_id)
            rb = self.radius(e.child_id)
            segs.append((pa, pb, ra, rb))
        return segs

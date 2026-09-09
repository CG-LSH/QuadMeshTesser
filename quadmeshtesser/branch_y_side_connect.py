"""Binary Y-fork: connect ring A → ring B or C (avoid self-intersection)."""

from __future__ import annotations

import numpy as np

from quadmeshtesser.branch_y_polar import _add_loft_quad
from quadmeshtesser.fork_connect_score import resolve_ab_connect_pairing
from quadmeshtesser.fork_rings import ForkRings

Vec3 = np.ndarray


def _connect_full_ring_strip_A_to_child(
    ring_A: list[int],
    ring_child: list[int],
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    only_open_edges: bool = False,
) -> None:
    """Full strip with A[i] paired to child[i] (rings pre-aligned)."""
    from quadmeshtesser.branch_convex import _can_add_face

    n = len(ring_A)
    if n < 2 or len(ring_child) != n:
        return
    for i in range(n):
        ni = (i + 1) % n
        ai, aj = ring_A[i], ring_A[ni]
        bi, bj = ring_child[i], ring_child[ni]
        if only_open_edges:
            face = [ai, aj, bj, bi]
            if not _can_add_face(face, edge_counts):
                continue
        _add_loft_quad(
            ai,
            aj,
            bj,
            bi,
            branch_pos,
            verts,
            quads,
            tris,
            edge_counts,
            allow_tri_fallback=False,
        )


def connect_A_to_B(
    rings: ForkRings,
    branch_pos: Vec3,
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    connection_debug: list[np.ndarray] | None = None,
    only_open_edges: bool = True,
    prefer_child_b: bool = True,
) -> str:
    """
    Connect ring **A** to ring **B** on mesh indices (no hull vertex reorder).

    Pairing is recomputed from actual vertex positions every time.
    """
    del prefer_child_b
    ring_A = rings.ring_A
    ring_B = rings.ring_B
    n = len(ring_A)
    if n < 3 or len(ring_B) != n:
        return "B"

    verts = np.array(vert_list, dtype=np.float64)
    frame = rings.fork_joint

    ring_a, ring_b = resolve_ab_connect_pairing(
        ring_A,
        ring_B,
        verts,
        frame,
        rings.spoke_B,
        rings.spoke_C,
    )

    if connection_debug is not None:
        for i in range(n):
            connection_debug.append(
                np.stack([verts[ring_a[i]], verts[ring_b[i]]], axis=0)
            )

    bp = np.asarray(branch_pos, dtype=np.float64)
    _connect_full_ring_strip_A_to_child(
        ring_a,
        ring_b,
        bp,
        verts,
        quads,
        tris,
        edge_counts,
        only_open_edges=only_open_edges,
    )
    return "B"


__all__ = ["connect_A_to_B"]

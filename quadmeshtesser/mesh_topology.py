"""Mesh topology checks (manifold / watertight)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from quadmeshtesser.meshgen import QuadMesh


def edge_face_counts(mesh: QuadMesh) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)

    def add_edge(a: int, b: int) -> None:
        key = (a, b) if a < b else (b, a)
        counts[key] += 1

    for f in mesh.quads:
        for i in range(4):
            add_edge(int(f[i]), int(f[(i + 1) % 4]))

    for f in mesh.triangles:
        for i in range(3):
            add_edge(int(f[i]), int(f[(i + 1) % 3]))

    return counts


def non_manifold_edges(mesh: QuadMesh) -> list[tuple[int, int]]:
    return [e for e, c in edge_face_counts(mesh).items() if c > 2]


def boundary_edges(mesh: QuadMesh) -> list[tuple[int, int]]:
    return [e for e, c in edge_face_counts(mesh).items() if c == 1]


def is_edge_manifold(mesh: QuadMesh) -> bool:
    return len(non_manifold_edges(mesh)) == 0


def is_watertight(mesh) -> bool:
    return len(boundary_edges(mesh)) == 0


def topology_summary(mesh: QuadMesh) -> dict[str, int]:
    counts = edge_face_counts(mesh)
    return {
        "non_manifold": sum(1 for c in counts.values() if c > 2),
        "boundary": sum(1 for c in counts.values() if c == 1),
        "internal": sum(1 for c in counts.values() if c == 2),
        "watertight": sum(1 for c in counts.values() if c == 1) == 0,
    }

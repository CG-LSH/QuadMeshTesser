"""Domain errors for QuadMeshTesser public API entry points."""

from __future__ import annotations


class QuadMeshTesserError(Exception):
    """Base exception for QuadMeshTesser."""


class InvalidSWCError(QuadMeshTesserError):
    """SWC path missing/invalid, or skeleton data structurally unusable."""


class MeshGenerationError(QuadMeshTesserError):
    """Mesh generation / pipeline step failed."""

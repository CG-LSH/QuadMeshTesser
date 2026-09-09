"""Backward-compatible alias — use ``branch_loft_patch`` (Frame-guided lofting)."""

from __future__ import annotations

from quadmeshtesser.branch_loft_patch import append_branch_loft_patch

__all__ = ["append_branch_loft_patch", "append_variational_branch_patch"]

append_variational_branch_patch = append_branch_loft_patch

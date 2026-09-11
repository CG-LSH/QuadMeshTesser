"""Selective / blended projection to preserve smooth subdiv mesh where already close."""

from __future__ import annotations

import numpy as np

__all__ = ["blend_projected_vertices"]


def blend_projected_vertices(
    original: np.ndarray,
    projected: np.ndarray,
    *,
    field_err: np.ndarray | None = None,
    iso: float = 1.0,
    edge_steps: np.ndarray | None = None,
    rel_tol: float = 0.12,
    move_tol: float = 0.30,
    soft: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Keep original verts when already close; otherwise take (or blend) projection.

    A vertex is treated as "close enough" when either:
      - |F-iso| / max(|iso|, eps) <= rel_tol, or
      - ||proj-orig|| <= move_tol * local_edge_step

    With soft=True, weights ramp from 0 at tol to 1 at 2*tol (field) / 2*move_tol (move).
    """
    orig = np.asarray(original, dtype=np.float64)
    proj = np.asarray(projected, dtype=np.float64)
    if orig.shape != proj.shape:
        raise ValueError("original/projected shape mismatch")
    n = len(orig)
    if n == 0:
        return orig.copy(), {"kept": 0.0, "projected": 0.0, "blended": 0.0}

    move = np.linalg.norm(proj - orig, axis=1)
    iso_abs = max(abs(float(iso)), 1e-9)
    rel_tol = max(float(rel_tol), 0.0)
    move_tol = max(float(move_tol), 0.0)

    if field_err is not None:
        rel_err = np.asarray(field_err, dtype=np.float64).reshape(-1) / iso_abs
    else:
        rel_err = np.full(n, np.inf, dtype=np.float64)

    if edge_steps is not None:
        steps = np.maximum(np.asarray(edge_steps, dtype=np.float64).reshape(-1), 1e-9)
        move_ratio = move / steps
    else:
        move_ratio = np.full(n, np.inf, dtype=np.float64)

    # Weight 0 = keep original, 1 = full projection.
    if soft and rel_tol > 0:
        w_field = np.clip((rel_err - rel_tol) / max(rel_tol, 1e-12), 0.0, 1.0)
    else:
        w_field = (rel_err > rel_tol).astype(np.float64)

    if soft and move_tol > 0:
        w_move = np.clip((move_ratio - move_tol) / max(move_tol, 1e-12), 0.0, 1.0)
    else:
        w_move = (move_ratio > move_tol).astype(np.float64)

    # Need projection only if BOTH field and move say it is not close.
    # If either criterion says close, suppress projection weight.
    w = np.minimum(w_field, w_move)

    out = orig + (proj - orig) * w[:, None]
    kept = float(np.mean(w <= 1e-6))
    full = float(np.mean(w >= 1.0 - 1e-6))
    blended = float(np.mean((w > 1e-6) & (w < 1.0 - 1e-6)))
    return out, {"kept": kept, "projected": full, "blended": blended, "mean_w": float(np.mean(w))}

"""Project / sample data path discovery."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "candidate_data_dirs",
    "default_sample_swc",
    "discover_sample_swc",
    "package_root",
    "project_root",
]

_PREFERRED_SAMPLES = ("test_linear_0.swc", "cell021.CNG.swc")


def package_root() -> Path:
    """Directory containing quadmeshtesser package."""
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """Dev tree root (parent of quadmeshtesser/)."""
    return package_root().parent


def candidate_data_dirs() -> list[Path]:
    """Search locations for bundled or nearby SWC samples."""
    dirs: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        try:
            r = p.resolve()
        except OSError:
            return
        if r in seen:
            return
        seen.add(r)
        dirs.append(p)

    add(package_root() / "data")
    add(project_root() / "data")

    cwd = Path.cwd()
    add(cwd / "data")
    add(cwd)

    cur = cwd
    for _ in range(6):
        add(cur / "data")
        add(cur / "QuadMeshTesser" / "data")
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    # Optional sibling repo (MorphTesser)
    add(project_root().parent / "morphtesser" / "data")

    return dirs


def discover_sample_swc() -> list[Path]:
    """All ``*.swc`` under candidate data directories (recursive)."""
    found: list[Path] = []
    seen: set[Path] = set()
    for d in candidate_data_dirs():
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.swc")):
            if not p.is_file():
                continue
            try:
                key = p.resolve()
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            found.append(p)
    return found


def default_sample_swc() -> Path | None:
    """Preferred demo SWC, or first discovered file."""
    samples = discover_sample_swc()
    if not samples:
        return None
    by_name = {p.name: p for p in samples}
    for name in _PREFERRED_SAMPLES:
        if name in by_name:
            return by_name[name]
    return samples[0]

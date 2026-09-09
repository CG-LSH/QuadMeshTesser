# QuadMeshTesser

**Skeleton-Driven Quadrilateral Surface Modeling of Neurons with Controllable Convolution Constraints**

QuadMeshTesser builds quadrilateral-dominant, topology-checkable neuronal membrane meshes from SWC skeletons. An explicit sweep and junction/soma assembly produce a base mesh; Catmull–Clark subdivision then increases resolution; controllable convolution isosurfaces constrain vertex positions so the mesh stays faithful to the skeleton radii while remaining editable.

![QuadMeshTesser workflow](workflow.png)

*Pipeline: SWC preprocessing → RMF quad sweep → fork/soma joining → Catmull–Clark subdivision → controllable convolution isosurface projection.*

## Features

- **Input:** SWC neuronal morphologies (radius-annotated tree skeletons)
- **Output:** Quadrilateral-dominant surface meshes (OBJ with quad faces)
- **Meshing:** Rotation-minimizing frame (RMF) cross-section sweep, Y-fork junction strips, soma sphere constraints, and multi-region watertight stitching
- **Refinement:** Catmull–Clark subdivision; optional projection of vertices onto a line-skeleton controllable convolution isosurface
- **UI:** PyQt5 / qtpy + PyVista parameter panel and 3D preview

## Install

```bash
cd QuadMeshTesser
python -m pip install -r requirements.txt
python -m app.main
# or run setup_and_run.bat / run.bat
```

Dependencies: `numpy`, `pandas`, `scipy`, `PyQt5`, `qtpy`, `pyvista`, `pyvistaqt`

> On Windows, if PySide6 / PyQt6 hit DLL issues, this project defaults to **PyQt5** via `qtpy`.

After launch, load data from **Quick samples** or **Open SWC** (e.g. `data/Gol.swc`, `data/test_linear_0.swc`).

## Main parameters

| Parameter | Description |
|-----------|-------------|
| Cross-section sides | Polygon sides per section; **4** yields a pure-quad sweep |
| Catmull–Clark levels | Subdivision depth (smoother, more vertices) |
| Radius scale | Global scale on SWC radii |
| Sweep only | Build the base mesh only; skip subdivision and convolution projection |
| SWC preprocess | Remove overlapping node spheres, short-edge issues, etc. |
| Approximation mode | Convolution constraint style (controllable field / line integral / subdivision-only, …) |
| Kernel | `quartic` (compact support) / `cauchy` |
| Iso value | Target isosurface \(F = T\) |
| Project to isosurface | Iteratively project subdivided vertices onto the convolution surface |

## Method highlights

- **Skeleton-driven explicit meshing:** RMF tube walls, Y-strips or convex transitions at forks, soma spheres joined to primary branches, edge-occupancy checks for a watertight manifold.
- **Controllable convolution constraints:** A fusion field from the same skeleton and radii; finite / variable support limits over-blending; after subdivision, vertices are pulled to the isosurface along the field gradient to reduce inflate/shrink from pure subdivision.
- **Quad-dominant output:** Suited to further subdivision, editing, and simulation preprocessing.

## Layout

```
QuadMeshTesser/
  app/              # GUI
  quadmeshtesser/   # Core library
  data/             # Sample SWC / reference meshes
  scripts/          # Tests and benchmarks
  workflow.png      # Pipeline figure
```

## Programmatic example

```python
from pathlib import Path
from quadmeshtesser import TreeQuadPipeline, PipelineParams

result = TreeQuadPipeline(
    PipelineParams(subdiv_levels=2, project=True, sweep_only=False)
).run(Path("data/test_linear_0.swc"))
print(result.mesh.n_vertices, result.mesh.n_quads)
```

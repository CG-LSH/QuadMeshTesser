"""End-to-end pipeline: SWC → BLT polyhedron → subdivision → approximation → optional OffsetSurf."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quadmeshtesser.errors import InvalidSWCError, MeshGenerationError

from quadmeshtesser.appr_parallel import (
    approximate_vertices_line,
    approximate_vertices_metaball,
    resolve_backend,
)
from quadmeshtesser.cpp_constants import (
    CPP_DEFAULT_APPR_STYLE,
    CPP_DEFAULT_BOUND_SCALE,
    CPP_DEFAULT_CONNECT_BRANCH_JUNCTION,
    CPP_DEFAULT_INIT_ROT,
    CPP_DEFAULT_INSERT_ASSIST,
    CPP_DEFAULT_KERNEL,
    CPP_DEFAULT_RESTORE_OFFSET,
    CPP_DEFAULT_SUBDIV_LEVELS,
    POST_PROJECT_SMOOTH_ITERS,
    POST_PROJECT_SMOOTH_LAMBDA,
    POST_PROJECT_SMOOTH_MU,
    CPP_DEFAULT_SUB_LMT_CNT,
    CPP_DEFAULT_SWEEP_ONLY,
    CPP_DEFAULT_USE_LMT_CNT,
    DEFAULT_CAUCHY_ISO,
    DEFAULT_CAUCHY_PROJ_ISO,
    DEFAULT_INFLU_S,
    DEFAULT_QUARTIC_ISO,
    MAX_APPR_ITER,
    projection_iso_for_kernel,
)
from quadmeshtesser.export_obj import export_obj
from quadmeshtesser.joint import Joint, build_joint_tree_from_nodes
from quadmeshtesser.meshgen import QuadMesh, build_base_quad_mesh_from_joint
from quadmeshtesser.metaball import (
    DEFAULT_METABALL_INTERVAL,
    DEFAULT_METABALL_ISO,
    METABALL_CPU_RADIUS_SCALE,
    build_metaball_field,
    collect_tree_skel_metaball,
)
from quadmeshtesser.nnls_weights import calc_line_skel_weights
from quadmeshtesser.obj_io import load_obj_mesh
from quadmeshtesser.offset_surf import (
    compute_vertex_normals,
    delete_bad_offsets_all,
    mesh_to_triangles,
    offset_surf,
)
from quadmeshtesser.skeleton import TreeSkeleton
from quadmeshtesser.subdivision import catmull_clark
from quadmeshtesser.swc_preprocess import PreprocessStats, load_swc_nodes
from quadmeshtesser.junction_methods import DEFAULT_BRANCH_JUNCTION, JunctionMethod, parse_branch_junction
from quadmeshtesser.morphtesser_field import DEFAULT_MORPH_ISO
from quadmeshtesser.tree_skel import collect_tree_segments, compute_appr_step_sizes, conv_field_tree

__all__ = ["PipelineParams", "PipelineResult", "TreeQuadPipeline"]


@dataclass
class PipelineParams:
    subdiv_levels: int = CPP_DEFAULT_SUBDIV_LEVELS
    radius_scale: float = 1.0
    min_radius: float = 0.01
    bound_scale: float = CPP_DEFAULT_BOUND_SCALE
    init_rot: float = CPP_DEFAULT_INIT_ROT
    insert_assist: bool = CPP_DEFAULT_INSERT_ASSIST
    bound_tet_scaled: bool = False
    restore_offset: bool = CPP_DEFAULT_RESTORE_OFFSET
    ref_obj_path: str | None = None
    offset_mode: str = "segment"
    weight_mode: str = "local"
    smooth_weights: bool = False
    sweep_only: bool = CPP_DEFAULT_SWEEP_ONLY  # 仅扫掠四边形管道，跳过后续细分/投影
    use_lmt_cnt: bool = CPP_DEFAULT_USE_LMT_CNT
    sub_lmt_cnt: int = CPP_DEFAULT_SUB_LMT_CNT
    appr_style: str = CPP_DEFAULT_APPR_STYLE
    kernel: str = CPP_DEFAULT_KERNEL
    cauchy_iso: float = DEFAULT_CAUCHY_ISO
    quartic_iso: float = DEFAULT_QUARTIC_ISO
    base_influ: float = DEFAULT_INFLU_S
    metaball_iso: float = DEFAULT_METABALL_ISO
    metaball_interval: float = DEFAULT_METABALL_INTERVAL
    project: bool = False  # EAS_Limit skips ApprConvSurf
    sample_iso_surface: bool = False  # viz target cloud; expensive for local/quartic
    project_backend: str = "numba"
    sides: int = 4
    ts: float = 1.2
    iso_value: float | None = None  # None → kernel default (C++ ApprConvSurf)
    project_iters: int = 10
    project_step: float = 0.35
    project_selective: bool = True  # skip verts already close to iso
    project_rel_tol: float = 0.12  # |F-iso|/|iso|
    project_move_tol: float = 0.45  # ||delta|| / local edge
    project_move_cap: float = 0.85  # max ||delta|| / local edge after blend
    post_project_smooth_iters: int = POST_PROJECT_SMOOTH_ITERS
    post_project_smooth_lambda: float = POST_PROJECT_SMOOTH_LAMBDA
    post_project_smooth_mu: float = POST_PROJECT_SMOOTH_MU
    swc_preprocess: bool = True
    branch_junction: str | JunctionMethod = DEFAULT_BRANCH_JUNCTION
    junction_hub_sides: int | None = None
    junction_child_sides: int | None = None
    connect_branch_junction: bool = CPP_DEFAULT_CONNECT_BRANCH_JUNCTION


@dataclass
class PipelineResult:
    root: Joint
    skeleton: TreeSkeleton
    mesh: QuadMesh
    params: PipelineParams
    preprocess_stats: PreprocessStats | None = None
    iso_surface_points: np.ndarray | None = None
    iso_surface_mesh: object | None = None  # QuadMesh on real convolution iso (ring mesh)
    branch_connection_debug_segments: list[np.ndarray] | None = None


class TreeQuadPipeline:
    def __init__(self, params: PipelineParams | None = None) -> None:
        self.params = params or PipelineParams()

    def run(self, swc_path: str | Path) -> PipelineResult:
        """Run the full pipeline: SWC → mesh.

        Raises:
            InvalidSWCError: Missing/invalid SWC path or unusable skeleton data.
            MeshGenerationError: Mesh generation / pipeline failure.
        """
        swc_path = Path(swc_path)
        if not swc_path.exists():
            raise InvalidSWCError(f"SWC file not found: {swc_path}")
        if not swc_path.is_file():
            raise InvalidSWCError(f"Path is not a file: {swc_path}")

        try:
            return self._run_impl(swc_path)
        except (InvalidSWCError, MeshGenerationError):
            raise
        except ValueError as e:
            msg = str(e)
            if "SWC" in msg or "树" in msg or "root" in msg.lower() or "nodes" in msg.lower():
                raise InvalidSWCError(f"Invalid SWC structure: {e}") from e
            raise MeshGenerationError(f"Mesh generation failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise MeshGenerationError(f"Mesh generation failed: {e}") from e
        except OSError as e:
            raise InvalidSWCError(f"Failed to read SWC file: {e}") from e

    def _run_impl(self, swc_path: Path) -> PipelineResult:
        p = self.params

        nodes, pp_stats = load_swc_nodes(
            str(swc_path),
            radius_scale=p.radius_scale,
            min_radius=p.min_radius,
            preprocess=p.swc_preprocess,
        )
        root = build_joint_tree_from_nodes(nodes)

        use_morphtesser = p.appr_style == "morphtesser"
        use_pure_limit = p.appr_style == "limit"
        use_lmt = (p.use_lmt_cnt or use_pure_limit) and not p.sweep_only
        sub_lmt = p.sub_lmt_cnt if p.sub_lmt_cnt > 0 else p.subdiv_levels
        branch_j = parse_branch_junction(p.branch_junction)
        skeleton = TreeSkeleton.from_nodes(nodes)

        if p.sweep_only:
            use_lmt_sw = (p.use_lmt_cnt or p.appr_style == "limit") and p.sweep_only
            sub_lmt_sw = p.sub_lmt_cnt if p.sub_lmt_cnt > 0 else p.subdiv_levels
            from quadmeshtesser.polyhedron_builder import build_polyhedron_surface

            build_result = build_polyhedron_surface(
                root,
                bound_scale=p.bound_scale,
                init_rot=p.init_rot,
                use_lmt_cnt=use_lmt_sw,
                sub_lmt_cnt=sub_lmt_sw if use_lmt_sw else 0,
                insert_assist=p.insert_assist,
                bound_tet_scaled=p.bound_tet_scaled or p.insert_assist,
                branch_junction=branch_j,
                junction_hub_sides=p.junction_hub_sides,
                junction_child_sides=p.junction_child_sides,
                connect_branch_junction=p.connect_branch_junction,
            )
            mesh = build_result.mesh
            return PipelineResult(
                root=root,
                skeleton=skeleton,
                mesh=mesh,
                params=p,
                preprocess_stats=pp_stats,
                iso_surface_points=None,
                iso_surface_mesh=None,
                branch_connection_debug_segments=build_result.branch_connection_debug_segments,
            )

        base = build_base_quad_mesh_from_joint(
            root,
            bound_scale=p.bound_scale,
            init_rot=p.init_rot,
            insert_assist=p.insert_assist,
            bound_tet_scaled=p.bound_tet_scaled or p.insert_assist,
            use_lmt_cnt=use_lmt,
            sub_lmt_cnt=sub_lmt if use_lmt else 0,
            branch_junction=branch_j,
            connect_branch_junction=p.connect_branch_junction,
        )
        from quadmeshtesser.hole_close import close_quad_mesh

        base = close_quad_mesh(base, max_passes=12)

        def _iso_surface_pts() -> np.ndarray:
            from quadmeshtesser.iso_surface_samples import sample_iso_surface_points

            morph_iso = p.iso_value if p.iso_value is not None else DEFAULT_MORPH_ISO
            ts_fixed = p.ts if p.ts > 0 else None
            return sample_iso_surface_points(
                root, iso=morph_iso, ts_fixed=ts_fixed, max_points=80_000
            )

        use_metaball_point = p.appr_style == "metaball"
        use_metaball_line = p.appr_style == "metaball_line"
        kernel = p.kernel if not use_metaball_point else "cauchy"

        need_line_field = (
            (not use_morphtesser)
            and (p.project or p.restore_offset)
            and not use_pure_limit
        )
        if need_line_field:
            calc_line_skel_weights(
                root,
                mode=p.weight_mode,
                kernel=kernel,
                cauchy_iso=p.cauchy_iso,
                quartic_iso=p.quartic_iso,
                base_influ=p.base_influ,
                bound_tet_scaled=p.bound_tet_scaled or p.insert_assist,
                smooth_weights=p.smooth_weights,
            )

        iso_pts: np.ndarray | None = None
        iso_mesh = None
        # Real convolution isosurface (same field as projection), only when UI asks.
        if p.sample_iso_surface and (p.project or use_morphtesser):
            if use_morphtesser:
                iso_pts = _iso_surface_pts()
            elif not use_metaball_point and not use_pure_limit:
                from quadmeshtesser.iso_surface_samples import extract_line_skel_iso_voxel_mesh

                line_iso = (
                    p.metaball_iso
                    if use_metaball_line
                    else (
                        p.iso_value
                        if p.iso_value is not None
                        else projection_iso_for_kernel(
                            kernel, cauchy_iso=p.cauchy_iso, quartic_iso=p.quartic_iso
                        )
                    )
                )
                line_kernel = "cauchy" if use_metaball_line else kernel
                iso_mesh = extract_line_skel_iso_voxel_mesh(
                    root,
                    kernel=line_kernel,
                    iso=float(line_iso),
                    sides=10,
                    spacing_factor=0.45,
                )
                if iso_mesh is not None and getattr(iso_mesh, "n_vertices", 0) > 0:
                    iso_pts = np.asarray(iso_mesh.vertices, dtype=np.float64)

        mesh = catmull_clark(base, levels=p.subdiv_levels)

        from quadmeshtesser.mesh_cleanup import cleanup_quad_mesh, taubin_smooth_mesh

        mesh = cleanup_quad_mesh(mesh)

        if use_morphtesser:
            iso = p.iso_value if p.iso_value is not None else DEFAULT_MORPH_ISO
        elif use_metaball_point or use_metaball_line:
            iso = p.metaball_iso
        elif p.iso_value is not None:
            iso = p.iso_value
            ref_iso = projection_iso_for_kernel(
                kernel, cauchy_iso=p.cauchy_iso, quartic_iso=p.quartic_iso
            )
            if iso > ref_iso * 8.0:
                iso = ref_iso
        else:
            iso = projection_iso_for_kernel(kernel, cauchy_iso=p.cauchy_iso, quartic_iso=p.quartic_iso)

        if p.restore_offset and mesh.n_vertices > 0 and not use_pure_limit:
            if p.ref_obj_path and Path(p.ref_obj_path).is_file():
                ref_verts, ref_tris = load_obj_mesh(p.ref_obj_path)
            else:
                ref_verts, ref_tris = mesh_to_triangles(mesh)
            root.create_offset(ref_verts, ref_tris)
            delete_bad_offsets_all(root)

        if p.project and mesh.n_vertices > 0 and not use_pure_limit:
            steps = compute_appr_step_sizes(mesh.vertices, mesh.quads, mesh.triangles)

            if use_morphtesser:
                from quadmeshtesser.morph_appr import approximate_vertices_morph

                ts_fixed = p.ts if p.ts > 0 else None
                verts = approximate_vertices_morph(
                    mesh.vertices,
                    steps,
                    root,
                    iso,
                    ts_fixed=ts_fixed,
                    max_iter=p.project_iters,
                )
            elif use_metaball_point:
                backend = resolve_backend(p.project_backend, n_vertices=mesh.n_vertices)
                samples = collect_tree_skel_metaball(root, interval=p.metaball_interval)
                ball_pos = np.array([s.pos for s in samples], dtype=np.float64)
                ball_r = np.array([s.radius for s in samples], dtype=np.float64)
                verts = approximate_vertices_metaball(
                    mesh.vertices,
                    steps,
                    ball_pos,
                    ball_r,
                    iso,
                    METABALL_CPU_RADIUS_SCALE,
                    backend=backend,
                )
            else:
                backend = resolve_backend(p.project_backend, n_vertices=mesh.n_vertices)
                segments = collect_tree_segments(root)
                line_kernel = "cauchy" if use_metaball_line else kernel
                line_iso = p.metaball_iso if use_metaball_line else iso
                if segments:
                    verts = approximate_vertices_line(
                        mesh.vertices,
                        steps,
                        segments,
                        line_iso,
                        kernel=line_kernel,
                        backend=backend,
                    )
                else:
                    verts = mesh.vertices.copy()

            if p.project_selective and verts is not None:
                from quadmeshtesser.selective_project import blend_projected_vertices

                orig_v = np.asarray(mesh.vertices, dtype=np.float64)
                proj_v = np.asarray(verts, dtype=np.float64)
                # Fast path: decide by displacement vs local edge (preserves smooth subdiv).
                # Optional hard cap: never move more than project_move_cap * edge in one shot.
                verts, _sel_stats = blend_projected_vertices(
                    orig_v,
                    proj_v,
                    field_err=None,
                    iso=float(iso),
                    edge_steps=steps,
                    rel_tol=float(p.project_rel_tol),
                    move_tol=float(p.project_move_tol),
                    soft=True,
                )
                cap = float(getattr(p, "project_move_cap", 1.25))
                if cap > 0:
                    delta = verts - orig_v
                    dist = np.linalg.norm(delta, axis=1)
                    lim = np.maximum(np.asarray(steps, dtype=np.float64) * cap, 1e-9)
                    scale = np.ones(len(orig_v), dtype=np.float64)
                    over = dist > lim
                    scale[over] = lim[over] / dist[over]
                    verts = orig_v + delta * scale[:, None]

            mesh = QuadMesh(verts, mesh.quads.copy(), mesh.triangles.copy())
            mesh = cleanup_quad_mesh(mesh)
            if use_morphtesser:
                mesh = close_quad_mesh(mesh, max_passes=32)
                mesh = cleanup_quad_mesh(mesh)

            if int(p.post_project_smooth_iters) > 0:
                mesh = taubin_smooth_mesh(
                    mesh,
                    iterations=int(p.post_project_smooth_iters),
                    lam=float(p.post_project_smooth_lambda),
                    mu=float(p.post_project_smooth_mu),
                )
                mesh = cleanup_quad_mesh(mesh)

            if p.restore_offset:
                normals = compute_vertex_normals(mesh.vertices, mesh.quads, mesh.triangles)
                mesh = offset_surf(mesh, root, normals=normals, mode=p.offset_mode)

        return PipelineResult(
            root=root,
            skeleton=skeleton,
            mesh=mesh,
            params=p,
            preprocess_stats=pp_stats,
            iso_surface_points=iso_pts,
            iso_surface_mesh=iso_mesh,
        )

    def export(self, result: PipelineResult, out_path: str | Path) -> None:
        export_obj(result.mesh, out_path)

    @staticmethod
    def estimate_iso(result: PipelineResult, sample_count: int = 64) -> float:
        """Suggest iso from current mesh vertex field (for projection tuning)."""
        p = result.params
        if p.appr_style == "morphtesser":
            return DEFAULT_MORPH_ISO
        kernel = p.kernel if p.appr_style not in ("metaball", "metaball_line") else "cauchy"
        segments = collect_tree_segments(result.root)
        if result.mesh.n_vertices > 0 and segments:
            n = result.mesh.n_vertices
            idx = np.linspace(0, n - 1, min(sample_count, n), dtype=int)
            vals = [
                conv_field_tree(result.mesh.vertices[i], segments, kernel)
                for i in idx
            ]
            if vals:
                return float(np.median(vals))
        if p.appr_style in ("metaball", "metaball_line"):
            mb = build_metaball_field(
                result.root,
                mode="point" if p.appr_style == "metaball" else "line",
                iso=p.metaball_iso,
            )
            pts = [j.pos for j in result.root.iter_all()]
            vals = [mb.conv_field(pt) for pt in pts]
            return float(np.median(vals) * 0.85) if vals else DEFAULT_METABALL_ISO

        return projection_iso_for_kernel(
            kernel,
            cauchy_iso=p.cauchy_iso,
            quartic_iso=p.quartic_iso,
        )

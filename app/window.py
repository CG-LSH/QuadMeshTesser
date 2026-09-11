"""PyQt + PyVista GUI for SWC → quad mesh pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from quadmeshtesser.cpp_constants import (
    CPP_DEFAULT_APPR_STYLE,
    CPP_DEFAULT_BOUND_SCALE,
    CPP_DEFAULT_CONNECT_BRANCH_JUNCTION,
    CPP_DEFAULT_INSERT_ASSIST,
    CPP_DEFAULT_KERNEL,
    CPP_DEFAULT_SUBDIV_LEVELS,
    POST_PROJECT_SMOOTH_ITERS,
    CPP_DEFAULT_SUB_LMT_CNT,
    CPP_DEFAULT_SWEEP_ONLY,
    CPP_DEFAULT_USE_LMT_CNT,
    DEFAULT_CAUCHY_PROJ_ISO,
    DEFAULT_QUARTIC_ISO,
    MAX_APPR_ITER,
    CPP_DEFAULT_PROJECT_ITERS,
)
from quadmeshtesser.export_obj import export_obj, mesh_to_pyvista_faces
from quadmeshtesser.mesh_viz import split_root_soma_mesh
from quadmeshtesser.metaball import DEFAULT_METABALL_INTERVAL, DEFAULT_METABALL_ISO
from quadmeshtesser.paths import candidate_data_dirs, discover_sample_swc
from quadmeshtesser.junction_methods import (
    BRANCH_JUNCTION_CHOICES,
    DEFAULT_BRANCH_JUNCTION,
    branch_junction_label,
    parse_branch_junction,
)
from quadmeshtesser.mesh_topology import topology_summary
from quadmeshtesser.pipeline import PipelineParams, PipelineResult, TreeQuadPipeline
from quadmeshtesser.morphtesser_field import DEFAULT_MORPH_ISO
from quadmeshtesser.skeleton_viz import add_joint_skeleton_wire_to_plotter

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("QuadMeshTesser — SWC → 四边形 Mesh (RMF Loft)")
        self.resize(1680, 820)

        self._result: PipelineResult | None = None
        self._swc_path: Path | None = None

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        self._plotter = QtInteractor(splitter)
        self._plotter.set_background("white")
        self._plotter.add_axes()
        splitter.addWidget(self._plotter.interactor)

        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)

        file_box = QGroupBox("文件")
        file_layout = QVBoxLayout(file_box)
        self._path_label = QLabel("未加载 SWC")
        self._path_label.setWordWrap(True)
        file_layout.addWidget(self._path_label)

        self._sample_combo = QComboBox()
        self._sample_combo.setToolTip("内置示例；任意 SWC 请点下方「打开 SWC 文件…」")
        self._sample_combo.currentIndexChanged.connect(self._on_sample_selected)
        file_layout.addWidget(QLabel("快速示例（可选）"))
        file_layout.addWidget(self._sample_combo)

        btn_load = QPushButton("打开 SWC 文件…")
        btn_load.setToolTip("从任意路径加载标准 SWC（支持 # 注释、表头行、额外列、GBK 编码等）")
        btn_load.clicked.connect(self._load_swc)
        btn_export = QPushButton("导出 OBJ…")
        btn_export.clicked.connect(self._export_obj)
        file_layout.addWidget(btn_load)
        file_layout.addWidget(btn_export)
        layout.addWidget(file_box)

        param_box = QGroupBox("网格参数")
        form = QFormLayout(param_box)

        self._sides = QSpinBox()
        self._sides.setRange(4, 16)
        self._sides.setValue(4)
        self._sides.setToolTip("截面边数；4 为纯四边形扫掠")
        form.addRow("截面边数", self._sides)

        self._subdiv = QSpinBox()
        self._subdiv.setRange(0, 6)
        self._subdiv.setValue(CPP_DEFAULT_SUBDIV_LEVELS)
        form.addRow("Catmull-Clark 细分", self._subdiv)

        self._radius_scale = QDoubleSpinBox()
        self._radius_scale.setRange(0.01, 100.0)
        self._radius_scale.setValue(1.0)
        self._radius_scale.setSingleStep(0.1)
        form.addRow("半径缩放", self._radius_scale)

        self._sweep_only = QCheckBox("仅初始 mesh")
        self._sweep_only.setChecked(CPP_DEFAULT_SWEEP_ONLY)
        self._sweep_only.setToolTip("只生成初始扫掠网格；跳过细分 / 卷积 / 投影")
        self._sweep_only.toggled.connect(self._on_sweep_only_changed)
        form.addRow(self._sweep_only)

        self._branch_junction = QComboBox()
        for method_id, label in BRANCH_JUNCTION_CHOICES:
            self._branch_junction.addItem(label, method_id)
        default_idx = self._branch_junction.findData(DEFAULT_BRANCH_JUNCTION.value)
        if default_idx >= 0:
            self._branch_junction.setCurrentIndex(default_idx)
        self._branch_junction.setToolTip(
            "内部分叉：RMF 截面 + 父/子环匹配 + Branch-aware loft patch；"
            "或凸包法；Root 1–2 子仍用四边形球"
        )
        self._branch_junction.currentIndexChanged.connect(self._on_branch_junction_changed)
        form.addRow("分叉衔接", self._branch_junction)

        self._connect_branch = QCheckBox("二分叉：Y-Fork 衔接")
        self._connect_branch.setChecked(CPP_DEFAULT_CONNECT_BRANCH_JUNCTION)
        self._connect_branch.setToolTip(
            "内部分叉处用 RMF Y-Fork loft 连接父/子管道环；"
            "关闭则仅生成管道侧壁与环预览端盖。"
        )
        form.addRow(self._connect_branch)

        self._swc_preprocess = QCheckBox("SWC 预处理 (去相交 / 重采样)")
        self._swc_preprocess.setChecked(True)
        self._swc_preprocess.setToolTip(
            "读取 SWC 后：移动子树、插点/删点/重采样、必要时缩小半径，"
            "使相邻与分叉处节点球不相交；最后可选自适应半径底线与沿树平滑"
        )
        form.addRow(self._swc_preprocess)

        layout.addWidget(param_box)

        conv_box = QGroupBox("卷积曲面")
        conv_form = QFormLayout(conv_box)

        self._appr_style = QComboBox()
        for value, label in (
            ("local", "线积分卷积 (local)"),
            ("morphtesser", "MorphTesser 卷积"),
            ("limit", "仅细分 (limit)"),
            ("metaball", "Metaball 点元"),
            ("metaball_line", "Metaball 线骨架"),
        ):
            self._appr_style.addItem(label, value)
        _appr_choices = ("local", "morphtesser", "limit", "metaball", "metaball_line")
        idx = self._appr_style.findData(
            CPP_DEFAULT_APPR_STYLE if CPP_DEFAULT_APPR_STYLE in _appr_choices else "local"
        )
        self._appr_style.setCurrentIndex(idx if idx >= 0 else 0)
        self._appr_style.currentIndexChanged.connect(
            lambda _: self._on_appr_style_changed(self._appr_style_value())
        )
        self._appr_style.setToolTip(
            "local=线积分卷积；morphtesser=MorphTesser；limit=仅细分；"
            "metaball=点元；metaball_line=线骨架 Metaball"
        )
        conv_form.addRow("逼近方式", self._appr_style)

        self._kernel = QComboBox()
        self._kernel.addItem("有限核 quartic", "quartic")
        self._kernel.addItem("无限核 cauchy", "cauchy")
        kidx = self._kernel.findData(CPP_DEFAULT_KERNEL)
        self._kernel.setCurrentIndex(kidx if kidx >= 0 else 0)
        self._kernel.currentIndexChanged.connect(
            lambda _: self._on_kernel_changed(self._kernel_value())
        )
        self._kernel.setToolTip("有限核=紧支撑 quartic；无限核=Cauchy")
        conv_form.addRow("核函数", self._kernel)

        self._iso = QDoubleSpinBox()
        self._iso.setRange(1e-6, 1e6)
        self._iso.setDecimals(6)
        _default_iso = (
            DEFAULT_MORPH_ISO
            if CPP_DEFAULT_APPR_STYLE == "morphtesser"
            else (
                DEFAULT_QUARTIC_ISO
                if CPP_DEFAULT_KERNEL == "quartic"
                else DEFAULT_CAUCHY_PROJ_ISO
            )
        )
        self._iso.setValue(_default_iso)
        self._iso.setToolTip(
            "等值面 F(p)=iso；MorphTesser 默认 0.5；"
            "线积分：Cauchy 用 m_dCauchyIso×5，Quartic 用 m_dQuarticIso"
        )
        conv_form.addRow("等值 iso", self._iso)

        self._metaball_iso = QDoubleSpinBox()
        self._metaball_iso.setRange(0.01, 10.0)
        self._metaball_iso.setDecimals(4)
        self._metaball_iso.setValue(DEFAULT_METABALL_ISO)
        self._metaball_iso.setToolTip("Metaball / metaball_line 模式的等值阈值")
        conv_form.addRow("Metaball iso", self._metaball_iso)

        self._project = QCheckBox("投影到等值面")
        self._project.setChecked(
            not CPP_DEFAULT_SWEEP_ONLY
            and CPP_DEFAULT_APPR_STYLE in ("local", "morphtesser")
        )
        self._project.setToolTip("关闭则只细分；limit 模式会自动关闭投影")
        conv_form.addRow(self._project)
        self._project_selective = QCheckBox("只逼近偏差大的顶点")
        self._project_selective.setChecked(True)
        self._project_selective.setToolTip(
            "场/位移已经接近等值面的顶点保持细分后的光滑位置；"
            "偏差大的才投影，并限制单步最大位移，减轻投影后网格质量下降"
        )
        conv_form.addRow(self._project_selective)

        self._project_move_tol = QDoubleSpinBox()
        self._project_move_tol.setRange(0.05, 2.0)
        self._project_move_tol.setSingleStep(0.05)
        self._project_move_tol.setValue(0.45)
        self._project_move_tol.setToolTip("||投影位移|| / 局部边长 低于此值则不逼近（软过渡到 2×）")
        conv_form.addRow("逼近位移阈值", self._project_move_tol)
        self._post_smooth_iters = QSpinBox()
        self._post_smooth_iters.setRange(0, 20)
        self._post_smooth_iters.setValue(POST_PROJECT_SMOOTH_ITERS)
        self._post_smooth_iters.setToolTip("投影后 Taubin 平滑次数（0=关闭）；用于校准网格质量、减轻锯齿")
        conv_form.addRow("投影后平滑", self._post_smooth_iters)



        self._proj_iters = QSpinBox()
        self._proj_iters.setRange(1, 100)
        self._proj_iters.setValue(CPP_DEFAULT_PROJECT_ITERS)
        conv_form.addRow("投影迭代", self._proj_iters)

        self._proj_step = QDoubleSpinBox()
        self._proj_step.setRange(0.01, 2.0)
        self._proj_step.setValue(0.35)
        conv_form.addRow("投影步长", self._proj_step)

        btn_auto_iso = QPushButton("自动估计 iso")
        btn_auto_iso.clicked.connect(self._auto_iso)
        conv_form.addRow(btn_auto_iso)

        layout.addWidget(conv_box)
        self._conv_box = conv_box

        vis_box = QGroupBox("显示")
        vis_layout = QVBoxLayout(vis_box)
        self._show_skel = QCheckBox("显示骨架线框")
        self._show_skel.setChecked(False)
        self._show_skel.setToolTip("细线骨架；默认关闭，避免生成时逐段 tube/sphere 渲染卡顿")
        self._show_skel.toggled.connect(lambda _: self._refresh_view())
        self._show_wire = QCheckBox("四边形线框")
        self._show_wire.setChecked(True)
        self._show_wire.toggled.connect(lambda _: self._refresh_view())
        self._show_surf = QCheckBox("显示曲面")
        self._show_surf.setChecked(True)
        self._show_surf.toggled.connect(lambda _: self._refresh_view())
        self._see_backfaces = QCheckBox("透视背面")
        self._see_backfaces.setChecked(False)
        self._see_backfaces.setToolTip(
            "开启：分支/管道网格可透视背面（半透明）；"
            "Root 球体始终不透明且背面剔除；"
            "关闭：全部仅显示朝向相机的一面（默认）"
        )
        self._see_backfaces.toggled.connect(lambda _: self._refresh_view())
        self._show_iso_surf = QCheckBox("显示目标卷积等值面")
        self._show_iso_surf.setChecked(False)
        self._show_iso_surf.setToolTip("真实投影卷积场等值面 F(p)=iso（与投影同一场）；骨架径向环+二分求根，非半径管粗预览")
        self._show_iso_surf.toggled.connect(lambda _: self._refresh_view())
        vis_layout.addWidget(self._show_skel)
        vis_layout.addWidget(self._show_wire)
        vis_layout.addWidget(self._show_surf)
        vis_layout.addWidget(self._see_backfaces)
        vis_layout.addWidget(self._show_iso_surf)
        layout.addWidget(vis_box)

        btn_gen = QPushButton("生成扫掠 Mesh")
        self._btn_gen = btn_gen
        btn_gen.setStyleSheet("font-weight: bold; padding: 8px;")
        btn_gen.clicked.connect(self._generate)
        layout.addWidget(btn_gen)
        layout.addStretch()

        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.setStatusBar(QStatusBar())

        self._refresh_sample_list()
        self.statusBar().showMessage("请打开任意 SWC 文件，或从快速示例中选择", 5000)
        self._on_sweep_only_changed(self._sweep_only.isChecked())

    def _try_load_default_sample(self) -> None:
        """保留供手动调用；启动时不自动加载 SWC。"""
        return

    def _on_branch_junction_changed(self, _: int) -> None:
        if self._swc_path and self._swc_path.is_file():
            self._generate()

    def _appr_style_value(self) -> str:
        v = self._appr_style.currentData()
        return str(v) if v else "local"

    def _kernel_value(self) -> str:
        v = self._kernel.currentData()
        return str(v) if v else "quartic"

    def _on_sweep_only_changed(self, checked: bool) -> None:
        self._conv_box.setEnabled(not checked)
        self._subdiv.setEnabled(not checked)
        if checked:
            self._subdiv.setValue(0)
            self._project.setChecked(False)
        else:
            if self._subdiv.value() == 0:
                self._subdiv.setValue(CPP_DEFAULT_SUBDIV_LEVELS)
            self._on_appr_style_changed(self._appr_style_value())
        self._btn_gen.setText("生成初始 Mesh" if checked else "生成四边形 Mesh")

    def _settings(self) -> QSettings:
        return QSettings()

    def _swc_dialog_dir(self) -> str:
        if self._swc_path and self._swc_path.parent.is_dir():
            return str(self._swc_path.parent)
        last = self._settings().value("swc/last_dir", "")
        if last and Path(str(last)).is_dir():
            return str(last)
        for d in candidate_data_dirs():
            if d.is_dir():
                return str(d)
        return str(Path.home())

    def _remember_swc_dir(self, path: Path) -> None:
        if path.parent.is_dir():
            self._settings().setValue("swc/last_dir", str(path.parent.resolve()))

    def _ensure_sample_combo_entry(self, path: Path) -> None:
        key = str(path.resolve())
        idx = self._sample_combo.findData(key)
        if idx >= 0:
            return
        if not self._sample_combo.isEnabled():
            self._sample_combo.setEnabled(True)
        self._sample_combo.insertItem(1, f"↳ {path.name}", key)

    def _refresh_sample_list(self) -> None:
        self._sample_combo.blockSignals(True)
        self._sample_combo.clear()
        samples = discover_sample_swc()
        if not samples:
            self._sample_combo.addItem("（无示例文件）")
            self._sample_combo.setEnabled(False)
        else:
            self._sample_combo.setEnabled(True)
            self._sample_combo.addItem("（请选择）", None)
            for p in samples:
                self._sample_combo.addItem(p.name, str(p.resolve()))
            self._sample_combo.setCurrentIndex(0)
        self._sample_combo.blockSignals(False)

    def _on_sample_selected(self, index: int) -> None:
        if index < 0 or not self._sample_combo.isEnabled():
            return
        path_str = self._sample_combo.itemData(index)
        if not path_str:
            return
        self._load_swc_path(Path(path_str), auto_generate=True)

    def _on_appr_style_changed(self, style: str) -> None:
        if self._sweep_only.isChecked():
            return
        if style == "limit":
            self._project.setChecked(False)
        elif style in ("local", "morphtesser", "metaball", "metaball_line"):
            self._project.setChecked(True)
        is_metaball = style in ("metaball", "metaball_line")
        self._metaball_iso.setEnabled(is_metaball)
        if style == "morphtesser":
            self._iso.setValue(DEFAULT_MORPH_ISO)
            self._kernel.setEnabled(False)
        elif is_metaball:
            self._kernel.setEnabled(style == "metaball_line")
            self._iso.setValue(self._metaball_iso.value())
        else:
            self._kernel.setEnabled(True)
            self._on_kernel_changed(self._kernel_value())

    def _on_kernel_changed(self, kernel: str) -> None:
        if self._appr_style_value() == "morphtesser":
            return
        self._iso.setValue(
            DEFAULT_CAUCHY_PROJ_ISO if kernel == "cauchy" else DEFAULT_QUARTIC_ISO
        )

    def _params(self) -> PipelineParams:
        return PipelineParams(
            sides=self._sides.value(),
            subdiv_levels=self._subdiv.value(),
            radius_scale=self._radius_scale.value(),
            bound_scale=CPP_DEFAULT_BOUND_SCALE,
            insert_assist=CPP_DEFAULT_INSERT_ASSIST,
            bound_tet_scaled=CPP_DEFAULT_INSERT_ASSIST,
            sweep_only=self._sweep_only.isChecked(),
            swc_preprocess=self._swc_preprocess.isChecked(),
            branch_junction=self._branch_junction.currentData() or DEFAULT_BRANCH_JUNCTION.value,
            connect_branch_junction=self._connect_branch.isChecked(),
            restore_offset=False,
            ref_obj_path=None,
            offset_mode="segment",
            weight_mode="local",
            smooth_weights=False,
            use_lmt_cnt=False,
            sub_lmt_cnt=CPP_DEFAULT_SUB_LMT_CNT,
            appr_style=self._appr_style_value(),
            kernel=self._kernel_value(),
            iso_value=self._iso.value(),
            metaball_iso=self._metaball_iso.value(),
            metaball_interval=DEFAULT_METABALL_INTERVAL,
            project=self._project.isChecked(),
            sample_iso_surface=self._show_iso_surf.isChecked() and self._project.isChecked(),
            project_backend="numba",
            project_iters=self._proj_iters.value(),
            project_step=self._proj_step.value(),
            project_selective=self._project_selective.isChecked(),
            project_move_tol=self._project_move_tol.value(),
            project_rel_tol=0.12,
            project_move_cap=0.85,
            post_project_smooth_iters=self._post_smooth_iters.value(),
        )

    def _load_swc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 SWC 文件",
            self._swc_dialog_dir(),
            "SWC 文件 (*.swc *.SWC);;所有文件 (*.*)",
        )
        if path:
            self._load_swc_path(Path(path), auto_generate=True)

    def _load_swc_path(self, path: Path, *, auto_generate: bool = True) -> None:
        path = path.resolve()
        if not path.is_file():
            QMessageBox.warning(self, "提示", f"文件不存在:\n{path}")
            return
        self._remember_swc_dir(path)
        self._swc_path = path
        self._path_label.setText(str(path))
        self._ensure_sample_combo_entry(path)
        idx = self._sample_combo.findData(str(path))
        if idx >= 0:
            self._sample_combo.blockSignals(True)
            self._sample_combo.setCurrentIndex(idx)
            self._sample_combo.blockSignals(False)
        self.statusBar().showMessage(f"已加载 {path.name}", 3000)
        if auto_generate:
            self._generate()

    def _generate(self) -> None:
        if not self._swc_path or not self._swc_path.is_file():
            QMessageBox.warning(self, "提示", "请先加载 SWC 文件")
            return
        try:
            pipeline = TreeQuadPipeline(self._params())
            self._result = pipeline.run(self._swc_path)
            m = self._result.mesh
            if not np.isfinite(m.vertices).all():
                n_bad = int(np.sum(~np.isfinite(m.vertices).all(axis=1)))
                raise RuntimeError(
                    f"网格含 {n_bad} 个无效顶点（NaN/Inf），"
                    "请尝试投影后端改为 numba 或关闭「投影到等值面」"
                )
            sk = self._result.skeleton
            n_branch = sum(1 for nid, ch in sk.children.items() if len(ch) > 1)
            stage = "初始 mesh" if self._result.params.sweep_only else "完整管线"
            junction = branch_junction_label(
                parse_branch_junction(self._result.params.branch_junction)
            )
            topo = topology_summary(m)
            topo_tag = "水密" if topo.get("watertight") else f"边界{topo.get('boundary', '?')}"
            if topo.get("non_manifold", 0):
                topo_tag += f" 非流形{topo['non_manifold']}"
            pp = self._result.preprocess_stats
            pp_msg = ""
            if pp is not None:
                pp_msg = (
                    f" | 预处理: -{pp.deleted} +{pp.inserted} 移{pp.moved_nodes} "
                    f"重叠 {pp.max_overlap_before:.3g}→{pp.max_overlap_after:.3g}"
                )
                if pp.radius_floor > 0.0:
                    pp_msg += f" r≥{pp.radius_floor:.3g}"
                if pp.radius_clamped:
                    pp_msg += f" 抬升{pp.radius_clamped}"
                if pp.warnings:
                    pp_msg += f" | SWC: {pp.warnings[0]}"
            self.statusBar().showMessage(
                f"[{stage}] {junction} | {m.n_vertices} 顶点, {m.n_quads} 四边形, "
                f"{m.n_triangles} 三角形 | {topo_tag} | "
                f"骨架 {len(sk.nodes)} 节点 / {n_branch} 分叉{pp_msg}",
                10000,
            )
            self._refresh_view()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, KeyError) and exc.args:
                detail = f"数据引用错误 — 节点 id {exc.args[0]} 不存在（可能 SWC 拓扑或预处理异常）"
            QMessageBox.critical(self, "生成失败", detail)

    def _auto_iso(self) -> None:
        if self._result is None:
            if not self._swc_path:
                return
            self._generate()
        if self._result is None:
            return
        iso = TreeQuadPipeline.estimate_iso(self._result)
        self._iso.setValue(iso)
        self.statusBar().showMessage(f"建议 iso = {iso:.6f}", 3000)

    def _export_obj(self) -> None:
        if self._result is None:
            QMessageBox.warning(self, "提示", "请先生成 mesh")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 OBJ", "tree_quad.obj", "OBJ (*.obj)"
        )
        if path:
            export_obj(self._result.mesh, path)
            self.statusBar().showMessage(f"已导出 {path}", 5000)

    def _refresh_view(self) -> None:
        self._plotter.clear()
        self._plotter.add_axes()

        if self._result is None:
            self._plotter.render()
            return

        if self._show_skel.isChecked():
            add_joint_skeleton_wire_to_plotter(self._plotter, self._result.root)

        if self._show_iso_surf.isChecked() and self._result is not None:
            iso_mesh = getattr(self._result, "iso_surface_mesh", None)
            if iso_mesh is not None and (
                getattr(iso_mesh, "n_quads", 0) > 0 or getattr(iso_mesh, "n_triangles", 0) > 0
            ):
                verts, faces = mesh_to_pyvista_faces(iso_mesh)
                if len(faces) > 0:
                    surf = pv.PolyData(verts, faces)
                    self._plotter.add_mesh(
                        surf,
                        color="#06d6a0",
                        opacity=0.35,
                        smooth_shading=True,
                        specular=0.15,
                        name="iso_surface",
                    )
            elif self._result.iso_surface_points is not None:
                pts = self._result.iso_surface_points
                if len(pts) > 0:
                    cloud = pv.PolyData(np.asarray(pts, dtype=np.float64))
                    self._plotter.add_mesh(
                        cloud,
                        style="points",
                        point_size=3,
                        render_points_as_spheres=False,
                        color="#06d6a0",
                        opacity=0.65,
                        name="iso_surface",
                    )

        body_mesh, root_mesh = split_root_soma_mesh(self._result.mesh, self._result.root)
        show_surf = self._show_surf.isChecked()
        show_wire = self._show_wire.isChecked()
        see_back = self._see_backfaces.isChecked()
        body_opacity = 0.82 if see_back and show_surf else 1.0

        if show_wire and not see_back:
            self._plotter.enable_hidden_line_removal()
        else:
            self._plotter.disable_hidden_line_removal()

        layers: list[tuple[str, object, bool, float, str]] = [
            ("body", body_mesh, see_back, body_opacity, "#4cc9f0"),
            ("root_soma", root_mesh, False, 1.0, "#4cc9f0"),
        ]
        for layer_name, sub_mesh, layer_see_back, opacity, color in layers:
            if sub_mesh.n_quads == 0 and sub_mesh.n_triangles == 0:
                continue
            verts, faces = mesh_to_pyvista_faces(sub_mesh)
            if len(faces) == 0:
                continue
            surf = pv.PolyData(verts, faces)
            cull_back = not layer_see_back

            def _apply_culling(actor) -> None:
                prop = actor.GetProperty()
                if cull_back:
                    prop.BackfaceCullingOn()
                    prop.FrontfaceCullingOff()
                else:
                    prop.BackfaceCullingOff()
                    prop.FrontfaceCullingOff()

            if show_surf and show_wire:
                actor = self._plotter.add_mesh(
                    surf,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    backface_culling=cull_back,
                    show_edges=True,
                    edge_color="black",
                    line_width=1,
                    name=f"{layer_name}_surface",
                )
                actor.GetProperty().SetAmbient(0.15)
                actor.GetProperty().SetDiffuse(0.85)
                _apply_culling(actor)
            elif show_surf:
                actor = self._plotter.add_mesh(
                    surf,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    backface_culling=cull_back,
                    name=f"{layer_name}_surface",
                )
                actor.GetProperty().SetAmbient(0.15)
                actor.GetProperty().SetDiffuse(0.85)
                _apply_culling(actor)
            elif show_wire:
                wire = self._plotter.add_mesh(
                    surf,
                    style="wireframe",
                    color="black",
                    line_width=1,
                    opacity=1.0 if cull_back else 0.95,
                    backface_culling=cull_back,
                    name=f"{layer_name}_wireframe",
                )
                wire.GetProperty().SetRepresentationToWireframe()
                _apply_culling(wire)

        self._plotter.reset_camera()
        self._plotter.render()

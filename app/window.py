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
    CPP_DEFAULT_SUBDIV_LEVELS,
    CPP_DEFAULT_SUB_LMT_CNT,
    CPP_DEFAULT_SWEEP_ONLY,
    CPP_DEFAULT_USE_LMT_CNT,
    DEFAULT_CAUCHY_PROJ_ISO,
    DEFAULT_QUARTIC_ISO,
    MAX_APPR_ITER,
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

        self._bound_scale = QDoubleSpinBox()
        self._bound_scale.setRange(0.5, 8.0)
        self._bound_scale.setValue(CPP_DEFAULT_BOUND_SCALE)
        self._bound_scale.setSingleStep(0.1)
        self._bound_scale.setToolTip("CreateBoundSweep 缩放 (C++ m_fBoundScale)")
        form.addRow("扫掠缩放", self._bound_scale)

        self._insert_assist = QCheckBox("分支辅助点 (InsertAssist)")
        self._insert_assist.setChecked(CPP_DEFAULT_INSERT_ASSIST)
        self._insert_assist.setToolTip("分支处插入辅助点并启用 EBTT_Scaled 扫掠 (bound×radius)")
        form.addRow(self._insert_assist)

        self._sweep_only = QCheckBox("仅扫掠管道 (BLT 初始四边形 mesh)")
        self._sweep_only.setChecked(CPP_DEFAULT_SWEEP_ONLY)
        self._sweep_only.setToolTip("沿 SWC 每段骨架扫掠四边形管道；跳过后续细分/卷积/投影")
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

        self._restore_offset = QCheckBox("恢复半径偏移 (OffsetSurf)")
        self._restore_offset.setToolTip("投影后按场加权沿法向恢复 SWC 半径")
        form.addRow(self._restore_offset)

        ref_row = QHBoxLayout()
        self._ref_obj_label = QLabel("（无）")
        self._ref_obj_label.setWordWrap(True)
        btn_ref_obj = QPushButton("参考 OBJ…")
        btn_ref_obj.setToolTip("CreateOffset 用外部三角 mesh；留空则用细分 mesh")
        btn_ref_obj.clicked.connect(self._pick_ref_obj)
        ref_row.addWidget(self._ref_obj_label, 1)
        ref_row.addWidget(btn_ref_obj)
        form.addRow("Offset 参考", ref_row)
        self._ref_obj_path: Path | None = None

        self._offset_mode = QComboBox()
        self._offset_mode.addItems(["segment", "branch", "combined"])
        self._offset_mode.setToolTip("FieldOffset_Segment / Branch / 组合")
        form.addRow("Offset 模式", self._offset_mode)

        layout.addWidget(param_box)

        conv_box = QGroupBox("卷积曲面")
        conv_form = QFormLayout(conv_box)

        self._appr_style = QComboBox()
        self._appr_style.addItems(
            ["morphtesser", "local", "limit", "metaball", "metaball_line"]
        )
        _appr_choices = ("morphtesser", "local", "limit", "metaball", "metaball_line")
        self._appr_style.setCurrentText(
            CPP_DEFAULT_APPR_STYLE
            if CPP_DEFAULT_APPR_STYLE in _appr_choices
            else "morphtesser"
        )
        self._appr_style.currentTextChanged.connect(self._on_appr_style_changed)
        self._appr_style.setToolTip(
            "morphtesser=MorphTesser 卷积场 iso=0.5; "
            "local=线积分投影; limit=仅细分不投影(EAS_Limit); "
            "metaball=点元; metaball_line=线骨架+Cauchy+Metaball iso"
        )
        conv_form.addRow("逼近方式", self._appr_style)

        self._weight_mode = QComboBox()
        self._weight_mode.addItems(["local", "global"])
        self._weight_mode.setToolTip("local=CreateConvLineSkel_Local; global=NNLS_Global")
        conv_form.addRow("权重模式", self._weight_mode)

        self._smooth_weights = QCheckBox("平滑全局权重 (SmoothWts)")
        conv_form.addRow(self._smooth_weights)

        self._use_lmt_cnt = QCheckBox("Limit 细分缩放 (use_lmt_cnt)")
        self._use_lmt_cnt.setChecked(CPP_DEFAULT_USE_LMT_CNT)
        self._use_lmt_cnt.setToolTip("EAS_Limit 默认启用；Limit_Global/Local 亦需勾选")
        conv_form.addRow(self._use_lmt_cnt)

        self._sub_lmt_cnt = QSpinBox()
        self._sub_lmt_cnt.setRange(0, 3)
        self._sub_lmt_cnt.setValue(CPP_DEFAULT_SUB_LMT_CNT)
        self._sub_lmt_cnt.setToolTip("0=与 Catmull-Clark 级数相同")
        conv_form.addRow("Limit 级数", self._sub_lmt_cnt)

        self._kernel = QComboBox()
        self._kernel.addItems(["cauchy", "quartic"])
        self._kernel.setCurrentText("cauchy")
        self._kernel.currentTextChanged.connect(self._on_kernel_changed)
        conv_form.addRow("核函数", self._kernel)

        self._iso = QDoubleSpinBox()
        self._iso.setRange(1e-6, 1e6)
        self._iso.setDecimals(6)
        self._iso.setValue(DEFAULT_CAUCHY_PROJ_ISO)
        self._iso.setToolTip(
            "等值面 F(p)=iso；MorphTesser 默认 0.5；"
            "BLT Cauchy 默认 m_dCauchyIso×5，Quartic 默认 m_dQuarticIso"
        )
        conv_form.addRow("等值 iso", self._iso)

        self._metaball_iso = QDoubleSpinBox()
        self._metaball_iso.setRange(0.01, 10.0)
        self._metaball_iso.setDecimals(4)
        self._metaball_iso.setValue(DEFAULT_METABALL_ISO)
        self._metaball_iso.setToolTip("Metaball 模式 iso (默认 0.7)")
        conv_form.addRow("Metaball iso", self._metaball_iso)

        self._project = QCheckBox("投影到等值面")
        self._project.setChecked(
            not CPP_DEFAULT_SWEEP_ONLY
            and CPP_DEFAULT_APPR_STYLE in ("local", "morphtesser")
        )
        self._project.setToolTip("EAS_Limit 默认关闭；local/limit&* 模式需开启")
        conv_form.addRow(self._project)

        self._project_backend = QComboBox()
        self._project_backend.addItems(["numba", "auto", "cupy", "serial"])
        self._project_backend.setCurrentText("numba")
        self._project_backend.setToolTip(
            "推荐 numba；auto 优先 Numba；cupy 需 CUDA 且大网格才可能有收益"
        )
        conv_form.addRow("投影后端", self._project_backend)

        self._proj_iters = QSpinBox()
        self._proj_iters.setRange(1, 100)
        self._proj_iters.setValue(MAX_APPR_ITER)
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
        self._show_wire.setChecked(False)
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
        self._show_iso_surf = QCheckBox("显示目标卷积面点云")
        self._show_iso_surf.setChecked(False)
        self._show_iso_surf.setToolTip("MorphTesser iso 等值面采样点（平面点渲染，非球体点云）")
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

    def _on_sweep_only_changed(self, checked: bool) -> None:
        self._conv_box.setEnabled(not checked)
        self._subdiv.setEnabled(not checked)
        if checked:
            self._subdiv.setValue(0)
            self._project.setChecked(False)
            self._use_lmt_cnt.setChecked(False)
        else:
            if self._subdiv.value() == 0:
                self._subdiv.setValue(CPP_DEFAULT_SUBDIV_LEVELS)
            self._on_appr_style_changed(self._appr_style.currentText())
        self._btn_gen.setText("生成扫掠 Mesh" if checked else "生成四边形 Mesh")

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
            self._use_lmt_cnt.setChecked(True)
        elif style in ("local", "metaball", "metaball_line", "morphtesser"):
            self._project.setChecked(True)
        if style == "morphtesser":
            self._iso.setValue(DEFAULT_MORPH_ISO)
        elif style == "local" and self._kernel.currentText() == "cauchy":
            self._iso.setValue(DEFAULT_CAUCHY_PROJ_ISO)

    def _on_kernel_changed(self, kernel: str) -> None:
        self._iso.setValue(
            DEFAULT_CAUCHY_PROJ_ISO if kernel == "cauchy" else DEFAULT_QUARTIC_ISO
        )

    def _pick_ref_obj(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择参考 OBJ", str(Path.cwd()), "OBJ (*.obj);;All (*.*)"
        )
        if path:
            self._ref_obj_path = Path(path)
            self._ref_obj_label.setText(self._ref_obj_path.name)
        else:
            self._ref_obj_path = None
            self._ref_obj_label.setText("（无）")

    def _params(self) -> PipelineParams:
        ref = str(self._ref_obj_path) if self._ref_obj_path else None
        return PipelineParams(
            sides=self._sides.value(),
            subdiv_levels=self._subdiv.value(),
            radius_scale=self._radius_scale.value(),
            bound_scale=self._bound_scale.value(),
            insert_assist=self._insert_assist.isChecked(),
            bound_tet_scaled=self._insert_assist.isChecked(),
            sweep_only=self._sweep_only.isChecked(),
            swc_preprocess=self._swc_preprocess.isChecked(),
            branch_junction=self._branch_junction.currentData() or DEFAULT_BRANCH_JUNCTION.value,
            connect_branch_junction=self._connect_branch.isChecked(),
            restore_offset=self._restore_offset.isChecked() and not self._sweep_only.isChecked(),
            ref_obj_path=ref,
            offset_mode=self._offset_mode.currentText(),
            weight_mode=self._weight_mode.currentText(),
            smooth_weights=self._smooth_weights.isChecked(),
            use_lmt_cnt=self._use_lmt_cnt.isChecked(),
            sub_lmt_cnt=self._sub_lmt_cnt.value(),
            appr_style=self._appr_style.currentText(),
            kernel=self._kernel.currentText(),
            iso_value=self._iso.value(),
            metaball_iso=self._metaball_iso.value(),
            metaball_interval=DEFAULT_METABALL_INTERVAL,
            project=self._project.isChecked(),
            project_backend=self._project_backend.currentText(),
            project_iters=self._proj_iters.value(),
            project_step=self._proj_step.value(),
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
            stage = "扫掠管道" if self._result.params.sweep_only else "完整管线"
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

        if (
            self._show_iso_surf.isChecked()
            and self._result.iso_surface_points is not None
        ):
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

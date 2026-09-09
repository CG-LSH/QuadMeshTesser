# QuadMeshTesser

SWC 树形骨架 → **四边形 mesh** 的 Python 独立项目，复现 ConvolutionProject 中 BLT 树形管线核心思路，卷积公式参考 [morphtesser](D:/morphtesser)。

## 功能

- **输入**：SWC 神经元/树形骨架
- **输出**：四边形主导 mesh（OBJ，含 `f v1 v2 v3 v4` 四边面）
- **流程**：扫掠基网格 → Catmull-Clark 细分 → 卷积分场等值面投影
- **界面**：PySide6 + PyVista 3D 可视化，参数面板可调

## 安装

```bash
cd QuadMeshTesser
python -m pip install -r requirements.txt
python -m app.main
# 或双击 setup_and_run.bat
```

依赖：`numpy`, `pandas`, `scipy`, `PyQt5`, `qtpy`, `pyvista`, `pyvistaqt`

> Windows 上若 `PySide6`/`PyQt6` 报 DLL 错误，本项目默认使用 **PyQt5**（通过 `qtpy`）。

启动后会自动加载 `data/test_linear_0.swc`（若存在）。

## 参数说明

| 参数 | 说明 |
|------|------|
| 截面边数 | 4 = 纯四边形扫掠（与 StrokeBar BLT 一致） |
| Catmull-Clark 细分 | 细分次数，越大曲面越光滑 |
| 半径缩放 | SWC 半径全局缩放 |
| 核函数 | `quartic`（morphtesser 线骨架四次核）/ `cauchy` |
| 支撑 ts | 紧支撑半径系数 (1.01–2.0) |
| 等值 iso | 卷积场等值面阈值 |
| 投影到等值面 | 将细分网格顶点投影到卷积曲面 |

## 与 morphtesser / ConvolutionProject 的关系

- **SWC 读取**：编码回退、parent=0→-1，与 morphtesser 一致
- **卷积场**：`quartic` 核 + 变半径线段解析积分，来自 morphtesser `line_skeleton.py`
- **四边形 mesh**：扫掠 + Catmull-Clark + 投影，对应 ConvolutionProject `Blt_to_polyhedron` + `CSubMesh`
- **分支拼接**：多分叉节点采用 C++ `CreateBranchConvex` 同思路的凸包三角形（`scipy.spatial.ConvexHull`）；根节点多分叉用扇形三角化

## 目录结构

```
QuadMeshTesser/
  app/              # GUI
  quadmeshtesser/   # 核心库
  data/             # 示例 SWC
```

## 命令行（可选扩展）

```python
from quadmeshtesser import TreeQuadPipeline, PipelineParams

result = TreeQuadPipeline(PipelineParams(subdiv_levels=2)).run("data/test_linear_0.swc")
TreeQuadPipeline().export(result, "out.obj")
```

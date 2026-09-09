# 凸包法（JunctionMethod.CONVEX_HULL）

> **标记日期**：2025-06  
> **标识符**：`convex_hull` / `JunctionMethod.CONVEX_HULL` / `CONVEX_HULL_METHOD_ID`  
> **代码登记**：`quadmeshtesser/junction_methods.py`

本文档记录当前 Python 扫掠 mesh 中 **分叉衔接** 所采用的 **凸包法** 完整快照，便于后续更换方法后仍可回退到此实现。

---

## 名称与 C++ 对应

| 中文 | 英文 ID | C++ 流程 |
|------|---------|----------|
| **凸包法** | `convex_hull` | `CreateBranchConvex` → `JoinConvexTri2Quad` → `Blt_to_polyhedron::AddConvexHull` |

参考源码：`ConvolutionProject/CSLib_BVH/CS_Joint.cpp`、`CS_Blt.cpp`、`CSLib_Subdivision/CS_BltToPolyhedron.h`。

---

## 适用范围（当前默认）

| 位置 | 子节点数 | 方法 |
|------|----------|------|
| **Root（soma）** | 1–2 | **非凸包法**：四边形球 mesh + portal 桥接（`branch_sphere.append_root_sphere_junction`） |
| **Root（soma）** | ≥3 | **凸包法**：每子分支球面插值四边形环 + 方向空间凸包 |
| **内部分叉** | 任意（`branch_id ≥ 0`） | **凸包法** |

---

## 凸包法流程概览

```
CreateBoundSweep          → 每节点截面环 bound_sweep[4]
CreateBoundSweepId        → bound_sweep_id / branch_id
CreateLocalAxis           → 截面局部坐标系（含 soma 一级分支修正）

内部分叉 CreateBranchConvex:
  方向空间采样（父环 + 各子环 [+ InsertAssist ±avg_normal]）
  → scipy ConvexHull(QJ)
  → vert_index_to_whole 最近邻匹配
  → JoinConvexTri2Quad（单遍、按法向点积排序、跳过最低分边）

Root ≥3 子 CreateRootConvex:
  每子：球面插值四边形环（与子管道同向，半径 root↔child 插值）
  + 各子 bound_sweep 环
  → 同上 ConvexHull + JoinConvexTri2Quad

Blt_to_polyhedron:
  AddQuad（侧向四边形；root 衔接处跳过端盖/首段侧 quad）
  → AddConvexHull（跨环面；边冲突过滤 _can_add_face）
  → fallback 扇形 + hole_close（Python 水密后处理）
```

---

## 关键文件

| 文件 | 职责 |
|------|------|
| `quadmeshtesser/joint.py` | `CreateBoundSweep` / `CreateBoundSweepId` / `CreateLocalAxis` |
| `quadmeshtesser/branch_convex.py` | `CreateBranchConvex`、`JoinConvexTri2Quad`、root 球面环采样 |
| `quadmeshtesser/polyhedron_builder.py` | `build_polyhedron_surface`（Blt_to_polyhedron 主入口） |
| `quadmeshtesser/segment_sweep.py` | `sweep_only` → polyhedron + `hole_close` |
| `quadmeshtesser/hole_close.py` | 边界星形补片 + 孔洞闭合（凸包法配套后处理） |

**非凸包法（勿混淆）**：

| 文件 | 说明 |
|------|------|
| `quadmeshtesser/branch_sphere.py` | Root 1–2 子：四边形球 + 桥接 |
| `quadmeshtesser/junction_hull.py` | 世界空间 3D 凸包（备用，当前 sweep 主路径未用） |

---

## 方向空间凸包采样（内部分叉）

- 父环：`parent.bound_sweep[j]`， spoke = `-normalize(joint.offset)`
- 子环：`child.bound_sweep[j]`， spoke = `normalize(child.offset)`
- InsertAssist（`insert_assist=True` 且 `bound_tet_scaled`）：assist  spoke 叉积求 `avg_normal`，追加 `±avg_normal`
- 环上点映射：`_hull_point(spoke, ring_center, ring_vertex)`，`QUAD_SIZE=0.01`

全局顶点映射见 `polyhedron_builder._map_hull_vert_to_global`（与 C++ `AddConvexHull` 一致）。

---

## Root ≥3 子：球面插值环（凸包法专用）

- 环心：`root.pos + normalize(child.offset) * root.radius`
- 环向：与 `child.axis[1/2]` 一致（与子管道截面同向）
- 环半径：`t = min(1, root.radius/|offset|)`，`r = ((1-t)*root.radius + t*child.radius) * f_scale`
- 全局顶点：追加在 `[sweep][branch_assist]` 之后的 `root_sphere_base` 槽位

---

## JoinConvexTri2Quad（与 C++ 对齐要点）

- 所有共边三角面算法向点积升序排序
- 从 **最高分边** 向 index=1 合并（**跳过** index=0 最低分边）
- **单遍**合并（非迭代至稳定）

---

## 测试基准

```powershell
cd QuadMeshTesser
python scripts/test_sweep_base.py
```

- `cell021`：`insert_assist=True`，多分叉 root 凸包 + 内部分叉凸包，要求 `watertight=True`
- `test_linear_0`：root 2 子 → 球 mesh（非凸包法）

---

## 回退凸包法时检查清单

1. `junction_methods.py` 中 `JunctionMethod.CONVEX_HULL` 仍为内部分叉默认
2. `polyhedron_builder.build_polyhedron_surface` 中 `use_root_hull = len(children) >= 3`
3. `branch_convex.collect_branch_convex` 仍只收集 `parent && children>1`
4. 未删除 `hole_close` / `_can_add_face` / fallback 若仍需 Python 水密
5. 跑通 `test_sweep_base.py`

---

## 变更记录

| 日期 | 说明 |
|------|------|
| 2025-06 | 初版标记：对齐 C++ polyhedron 路径；root 1–2 子改球 mesh；soma 轴向修正 |

# 数据与溯源契约

## 坐标约定

- 显示和路径编辑使用 GeoTIFF 降采样预览像素坐标。
- Qt 渲染纹理可以比逻辑预览更小，但图元变换必须使其场景边界仍等于逻辑预览尺寸；鼠标点、田块边界、掩膜 offset 和路径点不得存为纹理像素坐标。
- `GeoUtils` 使用与预览尺寸对应的仿射变换，将像素中心转换为源 CRS，再转换为 WGS84。
- GeoJSON、CSV、KML 和 JSON 使用带字段名或标准规定的 `lon, lat` 顺序。
- `$PATH` 路径按当前兼容格式写入 `lat, lon`；使用时按目标终端的字段定义逐项核对。
- 禁止无地理配准时写入伪坐标。

## 运行时与发布包（DECISION-018）

- 冻结版发布目录的入口固定为根目录 `align.exe`；PyInstaller 支持文件可以位于 `_internal`，不能要求用户调用 Python、Conda 或 pip；源码入口只做运行环境初始化，不提供依赖检测或安装界面。
- `runtime_paths.bundled_resource()` 只解析随程序发布的只读资源；`config.example.json`、`data/models/model_registry.json` 和内置模型不得在运行时直接改写。
- 用户配置、缓存、日志、用户模型和更新临时文件分别位于 `%LOCALAPPDATA%/ALIGN/config`、`cache`、`logs`、`models`、`updates`；首次运行从旧版 `config.json` 或内置示例迁移配置，更新不得覆盖这些目录。
- 冻结包的 `build_info.json` 至少包含版本、提交号、频道、构建时间和科学代码指纹。正式导出 manifest 必须保留这些构建信息；缺失科学代码指纹时不得生成可追溯正式清单。
- 源码 ZIP 与 Release 二进制 ZIP 是不同交付物。`update.bat` 只同步源代码；Release 工作流负责生成 `align-Windows-x64-<version>.zip` 和校验文件。

## 阶段证据

每个阶段记录 `algorithm_version`、`inputs`、`input_fingerprint`、`artifact_sha256` 和总 `fingerprint`。

| 阶段 | 必须纳入指纹的输入 | 产物 |
|---|---|---|
| inference | 源影像 SHA-256、模型 SHA-256、田块边界、切片参数、预处理参数 | 原始掩膜 |
| mask | 推理指纹、原始掩膜哈希、全部掩膜参数 | 处理后掩膜、作业区域/作业带几何 |
| path | 掩膜指纹、作业带/作业区域几何、中心线目录及操作员选择、农机参数、规划参数、调头策略、必选起点、可选终点和卸粮点 | 路径几何 |

缓存恢复和正式导出均重新计算并比较这些值。任何缺失或不一致都应失败关闭。

项目缓存中的 `turn_strategy_user_selected` 是调头策略来源标记：只有用户明确展开选择器选择或通过显式策略入口确认时才为 `true`。旧缓存缺少该字段时，不得把其中的非 `contour` 值当作用户意图；恢复为套圈默认，并使与旧策略绑定的路径缓存失效。

## 逐步操作历史（DECISION-025）

- `project_state.json` 是最近完成状态，不是撤销历史；`history/index.json` 分别保存有序 `undo_ids`、`redo_ids`、源影像 identity 和 SHA-256。两者必须具有相同的 `project_generation`、`save_session_id` 和 `save_revision`；同一进程中延迟完成的较旧异步保存不得回写覆盖较新的状态/历史。
- 每个 `history/snapshots/<history_id>.json` 是一个不可变的操作前状态，至少包含 `kind`、`label`、`created_at` 和 `values`。一个逻辑操作对应一个 ID；新操作清空重做分支。
- 数值数组不内嵌 JSON。数组以 dtype、shape 和字节内容计算 SHA-256，写入 `history/artifacts/<sha256>.npz`；快照只保存引用。相同原始掩膜或处理层跨多个操作共享同一个产物。
- 系统历史快照覆盖田块边界、原始掩膜、优化掩膜及语义层、中心线目录/选择、起终点和卸粮点、农机参数、路径、验证/provenance 与导出失效状态，保证一次撤销恢复完整一致的阶段边界。
- 历史只在源 GeoTIFF identity 与 SHA-256 均一致、快照 schema 正确、数组 dtype/shape/哈希均一致时恢复。损坏处之前的更旧历史不再加载；不得跳过缺口拼接一条看似连续的撤销链。
- 跨重启恢复到历史路径时保留几何用于审计和继续编辑，但必须设置 `auto_path_valid=false`、`simulation_done=false`、`export_done=false`，并标记 `path_validation.validation_source=history_snapshot`；不得把历史恢复等同于当前正式验证。
- 撤销栈和重做栈分别最多保留 50 步。索引淘汰后，仅清理当前项目 `history/` 下已无任何快照引用的快照和数组，不删除源数据或正式研究产物。

## 农机参数来源记录

- `planning_factors.harvester` 保存本次路径实际使用的数值；`planning_factors.harvester_param_sources` 逐字段保存来源，至少区分 `default`、`recommended`、`user_entered`、`user_measured` 和 `legacy_unverified`。
- `default`、`recommended`、`user_entered` 和 `legacy_unverified` 只能按原来源记录，不能被静默写成实测值。`field_trial`/`machine_candidate` 缺少完整 `user_measured` 来源时，路径结果必须明确 `machine_executable=false`；研究导出可以写出文件，但不得伪装成机器就绪结果。
- UI 必须列出仍未完成实测来源的具体参数；来源元数据与农机数值一同进入项目缓存、路径 provenance 和导出 manifest。

## 掩膜语义层

- 掩膜处理档位固定为 `standard`。旧配置中的 `light`、`strong` 和 `very_strong` 仅作为兼容输入，加载后统一归一为 `standard`，不得改变当前处理参数。

- `processed_mask`：经过验证、可作为作物主体和作业线依据的区域。
- `headland_mask`：方向和位置证据支持的田头作物候选，用于田头碾压统计和全田真实作物碾压并集，不等同于低质量主体；与主体重叠的像素只计一次。生成前先记录规则化主体的连通区域分区；只有通过区域轴向跨度、端部邻接和跨行支持率分类的候选种子才可进入该层。
- `headland_corridor_mask`：仅在规则化纵向行带的轴向端部、跨越足够多行且具有连续横向证据时重建的完整田头走廊；田头与主体粘连时可记录 `terminal_dense_seed` 作为补充种子来源。接受后的走廊按稳健端点墙在米制最大深度内展开，并与原始作物支持相交，不得填成矩形。端点墙只采纳达到 `headland_terminal_wall_min_axial_span_ratio` 的长跨度横截行，再以分箱 Theil-Sen 趋势替换超过 `headland_terminal_wall_outlier_tolerance_m` 的局部异常端点；候选和种子均不得越过最终端墙进入主体。当统一轴向重构区间把端点量化成直墙时，只允许采用田块多边形中跨行跨度足够长的物理端边恢复墙体斜率，并向田头侧补偿 `band_end_trim_m` 的重构裁边，不能把去噪裁边当成主作业区语义边界。存在规则化纵向主体作为端墙证据时，`headland_corridor_endpoint_margin_m` 的有效主体内切值固定为 0。诊断记录参考跨度、可信/被拒/异常替换行数、田块端边斜率来源、裁边补偿、稳健斜率和有效容差；稀疏纵向行尾、区域分界和内部短片段不得作为田头端点。`diagnostics.region_partition` 保存区域数量、边界框、端部分类、候选支持率和拒绝原因；未经区域分类接受的种子不进入 `headland_mask`。
- `headland_corridor_mask` 只是田头作物证据与转弯空间校验的辅助图层，不是独立的“通行廊道生成”任务，也不替代作业中心线、作业线或正式路径；用户界面和论文主流程仍以作业带、中心线和转弯连接为主。
- `uncertain_residual_mask`：不能可靠归属到主体或田头的原始残差，只用于显示和中性场地支撑。
- `planning_support_mask`：场地支撑并集，只用于越界/支撑判断，不得替代 `processed_mask` 计算作物覆盖或碾压。
- `forbidden_regions`：用户在显示坐标中确认的禁行多边形；规划时按掩膜偏移栅格化，不得由不确定残差自动生成。
- `mask_result.diagnostics.profile_gate_basis` 必须为 `run_edge_midpoints`：相邻剖面行带的局部占用门控按 profile run 边界中点分隔，避免宽等距块吞并旁侧四行带的中心线。
- 当前 `mask_result.diagnostics.algorithm_version` 为 `21`。版本 20 及更早的 `mask_processed`/`mask_result` 不包含粘连带瓶颈切割契约：宽度在 `adhesion_split_min_width_m`（默认 1.35 m）至 `adhesion_split_max_width_m`（默认 2.6 m）之间的疑似合并剖面 run，只有在距离变换宽度包络鞍点/平台比 ≤ `adhesion_split_dist_ratio`（默认 0.90）且切口两侧各自落在 `adhesion_split_min_half_m`（默认 0.45 m）与 `adhesion_split_max_child_m`（默认 1.5 m）之间时，才在鞍点处一分为二；未触发 run 必须逐位保留，正常案例不得改变行带几何；诊断记录于 `mask_result.diagnostics.adhesion_split`。版本 18 及更早的 `mask_processed`/`mask_result` 不包含行向估计近平分决胜契约与重建退化门槛：top1 与 top2 行向候选得分差低于 `angle_tiebreak_margin_ratio`（默认 0.05）时必须取前 4 个角度家族做 ±1.5°（步长 0.5°）横剖扫描，以宽度在 0.25–4.0 m 的“组合条带数”比较，仅当挑战家族领先 ≥ max(2, 当前家族 25%) 时才允许切换行向；候选带数 ≥8 而重建带数不足 40% 候选的退化重建必须显式失败（`种植带重建疑似退化`），不得静默输出塌陷几何。版本 17 及更早的 `mask_processed`/`mask_result` 还不包含分离种植区域多候选行向估计契约：全局 PCA 与主要连通轮廓内部主轴必须共同作为候选，再由横向投影周期性评分选择最终行向，区域质心连线不得主导行向。旧缓存恢复必须使掩膜处理和下游路径失效，不得仅因源影像、模型及数组形状相同而复用。
- 田块圈选同时定义三个不同边界：`mask_raw` 的推理/分析上下文可在圈外保留受控缓冲，供行向与宽窄带重构；前台叠加图层必须只显示原圈选内的副本，绝不能原地裁剪 `mask_raw`；行带轮廓、中心线、`processed_mask`、`planning_support_mask` 与规划路径必须按原圈选边界裁回，圈外分析像素不得成为作业范围。
- 田块边界必须是一个简单环（`field-boundary-geometry-v1`）。在闭合顶点上重复点击会在首点附近追加顶点，使环自相交；此时所有 shapely 判定都返回无效，中心线田界门禁会把**每一条**候选线判为越界，界面最终只显示"无有效中心线"。因此 `row_geometry.field_polygon_local` 统一经 `sanitize_field_polygon` 修复：连续重复顶点直接去除，自相交环用 `make_valid` 取最大连通多边形，**且仅当被丢弃面积不超过 2%（`FIELD_BOUNDARY_REPAIR_MAX_DROPPED_RATIO`）时才接受**。真实交叉（画成"8"字）会丢掉田块的真实部分，必须拒绝并让用户重画，不得猜测哪一瓣才是本意。修复结果写入 `layout.field_boundary_geometry`（`repaired`、`reason`、`dropped_area_ratio`、`duplicate_vertex_count`），并在圈选、缓存恢复、生成作业线三处写入日志。无法修复时 `layout.centerline_hard_reasons` 必须写明是田块边界几何问题，不得报成"作业线在田界外"。

这些数组与处理后主体保存在同一个原子 NPZ 产物中；项目 JSON 只保存轻量元数据。

## 路径段语义

### 正射影像双尺度推理证据（DECISION-004/009）

- `mask_raw` 仍以 640 px 原生源分辨率切片的语义结果为主体。
- `1280→640` 上下文切片只提供邻近弱检测恢复和相邻实例接触边界，不得单独生成整块语义掩膜。
- 弱检测阈值、上下文尺寸、重叠率、恢复像素数和实例分隔像素数必须写入 `raster_preprocessing`/推理 provenance。
- 实例接触边界需获得跨切片重复支持，并通过米制长度和主方向一致性筛选；单个切片外轮廓不能切断作业带。

### 种植区域分类、分区直线生成与全田转弯（DECISION-017）

未显式配置 `uniform_zone_min_width_m` 时，默认值为 `4.0 m`。该默认值用于把真实数据中约 `2.5 m` 的完整四行宽窄行带与明显更宽的等距块分开；显式配置仍按配置值和歧义保护带执行，不能用默认值替代人工/上游证据。

- `layout.planting_region_classification` 保存分类器版本、宽度阈值、模糊保护带、逐带宽度/法向位置/结论/理由、分类计数、是否混合田块以及自动规划是否允许。类型只能是 `wide_narrow_rows`、`uniform_block` 或 `ambiguous`。分类器还必须记录 `partition_basis` 和 `wide_narrow_region_count`；当一个 `uniform_block` 的法向位置位于两组宽窄行带之间时，两侧必须形成不同的 `wide_narrow_rows_N`，不能因类型相同而跨区合并。
- 作业带确定是中心线生成的前置算法步骤，不是新增的用户界面选择入口：宽窄行区域按四行宽行作业带确定，等距区域按割幅划分作业带，窄行只作为识别和分类证据，不直接生成作业线。完成作业带确定后，系统才生成整田候选中心线目录，再由左侧界面选择中心线。
- `layout.centerline_catalog` 保存由已确定作业带生成的整田候选中心线及稳定 ID；目录生成后才允许操作员选择，取消选择的中心线不得进入正式路径。历史 `work_band_catalog`/`selected_work_band_ids` 仅作为兼容字段，不代表新的界面流程。
- 当 `profile_width_m` 落入模糊保护带时，只有同时具备上游 `band_type=wide`、低于均匀区下界的局部重构宽度，以及连续中心线跨度证据，才可记录 `borderline_resolution=upstream_band_type_and_continuous_centerline_support` 并归入 `wide_narrow_rows`；缺少任一证据仍必须是 `ambiguous`，不得按田块左右位置猜测。
- `layout.work_regions` 保存区域标识、类型、来源带、法向位置、作业线数量和来源语义；不得内嵌重复的完整作业线数组。`layout.region_generation_order` 只记录直线生成来源顺序，`layout.region_execution_order` 必须为空，`layout.turn_planning_scope=all_generated_work_lines` 且 `region_boundaries_affect_turns=false`。必选起点在所有已生成直线的全田安全排序候选中优先选择最近可行作业线端点，相邻法向转弯数仅作次级排序；可选终点只参与末段方向参考。被田界阻断而必须收尾的唯一短线从旋转集合中移除后固定追加，不能用固定末线为由缩减其他首线候选；不得把“左区/右区”或区域访问次序写成执行拓扑。
- UI/项目快照中的带记录用 `planning_contour` 保存最多 256 点的轻量规划边界，原始稠密 `contour` 不进入轻量状态。等距区诊断的 `reference_geometry_source` 必须说明使用 `contour`、`planning_contour` 或历史兼容的 `compact_centerline_and_mask`；当 `profile_width_m` 明显大于该轻量边界时，必须以实测剖面宽度扩展横向规划包络，并记录 `effective_geometry_source=<reference>_expanded_from_profile_width`、`profile_width_px`、`reference_geometry_width_px` 和 `effective_transverse_width_m`，不得静默按窄边界截断等距区。若等距区与宽窄行区相邻，扩展包络仍必须受最近宽窄行中心线的共享边界安全带限制；该中心线归属边界作业带，禁止等距区再生成近重复作业线。
- 宽窄行区的 `line_source=four_row_band_centerlines`；等距区的 `line_source=cutter_width_uniform_swaths`。自动端点修复仅在短带自身纵向跨度、至少 3 条邻近完整中心线和已确认田界共同支持时执行，逐线记录仍保存在 `layout.endpoint_repair` 或 `layout.uniform_zones[].endpoint_repair`。系统不再提供自动“缺边处理/趋势补全”模式；影像缺失处由用户在路线编辑中拖动共享控制端点。控制点同时属于相邻作业段和调头段，拖动预览与编辑提交后的 `full_path` 必须使用重新生成的连续弧线；机器兼容文件所需的状态 `0` 直线弦单独保存在控制/协议投影中，不能覆盖连续显示路径。手工编辑会使自动规划科研指标和路径阶段指纹失效；正式科研导出仍需重新规划，但农机兼容测试导出必须使用当前手工控制点重新计算经纬度，不得退回编辑前路径。
- `turn_assessment.contour_ordering` 保存 `execution_scope=all_generated_work_lines`、`execution_topology_version`、全田直线数、来源区域列表和逐线 `line_provenance`。区域只参与生成证据，不能生成独立路线；因此 `turn_assessment.region_routes` 与 `turn_assessment.region_transfers` 必须为空，区域边界不得插入 `turn_aux`、整行 `turn_reverse` 或拼接弦。
- `layout.coverage_gap_recovery` 保存规划期漏线恢复版本、正常四行间距、异常间距倍数、来源带、目标法向位置、`phase_method`、左右稳定队列数量、中心线支持率、横截面轴向支持率/平均占用率、轴向长度比、被明确忽略的窄带标识与 `recovered_band_ids`；`layout.planning_recovered=true` 表示路径实际使用了恢复作业带。两侧相位证据不对称时从较长稳定队列外推，证据相当时才取中点；机器中心线可位于四行作物之间，不能因中心像素为空而漏掉整带。窄带中心不得提升为四行作业线，且处理后掩膜的原始分类不得被覆盖；后台规划掩膜只能由 `wide_bands` 构建，不能在四行带为空时回退为完整 `processed_mask`。`field_trial`/`machine_candidate` 仍默认关闭新的自动恢复；缓存中已有该标记时必须保留并显示该来源。默认研究导出不弹窗、不阻断，严格 `fail_closed` 模式才拒绝。
- `turn_assessment.service_points` 保存 `mode`、必选起点/终点是否参与排序与方向、入口/出口连接是否实际写入路径，以及 `entry_actual_start_local`、`entry_actual_distance_m`、`entry_nearest_feasible_verified`、`entry_side_constrained_by_terminal_short_line`。默认及桌面工作流使用 `service_point_mode=reference_only`：连续路径保留真实作业/转弯几何，起点用于选择全田最近可行作业线端点；共享控制折线和兼容 TXT 另写入起点入口边与最终作业段终点延伸。终点未预先设置时由 `include_route_finish_extension` 生成 `exit` 终点，不生成整行返程；显式拖动后使用该终点坐标。唯一受阻短线可以保留为最后一条作业线，但不得通过要求其从受阻端结束或通过作业线总数奇偶性覆盖起点；按起点优先的交替方向使受阻端成为末线起点时，由倒数第二条的局部正式转弯连接，并按相同田界、曲率、禁行区和足迹门禁失败关闭。所有作业线必须先统一主方向，再根据起点确定首线执行方向。诊断必须记录 `terminal_blocked_endpoint_role`、`terminal_blocked_endpoint_is_route_finish`、`terminal_finish_orientation_forced=false`，且 `entry_side_constrained_by_terminal_short_line=false`。最终端点修复/共享田头墙完成后若 `entry_nearest_feasible_verified` 不是严格布尔值 `true`，路线必须失败关闭。底层研究调用显式设置 `service_point_mode=connect` 时才允许输出服务点连接段。

- `work`：由作物中心线支持、启用割刀几何覆盖计算的作业段。
- `turn_approach`：从作物端点到校正转弯锚点的田头直行段，不计入割台作物覆盖。
- `turn`：满足策略几何和最小转弯半径的前进转弯段。
- `turn_reverse` / `turn_aux`：包含倒车或辅助折返动作的转弯段。
- `track_outside_field_pct`：履带扫掠超出田块硬边界的比例。
- `track_outside_support_pct`：履带扫掠离开语义支撑区的比例，不能替代田界指标。
- `track_uncertain_overlap_pct`：履带扫掠经过未确认残差的比例，仅在提供该图层时有效。
- `track_forbidden_overlap_pct`：履带扫掠进入人工确认禁行区的比例，仅在提供该图层时有效。
- `machine_outside_field_pct`：二维刚体模型中机身平面包络扫掠越出田块硬边界的比例，与履带接地越界率分别统计。
- `machine_forbidden_overlap_pct`：二维刚体模型中机身平面包络扫掠进入人工确认禁行区的比例，与履带禁行区重叠分别统计。
- `footprint_model=planar_rigid_footprint_v2`：路径点表示 `track_contact_center`；左右履带由宽度、中心距和接地长度构成有向矩形。割刀由横向有效割幅线段表示，其纵向位置来自实测 `cutter_forward_offset_m`。演示和验证调用同一几何定义；`cutter_visual` 仅用于 UI 中显示有厚度的割台外观，不参与覆盖率、碾压率或足迹校验。土壤压实、打滑、坡度、悬挂、惯性、GNSS 天线偏移和完整车身外廓可作为扩展评价因素。
- `rolling_body_pct`：履带扫掠与 `processed_mask` 主体重叠面积 / 主体作物面积。
- `rolling_headland_pct`：履带扫掠与 `headland_mask` 重叠面积 / 田头作物面积；缺少田头层时为 `null`。
- `rolling_canopy_pct`：履带扫掠与 `processed_mask | headland_mask` 去重并集的重叠面积 / 全部已确认作物面积；作为界面和论文中的“预计碾压率”，不得加入农业价值权重。
- `physical_crop_rolling_area_m2`：履带扫掠压过的全部已确认作物物理面积。
- `agronomic_damage_cost_m2`：`track_body_overlap_area_m2 + headland_damage_weight * track_headland_overlap_area_m2`；仅用于报告和论文对照，不参与转弯策略选择，不得称为碾压率。
- `headland_damage_weight`：田头单位物理碾压面积相对主体的规划权重，范围 `(0,1]`，无田间标定时必须使用 `1.0`。
- `rolling_core_pct`：履带扫掠与作物核心区重叠面积 / 作物核心区总面积；只作为详细风险指标。
- `track_core_overlap_pct`：履带扫掠中落在作物核心区的面积 / 履带总扫掠面积；始终保留为理论指标，不得称为预计碾压率。当前 `track_core_overlap_policy=report_only`，该指标不参与路径有效性和导出阻断；只有显式设置 `hard_gate` 才恢复阈值阻断。

碾压模型只使用左右履带矩形扫掠，不把机身或割刀计入碾压。割刀线只参与主体作物掩膜几何覆盖率。当前路径按选定转弯实现生成并执行硬约束校验，不做转弯候选排序；田头碾压及田头加权农业损伤只作报告，不得影响转弯策略。`rolling_crop_pct` 已废止，不得导出或作为兼容别名继续产生第二种“碾压率”定义。无掩膜区域不得仅凭分割结果自动视为确认无作物道路。

- `work_line_mode=footprint_optimized`时，`requested_work_line_mode`记录请求模式，`work_line_mode`记录实际执行模式。成功结果必须同时包含`optimization_complete=true`、`footprint_optimization_scope=centerline_anchored_lateral_search`、相等且大于零的`baseline_pass_count/optimized_pass_count`，以及逐线横移和目标代价诊断；不完整或无改进时如实回退为`band_centerline_with_footprint_validation`。
- 横向优化保留完整基准/优化工作线足迹关键指标摘要。显式请求 `footprint_optimized` 时，正式比较必须使用与最终完整路径验证相同的 `detected_mask`、`planned_mask`、规划支撑、不确定区和确认禁行区证据；桌面默认使用 `band_centerline_with_footprint_validation`，不隐式执行低碾压基准/优化比较。核心重叠、碾压率和覆盖率始终保留作结果字段；`footprint_metric_policy=report_only` 时不因其数值阻断路径或导出。田界越界、语义支撑区、不确定区和确认禁行区仍按当前策略处理；任一关键指标缺失、非有限或其他硬安全条件失败仍必须保留失败原因。
- 中心线锚定优化只能在`lateral_search_m`限定范围内横移，保留四行带基准线的数量、排序和纵向范围。确认禁行区候选不可选，非有限搜索参数必须在配置边界拒绝。
- `crop_mask_geometric_coverage_pct`：割刀有效割幅线段沿全部 `work` 段扫掠后，与处理后作物主体掩膜的交集面积 / 作物主体掩膜面积。
- `planned_target_geometric_coverage_pct`：同一割刀扫掠区域对规划目标掩膜的几何覆盖比例，是默认报告指标；只有显式 `footprint_metric_policy=hard_gate` 时才作为覆盖阈值比较。
- `detected_mask_geometric_coverage_pct`：同一割刀扫掠区域对原始识别掩膜的几何覆盖比例，只用于对照诊断。

上述三个指标描述路径割幅与作物掩膜的几何关系。实际收获率、产量和割台损失可通过田间收获调查联合评价。系统默认 `cutter_forward_offset_m=1.35 m` 仅作为规划/预览参数，来源必须保留为系统默认值，不能静默写成用户实测；用户明确清空参数时仍应在结果中保留该来源状态。

验证和演示必须调用 `path_pose` 中同一个按弧长插值、航向窗口和倒车方向算法。`footprint_pose_heading_window_m` 随验证结果记录；作业、转弯、进出场速度来自农机配置，播放倍率只改变观看快慢，不改变路径几何或足迹指标。

任何米制掩膜、路径和足迹函数缺少有效地理配准或掩膜状态时必须失败关闭，不得回退为约定像素比例。农机平面几何所需字段必须完整传入，不得在几何模块内部补造缺失尺寸。

视觉作物端点不得直接冒充物理转弯锚点；新增 `turn_approach` 必须保留在框选田块内并记录校正距离。

必选起点/可选终点拖动时，服务点在已确认田界内跟随鼠标的两个坐标轴自由移动；不得按主航线轴、田头深度或检测到的田头走廊强行吸附。与最近作业线端点相接的青色正交线只属于 UI `preview_only` 状态，不得写入 `auto_path`、路径 provenance、验证指标或导出文件。鼠标释放后保存光标位置（越界时仅投影到最近田界），使旧路径失效；只要起点已确认且其他规划条件完整，即由正式规划链重新生成 canonical 路径，不要求终点存在。

## 导出清单

`<route>.<ext>.manifest.json` 至少包含：

- 应用版本和 Git 提交号。
- 内部路径参考点 `track_contact_center` 与足迹模型版本；目标终端使用 GNSS 天线或其他参考点时，记录对应偏移。
- 输出文件 SHA-256、点数和坐标系。
- 源 GeoTIFF 完整 SHA-256、尺寸、波段、NoData、CRS、仿射和范围。
- 模型 SHA-256、模型注册信息。
- 三阶段完整证据、验证指标、农机参数和规划因素。
- Python 和核心依赖版本。
- 研究阶段、路径状态、详细检查项和已确认禁行区。

路径文件与清单必须作为一个归档单元保存。清单生成失败时，路径文件会被删除。

## 掩膜诊断包

掩膜处理完成后可导出 ZIP 诊断包，其中必须包含：

- `layers.npz`：原始掩膜、规则化主体、主体残差、残差合并结果、补缝新增、支撑约束移除、最终主体、田头、不确定残差和规划支撑等图层。
- `components.csv` / `components.json`：每个连通区域的面积、方向、主体距离、接触比例、端部外伸、最近行带和复核标志。
- `overview.png`：使用固定颜色区分处理阶段，并用编号标出需复核区域。
- `manifest.json`：算法版本、米制分辨率、处理阈值、各层像素数和复核区域总数。

诊断标志只用于定位异常，不得自动删除作物主体；算法调整必须结合原始影像和真实田块证据。

田头走廊修复必须满足端部位置、跨行连续性、原始田头种子占比和最大主体切入深度四项约束。走廊确认后按规则化主体端点走势建立平滑端墙，只保留真实原始作物支持；主体掩膜、行带轮廓和中心线必须同步裁剪，被移出的原始像素转入田头层，不得丢弃。内部横向结构和稀疏端部碎片不得自动升级为田头走廊。

## 自动离线实验包与完整论文证据

`align.paper-experiment.v1` 是软件中的离线实验 schema，当前默认计划只覆盖掩膜阶段消融。这个实现边界不等于论文研究范围：完整论文证据还包括作业带、中心线、路径、终端导入、无人收获机自主执行和作业后田间调查。统一流程见 [研究、算法、界面与田间验证统一流程](RESEARCH_SYSTEM_WORKFLOW.md)。

### 离线掩膜消融包

- `experiment_manifest.json` 使用 `align.paper-experiment.v1`，记录冻结输入摘要、实验矩阵、完成/失败计数和逐组参数。
- `results_long.csv` 是指标长表；`results_wide.csv` 和 `tables/*_comparison.csv` 是论文表格底稿。
- `cases/<case_id>/` 保存逐组 `result.json`、掩膜和路径预览；`figures/` 保存自动指标图。
- 掩膜消融组来自同一次正式规则化结果的原始掩膜、`regularized_body_mask`、`body_residual_mask` 和最终 `processed_mask`，不得伪造不存在的阶段开关。
- 默认计划的 `path_cases` 必须为空。底层运行器只有在未来显式设计独立路径对照时才接受路径策略组；各组必须共享完整规则化掩膜、田界、农机参数、服务点和禁行区，只改变请求策略。
- IoU、Dice、Precision、Recall 仅在提供同尺寸人工真值掩膜时输出；无真值时不得以原始预测代替。
- 失败实验组保留状态和错误原因，不能因局部失败丢弃其余结果。

### 完整软件田间资料包与外场记录

- `align.field-trial-package.v1` 归档源影像预览、原始/处理后掩膜、路径预览、候选导入路径、同名 manifest、推理/掩膜诊断和软件验证摘要，并由 `package_index.json` 记录文件哈希。
- 该资料包证明导出对象及其软件证据可追溯，不证明目标终端已接受路径，也不证明无人收获机完成自主作业。
- 终端导入回读、实际轨迹、执行时长、人工接管或中断、异常位置和作业后碾压调查必须作为外场记录另行保存，并与对应生育期、源影像、模型、中心线选择和路径 manifest 关联。
- 软件理论足迹与实际碾压调查必须分别保存和统计；没有田间记录时不得把理论几何指标改写为实测结果。

## `$PATH` 路径契约

- 每个路径点当前写为 `$PATH,route_id,index,total,lat,lon,offset_x,offset_y,status*`。
- `index` 从 `0` 开始；坐标系为 `EPSG:4326`；坐标字段顺序为纬度、经度。
- 状态码当前约定为：普通转弯 `0`、作业 `1`、进田 `2`、出田 `3`、倒车或辅助转弯 `4`。
- 文件末尾写入配置中的 `$INFM` 记录。
- 以上为当前软件兼容格式；目标收获机字段、状态码、索引和结束记录在 `path_protocol` 中记录。
- manifest 必须记录 `candidate_machine_path=true`、`terminal_protocol_validated` 及完整 `path_protocol`。
- manifest 的 `analysis.layout` 必须保留实际工作线模式、正式基准/优化比较和关键指标摘要；路径阶段 `artifact_sha256` 必须同时绑定 `full_path`、`segment_types` 与 `layout`。修改优化标签、比较结论或指标摘要后，正式导出和缓存恢复校验必须失败。

### 农机强制兼容 `$PATH` 变体

- `machine_harvester_straight_turn_v1` 是独立的终端适配投影，不替换科研路径或当前软件 `$PATH` 契约。
- 每条记录必须直接以 `$PATH` 开始，不允许插入独立索引行；严格格式为 `$PATH,0,index,total,lat,lon,offset_x,offset_y,2*`，第一列固定为 `0`，偏移量为纯数字，最后一列固定为 `2`。文件最后一行必须固定为 `$INFM,0,0.7,0.7,0.7,0.7,0,5,0*`，无额外空格；兼容文件使用 LF 换行。
- `index` 在每个分卷内从 `0` 开始，`total` 为该分卷 `$PATH` 记录数。内部 0/1 出边状态仅用于预览与清单，不得进入机子文件。
- 连续非作业段（包括真实转弯曲线、`turn_approach`、`turn_reverse`、`turn_aux` 以及进出场段）通常在兼容文件中压缩为上一条作业线终点到下一条作业线起点的一条直线弦；真实曲线仍保留在内部路径、运动学验证和履带评价中。唯一拓扑例外是完整折返上一条已收割作业线的 `turn_reverse`：必须保留该作业线另一端作为第二个状态 `0` 控制点，再以另一条状态 `0` 直线弦连接同田头的下一作业线，不能把两段重新压成跨田斜线。
- 画布“路径显示”固定区分“连续轨迹”和“农机文件”：前者显示 canonical 连续规划几何并作为模拟/理论评价来源，后者严格按最近一次成功写出的 `$PATH` 控制点、分卷边界和 0/1 出边状态显示。切换只改变显示，不得改写路径、验证或导出数据；跨分卷不得补画连接线。连续规划中的两点退化转弯必须保留 `display_only_connector` 等未验证诊断和原始两点几何，可按实线端点直连显示供人工调整；连续路径中超出已确认田块边界的折线子段以红色高亮，但不能伪造成未经验证的弧线或提升为机器就绪。套圈曲线因边界不足而保留供人工调整时不得压缩为两点，必须保留曲线点列，并由显示层仅将越界子段标红。
- 当前规划链不因混合种植区域边界生成 `turn_aux` 或整行 `turn_reverse`；历史 manifest 中的区域转场字段只作为旧结果证据读取，不得据此恢复或导出旧拓扑。兼容投影仍必须保留由真实服务点或运动学策略产生的独立 `turn_reverse` 控制点。
- 路线编辑器显示的 `state.path_points/path_status/path_geometry_modes` 是编辑器与兼容导出共享的唯一有序控制折线（DECISION-019）。相邻控制点构成一条可选边：`path_status` 保留终端兼容的 `1=作业/0=非作业` 语义，`path_geometry_modes` 逐边记录物理预览为 `straight` 或 `turn`，二者不能拆成第二套隐藏路线。自动结果进入编辑器时，每条主作业线先压缩为起终点；编辑器把连续状态 `1` 边即时派生为可选作业线对象，可从空路线新增，并支持整线平移、删除、调序、反向、端点移动、航点增删、边语义切换和直线/转弯几何切换。结构修改在相邻作业线之间重建局部状态 `0` 连接，不能维护第二套隐藏路线。编辑器点数、兼容预览文件点数和 manifest 的 `route_source=shared_route_control_polyline` 必须来自同一组控制点。
- 起点和终点在编辑覆盖层中与首、末控制点共用颜色、尺寸和交互层级；删除首/末控制点后，起终点引用自动切换到新的首/末控制点。未显式设置终点时不写入独立服务点状态，显示和兼容 TXT 仍使用当前控制折线最后一点作为自动终点。
- 两点状态 `0` 连接可根据当前最小转弯半径生成稠密连续预览；人工插入到三个及以上点的状态 `0` 点列是显式人工转弯，必须原样进入 canonical 显示、模拟与当前几何验证。结构调序/删除会重建相邻局部连接，但普通拖点和转弯插点不得把人工几何重新压成自动弓形。自动规划的稠密 `auto_path` 与人工重建后的连续 `auto_path` 都是各自当前路线的显示/评价几何，控制折线仍是人工编辑和兼容协议投影的权威来源。
- 每次人工修改都清空旧 `path_provenance/path_artifact` 和验证有效性。点击“验证人工路线”后，`path_artifact.manual_route.schema=align.manual-route.v1`，验证必须记录 `route_geometry_source=manual_control_edit_validated`、`manual_route_validated=true`、`automatic_planning_provenance_valid=false`、作业线数、控制点数和 0/1 状态；同一 `validate_path` 使用当前 `processed_mask`、`main_angle`、田界、确认禁行区、农机参数和语义掩膜，人工转弯另记录观测最小局部曲率半径。该记录证明当前软件几何检查已执行，不恢复自动规划阶段指纹，也不是田间执行证据。
- 路线编辑覆盖层必须叠加在当前 `processed_mask`、`headland_mask`、`uncertain_residual_mask`、田界和禁行区之上；进入编辑以及编辑中重绘不得清空这些前置证据或退回仅显示原始影像。紧凑工具区只常显选择/拖动、新增作业线、添加控制点、切换作业/转弯语义、切换直线/转弯几何和验证人工路线入口；添加控制点必须接受作业或转弯边，在线段上插入时保留原边模式，空白处新增时从当前路线末端追加直线边，低频结构操作由工具区常显的快捷键说明承接。工具区在已展开路径规划卡片中显示/隐藏后必须同步更新卡片最小高度和滚动内容高度，按钮几何不得相交或被卡片边界裁切。
- 编辑模式下“路径结果”列表由同一 `path_status` 与 `path_geometry_modes` 的连续运行段即时派生：每张卡片只保存 `edge_start/edge_end/node_start/node_end/work_index` 等临时索引，不得序列化为另一份路线。卡片单击与画布点选共享 `state.selected_type/selected_idx`；整段语义切换、直线/转弯几何切换、航点增删、作业线增删和作业线卡片拖拽排序必须调用同一人工编辑函数并刷新 `auto_path`、列表和画布。状态 `0` 转弯卡片不接受独立拖拽；作业线重排后局部转弯按共享端点重建。
- 兼容投影只在导出副本上清理相邻距离小于 `0.02 m` 的点，并重新计算 `index` 与 `total`；不改变内部路径。
- 桌面兼容 TXT 的共享控制折线必须包含已确认起点到第一条作业线的状态 `0` 入口边，以及最终作业段沿行进方向的终点延伸；未预先布设终点时由 `include_route_finish_extension` 生成，拖动后使用显式终点坐标。该入口/终点投影不替换连续 canonical 曲线，但必须进入 `001.txt`/`002.txt` 的实际记录。
- 农机兼容文件每份最多写入 50 条 `$PATH` 记录；超出时从保存目录现有最大三位数字 TXT 编号之后继续生成 `001.txt`、`002.txt`……（已有 `001.txt` 时从 `002.txt` 起）。每个分卷独立从 `index=0` 编号并声明本卷 `total`，manifest 的 `output_files`、`file_count`、`points_per_file` 和 `output_file_integrity` 必须记录实际分卷结果、每个文件的 SHA-256 与点数。
- 兼容测试投影不解释内部 `turn_reverse`/`turn_aux` 的车辆控制语义；所有 `work` 段压缩为本段两端点，普通非 `work` 连续段压缩为相邻主作业线端点之间的一条直线并输出状态 `0`。仅当 `turn_reverse` 的起终点精确对应上一条作业线的终点和起点时，保留这条已收割行折返边及后续同田头转场边两个状态 `0` 线段。这只改变兼容测试副本，不改变内部路径，也不把该副本宣称为机器执行就绪。
- 该变体的最后一列语义来自目标终端样例，尚未由终端实机确认；在 `terminal_protocol_validated` 为 `true` 前，manifest 的 `machine_executable` 必须保持 `false`。
- 界面的“农机兼容测试文件”允许在参数来源、机器就绪和终端协议尚未完成时生成测试文本，但 manifest 必须标记 `preview_only=true`、`machine_executable=false` 和 `COMPATIBILITY_TEST_ONLY_MANUAL_REVIEW`；这不是正式田间试验导出。
- 兼容 TXT 导出完成前必须回读全部分卷，验证 `output_file_integrity`、`points_per_file`、连续索引、每卷 `total`、固定记录状态码和 `$INFM`。画布切换到“TXT 路径”时再次按 manifest 校验；文件被替换、截断或与清单错配时，清除旧 TXT 预览并要求重新导出。
## 分段足迹和田头转弯指标

- `segment_metrics` 与 `turn_segment_metrics` 来自同一 `plan_pose` 采样和 `planar_rigid_footprint_v2` 有向左右履带矩形；每段至少记录 `segment_type`、`length_m`、独占 `track_swept_area_m2`、主体/田头/不确定区/禁行区重叠面积、田界越界面积与 `agronomic_damage_cost_m2`。
- `turn_metrics_attribution_rule=exclusive_first_segment` 表示重叠履带像素只归首次经过的段；各段独占面积和为全路径履带并集，不能与重复访问面积相加后再次当作并集。
- `track_pass_count_dtype=uint32` 的逻辑通过计数图用于 `track_rolling_visit_once_area_m2` 和 `track_rolling_visit_twice_or_more_area_m2`；每个逻辑路径段先把全部姿态合并为二值履带扫掠，再只累加一次。姿态覆盖次数仅作为调试信息，不得进入正式指标；这些面积是二维几何模型指标，不等于土壤压实或产量损失。
- `footprint_repeat_count_basis=logical_path_segment_union` 固定说明重复访问按逻辑路径段并集计算；`turn_repeat_twice_or_more_area_m2` 同样使用该逻辑通过次数。
- 转弯层另报 `headland_turn_track_overlap_area_m2`、`headland_turn_rolling_pct`、`turn_body_overlap_area_m2`、`turn_uncertain_overlap_area_m2`、`turn_forbidden_overlap_area_m2`、`turn_outside_field_pct`、`turn_cost_m2` 和 `turn_repeat_twice_or_more_area_m2`。田头指标只作报告，不参与策略选择；田界、确认禁行区和运动学仍是硬约束。
- 规划器保留 `requested_strategy`、`candidate_strategy`、`effective_strategy` 和 `fallback_reason`；固定策略生成后只执行田界、禁行区、运动学和足迹几何校验，不搜索田头偏移、不比较替代转弯，也不改变所选转弯几何。`auto` 仅按当前相邻作业线的几何条件逐段选择实现。
- `contour` 套圈策略记录 `ordering_strategy=contour`、`contour_ordering.algorithm_version`、`split_index`、`low_block_original_indices`、`high_block_original_indices`、`ordered_original_indices`、`corresponding_pairs`、`pair_count`、`normal_positions`、`adjacent_transition_count`、`final_order_validated`、`entry_selection_policy`、`minimum_possible_adjacent_transition_count`、`selected_adjacent_transition_count`、`entry_priority_overrode_minimum_adjacency`、`entry_selected_endpoint_before_headland_alignment_local`、`entry_selected_endpoint_local`、`entry_nearest_feasible_endpoint_local`、`entry_nearest_feasible_verified`、`terminal_blocked_endpoint_role`、`terminal_blocked_endpoint_is_route_finish`、`terminal_finish_orientation_forced`、`headland_wall_projections_px`、`shared_turn_clearance_m_by_side`、`shared_turn_clearance_scope`、`turn_distances_m` 和 `center_single_finish`。规划先形成完整的平行主作业线集合，再按法向排序后两半交错（8 行为 `0,4,1,5,2,6,3,7`，9 行为 `0,5,1,6,2,7,3,8,4`）；必选起点距离是循环/反向候选的第一排序键，相邻秩差数量是第二排序键。所有直线先统一到全田主方向，再根据起点确定首线执行方向；最终田头墙校正后还须对 canonical 首端点进行最近可行复核。`work_line_endpoint_extension_m` 只沿主航线向共享田头墙外延，且必须经过田界检查；可选外延越过已选田界时统一退回作物证据田头墙。被田界阻断且不能安全延伸的唯一短线可记录 `terminal_short_line` 并作为最终作业线；该线必须先从旋转集合中移除再追加，但不得反推全局进入侧。若受阻端成为末线起点，则保留该端不做田头墙延长，并由倒数第二条的局部正式转弯连接；若受阻端成为末线终点，则在该点正常结束。两种情况均不得生成跨田斜线，且继续受正式田界、曲率、禁行区和足迹门禁。宽间距局部转弯请求为 `bow`，中心间距仅在显式米制容差内允许用保持共同切线位置的小半圆，超过容差仍受最小半径安全回退。横向切线属于正式 `turn` 段并进入足迹归因，不得产生绕过路径协议的 `connector` 类型，也不得把合成闭合轮廓当作作业段。
- 等距种植区未指定 `uniform_zone_line_spacing_m` 时，间距固定取 `track_gauge_m`，并记录 `line_spacing_basis=track_gauge_overlap` 与 `adjacent_track_centerline_offset_m=0`；这使上一趟右履带中心线与下一趟左履带中心线重合。生成器可为整块选择一个统一相位以跨过局部掩膜缝隙，但不得逐条移动作业线；manifest 必须记录 `actual_spacing_m`、`phase_offset_m` 和 `spacing_is_uniform=true`。显式间距仅作为有记录的研究对照，不得冒充该默认重叠口径。
- 等距区边缘若只有一段明显短于主体作业跨度的掩膜碎片，生成器可在区域生成阶段丢弃该外侧碎片，避免把它与下一条完整作业线连接成跨田长斜线；内部短段和内部缺口不得被静默填补。`layout.uniform_zones` 必须记录 `line_lengths_m`、`median_supported_line_length_m`、`short_edge_line_indices`、`short_edge_line_filter_applied` 和 `accepted_pass_indices`，以便区分“边缘碎片被排除”和“推理/掩膜漏线”；若使用上述邻线端墙恢复，还必须同时保留 `pre_repair_line_lengths_m`，不得覆盖原始截短证据。
<!-- DECISION-016: contour ordering remains bound to crop-supported centerlines. -->

- 混合种植田块先在各分类区域内生成作业直线，再把全部直线展平并执行一次全田半分交错套圈。区域边界不得拥有独立入口、奇偶性、套圈顺序或转场；`contour_ordering.ordered_original_indices` 和 `adjacent_transition_count` 必须针对完整全田直线集合计算。逐线区域来源继续通过 `line_provenance` 保留，以审计不同密度生成规则。
- 每个固定转弯实现必须携带正式足迹验证的显式 `valid`、`issues`、`track_core_overlap_pct`、`planned_target_geometric_coverage_pct` 和 `track_outside_field_pct`；配置提供 `planning_support_mask`、`uncertain_mask` 或 `forbidden_mask` 时，还必须分别提供对应的履带重叠指标。缺失或其他硬条件超限保留失败原因，核心重叠在 `report_only` 模式下不得单独阻断路径。
<!-- DECISION-015: segment footprint and headland-turn metric contract. -->

## 人工参考线与中心线评价（DECISION-028）

### 中心线输出简化（DECISION-017/024）

`centerline-adaptive-polyline-v3` 把 `centerline_simplify_max_points`（默认 8）作为软预算、`centerline_simplify_hard_max_points`（默认 64）作为硬上限：达到软预算后只对仍不满足横向偏差或局部方向条件的区间继续加点，直到满足或到硬上限；硬上限仍不满足时才保留简化前完整的平滑点列。软预算点列作为端点修复锚点，额外加点不移动端点；一条带仍只生成一个候选目录项。`layout.centerline_simplification.point_budget_policy=soft_budget_then_hard_cap_then_smoothed_trace` 记录该规则，逐带记录输入/输出点数及 `budget_exceeded`、`hard_cap_reached`、`dense_fallback`。该变化只控制简化的附加误差，不证明相对人工真值的误差改善。路径算法版本为 `footprint-path-v33`，旧路径缓存重新规划；掩膜处理版本仍为 20。


- 参考线 sidecar 固定在 `<cache>/<project>/reference_lines.json`，schema `centerline-reference-lines-v1`：`{schema, tif_path, station_spacing_hint_m, updated_at, lines:[{id, band_id, band_label, pass, operator, created_at, points:[[x,y]...], fit:{p1,p2,angle_rad}}]}`。`fit` 一律由 `points` 重算，`points` 为全幅显示坐标。
- 参考线是人工研究证据，不是作业输入：不得进入 `project_state.json`、撤销历史、路径 provenance、导出闸门或缓存恢复校验；桌面端只以独立"参考线（人工）"图层显示，可在图层面板显隐。
- 离线评价包（`centerline-evaluation` v1，用 `--cache-dir/--tif` 复现）固定包含：`per_line.csv`（逐线 Eθ、平均/最大横向偏差、重叠率、匹配距离与未匹配说明）、`noise_floor.csv`（两遍人工参考线间偏差 = 判读噪声底）、`summary.json`（含 `mask_regularization_version` 与 mpp 溯源）、`reference_lines.json` 副本、`overlay.png`、`boxplot.png`、`README.md`，并打包同名 zip。
- 横向偏差口径：沿参考线 1 m 等间隔采样到提取中心线（折线）的最短距离；方向角取两线最小二乘方向差（180° 取小）。报告平均与最大，不得只报其一，也不得替换为文献未背书的指标。

# 数据与溯源契约

## 坐标约定

- 显示和路径编辑使用 GeoTIFF 降采样预览像素坐标。
- Qt 渲染纹理可以比逻辑预览更小，但图元变换必须使其场景边界仍等于逻辑预览尺寸；鼠标点、田块边界、掩膜 offset 和路径点不得存为纹理像素坐标。
- `GeoUtils` 使用与预览尺寸对应的仿射变换，将像素中心转换为源 CRS，再转换为 WGS84。
- GeoJSON、CSV、KML 和 JSON 使用带字段名或标准规定的 `lon, lat` 顺序。
- `$PATH` 路径按当前兼容格式写入 `lat, lon`；使用时按目标终端的字段定义逐项核对。
- 禁止无地理配准时写入伪坐标。

## 运行时与发布包（DECISION-018）

- 冻结版发布目录的入口固定为根目录 `reUI.exe`；PyInstaller 支持文件可以位于 `_internal`，不能要求用户调用 Python、Conda 或 pip。
- `runtime_paths.bundled_resource()` 只解析随程序发布的只读资源；`config.example.json`、`data/models/model_registry.json` 和内置模型不得在运行时直接改写。
- 用户配置、缓存、日志、用户模型和更新临时文件分别位于 `%LOCALAPPDATA%/reUI/config`、`cache`、`logs`、`models`、`updates`；首次运行从旧版 `config.json` 或内置示例迁移配置，更新不得覆盖这些目录。
- 冻结包的 `build_info.json` 至少包含版本、提交号、频道、构建时间和科学代码指纹。正式导出 manifest 必须保留这些构建信息；缺失科学代码指纹时不得生成可追溯正式清单。
- 源码 ZIP 与 Release 二进制 ZIP 是不同交付物。`update.bat` 只同步源代码；Release 工作流负责生成 `reUI-Windows-x64-<version>.zip` 和校验文件。

## 阶段证据

每个阶段记录 `algorithm_version`、`inputs`、`input_fingerprint`、`artifact_sha256` 和总 `fingerprint`。

| 阶段 | 必须纳入指纹的输入 | 产物 |
|---|---|---|
| inference | 源影像 SHA-256、模型 SHA-256、田块边界、切片参数、预处理参数 | 原始掩膜 |
| mask | 推理指纹、原始掩膜哈希、全部掩膜参数 | 处理后掩膜 |
| path | 掩膜指纹、农机参数、规划参数、调头策略、起终点和卸粮点 | 路径几何 |

缓存恢复和正式导出均重新计算并比较这些值。任何缺失或不一致都应失败关闭。

## 农机参数来源门禁

- `planning_factors.harvester` 保存本次路径实际使用的数值；`planning_factors.harvester_param_sources` 逐字段保存来源，至少区分 `default`、`recommended`、`user_entered`、`user_measured` 和 `legacy_unverified`。
- `default`、`recommended`、`user_entered` 和 `legacy_unverified` 只能按原来源记录，不能被静默写成实测值。`field_trial`/`machine_candidate` 缺少完整 `user_measured` 来源时，路径结果必须明确 `machine_executable=false`；研究导出可以写出文件，但不得伪装成机器就绪结果。
- UI 必须列出仍未完成实测来源的具体参数；来源元数据与农机数值一同进入项目缓存、路径 provenance 和导出 manifest。

## 掩膜语义层

- 掩膜处理档位固定为 `standard`。旧配置中的 `light`、`strong` 和 `very_strong` 仅作为兼容输入，加载后统一归一为 `standard`，不得改变当前处理参数。

- `processed_mask`：经过验证、可作为作物主体和作业线依据的区域。
- `headland_mask`：方向和位置证据支持的田头作物候选，用于田头碾压统计和全田真实作物碾压并集，不等同于低质量主体；与主体重叠的像素只计一次。生成前先记录规则化主体的连通区域分区；只有通过区域轴向跨度、端部邻接和跨行支持率分类的候选种子才可进入该层。
- `headland_corridor_mask`：仅在规则化纵向行带的轴向端部、跨越足够多行且具有连续横向证据时重建的完整田头走廊；田头与主体粘连时可记录 `terminal_dense_seed` 作为补充种子来源。接受后的走廊按稳健端点墙在米制最大深度内展开，并与原始作物支持相交，不得填成矩形。端点墙只采纳达到 `headland_terminal_wall_min_axial_span_ratio` 的长跨度横截行，再以分箱 Theil-Sen 趋势替换超过 `headland_terminal_wall_outlier_tolerance_m` 的局部异常端点；候选和种子均不得越过最终端墙进入主体。当统一轴向重构区间把端点量化成直墙时，只允许采用田块多边形中跨行跨度足够长的物理端边恢复墙体斜率，并向田头侧补偿 `band_end_trim_m` 的重构裁边，不能把去噪裁边当成主作业区语义边界。存在规则化纵向主体作为端墙证据时，`headland_corridor_endpoint_margin_m` 的有效主体内切值固定为 0。诊断记录参考跨度、可信/被拒/异常替换行数、田块端边斜率来源、裁边补偿、稳健斜率和有效容差；稀疏纵向行尾、区域分界和内部短片段不得作为田头端点。`diagnostics.region_partition` 保存区域数量、边界框、端部分类、候选支持率和拒绝原因；未经区域分类接受的种子不进入 `headland_mask`。
- `uncertain_residual_mask`：不能可靠归属到主体或田头的原始残差，只用于显示和中性场地支撑。
- `planning_support_mask`：场地支撑并集，只用于越界/支撑判断，不得替代 `processed_mask` 计算作物覆盖或碾压。
- `forbidden_regions`：用户在显示坐标中确认的禁行多边形；规划时按掩膜偏移栅格化，不得由不确定残差自动生成。
- `mask_result.diagnostics.profile_gate_basis` 必须为 `run_edge_midpoints`：相邻剖面行带的局部占用门控按 profile run 边界中点分隔，避免宽等距块吞并旁侧四行带的中心线。
- 当前 `mask_result.diagnostics.algorithm_version` 为 `17`。版本 16 及更早的 `mask_processed`/`mask_result` 不包含田块端边斜率恢复与重构裁边补偿契约，缓存恢复必须使掩膜处理和下游路径失效，不得仅因源影像、模型及数组形状相同而复用。

这些数组与处理后主体保存在同一个原子 NPZ 产物中；项目 JSON 只保存轻量元数据。

## 路径段语义

### 正射影像双尺度推理证据（DECISION-004/009）

- `mask_raw` 仍以 640 px 原生源分辨率切片的语义结果为主体。
- `1280→640` 上下文切片只提供邻近弱检测恢复和相邻实例接触边界，不得单独生成整块语义掩膜。
- 弱检测阈值、上下文尺寸、重叠率、恢复像素数和实例分隔像素数必须写入 `raster_preprocessing`/推理 provenance。
- 实例接触边界需获得跨切片重复支持，并通过米制长度和主方向一致性筛选；单个切片外轮廓不能切断作业带。

### 种植区域分类与分区路径（DECISION-017）

未显式配置 `uniform_zone_min_width_m` 时，默认值为 `4.0 m`。该默认值用于把真实数据中约 `2.5 m` 的完整四行宽窄行带与明显更宽的等距块分开；显式配置仍按配置值和歧义保护带执行，不能用默认值替代人工/上游证据。

- `layout.planting_region_classification` 保存分类器版本、宽度阈值、模糊保护带、逐带宽度/法向位置/结论/理由、分类计数、是否混合田块以及自动规划是否允许。类型只能是 `wide_narrow_rows`、`uniform_block` 或 `ambiguous`。分类器还必须记录 `partition_basis` 和 `wide_narrow_region_count`；当一个 `uniform_block` 的法向位置位于两组宽窄行带之间时，两侧必须形成不同的 `wide_narrow_rows_N`，不能因类型相同而跨区合并。
- 当 `profile_width_m` 落入模糊保护带时，只有同时具备上游 `band_type=wide`、低于均匀区下界的局部重构宽度，以及连续中心线跨度证据，才可记录 `borderline_resolution=upstream_band_type_and_continuous_centerline_support` 并归入 `wide_narrow_rows`；缺少任一证据仍必须是 `ambiguous`，不得按田块左右位置猜测。
- `layout.work_regions` 保存区域标识、类型、来源带、法向位置、作业线数量和来源语义；不得内嵌重复的完整作业线数组。`layout.region_execution_order` 保存实际完成顺序；一至两个区域时可由起点选择首区，有起终点时还用终点参考选择首尾方向。三个及以上法向连续区域只能在正向、反向两种单调顺序中择优，并记录 `region_order_rule=monotonic_normal_order_minimum_entry_exit_connection`，不得从中间区出发后跨越已完成区域。无起点/终点时使用稳定回退顺序，不得以“左区/右区”表达类型。
- UI/项目快照中的带记录用 `planning_contour` 保存最多 256 点的轻量规划边界，原始稠密 `contour` 不进入轻量状态。等距区诊断的 `reference_geometry_source` 必须说明使用 `contour`、`planning_contour` 或历史兼容的 `compact_centerline_and_mask`；当 `profile_width_m` 明显大于该轻量边界时，必须以实测剖面宽度扩展横向规划包络，并记录 `effective_geometry_source=<reference>_expanded_from_profile_width`、`profile_width_px`、`reference_geometry_width_px` 和 `effective_transverse_width_m`，不得静默按窄边界截断等距区。若等距区与宽窄行区相邻，扩展包络仍必须受最近宽窄行中心线的共享边界安全带限制；该中心线归属边界作业带，禁止等距区再生成近重复作业线。
- 宽窄行区的 `line_source=four_row_band_centerlines`；等距区的 `line_source=cutter_width_uniform_swaths`。混合田块的套圈排序仅在各区域内部进行，`turn_assessment.contour_ordering.no_cross_region_pairing=true`；所有区域的 `shared_headland_wall_projections_px` 和逐侧 `shared_turn_clearance_m_by_side` 必须相同，后者记录 `shared_turn_clearance_scope=all_classified_regions`。套圈调头方向使用同一主航线轴，不得由单条短线尾部改变；局部凹边界只能使全田同侧切线共同内移。若规则化带的 `longitudinal_extent_m` 与至少 3 条完整邻线共同证明中心线被内部缝隙截短，可在短线过滤前沿主航向恢复到邻线中位田头墙；每个延长段必须完全位于已确认田界内，并在 `layout.endpoint_repair` 或 `layout.uniform_zones[].endpoint_repair` 记录来源、目标墙、延长距离和结果跨度。等距线恢复只改变轴向端点，不得改变法向位置和 `track_gauge_m` 等距间隔。
- `turn_assessment.region_routes` 保存每区独立套圈诊断；`turn_assessment.region_transfers` 保存相邻区域间唯一 `turn_aux` 的起终点、策略、长度、田界检查、曲线非退化检查和机器可执行结论。任何转场失败都使整条候选失败关闭。
- `layout.coverage_gap_recovery` 保存规划期漏线恢复版本、正常四行间距、异常间距倍数、来源带、目标法向位置、`phase_method`、左右稳定队列数量、中心线支持率、横截面轴向支持率/平均占用率、轴向长度比、被明确忽略的窄带标识与 `recovered_band_ids`；`layout.planning_recovered=true` 表示路径实际使用了恢复作业带。两侧相位证据不对称时从较长稳定队列外推，证据相当时才取中点；机器中心线可位于四行作物之间，不能因中心像素为空而漏掉整带。窄带中心不得提升为四行作业线，且处理后掩膜的原始分类不得被覆盖。`field_trial`/`machine_candidate` 仍默认关闭新的自动恢复；缓存中已有该标记时必须保留并显示该来源。默认研究导出不弹窗、不阻断，严格 `fail_closed` 模式才拒绝。
- `turn_assessment.service_points` 保存 `mode`、起终点参考是否参与排序/方向、以及入口/出口连接是否实际写入路径。默认 `service_point_mode=connect`：起点和终点可以自由拖动，分别参与首尾作业线/方向参考，并将服务点与路线首尾实际连接；若仅需要参考坐标而不输出连接段，必须显式设置 `service_point_mode=reference_only`。套圈排序会先将离起点最近的完整两行组移到首组，不改变对应间隔。

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

碾压模型只使用左右履带矩形扫掠，不把机身或割刀计入碾压。割刀线只参与主体作物掩膜几何覆盖率。通过硬约束的候选路径按主体碾压、核心区重叠、语义风险、几何覆盖率和长度排序；田头碾压及田头加权农业损伤只作报告，不得影响转弯策略。`rolling_crop_pct` 已废止，不得导出或作为兼容别名继续产生第二种“碾压率”定义。无掩膜区域不得仅凭分割结果自动视为确认无作物道路。

- `work_line_mode=footprint_optimized`时，`requested_work_line_mode`记录请求模式，`work_line_mode`记录实际执行模式。成功结果必须同时包含`optimization_complete=true`、`footprint_optimization_scope=centerline_anchored_lateral_search`、相等且大于零的`baseline_pass_count/optimized_pass_count`，以及逐线横移和目标代价诊断；不完整或无改进时如实回退为`band_centerline_with_footprint_validation`。
- 横向优化保留完整基准/优化工作线足迹关键指标摘要。正式比较必须使用与最终完整路径验证相同的 `detected_mask`、`planned_mask`、规划支撑、不确定区和确认禁行区证据。核心重叠保留作对照记录；当前默认不因该指标数值阻断路径或导出。田界越界、语义支撑区、不确定区和确认禁行区仍按当前策略处理，规划目标几何覆盖率不得下降；任一关键指标缺失、非有限或其他硬安全条件失败仍必须保留失败原因。
- 中心线锚定优化只能在`lateral_search_m`限定范围内横移，保留四行带基准线的数量、排序和纵向范围。确认禁行区候选不可选，非有限搜索参数必须在配置边界拒绝。
- `crop_mask_geometric_coverage_pct`：割刀有效割幅线段沿全部 `work` 段扫掠后，与处理后作物主体掩膜的交集面积 / 作物主体掩膜面积。
- `planned_target_geometric_coverage_pct`：同一割刀扫掠区域对规划目标掩膜的几何覆盖比例，是路径有效性的默认覆盖门限。
- `detected_mask_geometric_coverage_pct`：同一割刀扫掠区域对原始识别掩膜的几何覆盖比例，只用于对照诊断。

上述三个指标描述路径割幅与作物掩膜的几何关系。实际收获率、产量和割台损失可通过田间收获调查联合评价。缺少实测 `cutter_forward_offset_m` 时不生成几何覆盖率或路径；系统配置中该字段默认 `null`。

验证和演示必须调用 `path_pose` 中同一个按弧长插值、航向窗口和倒车方向算法。`footprint_pose_heading_window_m` 随验证结果记录；作业、转弯、进出场速度来自农机配置，播放倍率只改变观看快慢，不改变路径几何或足迹指标。

任何米制掩膜、路径和足迹函数缺少有效地理配准或掩膜状态时必须失败关闭，不得回退为约定像素比例。农机平面几何所需字段必须完整传入，不得在几何模块内部补造缺失尺寸。

视觉作物端点不得直接冒充物理转弯锚点；新增 `turn_approach` 必须保留在框选田块内并记录校正距离。

起点/终点拖动时，服务点在已确认田界内跟随鼠标的两个坐标轴自由移动；不得按主航线轴、田头深度或检测到的田头走廊强行吸附。与最近作业线端点相接的青色正交线只属于 UI `preview_only` 状态，不得写入 `auto_path`、路径 provenance、验证指标或导出文件。鼠标释放后保存光标位置（越界时仅投影到最近田界），使旧路径失效，并由正式规划链重新生成 canonical 路径。

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

## 论文实验资料包

- `experiment_manifest.json` 使用 `reui.paper-experiment.v1`，记录冻结输入摘要、实验矩阵、完成/失败计数和逐组参数。
- `results_long.csv` 是指标长表；`results_wide.csv` 和 `tables/*_comparison.csv` 是论文表格底稿。
- `cases/<case_id>/` 保存逐组 `result.json`、掩膜和路径预览；`figures/` 保存自动指标图。
- 掩膜消融组来自同一次正式规则化结果的原始掩膜、`regularized_body_mask`、`body_residual_mask` 和最终 `processed_mask`，不得伪造不存在的阶段开关。
- 路径策略组共享完整规则化掩膜、田界、农机参数、服务点和禁行区，只改变请求策略。
- IoU、Dice、Precision、Recall 仅在提供同尺寸人工真值掩膜时输出；无真值时不得以原始预测代替。
- 失败实验组保留状态和错误原因，不能因局部失败丢弃其余结果。

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
- 记录固定为 `$PATH,0,index,total,lat,lon,offset_x,offset_y,status*`；第一列固定为 `0`，最后一列只允许 `0`（转弯及其他非作业段）和 `1`（作业段）。
- 状态码按当前点到下一点的出边解释，末点固定为 `0`；因此状态切换应与示例中的 `1,0,1,0` 交替边界一致。
- 连续非作业段（包括真实转弯曲线、`turn_approach`、`turn_reverse`、`turn_aux` 以及进出场段）在兼容文件中压缩为上一条作业线终点到下一条作业线起点的一条直线弦；真实曲线仍保留在内部路径、运动学验证和履带评价中。
- 路线编辑器显示的 `state.path_points/path_status` 是编辑器与兼容导出共享的控制折线：每条主作业线只保留起点、终点两点，连续非作业段只保留首尾点并绘制为一条直线；因此编辑器点数、兼容预览文件点数和 manifest 的 `route_source=shared_route_control_polyline` 必须一致。密集 `auto_path` 仅作为科研验证、动画和履带评价的 canonical 路径保留。
- 兼容投影只在导出副本上清理相邻距离小于 `0.02 m` 的点，并重新计算 `index` 与 `total`；不改变内部路径。
- 农机兼容文件每份最多写入 50 条 `$PATH` 记录；超出时从保存目录现有最大三位数字 TXT 编号之后继续生成 `001.txt`、`002.txt`……（已有 `001.txt` 时从 `002.txt` 起）。每个分卷独立从 `index=0` 编号并声明本卷 `total`，manifest 的 `output_files`、`file_count` 和 `points_per_file` 必须记录实际分卷结果。
- 兼容测试投影不解释内部 `turn_reverse`/`turn_aux` 的运动语义；所有 `work` 段压缩为本段两端点，所有非 `work` 段统一压缩为相邻主作业线端点之间的一条直线，并输出状态 `0`。这只改变兼容测试副本，不改变内部路径，也不把该副本宣称为机器执行就绪。
- 该变体的最后一列语义来自目标终端样例，尚未由终端实机确认；在 `terminal_protocol_validated` 为 `true` 前，manifest 的 `machine_executable` 必须保持 `false`。
- 界面的“农机兼容测试文件”允许在参数来源、机器就绪和终端协议尚未完成时生成测试文本，但 manifest 必须标记 `preview_only=true`、`machine_executable=false` 和 `COMPATIBILITY_TEST_ONLY_MANUAL_REVIEW`；这不是正式田间试验导出。
## 分段足迹和田头转弯指标

- `segment_metrics` 与 `turn_segment_metrics` 来自同一 `plan_pose` 采样和 `planar_rigid_footprint_v2` 有向左右履带矩形；每段至少记录 `segment_type`、`length_m`、独占 `track_swept_area_m2`、主体/田头/不确定区/禁行区重叠面积、田界越界面积与 `agronomic_damage_cost_m2`。
- `turn_metrics_attribution_rule=exclusive_first_segment` 表示重叠履带像素只归首次经过的段；各段独占面积和为全路径履带并集，不能与重复访问面积相加后再次当作并集。
- `track_pass_count_dtype=uint32` 的逻辑通过计数图用于 `track_rolling_visit_once_area_m2` 和 `track_rolling_visit_twice_or_more_area_m2`；每个逻辑路径段先把全部姿态合并为二值履带扫掠，再只累加一次。姿态覆盖次数仅作为调试信息，不得进入正式指标；这些面积是二维几何模型指标，不等于土壤压实或产量损失。
- `footprint_repeat_count_basis=logical_path_segment_union` 固定说明重复访问按逻辑路径段并集计算；`turn_repeat_twice_or_more_area_m2` 同样使用该逻辑通过次数。
- 转弯层另报 `headland_turn_track_overlap_area_m2`、`headland_turn_rolling_pct`、`turn_body_overlap_area_m2`、`turn_uncertain_overlap_area_m2`、`turn_forbidden_overlap_area_m2`、`turn_outside_field_pct`、`turn_cost_m2` 和 `turn_repeat_twice_or_more_area_m2`。田头指标只作报告，不进入候选排序；田界、确认禁行区和运动学仍是硬约束。
- 规划器保留 `requested_strategy`、`candidate_strategy`、`effective_strategy` 和 `fallback_reason`；`headland_turn_optimization_enabled` 只控制有限转弯候选选择，默认启用，不能把工作线推离四行带中心线。
- `contour` 套圈策略记录 `ordering_strategy=contour`、`contour_ordering.algorithm_version`、`split_index`、`low_block_original_indices`、`high_block_original_indices`、`ordered_original_indices`、`corresponding_pairs`、`pair_count`、`normal_positions`、`adjacent_transition_count`、`headland_wall_projections_px`、`shared_turn_clearance_m_by_side`、`shared_turn_clearance_scope`、`turn_distances_m` 和 `center_single_finish`。规划先形成完整的平行主作业线集合，再按法向排序后两半交错（8 行为 `0,4,1,5,2,6,3,7`，9 行为 `0,5,1,6,2,7,3,8,4`）；`work_line_endpoint_extension_m` 只沿主航线向共享田头墙外延，且必须经过田界检查。宽间距局部转弯请求为 `bow`，中心间距仅在显式米制容差内允许用保持共同切线位置的小半圆，超过容差仍受最小半径安全回退。横向切线属于正式 `turn` 段并进入足迹归因，不得产生绕过路径协议的 `connector` 类型，也不得把合成闭合轮廓当作作业段。
- 等距种植区未指定 `uniform_zone_line_spacing_m` 时，间距固定取 `track_gauge_m`，并记录 `line_spacing_basis=track_gauge_overlap` 与 `adjacent_track_centerline_offset_m=0`；这使上一趟右履带中心线与下一趟左履带中心线重合。生成器可为整块选择一个统一相位以跨过局部掩膜缝隙，但不得逐条移动作业线；manifest 必须记录 `actual_spacing_m`、`phase_offset_m` 和 `spacing_is_uniform=true`。显式间距仅作为有记录的研究对照，不得冒充该默认重叠口径。
- 等距区边缘若只有一段明显短于主体作业跨度的掩膜碎片，生成器可在区域生成阶段丢弃该外侧碎片，避免把它与下一条完整作业线连接成跨田长斜线；内部短段和内部缺口不得被静默填补。`layout.uniform_zones` 必须记录 `line_lengths_m`、`median_supported_line_length_m`、`short_edge_line_indices`、`short_edge_line_filter_applied` 和 `accepted_pass_indices`，以便区分“边缘碎片被排除”和“推理/掩膜漏线”；若使用上述邻线端墙恢复，还必须同时保留 `pre_repair_line_lengths_m`，不得覆盖原始截短证据。
<!-- DECISION-016: contour ordering remains bound to crop-supported centerlines. -->

- 混合区域的每个区域仍独立执行半分交错套圈；若上一区域最后一条作业线与下一区域默认首条线的法向间距小于跨区域安全间隔，规划器必须尝试将下一区域整体反向，从远侧开始套圈。诊断写入 `contour_ordering.cross_region_transition_records`、区域级 `cross_region_transition_spacing_m` 与 `normal_order_reversed`；这不是跨区域配圈。
- 每个转弯候选在软代价排序前必须携带正式足迹验证的显式 `valid=True`、空 `issues`、`track_core_overlap_pct`、`planned_target_geometric_coverage_pct` 和 `track_outside_field_pct`；配置提供 `planning_support_mask`、`uncertain_mask` 或 `forbidden_mask` 时，还必须分别提供对应的履带重叠指标。缺失或其他硬条件超限进入 `rejected_candidates`，核心重叠在 `report_only` 模式下不得单独拒绝候选。
<!-- DECISION-015: segment footprint and headland-turn metric contract. -->

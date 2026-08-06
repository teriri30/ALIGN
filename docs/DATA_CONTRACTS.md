# 数据与溯源契约

## 坐标约定

- 显示和路径编辑使用 GeoTIFF 降采样预览像素坐标。
- Qt 渲染纹理可以比逻辑预览更小，但图元变换必须使其场景边界仍等于逻辑预览尺寸；鼠标点、田块边界、掩膜 offset 和路径点不得存为纹理像素坐标。
- `GeoUtils` 使用与预览尺寸对应的仿射变换，将像素中心转换为源 CRS，再转换为 WGS84。
- GeoJSON、CSV、KML 和 JSON 使用带字段名或标准规定的 `lon, lat` 顺序。
- `$PATH` 路径按当前兼容格式写入 `lat, lon`；使用时按目标终端的字段定义逐项核对。
- 禁止无地理配准时写入伪坐标。

## 阶段证据

每个阶段记录 `algorithm_version`、`inputs`、`input_fingerprint`、`artifact_sha256` 和总 `fingerprint`。

| 阶段 | 必须纳入指纹的输入 | 产物 |
|---|---|---|
| inference | 源影像 SHA-256、模型 SHA-256、田块边界、切片参数、预处理参数 | 原始掩膜 |
| mask | 推理指纹、原始掩膜哈希、全部掩膜参数 | 处理后掩膜 |
| path | 掩膜指纹、农机参数、规划参数、调头策略、起终点和卸粮点 | 路径几何 |

缓存恢复和正式导出均重新计算并比较这些值。任何缺失或不一致都应失败关闭。

## 掩膜语义层

- `processed_mask`：经过验证、可作为作物主体和作业线依据的区域。
- `headland_mask`：方向和位置证据支持的田头作物候选，用于田头碾压统计和全田真实作物碾压并集，不等同于低质量主体；与主体重叠的像素只计一次。生成前先记录规则化主体的连通区域分区；只有通过区域轴向跨度、端部邻接和跨行支持率分类的候选种子才可进入该层。
- `headland_corridor_mask`：仅在规则化纵向行带的轴向端部、跨越足够多行且具有连续横向证据时重建的完整田头走廊；田头与主体粘连时可记录 `terminal_dense_seed` 作为补充种子来源，其语义优先于纵向规则化主体。接受后的走廊按平滑逐列端点墙在米制最大深度内展开，并与原始作物支持相交，不得填成矩形。端点墙只采纳达到 `headland_terminal_wall_min_axial_span_ratio` 的长跨度横截行，并记录参考跨度、可信行数和被拒短行数；稀疏纵向行尾、区域分界和内部短片段不得作为田头端点。`diagnostics.region_partition` 保存区域数量、边界框、端部分类、候选支持率和拒绝原因；未经区域分类接受的种子不进入 `headland_mask`。
- `uncertain_residual_mask`：不能可靠归属到主体或田头的原始残差，只用于显示和中性场地支撑。
- `planning_support_mask`：场地支撑并集，只用于越界/支撑判断，不得替代 `processed_mask` 计算作物覆盖或碾压。
- `forbidden_regions`：用户在显示坐标中确认的禁行多边形；规划时按掩膜偏移栅格化，不得由不确定残差自动生成。

这些数组与处理后主体保存在同一个原子 NPZ 产物中；项目 JSON 只保存轻量元数据。

## 路径段语义

### 种植区域分类与分区路径（DECISION-017）

- `layout.planting_region_classification` 保存分类器版本、宽度阈值、模糊保护带、逐带宽度/法向位置/结论/理由、分类计数、是否混合田块以及自动规划是否允许。类型只能是 `wide_narrow_rows`、`uniform_block` 或 `ambiguous`。
- `layout.work_regions` 保存区域标识、类型、来源带、法向位置、作业线数量和来源语义；不得内嵌重复的完整作业线数组。`layout.region_execution_order` 保存实际完成顺序；有起点时 `region_order_rule=nearest_entry_region_then_distance_tiebreak`，无起点时使用稳定回退顺序，不得以“左区/右区”表达类型。
- UI/项目快照中的带记录用 `planning_contour` 保存最多 256 点的轻量规划边界，原始稠密 `contour` 不进入轻量状态。等距区诊断的 `reference_geometry_source` 必须说明使用 `contour`、`planning_contour` 或历史兼容的 `compact_centerline_and_mask`。
- 宽窄行区的 `line_source=four_row_band_centerlines`；等距区的 `line_source=cutter_width_uniform_swaths`。混合田块的套圈排序仅在各区域内部进行，`turn_assessment.contour_ordering.no_cross_region_pairing=true`；所有区域的 `shared_headland_wall_projections_px` 和逐侧 `shared_turn_clearance_m_by_side` 必须相同，后者记录 `shared_turn_clearance_scope=all_classified_regions`。套圈调头方向使用同一主航线轴，不得由单条短线尾部改变；局部凹边界只能使全田同侧切线共同内移。
- `turn_assessment.region_routes` 保存每区独立套圈诊断；`turn_assessment.region_transfers` 保存相邻区域间唯一 `turn_aux` 的起终点、策略、长度、田界检查、曲线非退化检查和机器可执行结论。任何转场失败都使整条候选失败关闭。
- `layout.coverage_gap_recovery` 保存规划期漏线恢复版本、正常四行间距、异常间距倍数、来源带、目标法向位置、`phase_method`、左右稳定队列数量、中心线支持率、横截面轴向支持率/平均占用率、轴向长度比、被明确忽略的窄带标识与 `recovered_band_ids`。两侧相位证据不对称时从较长稳定队列外推，证据相当时才取中点；机器中心线可位于四行作物之间，不能因中心像素为空而漏掉整带。窄带中心不得提升为四行作业线，且处理后掩膜的原始分类不得被覆盖。
- `turn_assessment.service_points.exit_route` 保存离场模式。`direct_same_headland` 只允许短边田头内接驳；若直达存在显著主航线轴向跨度，则使用 `retrace_last_harvested_work_then_exit_headland`，把沿已收割作业线的返回段标为 `turn_reverse`、只把田头接驳标为 `exit`，并记录返回距离和田界检查，禁止输出跨田斜向 `exit`。

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
- `track_core_overlap_pct`：履带扫掠中落在作物核心区的面积 / 履带总扫掠面积；用于路径安全门限，不得称为预计碾压率。

碾压模型只使用左右履带矩形扫掠，不把机身或割刀计入碾压。割刀线只参与主体作物掩膜几何覆盖率。通过硬约束的候选路径按主体碾压、核心区重叠、语义风险、几何覆盖率和长度排序；田头碾压及田头加权农业损伤只作报告，不得影响转弯策略。`rolling_crop_pct` 已废止，不得导出或作为兼容别名继续产生第二种“碾压率”定义。无掩膜区域不得仅凭分割结果自动视为确认无作物道路。

- `work_line_mode=footprint_optimized`时，`requested_work_line_mode`记录请求模式，`work_line_mode`记录实际执行模式。成功结果必须同时包含`optimization_complete=true`、`footprint_optimization_scope=centerline_anchored_lateral_search`、相等且大于零的`baseline_pass_count/optimized_pass_count`，以及逐线横移和目标代价诊断；不完整或无改进时如实回退为`band_centerline_with_footprint_validation`。
- 横向优化保留完整基准/优化工作线足迹关键指标摘要。正式比较必须使用与最终完整路径验证相同的 `detected_mask`、`planned_mask`、规划支撑、不确定区和确认禁行区证据。接受条件为：履带核心重叠、田界越界、语义支撑区、不确定区和确认禁行区均不得恶化，规划目标几何覆盖率不得下降，且至少一项指标严格改善；任一关键指标缺失、非有限或恶化都必须回退基准中心线。`machine_readiness` 只有在 `formal_comparison.accepted=true`、无退化/无效指标、至少一项严格改善且基准/优化摘要的核心重叠、规划覆盖率和田界越界均为有限数值时，才能通过该优化证据门禁。
- 中心线锚定优化只能在`lateral_search_m`限定范围内横移，保留四行带基准线的数量、排序和纵向范围。确认禁行区候选不可选，非有限搜索参数必须在配置边界拒绝。
- `crop_mask_geometric_coverage_pct`：割刀有效割幅线段沿全部 `work` 段扫掠后，与处理后作物主体掩膜的交集面积 / 作物主体掩膜面积。
- `planned_target_geometric_coverage_pct`：同一割刀扫掠区域对规划目标掩膜的几何覆盖比例，是路径有效性的默认覆盖门限。
- `detected_mask_geometric_coverage_pct`：同一割刀扫掠区域对原始识别掩膜的几何覆盖比例，只用于对照诊断。

上述三个指标描述路径割幅与作物掩膜的几何关系。实际收获率、产量和割台损失可通过田间收获调查联合评价。缺少实测 `cutter_forward_offset_m` 时不生成几何覆盖率或路径；系统配置中该字段默认 `null`。

验证和演示必须调用 `path_pose` 中同一个按弧长插值、航向窗口和倒车方向算法。`footprint_pose_heading_window_m` 随验证结果记录；作业、转弯、进出场速度来自农机配置，播放倍率只改变观看快慢，不改变路径几何或足迹指标。

任何米制掩膜、路径和足迹函数缺少有效地理配准或掩膜状态时必须失败关闭，不得回退为约定像素比例。农机平面几何所需字段必须完整传入，不得在几何模块内部补造缺失尺寸。

视觉作物端点不得直接冒充物理转弯锚点；新增 `turn_approach` 必须保留在框选田块内并记录校正距离。

起点/终点拖动时显示的田头吸附点和青色正交接驳只属于 UI `preview_only` 状态，不得写入 `auto_path`、路径 provenance、验证指标或导出文件。鼠标释放后保存纠正后的最终服务点，使旧路径失效，并由正式规划链重新生成 canonical 路径。

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
## 分段足迹和田头转弯指标

- `segment_metrics` 与 `turn_segment_metrics` 来自同一 `plan_pose` 采样和 `planar_rigid_footprint_v2` 有向左右履带矩形；每段至少记录 `segment_type`、`length_m`、独占 `track_swept_area_m2`、主体/田头/不确定区/禁行区重叠面积、田界越界面积与 `agronomic_damage_cost_m2`。
- `turn_metrics_attribution_rule=exclusive_first_segment` 表示重叠履带像素只归首次经过的段；各段独占面积和为全路径履带并集，不能与重复访问面积相加后再次当作并集。
- `track_access_count_dtype=uint32` 的访问计数图用于 `track_rolling_visit_once_area_m2` 和 `track_rolling_visit_twice_or_more_area_m2`；这些是二维几何模型指标，不等于土壤压实或产量损失。
- 转弯层另报 `headland_turn_track_overlap_area_m2`、`headland_turn_rolling_pct`、`turn_body_overlap_area_m2`、`turn_uncertain_overlap_area_m2`、`turn_forbidden_overlap_area_m2`、`turn_outside_field_pct`、`turn_cost_m2` 和 `turn_repeat_twice_or_more_area_m2`。田头指标只作报告，不进入候选排序；田界、确认禁行区和运动学仍是硬约束。
- 规划器保留 `requested_strategy`、`candidate_strategy`、`effective_strategy` 和 `fallback_reason`；`headland_turn_optimization_enabled` 只控制有限转弯候选选择，默认启用，不能把工作线推离四行带中心线。
- `contour` 套圈策略记录 `ordering_strategy=contour`、`contour_ordering.algorithm_version`、`ordered_original_indices`、`normal_positions`、`headland_wall_projections_px`、`shared_turn_clearance_m_by_side`、`shared_turn_clearance_scope`、`turn_distances_m`、`turn_distances_nonincreasing` 和 `center_single_finish`。顺序固定为法向两侧向中心配对；宽间距局部转弯请求为 `bow`，中心间距仅在显式米制容差内允许用保持共同切线位置的小半圆，超过容差仍受最小半径安全回退。横向切线属于正式 `turn` 段并进入足迹归因，不得产生绕过路径协议的 `connector` 类型，也不得把合成闭合轮廓当作作业段。
<!-- DECISION-016: contour ordering remains bound to crop-supported centerlines. -->
- 每个转弯候选在软代价排序前必须携带正式足迹验证的显式 `valid=True`、空 `issues`、`track_core_overlap_pct`、`planned_target_geometric_coverage_pct` 和 `track_outside_field_pct`；配置提供 `planning_support_mask`、`uncertain_mask` 或 `forbidden_mask` 时，还必须分别提供对应的履带重叠指标。缺失或超限均进入 `rejected_candidates`，不得按零值补齐。
<!-- DECISION-015: segment footprint and headland-turn metric contract. -->

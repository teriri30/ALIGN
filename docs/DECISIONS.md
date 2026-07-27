# reUI 关键设计决策

本文件记录不能被随意删除、绕过或以“简化”为理由弱化的系统级决策。修改决策相关代码时，必须同步核对代码锚点和不变量测试。

## DECISION-001：正式路径导出必须失败关闭

**结论**：GeoJSON、CSV、KML、JSON 和 `$PATH` 必须先经过 `_geo_export_payload()`；输出成功后必须生成 manifest。

**原因**：路径可能进入受控田间试验。可用性不能优先于坐标、参数和路径安全。

**不变量**：

- 无有效地理配准、路径未验证或 provenance 不一致时禁止导出。
- 农机参数、调头策略与规划时不一致时禁止导出。
- 路径点超出影像范围时禁止仿射外推。
- manifest 写入失败时不得保留正式路径文件。
- 旧 `ExportEngine` 不得成为旁路。

**代码锚点**：`MainWindow._geo_export_payload`、`MainWindow._write_export_manifest`、`integrity_check.verify_manifest`。

**测试锚点**：`test_invalid_or_manually_edited_path_is_blocked_from_export`、`test_offline_manifest_checker_detects_route_tampering`、`test_legacy_export_engine_is_fail_closed`。

## DECISION-002：缓存恢复必须验证完整阶段证据

**结论**：缓存必须绑定源影像、模型、配置、上游指纹和产物哈希；缺失或不一致时逐级失效。

**原因**：同名模型、同尺寸影像或同一路径不代表内容相同，错误恢复会污染全部下游结果。

**不变量**：

- 模型内容变化后旧推理掩膜失效。
- 不按图像尺寸跨源复用缓存。
- 推理失效必须使掩膜和路径失效；掩膜失效必须使路径失效。
- 旧模式缓存失败关闭，不猜测兼容。
- 恢复的路径几何和指标只能作为历史快照展示；在当前输入与当前验证规则下重新执行完整路径验证前，`auto_path_valid` 必须为 `False`，且禁止正式导出。

**代码锚点**：`CacheRestoreWorker`、`cache.load_project_state`、`provenance.verify_stage_record`。

**测试锚点**：`test_cache_restore_rejects_old_mask_when_model_file_changes`、`test_cache_loader_never_reuses_same_shape_mask_from_another_source`、`test_stage_record_rejects_changed_model_or_artifact`。

## DECISION-003：无可靠地理配准时禁止米制处理

**结论**：掩膜正则化、路径规划和地理导出必须建立在有效 CRS、仿射和可接受 GSD 上。

**原因**：履带宽度、转弯半径、覆盖率和碾压率都是米制量，像素假尺度会直接产生错误结论。

**不变量**：

- 无地理配准时 `require_metric_scale()` 必须失败。
- 非有限、超范围、横纵尺度差异过大或图幅内 GSD 变化过大时必须失败。
- 不允许回退到猜测的 `1 m/px` 或 `0.01 m/px`。

**代码锚点**：`require_metric_scale`、`GeoUtils`、`TifLoadWorker`。

**测试锚点**：`test_metric_processing_requires_valid_georeference`、`test_metric_processing_rejects_anisotropic_raster_pixels`、`test_tif_loader_rejects_crs_without_valid_affine_geotransform`。

## DECISION-004：正式推理必须读取原始 GeoTIFF

**结论**：UI 预览图只用于显示；正式识别从原始 GeoTIFF 田块窗口读取。超大田块使用原始分辨率流式切片，不整体缩图。

**原因**：预览降采样或整体缩图会丢失细行带信息，改变分割结果和后续路径。

**不变量**：

- 模型看到的 tile 保持配置的原始像素尺寸，默认 `640 x 640`。
- 大裁剪不得因为 36MP 阈值直接失败，也不得先整体下采样。
- 任一 tile 推理失败时整次推理失败。
- 田块窗口、切片参数和预处理信息进入 provenance。
- 超大 GeoTIFF 的 Qt 渲染纹理最长边限制为 4096；该纹理只用于绘制，逻辑预览尺寸、场景坐标、源仿射和正式推理输入不得随纹理缩小。
- GeoTIFF 已有 overview 时允许 GDAL/Rasterio 用于预览读取；不得改写源 TIFF，也不得用 overview 替代源分辨率推理 tile。

**代码锚点**：`InferenceRunner.start_from_tif`、`InferenceRunner._run_source_tif_tiles`、`InferenceRunner._run_from_tif`、`bounded_render_preview`、`ImageViewer.set_image`。

**测试锚点**：`test_segment_uses_source_geotiff_instead_of_downsampled_preview`、`test_source_tif_inference_streams_oversized_crop_at_source_tile_size`、`test_failed_inference_tile_aborts_instead_of_becoming_background`、`test_preview_shape_bounds_render_pixels_without_changing_source_footprint`、`test_low_resolution_render_keeps_logical_analysis_coordinates`。

## DECISION-005：栅格预处理必须确定且可追溯

**结论**：RGB 波段选择、NoData、位深和归一化必须通过统一模块处理并记录元数据。

**原因**：不同 GeoTIFF 的波段解释和位深不同，隐式转换会让同一模型得到不可比较的输入。

**不变量**：

- 优先依据 `colorinterp` 选择 RGB；缺失时使用明确回退。
- NoData 不得进入有效像素统计。
- 原生 8 位数据保持字节值；非 8 位处理规则必须版本化。
- 非有限元数据写入 provenance 时规范为 `null`。

**代码锚点**：`raster_preprocessing.normalise_raster_bands`、`raster_preprocessing.read_rgb_raster`。

**测试锚点**：`test_uint16_preprocessing_is_per_band_deterministic_and_masks_nodata`、`test_provenance_canonical_json_normalises_nonfinite_metadata`。

## DECISION-006：模型权重必须按信任和哈希管理

**结论**：内置模型必须匹配注册表 SHA-256；外部 `.pt` 必须经过用户信任确认，推理时记录模型哈希。

**原因**：同路径权重可能被替换，且 PyTorch `.pt` 不是纯数据格式。

**不变量**：

- 内置模型哈希不匹配时拒绝加载。
- 外部模型默认不信任。
- 模型内容变化后禁止复用旧推理结果和导出旧路径。
- manifest 必须记录模型哈希和注册信息。

**代码锚点**：`MainWindow._on_load_model_at`、`data/models/model_registry.json`、推理 provenance。

**测试锚点**：`test_builtin_model_registry_matches_present_weight_files`、`test_export_rejects_model_content_change_after_inference`、`test_cache_restore_rejects_old_mask_when_model_file_changes`。

## DECISION-007：科学输入变化必须使下游结果失效

**结论**：模型、掩膜参数、农机参数、调头策略、起终点和卸粮点变化后，不能继续使用旧下游结果。

**原因**：路径由完整输入集合决定；只更新界面参数但保留旧路径会产生显示与计算不一致。

**不变量**：

- 失效顺序固定为 inference -> mask -> path -> simulation/export。
- 重新推理不保留旧掩膜作为成功结果。
- 手工编辑路线后路径状态必须为未验证，禁止正式导出。

**代码锚点**：`MainWindow._invalidate_analysis_from`、`_on_params_dialog`、`_on_turn_strategy_changed`、`_place_entry_exit_point`。

**测试锚点**：`test_input_changes_invalidate_stale_downstream_results`、`test_export_blocks_when_harvester_parameters_changed_after_planning`、`test_export_rejects_service_point_change_after_path_planning`。

## DECISION-008：科研适用范围与指标表述

**结论**：项目定位为科研软件，用于方法验证、路径方案比较和田间试验准备。结果表述应说明数据来源、计算模型和评价条件。

**原因**：GCP/RTK 绝对精度、目标终端格式、真实轨迹 RMSE、作业后碾压和异常工况会影响评价尺度，应作为对应试验的记录项。

**不变量**：

- README 必须说明系统的科研用途和模型计算条件。
- 文档应区分内部一致性指标、影像坐标精度和田间实测指标。
- 农机候选参数应标注其计算依据和适用的行带、机具参数范围。

**文档锚点**：`README.md`、`docs/SCIENTIFIC_VALIDATION.md`、`AI_RULES.md`。

**测试锚点**：`test_readme_preserves_research_prototype_boundary`、`test_machine_parameter_output_is_explicitly_unvalidated_heuristic`。

## DECISION-009：掩膜证据层和规划支撑层必须分离

**结论**：原始残差只有在方向、主体投影和米制邻近关系同时成立时，才能并入 `processed_mask`。无法可靠归属的残差进入 `uncertain_residual_mask`，不得伪装成主作业区或田头。

**原因**：仅凭方向和纵向投影会把田头孤岛、道路边缘或漏检碎片错误提升为主作业区；反过来把所有残差标成田头也会制造错误语义，并污染覆盖率和碾压率评价。

**不变量**：

- 与重构主体不邻近的平行残差不得进入主作业区。
- 内部断苗和端部边界使用不同补缝阈值，端部不得借用 6～9 m 的内部阈值。
- `processed_mask` 用于作物主体和作物重叠指标；`planning_support_mask` 只用于场地支撑和越界判断。
- 不确定残差必须独立显示，不能显示成绿色主体或黄色田头。
- 处理后主体、田头和不确定层作为同一个原子缓存产物保存，JSON 不得承载大掩膜数组。

**代码锚点**：`row_geometry.residual_mask_layers`、`row_geometry.close_band_support_gaps`、`footprint_planner.validate_footprints`、处理后掩膜缓存。

**测试锚点**：`test_detached_parallel_fragment_inside_body_extent_becomes_uncertain`、`test_terminal_band_gap_uses_stricter_threshold_than_internal_gap`、`test_validate_footprints_uses_neutral_support_without_counting_it_as_crop`、`test_semantic_mask_layers_roundtrip_without_entering_json_summary`。

**允许的变更方式**：可通过真实标注样本调整米制邻近阈值和端部阈值，但必须同时报告主体误增面积、主体漏保留面积、田头误分类面积和运行时间。

## DECISION-010：转弯锚点和曲率约束必须物理可解释

**结论**：作物中心线端点是识别证据，不必然是转弯起点。端点必须按同侧端点分布独立判断；补偿距离受限且补偿段完全位于田块内时，才允许建立独立 `turn_approach`。任何半圆半径不得小于农机最小转弯半径。

**原因**：单行识别残缺会让蛇形连接在某一处提前转弯；而以 `ω/2 < R` 的半圆强行连接虽然视觉连续，真实农机无法执行。边界内直线连接也不能替代满足曲率约束的转弯。

**不变量**：

- 原 `work` 段保持作物证据范围，端点补偿单独标记为 `turn_approach`。
- 少于三条作业线、没有田块边界、补偿超限或补偿段越界时禁止自动校正。
- `ω < 2R` 时禁止选择或生成普通半圆，前进模式使用满足半径的梨形，受限场景使用明确含倒车动作的策略。
- 转弯曲线退化为端点直线时必须产生运动学硬错误，并使候选路径验证失败。
- 端点校正数量、补偿距离、阻止原因、实际策略、`ω` 和 `R` 必须进入规划诊断。

**代码锚点**：`path_planner._align_short_turn_anchors`、`path_planner._select_turn_strategy`、`path_planner._turn_semicircle`、`path_planner._connect_work_lines`。

**测试锚点**：`test_single_short_line_uses_separate_headland_approach_before_turn`、`test_endpoint_alignment_refuses_target_outside_field_boundary`、`test_spacing_below_two_radius_never_selects_semicircle`、`test_boundary_direct_fallback_is_reported_as_kinematic_failure`。

**允许的变更方式**：阈值只能依据带人工核验端点的真实样本调整；至少报告提前转弯检出率、误校正率、最小曲率、越界率和路线覆盖率。

## DECISION-011：作业线模式、语义足迹与机器执行门禁

**结论**：田块硬边界、语义支撑区、不确定区和人工确认禁行区必须分别统计，不得互相替代。通用地理导出不是机器执行协议。

**不变量**：

- `work_line_mode=band_centerline` 保持现有中心线流程；`band_centerline_with_footprint_validation` 也保留四行带中心线作为路径参考，足迹模型用于验证和风险评分，不把作业线横向推到窄水道。旧配置键 `footprint_optimized` 仅作为兼容别名读取。
- 用户界面不暴露“快速预览、田间试验、高级校验”等内部模式；默认按田间试验要求执行，内部配置细节不要求普通用户理解。
- 验证等级分为 `research`、`field_trial` 和 `machine_candidate`；结果界面只显示路径状态、作业线模式和关键风险，内部阻断项放入详情提示。
- 未确认的不确定残差不得自动升级为禁行区；提供禁行区时，履带重叠默认硬门限为 0%。
- 整机包络阈值由 `machine_candidate` 验证配置固定为 0%，不提供普通用户可编辑的整机门限。
- 禁行区必须来自用户确认多边形，并进入项目缓存、路径 provenance、足迹验证和导出 manifest；任何修改必须使旧路径失效。
- GeoJSON、CSV、KML、JSON 和 `$PATH` 均记录坐标基准、车辆参考点和规划参数，便于与目标终端格式进行核对。
- 机器执行就绪评估基于外部定位精度、车辆参考几何、运动学、终端适配和实车跟踪等记录项计算；记录不完整时，界面应明确显示待补充项。
- 足迹验证必须记录 `planar_rigid_footprint_v2` 和参考点 `track_contact_center`。演示与验证共用左右履带、机身和割刀线；履带碾压使用有限接地长度的有向矩形扫掠，作物掩膜几何覆盖使用具有实测前置距离的有效割幅线段扫掠。
- 二维刚体模型用于计算平面包络、履带扫掠和割幅覆盖；轴距影响机身与割台前后位置，履带接地长度独立决定履带接地矩形，避免重复计量。车辆动力学、土壤压实和 CAD 外廓可作为后续扩展模型。
- 进入候选软排序前，正式足迹验证必须显式给出 `valid=True`、空 `issues` 和完整硬门禁指标；缺少有效性证据、必需指标、对应语义层指标或任一门限失败均须 fail-closed，并进入候选拒绝诊断。

**代码锚点**：`path_planner._prepare_work_line_layout`、`harvester_geometry.planar_geometry_m`、`footprint_planner.validate_footprints`、`footprint_planner.rasterize_forbidden_regions`、`machine_route_validator.assess_machine_readiness`、`RouteInfoPanel`。

**测试锚点**：`test_planar_geometry_uses_the_same_configured_dimensions_as_animation`、`test_track_contact_length_changes_terminal_rolling_area`、`test_validation_identifies_rigid_planar_footprint_model`。

## DECISION-012：碾压目标、路径姿态与结果状态必须形成单一数学链

**结论**：预计碾压只定义为两条履带有限矩形沿路径扫掠后与全部已确认作物像素（主体与田头）的交集比例。机身不计入碾压，割刀线只计算主体作物掩膜几何覆盖。验证和演示共用唯一的分段路径姿态采样器；路径排序中的田头农业价值差异由 DECISION-013 的独立代价表达，不得改写物理碾压率。

**不变量**：

- 不得使用机身或割台面积增加或替代履带碾压率。
- 验证口径中的割刀只建模为横向有效割幅线段。`cutter_forward_offset_m` 必须实测；禁止由轴距、机身长度或固定比例推算割台纵向深度。UI 可绘制 `cutter_visual` 作为显示用割台面板，但不得把它计入覆盖率、碾压率或足迹校验。
- “作物掩膜几何覆盖率”不得表述为实际收获率。覆盖结果只说明二维割刀扫掠是否掠过对应掩膜。
- `rolling_crop_pct` 废止；界面、报告和导出只把 `rolling_canopy_pct` 称为预计碾压率。
- `rolling_canopy_pct` 必须使用 `processed_mask | headland_mask` 的去重并集；主体和田头碾压同时单列，任何权重不得进入该物理比例。
- 自动规划完成后，演示状态中的覆盖、碾压、效率和各类路径长度必须来自同一次 `validation`，不得读取初始值或历史值。
- `path_pose` 是位置插值、航向窗口和倒车方向的唯一实现；验证和演示不得各自计算航向。
- 缺少有效米制尺度或完整农机几何字段时失败关闭，不得使用 `0.01 m/px` 或模块私有默认尺寸。
- 作业、转弯和进出场速度是明确农机配置；播放倍率仅用于演示控制。
- 受控试验就绪检查必须核验足迹模型版本与内部路径参考点。

**代码锚点**：`path_pose.py`、`path_planner._rank_path_candidates`、`footprint_planner.validate_footprints`、`workflow.AnimationEngine`、`MainWindow._store_path_result`、`machine_route_validator.assess_machine_readiness`。

**测试锚点**：`tests/test_cross_module_math.py`、`tests/test_animation_controls.py`。

## DECISION-013：田头碾压只作报告，不参与转弯策略选择

**结论**：主作业区作物、田头作物、未确认无掩膜区和人工禁行区必须保持不同语义。转弯策略仍由运动学、田块边界、禁行区、自交、主体/核心区碾压、覆盖率和路径效率决定；`rolling_headland_pct` 与 `agronomic_damage_cost_m2` 仅作报告和论文对照，不进入局部转弯候选或全局策略排序。

**原因**：转弯策略用于指导农机执行，必须首先满足运动学和安全约束。当前课题没有田头碾压对转弯策略影响的实测标定依据，因此不能让田头作物分布或田头权重改变转弯几何；田头碾压仍保留为可审计统计指标。

**不变量**：

- `headland_damage_weight` 必须在 `(0, 1]` 内，默认 `1.0`；低于 `1.0` 只允许依据田间对照产量标定后使用。
- 调整田头权重不得改变主体、田头、全田作物的真实碾压面积和 `rolling_canopy_pct`。
- `agronomic_damage_cost_m2` 是报告/实验对照指标，不参与转弯策略排序，也不得称为碾压率。
- `headland_mask` 只能使用与主体掩膜同尺度的确认作物证据；重叠像素必须去重，不得重复计量。
- 无掩膜像素不得自动认定为无作物道路；未确认区域继续接受语义支撑、禁行区和人工复核约束。
- 实际执行配置与路径 provenance 中记录的配置必须一致，田头权重变化必须使路径阶段指纹变化。

**代码锚点**：`footprint_planner.validate_footprints`、`path_planner._rank_path_candidates`、`pyside6_app.workers.PlanWorker`。

**测试锚点**：`tests/test_cross_module_math.py`。

**允许的变更方式**：未来只有在获得田头/主体碾压对照试验并明确要求将其纳入决策后，才可重新讨论田头代价是否参与排序；届时必须报告样本、估计方法、置信区间和安全约束优先级。

## DECISION-014 论文实验必须冻结输入并保持单变量可比

- 结论：论文实验从当前项目读取源影像身份、原始掩膜、地理尺度、田界、农机参数和服务点的冻结快照，在隔离状态中运行；掩膜组只改变处理阶段，路径组统一使用完整处理掩膜并只改变请求调头策略。
- 原因：批量功能如果同时改变模型、掩膜、参数或田界，生成的表格不能解释为模块贡献；后台实验若写回当前项目，还会破坏现有科研链路证据。
- 不变量：
  1. 实验运行不得修改当前 `AppState` 的模型、掩膜、路径或策略。
  2. 原始掩膜、行带几何重构、主体残差恢复和完整规则化组必须来自同一次正式掩膜处理及其真实中间证据层，不得用伪参数或合成结果冒充消融。
  3. 路径策略组必须共享同一完整处理掩膜、田界、服务点、禁行区和农机参数；每组记录请求策略与实际选择策略。
  4. IoU、Dice、Precision、Recall 仅在提供同尺寸人工真值掩膜时生成；没有真值时保持缺失，不得用原始模型输出充当真值。
  5. 单个实验组失败必须保留失败原因并继续其余组；输出必须包含输入摘要、组参数、长表、宽表、逐组结果和效果图。
  6. 当前入口不具备隔离原始 GeoTIFF 多模型推理时，不得把同一原始掩膜复制并标记为不同模型结果。
- 代码锚点：`paper_experiments.py`、`pyside6_app/workers.py`、`pyside6_app/main_window.py`。
- 测试锚点：`tests/test_paper_experiments.py`。
- 允许变更方式：可增加新的真实中间层、模型隔离推理和统计图，但必须保持固定输入、人工真值门槛和失败记录，并同步更新实验 schema 与测试。

## 修改决策的方式

决策不是永久禁止演进。需要改变时，必须在同一个变更中：

1. 说明旧决策为什么不再成立及新增证据。
2. 更新本文件中的结论和不变量。
3. 更新所有相关代码锚点。
4. 修改或新增行为测试。
5. 在交付说明中列出风险变化和仍未验证条件。
## DECISION-015：分段足迹归因和重复访问必须可求和核对

- 结论：路径验证以同一 `plan_pose` 和 `planar_rigid_footprint_v2` 产生整型履带访问计数图，并对 `work`、`turn_approach`、`turn`、`turn_reverse`、`turn_aux`、`entry`、`exit` 等实际段分别记录长度和足迹归因。
- 原因：整条路径单一统计无法审计田头转弯代价，也不能把二值并集误称为重复碾压。
- 不变量：
  1. `segment_metrics` 使用 `exclusive_first_segment` 规则；分段独占履带并集和为全路径 `track_swept_area_m2`，重复访问另由 `uint32` 访问计数图统计。
  2. 转弯层单独报告 `headland_turn_track_overlap_area_m2`、`headland_turn_rolling_pct`、主体/不确定区/禁行区重叠、田界越界和 `turn_cost_m2`；田头仍是软代价，不是硬障碍。
  3. 候选搜索只在端点保持、策略和最小半径不变的有限走廊内进行；田界、禁行区、运动学不可行、无有效米制尺度和退化直线均 fail-closed。没有安全改进候选时保留原候选并记录 `fallback_reason`。
  4. 一次与两次及以上履带访问面积是二维几何模型指标，不等于土壤压实、产量损失、实机可执行结果或田间验证结果。
  5. 候选必须通过与正式路径一致的足迹硬门禁：始终要求核心区重叠、规划目标覆盖和田界越界指标，并在提供支撑区、不确定区或禁行区时要求对应指标；`valid`、`issues` 或指标缺失不得默认放行。
- 代码锚点：`footprint_planner.validate_footprints`、`path_planner._connect_work_lines`、`headland_turn_optimizer.select_turn_candidate`。
- 测试锚点：`tests/test_headland_turn.py`、`tests/test_headland_turn_contract.py`、`tests/test_headland_turn_integration.py`。
- 允许变更方式：可增加有实测依据的足迹模型或标定代价，但必须保留整路径/分段核对、访问计数类型、硬约束诊断和实验输入冻结。

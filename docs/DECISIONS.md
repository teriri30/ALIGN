# reUI 关键设计决策

本文件记录不能被随意删除、绕过或以“简化”为理由弱化的系统级决策。修改决策相关代码时，必须同步核对代码锚点和不变量测试。

## DECISION-001：路径导出必须保持数据一致性，安全指标不作为研究输出硬门禁

**结论**：GeoJSON、CSV、KML、JSON 和 `$PATH` 必须先经过 `_geo_export_payload()`；输出成功后必须生成 manifest。研究作品默认允许导出，理论履带重叠率、覆盖率、碾压率和机器就绪证据只作为结果字段保存，不再阻止研究输出。桌面默认 `footprint_metric_policy=report_only`：碾压率、核心区重叠和规划覆盖率继续计算并显示，但不与固定阈值比较；只有显式严格流程才允许启用 `hard_gate`。`fail_closed` 仅用于严格 CI 或明确的受控试验流程。

**原因**：路径可能进入受控田间试验。可用性不能优先于坐标、参数和路径安全。

**兼容测试文件边界**：`machine_harvester_straight_turn_v1` 是给目标终端做格式和几何试跑的文本投影。它仍必须经过 `_geo_export_payload(formal=False)` 的地理配准、当前路径、provenance、有限坐标、点序和压缩后线段格式检查；研究指标、参数来源、机器就绪、恢复状态和终端协议不再弹窗或阻断。manifest 必须保留 `preview_only=true`、`machine_executable=false` 和 `COMPATIBILITY_TEST_ONLY_MANUAL_REVIEW`，不能被误称为田间试验就绪证据。

**不变量**：

- 无有效地理配准、源影像已变化、provenance 无法重算、坐标非有限或点序/状态码不一致时禁止导出。
- 农机参数、调头策略与规划时不一致时保留在 manifest；严格 `fail_closed` 模式下才禁止导出。
- 路径点超出影像范围时禁止仿射外推；研究模式不以理论履带指标替代坐标一致性检查。
- manifest 写入失败时不得保留正式路径文件。
- 农机兼容 TXT 写入后必须回读 manifest 和全部分卷，核对文件 SHA-256、点数、索引、`total`、固定状态码及 `$INFM`；校验失败不得报告导出成功或保留本次新生成的文件。再次切换到“TXT 路径”视图时也必须复核，发现文件被外部替换则停止显示旧预览。
- 旧 `ExportEngine` 不得成为旁路。

**代码锚点**：`MainWindow._geo_export_payload`、`MainWindow._write_export_manifest`、`MainWindow._verify_machine_export_manifest`、`MainWindow._machine_route_preview_integrity_errors`、`integrity_check.verify_manifest`。

**测试锚点**：`test_invalid_or_manually_edited_path_is_blocked_from_export`、`test_offline_manifest_checker_detects_route_tampering`、`test_offline_manifest_checker_checks_every_machine_file_and_point_count`、`test_machine_compatibility_export_fails_closed_on_manifest_mismatch`、`test_machine_route_preview_rechecks_manifest_before_display`、`test_legacy_export_engine_is_disabled_to_prevent_bypassing_export_gate`、`test_legacy_export_engine_is_fail_closed`。

## DECISION-002：缓存恢复必须验证完整阶段证据

**结论**：缓存必须绑定源影像、模型、配置、上游指纹和产物哈希；缺失或不一致时逐级失效。

**原因**：同名模型、同尺寸影像或同一路径不代表内容相同，错误恢复会污染全部下游结果。

**不变量**：

- 模型内容变化后旧推理掩膜失效。
- 历史缓存模型与当前加载模型不一致时，只禁止复用历史掩膜；当前模型的路径、哈希与实际加载权重一致时，允许直接开始全新推理，不要求重开项目或改回历史模型。
- 不按图像尺寸跨源复用缓存。
- 推理失效必须使掩膜和路径失效；掩膜失效必须使路径失效。
- 旧模式缓存失败关闭，不猜测兼容。
- 恢复的路径几何和指标必须标记为历史快照；在当前输入与当前验证规则下重新执行完整路径验证前，`auto_path_valid` 必须为 `False`。默认研究导出保留该状态并写入 manifest，不弹窗、不阻断；只有显式 `fail_closed` 模式才拒绝导出。

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

- 正式语义掩膜的主推理 tile 保持配置的原始像素尺寸，默认 `640 x 640`。当源 GSD 使 640 px 不足以覆盖一个完整四行带时，允许增加一个显式记录的 `1280→640` 上下文辅助通道；它只能恢复主掩膜邻近弱证据和提取相邻实例接触边界，不得替代原生分辨率主推理。
- 大裁剪不得因为 36MP 阈值直接失败，也不得先整体下采样。
- 任一 tile 推理失败时整次推理失败。
- 田块窗口、切片参数和预处理信息进入 provenance。
- 超大 GeoTIFF 的 Qt 渲染纹理最长边限制为 4096；该纹理只用于绘制，逻辑预览尺寸、场景坐标、源仿射和正式推理输入不得随纹理缩小。
- GeoTIFF 已有 overview 时允许 GDAL/Rasterio 用于预览读取；不得改写源 TIFF，也不得用 overview 替代源分辨率推理 tile。
- 画布“对比”固定切换“不可变原始影像”和“当前应用状态的可视结果”；不得把当前步骤与前一步中间产物冒充原图对比。
- 对比切层前后必须保持相同图像坐标中心、缩放倍率和旋转角；不得因底图像素尺寸或图层切换调用自动适配全图。

**代码锚点**：`InferenceRunner.start_from_tif`、`InferenceRunner._run_source_tif_tiles`、`_fuse_context_evidence`、`InferenceRunner._run_from_tif`、`bounded_render_preview`、`ImageViewer.set_image`、`MainWindow._on_toggle_mask_compare`。

**测试锚点**：`test_segment_uses_source_geotiff_instead_of_downsampled_preview`、`test_source_tif_inference_streams_oversized_crop_at_source_tile_size`、`test_context_fusion_recovers_nearby_support_and_preserves_long_separator`、`test_failed_inference_tile_aborts_instead_of_becoming_background`、`test_preview_shape_bounds_render_pixels_without_changing_source_footprint`、`test_low_resolution_render_keeps_logical_analysis_coordinates`、`test_canvas_compare_toggles_source_image_and_current_result`、`test_canvas_compare_preserves_zoom_rotation_and_image_location`。

## DECISION-005：栅格预处理必须确定且可追溯

**结论**：RGB 波段选择、NoData、位深和归一化必须通过统一模块处理并记录元数据。

**原因**：不同 GeoTIFF 的波段解释和位深不同，隐式转换会让同一模型得到不可比较的输入。

**不变量**：

- 优先依据 `colorinterp` 选择 RGB；缺失时使用明确回退。
- `read_rgb_raster` 返回数组已经是 RGB，reUI 内部和切片融合过程始终保持 RGB。
- 当前 Ultralytics 8.x 将 `numpy.ndarray` 解释为 OpenCV BGR，并在 `BasePredictor.preprocess` 内转换为 RGB；因此只能由 `ModelEngine.predict` 在外部库边界执行一次 RGB→BGR 适配，其他推理入口不得自行交换通道。
- NoData 不得进入有效像素统计。
- 原生 8 位数据保持字节值；非 8 位处理规则必须版本化。
- 非有限元数据写入 provenance 时规范为 `null`。

**代码锚点**：`raster_preprocessing.normalise_raster_bands`、`raster_preprocessing.read_rgb_raster`、`ModelEngine.predict`。

**测试锚点**：`test_uint16_preprocessing_is_per_band_deterministic_and_masks_nodata`、`test_provenance_canonical_json_normalises_nonfinite_metadata`、`test_model_engine_adapts_rgb_to_ultralytics_bgr_numpy_contract`、`test_tiled_inference_keeps_rgb_channel_order`、`test_source_tif_inference_keeps_rgb_channel_order`。

## DECISION-006：模型权重必须按信任和哈希管理

**结论**：内置模型必须匹配注册表 SHA-256；外部 `.pt` 必须经过用户信任确认，推理时记录模型哈希。

**原因**：同路径权重可能被替换，且 PyTorch `.pt` 不是纯数据格式。

**不变量**：

- 内置模型哈希不匹配时拒绝加载。
- 外部模型默认不信任。
- 模型内容变化后禁止复用旧推理结果和导出旧路径。
- 已注册或经用户确认信任的新模型可替换历史缓存模型并重新推理；模型切换必须先逐级清除旧推理、掩膜和路径，但不得把历史模型身份误作新推理门禁。
- manifest 必须记录模型哈希和注册信息。

**代码锚点**：`MainWindow._on_load_model_at`、`data/models/model_registry.json`、推理 provenance。

**测试锚点**：`test_builtin_model_registry_matches_present_weight_files`、`test_export_rejects_model_content_change_after_inference`、`test_cache_restore_rejects_old_mask_when_model_file_changes`。

## DECISION-007：科学输入变化必须使下游结果失效

**结论**：模型、掩膜参数、农机参数、调头策略、起终点和卸粮点变化后，不能继续使用旧下游结果。

**原因**：路径由完整输入集合决定；只更新界面参数但保留旧路径会产生显示与计算不一致。

**不变量**：

- 失效顺序固定为 inference -> mask -> path -> simulation/export。
- 重新推理不保留旧掩膜作为成功结果。
- 手工编辑路线后路径状态必须为未验证。默认研究导出保留该状态并写入 manifest，不弹窗、不阻断；只有显式 `fail_closed` 模式才拒绝导出。
- 系统自动推荐必选起点；终点标记默认沿最终作业段生成短延伸并写入共享控制折线和农机兼容 TXT，用户也可在影像上直接拖动该终点改为显式目标。起点、终点和卸粮点拖动受已确认田界约束，一次按下—移动—释放只形成一个撤销事务，并在释放时使旧路径及其验证/导出证据失效。
- 起点和终点拖动期间只受已确认田界约束，两个坐标轴均跟随鼠标，不使用主航线轴、田头深度或田头走廊把点吸附到一条线；连续 canonical 路径保留真实作业/转弯几何，入口连接和最终终点延伸由共享控制折线写入兼容 TXT。释放后旧路径立即失效；必选起点和其他任务条件完整时短延迟自动启动正式路径规划和完整足迹验证。
- 路线编辑中的起点和终点与 `state.path_points` 的首、末控制节点共享同一显示级别和状态颜色；删除首/末节点后，对应服务点自动跟随新的首/末节点。未显式设置终点时仍保留自动终点语义，只从当前共享控制折线的末节点显示和导出。
- 释放后的短延迟期间允许保留一份显式标记为“待重新规划预览（不可导出）”的旧路线，仅供连续交互观察；canonical `auto_path`、验证、仿真和导出状态仍为空/失效，新规划提交后预览必须被清除。
- 项目缓存恢复 `turn_strategy` 后，路径规划卡片和其他可见策略选择器必须无信号回写地同步到同一个 `AppState.turn_strategy`；不得让界面显示“套圈收割”而后台继续按缓存中的 `bow` 生成相邻行往返路线。
- 调头方式选择器关闭时必须忽略滚轮，只有用户明确展开列表并选择或通过显式策略入口确认时，才设置 `turn_strategy_user_selected=true`。缺少该标记的旧缓存不能用非 `contour` 值覆盖“套圈收割”默认策略；与该旧值绑定的缓存路径同时失效，避免用新标签解释旧几何。

**代码锚点**：`ImageViewer.set_direct_drag_hit_test`、`MainWindow._invalidate_analysis_from`、`MainWindow._restore_cache_payload`、`_on_params_dialog`、`_on_turn_strategy_changed`、`_place_entry_exit_point`、`_begin_service_point_drag`、`_constrain_service_point`、`_draw_service_point_drag_preview`、`_finish_service_point_drag`。

**测试锚点**：`test_input_changes_invalidate_stale_downstream_results`、`test_export_blocks_when_harvester_parameters_changed_after_planning`、`test_export_rejects_service_point_change_after_path_planning`、`test_recommended_service_point_can_be_dragged_directly_and_invalidates_path`、`test_service_point_drag_preserves_free_2d_position_and_schedules_replan`、`test_viewer_captures_service_point_drag_before_left_button_pan`、`test_cache_restore_keeps_visible_turn_strategy_equal_to_restored_state`。

## DECISION-008：科研适用范围与指标表述

**结论**：项目定位为面向无人自主收获收割机的科研系统，用于从影像感知、作业带确定、中心线与路径生成，到导航终端导入、田间自主作业试验和结果记录。它不是成熟生产系统；终端导入和田间执行仍需按数据来源、终端协议、机器参数和安全条件核验。结果表述应说明数据来源、计算模型和评价条件。

**原因**：GCP/RTK 绝对精度、目标终端格式、真实轨迹 RMSE、作业后碾压和异常工况会影响评价尺度，应作为对应试验的记录项。

**不变量**：

- README 必须说明系统的科研用途和模型计算条件。
- 文档应区分内部一致性指标、影像坐标精度和田间实测指标。
- 农机候选参数应标注其计算依据和适用的行带、机具参数范围。

**文档锚点**：`README.md`、`docs/SCIENTIFIC_VALIDATION.md`、`AI_RULES.md`。

**测试锚点**：`test_readme_preserves_research_use_and_model_conditions`、`test_machine_parameter_output_states_concise_calculation_basis`。

## DECISION-009：掩膜证据层和规划支撑层必须分离

**结论**：原始残差只有在方向、主体投影和米制邻近关系同时成立时，才能并入 `processed_mask`。无法可靠归属的残差进入 `uncertain_residual_mask`，不得伪装成主作业区或田头。

**原因**：仅凭方向和纵向投影会把田头孤岛、道路边缘或漏检碎片错误提升为主作业区；反过来把所有残差标成田头也会制造错误语义，并污染覆盖率和碾压率评价。

**不变量**：

- 与重构主体不邻近的平行残差不得进入主作业区。
- 内部断苗和端部边界使用不同补缝阈值，端部不得借用 6～9 m 的内部阈值。
- `processed_mask` 用于作物主体和作物重叠指标；`planning_support_mask` 只用于场地支撑和越界判断。
- 不确定残差必须独立显示，不能显示成绿色主体或黄色田头。
- 田头与纵向主体粘连、导致残差方向失真时，只允许在规则化纵向行带的轴向端部，用跨越足够多行的高占用横向连续带补充田头种子；不得使用已合并残差后的外包络作为端部参照，也不得把稀疏纵向行尾提升为田头。
- 已确认的端部田头走廊必须沿规则化纵向主体的逐列端点墙平滑展开，并与原始作物支持逐像素相交；不得用矩形外包络填平真实弯曲边缘或凭空补出无作物区域。
- 端点墙只能由轴向跨度达到全田参考跨度显式比例的横截行建立；区域分界、断行或孤立短作业片段不得贡献端点，缺失横截行只在可信长跨度端点之间插值。诊断必须记录参考跨度、可信行数和被拒短行数。
- 达到跨度门限但明显偏离共同端墙的局部端点，必须经分箱 Theil-Sen 趋势和米制偏差门限剔除后再插值；田头候选及其原始种子只能保留在最终稳健端墙外侧。存在规则化纵向主体作为端墙证据时，不得再向主体方向叠加端点余量。
- 统一轴向重构区间把可信端点量化成直墙时，端墙斜率只能由田块多边形中跨越足够行宽、位于对应物理田端且斜率有限的边恢复；同时必须向田头侧补回 `band_end_trim_m` 的重构裁边。田块侧边、短倒角和未量化的真实端点趋势不得触发该替代。
- 行向短缺口补缝必须先把掩膜旋转到行向坐标系再执行闭运算；不得在原尺寸的细长核画布中旋转并裁断核。补出的像素仍须同时落在规则化主体内，不能形成端部单侧外伸。
- 横向剖面相邻行带的局部占用门控必须按相邻 profile run 的边界中点划分，并记录 `profile_gate_basis=run_edge_midpoints`；不得用相邻中心点中点为宽等距块旁的四行带建立门控，否则宽块尾部会把真实四行中心线中位数拉偏。
- YOLO 相邻实例在进入语义并集前必须保留接触边界证据。上下文切片边界只有在跨切片重复支持、长度达到米制阈值且方向与主作业方向一致时才能从语义掩膜中切开；普通实例外轮廓、切片边缘和短边界不得作为分带依据。
- 田块端边斜率恢复与重构裁边补偿自掩膜规则化算法版本 17 生效；版本 16 及更早的处理缓存不得恢复为当前掩膜证据，必须从同一原始推理掩膜重新执行掩膜处理后再规划。
- 田头分类必须先对规则化主体做连通区域分区，再按区域的轴向跨度、端部邻接和跨行支持率分类；只有被可信端部区域接受的残差种子才能进入 `headland_mask`。被拒的内部区域种子不得因“曾经是方向残差”而继续显示为黄色田头。
- 处理后主体、田头和不确定层作为同一个原子缓存产物保存，JSON 不得承载大掩膜数组。

**代码锚点**：`_contact_boundary_from_instances`、`_fuse_context_evidence`、`row_geometry.regularize_crop_mask`、`row_geometry._profile_run_gates`、`row_geometry.residual_mask_layers`、`row_geometry.close_band_support_gaps`、`row_geometry.resolve_terminal_headland_overlap`、`footprint_planner.validate_footprints`、处理后掩膜缓存。

**测试锚点**：`test_contact_boundary_keeps_only_evidence_between_two_instances`、`test_context_fusion_recovers_nearby_support_and_preserves_long_separator`、`test_profile_gate_edges_isolate_band_next_to_broad_uniform_block`、`test_detached_parallel_fragment_inside_body_extent_becomes_uncertain`、`test_terminal_band_gap_uses_stricter_threshold_than_internal_gap`、`test_repair_row_gaps_closes_short_breaks_at_oblique_row_angle`、`test_headland_corridor_recovers_dense_second_terminal_without_angle_seed`、`test_headland_dense_seed_does_not_relabel_parallel_row_tails`、`test_terminal_headland_follows_curved_crop_support_not_rectangle`、`test_terminal_headland_wall_ignores_short_internal_work_fragment`、`test_terminal_headland_wall_rejects_long_span_inward_endpoint_wedge`、`test_terminal_headland_restores_quantized_wall_from_field_end_edge`、`test_terminal_headland_classifies_regions_before_exposing_seed`、`test_validate_footprints_uses_neutral_support_without_counting_it_as_crop`、`test_semantic_mask_layers_roundtrip_without_entering_json_summary`。

**允许的变更方式**：可通过真实标注样本调整米制邻近阈值和端部阈值，但必须同时报告主体误增面积、主体漏保留面积、田头误分类面积和运行时间。

## DECISION-010：转弯锚点和曲率约束必须物理可解释

**结论**：作物中心线端点是识别证据，不必然是转弯起点。端点必须按同侧端点分布独立判断；补偿距离受限且补偿段完全位于田块内时，才允许建立独立 `turn_approach`。任何半圆半径不得小于农机最小转弯半径。

**原因**：单行识别残缺会让蛇形连接在某一处提前转弯；而以 `ω/2 < R` 的半圆强行连接虽然视觉连续，真实农机无法执行。边界内直线连接也不能替代满足曲率约束的转弯。

**不变量**：

- 原 `work` 段保持作物证据范围，端点补偿单独标记为 `turn_approach`。
- 少于三条作业线、没有田块边界、补偿超限或补偿段越界时禁止自动校正。
- `ω < 2R` 时禁止选择或生成普通半圆，前进模式使用满足半径的梨形，受限场景使用明确含倒车动作的策略。
- 转弯曲线退化为端点直线时必须产生运动学硬错误，并使候选路径验证失败。
- 弓形、半圆、梨形或套圈候选因田界不足而无法形成边界内曲线时，连续路径必须保留原候选曲线并记录越界段供人工调整；不得为了画面看似在田界内而替换成两点直连。仅当几何生成器本身实际只返回两点时才建立 `display_only_connector`，两者都不得伪装成机器可执行。
- 端点校正数量、补偿距离、阻止原因、实际策略、`ω` 和 `R` 必须进入规划诊断。

**代码锚点**：`path_planner._align_short_turn_anchors`、`path_planner._select_turn_strategy`、`path_planner._turn_semicircle`、`path_planner._connect_work_lines`。

**测试锚点**：`test_single_short_line_uses_separate_headland_approach_before_turn`、`test_endpoint_alignment_refuses_target_outside_field_boundary`、`test_spacing_below_two_radius_never_selects_semicircle`、`test_boundary_direct_fallback_is_reported_as_kinematic_failure`。

**允许的变更方式**：阈值只能依据带人工核验端点的真实样本调整；至少报告提前转弯检出率、误校正率、最小曲率、越界率和路线覆盖率。

## DECISION-011：作业线模式、语义足迹与研究导出边界

**结论**：田块硬边界、语义支撑区、不确定区和人工确认禁行区必须分别统计，不得互相替代。通用地理导出不是机器执行协议；面向无人自主收获收割机的田间试验还必须核对终端导入、实际自主执行和作业后调查。

**不变量**：

- `work_line_mode=band_centerline` 保持原中心线流程；`band_centerline_with_footprint_validation`保留四行带中心线并只进行足迹验证和风险评分；`footprint_optimized`先筛选、平滑并排序四行带中心线，再在每条基准线两侧的有限横向走廊内，以履带核心重叠、冠层重叠、割台覆盖、场地支撑、确认禁行区、偏移幅度和连续平滑性联合选择实际作业线。优化不得改变基准作业线数量与顺序，横移不得超过`lateral_search_m`；核心重叠始终记录，但默认 `track_core_overlap_policy=report_only`，不单独使路径无效或阻断导出。`footprint_metric_policy=report_only`时，规划覆盖率和碾压相关指标只保留为结果字段，不参与默认有效性比较；田界、支撑区、不确定区、禁行区和其他坐标完整性仍按正式验证处理。任一作业线没有完整有限候选、平滑后进入禁行候选、关键指标缺失/非有限/其他硬条件失败时，整组回退为`band_centerline_with_footprint_validation`并记录原因。
- 用户界面不暴露“快速预览、田间试验、高级校验”等内部模式；默认按田间试验要求执行，内部配置细节不要求普通用户理解。
- 验证等级分为 `research`、`field_trial` 和 `machine_candidate`；默认结果界面只显示路径校验状态和必要问题提示，不显示试验记录、机器就绪或参数来源门禁。
- 未确认的不确定残差不得自动升级为禁行区；提供禁行区时，履带重叠默认硬门限为 0%。
- 整机包络阈值由 `machine_candidate` 验证配置固定为 0%，不提供普通用户可编辑的整机门限。
- 禁行区必须来自用户确认多边形，并进入项目缓存、路径 provenance、足迹验证和导出 manifest；任何修改必须使旧路径失效。
- GeoJSON、CSV、KML、JSON 和 `$PATH` 均记录坐标基准、车辆参考点和规划参数，便于与目标终端格式进行核对。
- 机器执行就绪评估可作为内部研究记录保存，但不是默认界面状态，也不是科研路径导出的条件。
- 农机几何参数必须逐字段携带来源元数据；`default`、`recommended`、`user_entered` 和历史未记录来源不得静默写成实测值。`field_trial`/`machine_candidate` 缺少完整的显式 `user_measured` 来源时，路径结果必须为 `machine_executable=false`，但研究导出仍可保留原始来源信息。
- 兼容测试可以在上述参数来源、机器就绪和终端协议记录不完整时生成 `$PATH` 文本；它不得提升 `machine_executable`，但不再通过警告或理论指标阻断研究作品导出。
- 足迹验证必须记录 `planar_rigid_footprint_v2` 和参考点 `track_contact_center`。演示与验证共用左右履带、机身和割刀线；履带碾压使用有限接地长度的有向矩形扫掠，作物掩膜几何覆盖使用具有实测前置距离的有效割幅线段扫掠。
- 二维刚体模型用于计算平面包络、履带扫掠和割幅覆盖；轴距影响机身与割台前后位置，履带接地长度独立决定履带接地矩形，避免重复计量。车辆动力学、土壤压实和 CAD 外廓可作为后续扩展模型。
- 进入候选软排序前，正式足迹验证必须显式给出有效性状态和完整指标；缺少有效性证据、必需指标、对应语义层指标或真实几何完整性失败仍须 fail-closed，并进入候选拒绝诊断。默认 `footprint_metric_policy=report_only` 时，碾压率、核心区重叠和规划覆盖率低于配置值不再单独造成无效；显式 `hard_gate` 才恢复相应阈值比较。

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

**代码锚点**：`path_pose.py`、`path_planner._select_turn_strategy`、`footprint_planner.validate_footprints`、`workflow.AnimationEngine`、`MainWindow._store_path_result`、`machine_route_validator.assess_machine_readiness`。

**测试锚点**：`tests/test_cross_module_math.py`、`tests/test_animation_controls.py`。

## DECISION-013：田头碾压只作报告，不参与转弯策略选择

**结论**：主作业区作物、田头作物、未确认无掩膜区和人工禁行区必须保持不同语义。转弯几何按用户选定策略或明确的单步几何规则生成，再由运动学、田块边界、禁行区和足迹验证检查；当前不做转弯候选排序或路径优化。`rolling_headland_pct` 与 `agronomic_damage_cost_m2` 仅作报告和论文对照。

**原因**：转弯策略用于指导农机执行，必须首先满足运动学和安全约束。当前课题没有田头碾压对转弯策略影响的实测标定依据，因此不能让田头作物分布或田头权重改变转弯几何；田头碾压仍保留为可审计统计指标。

**不变量**：

- `headland_damage_weight` 必须在 `(0, 1]` 内，默认 `1.0`；低于 `1.0` 只允许依据田间对照产量标定后使用。
- 调整田头权重不得改变主体、田头、全田作物的真实碾压面积和 `rolling_canopy_pct`。
- `agronomic_damage_cost_m2` 是报告/实验对照指标，不参与转弯策略选择，也不得称为碾压率。
- `headland_mask` 只能使用与主体掩膜同尺度的确认作物证据；重叠像素必须去重，不得重复计量。
- 无掩膜像素不得自动认定为无作物道路；未确认区域继续接受语义支撑、禁行区和人工复核约束。
- 实际执行配置与路径 provenance 中记录的配置必须一致，田头权重变化必须使路径阶段指纹变化。

**代码锚点**：`footprint_planner.validate_footprints`、`path_planner._select_turn_strategy`、`pyside6_app.workers.PlanWorker`。

**测试锚点**：`tests/test_cross_module_math.py`。

**允许的变更方式**：未来只有在获得田头/主体碾压对照试验并明确要求将其纳入决策后，才可重新讨论田头代价是否参与排序；届时必须报告样本、估计方法、置信区间和安全约束优先级。

## DECISION-014 软件自动实验包与完整论文证据必须分层

- 结论：`reui.paper-experiment.v1` 从当前项目读取源影像身份、原始掩膜、地理尺度、田界、农机参数和服务点的冻结快照，在隔离状态中运行；其当前默认计划只改变掩膜处理阶段。这个自动化工具的实现边界不等于论文研究范围。完整论文证据另包括作业带确定、中心线生成、路径规划、终端导入、无人收获机自主执行和作业后田间调查。转弯策略已实现，但系统不自动开展低碾压优化或策略对比。
- 解释性修订（2026-09-01）：旧标题容易把“默认离线实验工具只生成掩膜资料包”误读成“论文只研究掩膜处理”。本次修订不改变代码和 schema，只明确自动离线包、完整软件田间资料包与实际外场证据的边界。
- 原因：批量功能如果同时改变模型、掩膜、参数或田界，生成的表格不能解释为模块贡献；后台实验若写回当前项目，还会破坏现有科研链路证据。软件资料包如果被直接当成终端执行或田间实测，又会混淆可追溯软件结果与真实外场证据。
- 不变量：
  1. 实验运行不得修改当前 `AppState` 的模型、掩膜、路径或策略。
  2. 原始掩膜、行带几何重构、主体残差恢复和完整规则化组必须来自同一次正式掩膜处理及其真实中间证据层，不得用伪参数或合成结果冒充消融。
  3. 论文默认计划不生成路径策略组；底层运行器仍可接受显式 `path_cases`，仅供未来单独设计的离线实验使用，不得把默认资料包解释为转弯策略比较。
  4. IoU、Dice、Precision、Recall 仅在提供同尺寸人工真值掩膜时生成；没有真值时保持缺失，不得用原始模型输出充当真值。
  5. 单个实验组失败必须保留失败原因并继续其余组；输出必须包含输入摘要、组参数、长表、宽表、逐组结果和效果图。
  6. 当前入口不具备隔离原始 GeoTIFF 多模型推理时，不得把同一原始掩膜复制并标记为不同模型结果。
  7. `reui.field-trial-package.v1` 中的路径、manifest、掩膜和诊断只证明软件证据链完整；终端接收、自主执行和实际碾压必须以独立田间记录证明。
- 代码锚点：`src/paper_experiments.py`、`src/pyside6_app/workers.py`、`src/trial_package.py`；桌面主界面不再提供消融实验入口，离线实验模块当前只生成掩膜阶段资料包，完整田间资料包由独立导出链生成。
- 测试锚点：`tests/test_paper_experiments.py`、`tests/test_trial_package.py`。
- 文档锚点：`docs/RESEARCH_SYSTEM_WORKFLOW.md`、`docs/DATA_CONTRACTS.md`、`docs/SCIENTIFIC_VALIDATION.md`。
- 允许变更方式：可增加新的真实中间层、模型隔离推理、统计图或外场证据清单，但必须保持固定输入、人工真值门槛、失败记录和软件/田间证据分层，并同步更新相关 schema、测试和验证文档。

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
    1. `segment_metrics` 使用 `exclusive_first_segment` 规则；分段独占履带并集和为全路径 `track_swept_area_m2`，重复访问另由 `uint32` 逻辑通过计数图统计。一个逻辑路径段内部的相邻姿态先取二值并集，只能计为一次通过；姿态覆盖次数不得进入正式指标。
  2. 转弯层单独报告 `headland_turn_track_overlap_area_m2`、`headland_turn_rolling_pct`、主体/不确定区/禁行区重叠、田界越界和 `turn_cost_m2`；田头仍是软代价，不是硬障碍。
  3. 用户所选策略只生成一套固定转弯几何，不搜索田头偏移，也不按软代价比较替代转弯；田界、禁行区、运动学不可行、无有效米制尺度和退化直线均进入校验诊断并记录 `fallback_reason`。
  4. 一次与两次及以上履带访问面积是二维几何模型指标，不等于土壤压实、产量损失、实机可执行结果或田间验证结果。
  5. 候选必须使用与正式路径一致的足迹模型并保留核心区重叠、规划目标覆盖和田界越界指标；缺少 `valid`、`issues` 或必需指标不得默认放行。默认报告模式不把碾压/覆盖率阈值当作硬门禁，真实田界、禁行区、交叉和坐标完整性仍按配置检查。
- 代码锚点：`footprint_planner._accumulate_logical_segment_pass`、`footprint_planner.validate_footprints`、`path_planner._connect_work_lines`、`headland_turn_validation.validate_fixed_turn_candidate`。
- 测试锚点：`tests/test_track_pass_count.py`、`tests/test_headland_turn.py`、`tests/test_headland_turn_contract.py`、`tests/test_headland_turn_integration.py`。
- 允许变更方式：可增加有实测依据的足迹模型或标定代价，但必须保留整路径/分段核对、访问计数类型、硬约束诊断和实验输入冻结。

## DECISION-016：套圈按两半交错作业线并共享田头转弯线

- **当前 v17（2026-08-12）覆盖 v16 的末线奇偶约束**：`001.txt` 的 20 条偶数作业线证据表明，若为保证唯一受阻短线“从受阻端结束”而反推交替方向，会把已正确选中的约 `0.48 m` 入口翻到约 `82.22 m` 外；原始中心线端点顺序若与全田主方向相反，先按原顺序选择进入端、随后再做轴向归一化也会产生同样的二次翻转。因此法向排序后的 `n` 条线仍取 `split=ceil(n/2)`，输出 `0, split, 1, split+1, ...`；8 条为 `0,4,1,5,2,6,3,7`，9 条为 `0,5,1,6,2,7,3,8,4`，4 条线使用 `1,3,0,2` 避免中心相邻转弯，但规划优先级改为：先保留各区域生成的全部直线；唯一受阻短线从循环候选中移除后固定追加；随后以必选起点到最终首端点的距离为第一排序键，相邻法向转弯数只作第二排序键；所有直线先统一到全田主方向，再依据起点确定交替进入方向；最后统一构造共享田头墙和全田转弯。受阻短线最后执行，但不得通过偶数/奇数条数反推并覆盖全局进入方向；若按起点优先的交替方向使受阻端成为末线起点，只允许由倒数第二条的局部正式转弯连接该端，并继续接受田界、曲率、禁行区和足迹失败关闭。每次排序变换后都必须重算法向秩间隔，端点修复与共享田头墙后必须用 canonical 作业线复核 `entry_nearest_feasible_verified=true`。诊断记录 `terminal_blocked_endpoint_role`、`terminal_blocked_endpoint_is_route_finish` 和 `terminal_finish_orientation_forced=false`；`entry_side_constrained_by_terminal_short_line` 必须为 `false`。终点标记可以不预先布设，桌面实现会沿最终作业段生成短延伸并写入共享控制折线和兼容 TXT；显式拖动后该点作为终点目标使用。桌面工作流固定使用 `service_point_mode=reference_only`，连续 canonical 路径仍保留作业/转弯几何，入口与终点延伸由兼容控制投影承载；底层研究调用只有显式指定 `service_point_mode=connect` 才生成服务连接段。等距区外侧若仅存在短于主体跨度的掩膜碎片，必须在套圈排序前记录并排除该边缘碎片，不能自动把它推断成完整作业行；同一条带的内部断行只有在两侧均有足够长作业证据时才允许在掩膜阶段桥接。自动端点修复只保留原有证据约束：至少 3 条邻近完整中心线、短带自身纵向跨度和已确认田界三项证据同时成立时，才可沿主航向恢复到共同田头墙。正射影像缺失且自身证据不足时，不再提供自动“缺边处理/趋势补全”模式；用户进入路线编辑后可拖动作业线与调头段共用的控制端点，画布在拖动过程中用相同转弯半径实时重建橙色物理弧线，退出编辑后仍显示连续弧线。协议导出仍从同一组控制端点生成状态 `0` 的直线弦，不得用协议直线替换画布连续路线。种植区域只控制作业直线的来源和密度；全部区域直线生成后必须合并成一个全田集合，再统一执行本决策的排序、方向、共享田头墙和转弯生成，不得因区域边界插入 `turn_aux` 或整行 `turn_reverse`。

- 结论：`contour`（套圈收割）保留现有作物证据支持的平行作业线，先按田块法向轴排序，再按两半交错访问；相邻访问段在对应短边田头连接，同一端的转弯共用同一条轴向田头墙。它不生成脱离作物证据的闭合偏置轮廓，也不绕过运动学和正式足迹验证。
- **v3（2026-08-05 修订）**：替代 v2 的单侧连续扫掠。v2 虽能保证相邻法向位置连续，却仍是蛇形行走，不符合外侧两行组成第 1 圈、次外侧两行组成第 2 圈的回字形语义。v3 同时规定：
  1. 作业线顺序沿主航线法向取 `low, high, low+1, high-1, ...`；入口点只允许交换每一层的左右次序，不得把整个序列反转为从中心向外。
  2. 所有作业线先统一主航线方向再交替反向；同一田头的端点投影到同一轴向墙。宽间距采用两段最小半径四分之一圆弧加横向切线的弓形转弯，使同一端的横向切线重叠；中心间距小于 `2R` 时仍按 DECISION-010 选择半径安全的替代几何。
  3. 横向田头跨度代表田块宽度，不设旧的 `4 m` 短连接上限；只限制沿主航线的端点补偿，并要求邻近/群体证据和田界内线段。无法建立共享田头墙时失败关闭，不把普通蛇形路线标成套圈成功。
  4. 奇数作业线允许中心单行收尾；不得因未配对中心线拒绝整条套圈路线。
  5. 等距种植区细分线使用相对质心的局部主向坐标扫描，禁止把绝对投影再次叠加质心而只生成半段主航线。
  6. 等距区未给出显式间距时，细分线间距取 `track_gauge_m`，使相邻两趟的右履带/左履带中心线重合；掩膜缝隙补偿只能选择整块统一相位，不得逐条移动作业线；诊断必须记录 `actual_spacing_m`、`phase_offset_m` 和 `spacing_is_uniform=true`。显式 `uniform_zone_line_spacing_m` 必须在诊断中标记为研究对照。
  7. 兼容农机控制折线中每条主作业线只保留两个端点，套圈非作业段只保留一条直线弦；canonical 稠密路径仍单独用于科研足迹验证。
- **v4（2026-08-06 修订）**：同一侧田头的全部转弯先按田界逐段审计安全余量，再采用该侧最严格的共同余量；混合种植田块还必须跨全部已分类区域共同审计。接近 `2R`、且半径短缺不超过显式米制几何容差时，用向外平移的小半圆保持最小半径切线位置；超过容差仍失败关闭或使用原有安全回退，不得产生单独梨形环。
- 原因：仅连接相邻法向作业线会把最后两条相邻作业线硬接在一起并产生过大的转弯角；两半交错顺序、共享田头墙和弓形横向切线共同表达用户确认的套圈拓扑，同时仍以作物中心线、田界和农机曲率作为安全证据。
- 不变量：
  1. `contour` 不改变作业线数量和法向层级；只允许已有端点群体/邻线证据支持、位于田界内的主向端点校正，不允许横向移动作业线或伪造闭合轮廓。
  2. 诊断必须记录原始索引、分半索引、低/高半原始索引、交错原始索引、对应作业线对、两条田头墙投影、逐次横向转弯距离和奇数中心收尾状态；`minimum_possible_adjacent_transition_count` 与实际 `selected_adjacent_transition_count` 必须同时记录。起点最近可行首端点优先于相邻数最小值，发生覆盖时记录 `entry_priority_overrode_minimum_adjacency=true`，并仍由逐转弯田界、曲率、自交和足迹验证失败关闭；`turn_distances_m` 按实际交错顺序记录，不再强行要求单调不增。
  3. 同一田头的转弯锚点必须具有相同主向投影；固定转弯校验不得移动宽间距套圈转弯的横向切线或共享田头线。
  3a. 同一田头侧的 `shared_turn_clearance_m_by_side` 必须在全田已分类区域中一致，诊断记录 `shared_turn_clearance_scope=all_classified_regions`；局部凹边界只允许整体内移该侧共同切线，不得只改变一个转弯的曲率。
  4. 套圈连接继续标记为 `turn`/`turn_reverse`/`turn_aux` 等正式转弯段，必须进入 DECISION-015 的长度、足迹和重复访问归因；不得新增绕过统计与协议的 `connector` 段类型。
  5. 套圈路线必须经过与其他策略相同的曲率、田界、禁行区和足迹校验；失败时保留诊断，不把显示用直连段标记为可执行。混合种植田块不得先生成区域路线再拼接；全田统一排序后的每个相邻作业线对都按同一套转弯约束处理。
  6. `auto` 只按当前相邻作业线的几何条件逐段选择实现，不比较其他转弯策略，也不计算全局策略得分。
- 代码锚点：`path_planner._contour_order_work_lines`、`path_planner._build_contour_connector_anchors`、`path_planner._connect_work_lines`、`path_planner._generate_uniform_zone_lines`、`path_planner._align_short_turn_anchors`、`path_planner._select_turn_strategy`、`pyside6_app.panels.TaskPanel`。
- 测试锚点：`tests/test_contour_strategy.py`。
- 允许的变更方式：可用带人工核验田界、作业线端点和实机转弯轨迹的样本调整端点证据阈值或几何容差；不得移除中心线证据、最小半径、共享田头线诊断、正式足迹验证或失败关闭。

<!-- DECISION-016: nested contour ordering remains centerline-anchored and fail-closed. -->
<!-- DECISION-016 v3: outside-pair ordering shares headland walls and supports a center single. -->

## DECISION-017：先按区域生成作业直线，再全田统一计算套圈与转弯

- 结论：路径规划必须先依据每个候选带的实测宽度证据判定 `wide_narrow_rows`（宽窄行四行带）、`uniform_block`（大片等距种植区）或 `ambiguous`，不得用影像左侧/右侧作为类型规则。混合田块中，宽窄行保留四行中心线，等距区按割幅生成自定义平行线；所有区域的直线全部生成后展平成一个全田作业线集合，统一执行 DECISION-016 的套圈排序、进入方向、共享田头墙和转弯计算。区域归属只保留为直线生成与 provenance 证据，不是路线执行边界。
- 原因：`002.txt` 的复现证据表明，先完成区域路线再通过 `turn_reverse + turn_aux` 拼接，会在本田块形成约 `82.78 m` 的状态 0 跨田边；即使转场端点同属一侧田头，区域奇偶性仍会制造不必要的整行折返。用户确认的真实语义是“区域决定直线密度，转弯全田统一计算”。全田排序不会横移或改写已生成直线，因此不破坏四行中心线和等距割幅约束。
- 不变量：
  1. 分类记录必须包含原始带标识、`profile_width_m`、`width_m`、有效判定宽度、主航线法向位置、阈值、模糊保护带、结论和理由；镜像或交换左右位置不得改变相同宽度证据的类型。
  2. 阈值保护带内或同一实测队列中缺失宽度证据的候选必须标为 `ambiguous` 并失败关闭，不得猜测路径策略。若边界剖面同时有上游 `band_type=wide`、低于均匀区下界的局部重构宽度和连续中心线跨度三项独立证据，允许将其记录为边界消歧后的 `wide_narrow_rows`；缺少任一项仍失败关闭。仅在旧调用完全没有宽度字段时，允许按“四行带输入契约”兼容为宽窄行并如实记录理由。
  3. `wide_narrow_rows` 作业线必须继续锚定对应四行中心线；`uniform_block` 作业线必须按收割机割幅和该区掩膜/轮廓生成，不得拿大片区域的一条中心线代替全覆盖。
  4. 混合田块的全部有效作业线必须在转弯计算前展平为一个集合；`_connect_work_lines` 只能调用一次。起点从全田最终安全排序候选中选择最近可行作业线与进入端，终点只作为可选末段方向参考。被田界阻断的唯一短线若固定为末线，必须从可旋转集合中单独移除后再追加，并在最终田头墙校正后重新核对实际首端点；不得用末线受阻端或作业线总数奇偶性反推全局进入侧。所有作业线端点必须先按统一主方向归一化，再根据起点决定首线执行方向。区域不得拥有独立的访问奇偶性、入口方向或执行顺序，也不得因区域边界改变全田法向排序。
  5. 区域边界不得产生 `turn_aux`、`turn_reverse`、区域路线拼接弦或其他特殊连接。每个正式转弯只连接全田统一排序中相邻访问的两条作业线，并与整条路线一起进入 DECISION-015 的足迹、长度、禁行区、曲率和导出门禁；真实服务点或运动学策略需要的倒车段不受此条禁止，但必须有与区域边界无关的明确原因。
  6. 单一种植类型仍使用同一分类协议和原有规划/验证链路，不得因为没有混合区域而绕过证据记录。
  7. UI/缓存可删除稠密带轮廓，但必须保留至多 256 点的 `planning_contour`；历史紧凑缓存缺少该字段时，等距区必须能以中心线确定主向范围、以原始/处理掩膜确定真实横向范围，不得因此生成 0 条作业线。
  8. 两行窄带中心不得提升为四行作业线。宽窄行相邻正式四行中心线间距成为孤立异常大值、且约等于两个正常四行间距时，允许恢复一个规划期四行晶格中心：两侧稳定晶格相位支持不对称时必须从较长队列外推，不得机械取缺口中点；相位证据相当时才允许中点。支持校验必须沿候选四行带宽度做横截面占用并检查轴向连续性，不得要求机器中心线本身落在作物像素上。间隙中的 `all_bands` 窄带只能作为“已检查但忽略”的负证据。原分类、相位方法、队列证据、中心线/横截面支持、轴向长度和补线理由必须同时保留；连续多条缺失或证据不足时失败关闭。
  9. 终点缺省时，连续 canonical 路径完成最后一条作业线后，桌面实现沿该作业段方向生成一小段 `exit` 终点延伸，并把它写入共享控制折线和兼容 TXT；不得新增整行返程。只有底层研究调用显式选择 `service_point_mode=connect` 且提供终点时，才按服务连接语义生成更长的 `exit`/`turn_reverse` 段；此时仍不得以长斜线跨越田块，终点位于最后作业线另一侧田头时先用显式 `turn_reverse` 沿最后一条已收割作业线返回对应田头，再用 `exit` 在田头横向接驳，并记录直达轴向距离、选择模式和田界检查。
  10. 漏线恢复算法保留用于 `research` 模式的显式手动复现；`field_trial` 和 `machine_candidate` 仍强制关闭自动恢复。缓存中存在 `planning_recovered=true` 的作业带时，结果必须保留该标记；严格 `fail_closed` 模式拒绝导出，研究导出不再弹窗或阻断。
  11. 种植类型相同不代表属于同一物理生成区。若等距区的主航线法向位置位于两组宽窄行四行带之间，分类器必须把两侧宽窄行拆成独立且连续的 `wide_narrow_rows_N` 区域，以保留各自来源带和生成密度；生成完成后的直线仍进入同一个全田转弯集合。分区依据和区域数量必须写入分类诊断。
  12. `layout.work_regions` 与 `layout.region_generation_order` 保存区域生成来源；`layout.region_execution_order` 必须为空，`turn_planning_scope=all_generated_work_lines`。`turn_assessment.contour_ordering` 必须记录 `execution_scope`、全田直线数、逐线区域 provenance 和 `region_boundaries_affect_turns=false`；`region_routes`、`region_transfers` 必须为空，并进入路径阶段指纹。
  13. 每个最终候选中心线目录项必须绑定一整条种植带的有序折线。直带默认只保留首尾 2 点；只有横向弦偏离或局部航向变化超过显式阈值时才增加有限折点。折点不得拆成多个可勾选条目，选择、覆盖栅格和路径规划必须按同一目录 ID 整体处理，并记录逐带输入/输出点数。点数预算不能覆盖横向偏差/方向条件：`centerline_simplify_max_points`（默认 8）只是软预算，达到后只对仍不满足条件的区间继续加点，直到满足或达到硬上限 `centerline_simplify_hard_max_points`（默认 64）；硬上限仍不满足时才返回简化前完整的平滑点列。软预算点列同时作为端点修复锚点，额外加点不得移动端点。该保证只消除简化附加误差，不等于人工真值精度已经提高。对应 `centerline-adaptive-polyline-v3`、`footprint-path-v33`；旧路径缓存重新规划，原掩膜不失效。
- 代码锚点：`MainWindow._compact_mask_result_for_ui`、`row_geometry.resolve_terminal_headland_overlap`、`path_planner._centerline_simplification_settings`、`path_planner._simplify_centerline_polyline`、`path_planner.prepare_band_centerlines`、`path_planner.build_band_mask`、`path_planner._uniform_zone_reference_geometry`、`path_planner._classify_planting_regions`、`path_planner._recover_lattice_supported_work_bands`、`path_planner._prepare_work_line_layout`、`path_planner._connect_work_regions`、`path_planner._connect_work_lines`、`path_planner._build_exit_service_route`、`path_planner.plan_path`。
- 测试锚点：`tests/test_mixed_region_strategy.py`、`tests/test_contour_strategy.py`、`tests/test_path_boundary.py::test_centerline_simplification_keeps_a_straight_band_to_two_points`、`tests/test_path_boundary.py::test_curved_centerline_keeps_extra_points_but_one_checkbox_item`、`tests/test_path_boundary.py::test_soft_budget_spends_extra_points_before_dense_fallback`、`tests/test_path_boundary.py::test_hard_cap_falls_back_to_smoothed_trace`、`test_compact_mask_result_keeps_small_planning_contour`。
- 允许的变更方式：可用人工标注的宽窄行/等距区样本重新标定宽度阈值、模糊带、异常间距或直线生成规则，但必须版本化分类器与补线器、保存逐带证据、维持不按左右硬编码、先完成全部直线生成再统一转弯和不确定时失败关闭。

默认 `uniform_zone_min_width_m=4.0 m`，用于避免真实约 `2.5 m` 的完整四行带落入默认歧义保护带；显式阈值、逐带证据和缺证据时失败关闭规则保持不变。

<!-- DECISION-017: generate lines by region before one field-wide turn plan. -->

## DECISION-018：冻结发布包与用户数据必须分离

- 结论：Windows 便携版使用 PyInstaller `onedir` 构建，`reUI.exe` 位于发布目录根部，运行库位于 `_internal`；依赖检测与一键安装界面已移除，源码启动入口只做运行环境初始化（`PYTHONPATH` 隔离与地理环境配置），冻结入口同样不得调用 pip、Conda 或环境安装界面。
- v2（2026-09-10 修订）：普通用户统一通过带版本标签的 Release 便携版运行，源码入口不再承担依赖补齐职责；`src/startup.py` 的环境检测与一键安装（Tk 对话框与 pip 子进程）整体删除，v1 的“冻结入口不得调用”约束不变。原因：安装器依赖的 tkinter 界面本就不进入发布包，与“每次发布提供开箱即用便携版”的交付方式重复；源码缺依赖时按 `developer/requirements/` 自行安装。风险变化：源码模式缺少依赖时不再有引导安装，只会以导入错误退出。
- 原因：普通用户需要解压后直接双击运行；程序更新时不能覆盖农机参数、项目缓存、日志和模型导入记录。
- 不变量：
  1. 内置 `config.example.json`、模型注册表和模型权重属于只读 bundle 资源；用户配置、缓存、日志、用户模型和更新包写入 `%LOCALAPPDATA%\\reUI`（测试和便携实验可用 `REUI_USER_DATA_DIR` 覆盖）。
  2. 源码与冻结版必须通过同一套 `runtime_paths` 解析资源；冻结版缺少构建指纹时，正式 manifest 不能伪装为可追溯发布。
  3. `$PATH`、GeoJSON、CSV、KML 和 manifest 的科学路径验证不因打包而旁路；冻结 manifest 使用构建时生成的提交号和科学代码指纹。
  4. Release 构建必须先通过静态检查、编译、测试、依赖检查和 `reUI.exe --self-test`，失败时不得上传可下载版本。
  5. `update.bat` 只负责开发者将源代码提交/推送到 GitHub，不负责生成或替换用户端 EXE；稳定用户通过带版本标签的 GitHub Release 获取二进制包。
  6. 源码入口只允许运行环境初始化，不得恢复依赖检测或安装界面；缺失依赖按 `developer/requirements/` 处理。
- 代码锚点：`src/runtime_paths.py`、`src/app_entry.py`、`src/frozen_main.py`、`src/startup.py`、`src/main.py`、`src/config.Config.load`、`src/cache.CACHE_ROOT`、`src/pyside6_app.main_window.MainWindow._write_export_manifest`。
- 测试锚点：`tests/test_runtime_packaging.py`、`tests/test_application_entry.py`。
- 允许的变更方式：可替换 PyInstaller 版本或 Release 编排，但必须保留根目录 EXE、用户数据隔离、构建指纹、冻结包自检和失败不发布条件。

## DECISION-019：人工编辑使用与自动规划等价的作业线结构

- 结论：人工编辑不能只提供控制点微调；编辑器必须允许从空路线开始新增完整作业线，并将连续状态 `1` 边作为可选择的作业线对象，支持整线平移、删除、调整执行顺序、反向、端点自由移动、航点增删、边语义切换和逐边直线/转弯几何切换。`state.path_points/path_status/path_geometry_modes` 仍是唯一序列化控制路线：前两者保持终端 0/1 语义，后者表达物理预览几何，不建立与导出路线脱节的第二套隐藏对象。
- 原因：自动规划能够表达“有序作业线 + 相邻转弯”，人工编辑也应能表达同一结构。旧的点级编辑虽然能修端点，却不能从零搭建路线、整线调整或保存人工转弯形状；把人工转弯再次压成自动弓形还会抹掉用户明确标注的避让意图。
- 不变量：
  1. 连续状态 `1` 边从共享控制折线即时派生为作业线对象；状态 `0` 边是相邻作业线间的连接几何。每条相邻点对只有一个 `path_geometry_modes` 值（`straight` 或 `turn`），可独立于作业语义切换。所有画布操作最终回写同一 `path_points/path_status/path_geometry_modes`，兼容测试导出不得读取另一份旧几何。
  2. 新增、删除、调序或反向作业线时，只重建受影响作业线之间的局部状态 `0` 连接；端点是作业段和相邻转弯的共享节点，不得形成视觉上相接但数据中断开的两套点。起点/终点标记与首/末控制节点使用同一颜色、尺寸和拖动层级；删除首/末节点后服务点引用自动改到新的首/末节点。
  3. 只有两个端点且该边的 `path_geometry_modes` 值为 `turn` 时，连接可按当前最小转弯半径生成连续预览；`straight` 模式保留精确点对。人工增加到三个及以上点的 `turn` 点列是显式 canonical 转弯几何，显示、模拟、验证和当前几何导出必须保留其点位，不得静默替换为自动弓形。该物理预览模式不改变兼容导出的 0/1 终端语义。
  4. 每个结构、航点或几何模式修改都必须进入路线撤销栈，并立即使旧验证、自动路径阶段指纹、模拟结果和导出完成状态失效；人工编辑不得继承自动规划 provenance。
  5. “验证人工路线”复用当前掩膜、田界、禁行区、农机足迹和路径验证器，另审计人工转弯局部曲率；结果标记 `route_geometry_source=manual_control_edit_validated`、`manual_route_validated=true` 和 `automatic_planning_provenance_valid=false`。通过只表示当前几何在软件模型下通过，不等于恢复自动规划来源，更不等于田间实测或机器就绪。
  6. 编辑过程中允许先画出尚未通过田界或影像范围检查的草稿，以便自由修形；验证与导出继续检查坐标、田界和 provenance。兼容测试文件可以按 DECISION-001 使用当前人工控制几何并保留预览标记；普通科研导出不要求机器就绪记录，机器兼容/受控试验流程另行使用其专门检查。
  7. 没有自动规划结果时也必须能进入编辑器并直接在画布两次点击构造第一条作业线。进入编辑后不再显示重复的上方工具宫格；选择与点/整条作业线拖动由画布直接承担，新增、删除、反向、调序、插入控制点、切换作业/转弯语义和验证由下方“路径结果”卡片的右键菜单或拖动承担。现有快捷键可保留为加速入口，但不能是完成核心编辑的唯一入口。
  8. 路线编辑是处理结果之上的增量显示层。进入编辑以及每次拖动、插点或结构调整后的重绘必须继续显示当前处理后主体、田头、不确定区、田界和禁行区；不得以 `clear_overlays` 造成前置证据视觉消失，更不得改写或清空对应状态。拖动热路径只允许替换路线动态项，不得逐帧重新降采样或生成掩膜证据图层；悬停于控制点、可拖动作业线或其他可选路径边时必须分别给出可移动、可点击的光标反馈。
  9. 编辑模式下的“路径结果”卡片是共享控制折线的对象视图，不是第二份路线。卡片与画布必须双向同步选中和高亮；右键切换整段语义、直线/转弯几何、添加或删除内部航点、新增或删除作业线，以及拖拽调整作业线顺序，都必须回写 `state.path_points/path_status/path_geometry_modes`、进入同一撤销栈、使旧验证失效并即时重绘两侧。只有作业线卡片允许拖拽排序；状态 `0` 连接是相邻作业线的共享连接，不得脱离前后作业线独立搬动。结构排序后按既有规则重建相邻转弯。
- 代码锚点：`pyside6_app.main_window.MainWindow._route_edit_work_records`、`MainWindow._route_edit_segment_records`、`MainWindow._route_edge_geometry_modes`、`MainWindow._route_edit_set_segment_geometry_mode`、`MainWindow._on_route_list_edit_action`、`MainWindow._route_edit_replace_work_lines`、`MainWindow._route_control_endpoint`、`MainWindow._sync_service_points_to_route_control_endpoints`、`MainWindow._simulation_segments_from_control_route`、`MainWindow._draw_mask_evidence_layers`、`MainWindow._draw_route_edit_overlay`、`MainWindow._validate_manual_route`、`pyside6_app.panels.StepCard`、`pyside6_app.panels.TaskPanel`、`pyside6_app.panels.RouteInfoPanel`。
- 测试锚点：`tests/test_route_edit_pyside.py`、`test_route_edit_can_append_free_point_and_create_a_straight_edge`、`test_route_edit_switches_one_edge_between_straight_and_turn_preview`、`test_route_result_edit_cards_expose_linked_actions_and_work_drag_order`、`test_route_result_card_operations_immediately_update_shared_route_and_canvas`、`test_planning_card_fully_expands_in_scroll_area_without_overlap`。
- 允许的变更方式：可增加框选、多选、吸附、复制和曲线手柄等编辑能力，但必须维持单一控制路线、共享端点、人工来源显式标记、修改即失效和同一验证/导出门禁；不得用“更自由”旁路坐标、田界、禁行区、曲率或 provenance 审计。

## DECISION-020：推理裁剪与掩膜分析使用外扩田块多边形，行带仍裁剪回原边界

- 结论：田块多边形在系统中承担两个角色——作业范围（行带、中心线、路径必须落在多边形内）与掩膜分析范围（掩膜裁剪、剖面/行带检测的区域）。DECISION-020 将两者解耦：**推理阶段 `crop_field_mask` 与掩膜处理阶段的掩膜裁剪统一改用"田块多边形外扩"后的分析多边形（默认外扩 6 m，配置键 `field_analysis_expand_m`，推理从 `model` 段读取、掩膜处理从 `mask_processing` 段读取，均可省略走默认值），而行带多边形裁剪（`clip_band_polygon`）与中心线田界检查仍使用原始田块多边形**。外扩实现为 `row_geometry.expand_polygon_px`（bbox 局部膨胀取外轮廓，避免大图全尺寸掩膜内存开销）。
- 原因：紧贴圈选（25–32 顶点、约 1100 m² 的田块）时，推理裁剪窗口缩到"田块几乎占满视野"，模型失去外部参照后行间混连；更关键的是推理阶段的田块裁剪把模型输出从 7.2 Mpx 砍到 1.9 Mpx，只剩一个 99% 像素连通的 blob，掩膜正则化无法从中重构出 ≥2 条种植带，报"种植带重构失败"。实测（5.6 田5三模式 12m，返青期模型）：紧贴 32 顶点多边形直接失败；多边形外扩 250–500 px 后掩膜处理成功（22 条带）。同时推理裁剪窗口增加最小尺寸约束（`min_infer_window_display_px`，默认 4000 显示像素，以田块 bbox 中心展开、截断到显示图），保证推理上下文。
- 不变量：
  1. 作业范围语义不变：所有行带多边形、中心线仍按原始田块多边形裁剪/校验，路径不会越出用户圈选区域（端到端实测 1617/1617 中心线点 100% 落在原多边形内）。
  2. 外扩多边形只扩大"掩膜分析范围"，不改变田块面积、面积报告与路径统计口径。
  3. 外扩量用米（默认 6.0 m），按掩膜/显示 GSD 换算像素；外扩后多边形截断到掩膜/显示图边界。
  4. 推理裁剪窗口的放大不改变缓存键（仍按 tif 路径），旧掩膜缓存由现有模型指纹/失效机制处理。
- 代码锚点：`inference.InferenceRunner._run_from_tif`（`min_infer_window_display_px`、流式分支 `crop_field_mask`）、`inference.InferenceRunner._run_from_image`（整块分支 `crop_field_mask`）、`row_geometry.expand_polygon_px`、`row_geometry.regularize_crop_mask`（`analysis_polygon` 掩膜裁剪；`clip_band_polygon`/中心线检查仍用 `field_polygon`）。
- 测试锚点：`tests/test_align_logic.py::test_source_tif_inference_streams_oversized_crop_at_source_tile_size`（窗口 clamp 行为不回归）、`tests/test_reliability_hardening.py`（inference 指纹/缓存提交门禁不回归）。
- 允许的变更方式：可调整默认外扩量、最小窗口像素与配置键位置；不得让分析多边形缩回原多边形（紧贴圈选回归失败），也不得让行带/中心线改用外扩多边形（作业范围越界）。

## DECISION-021: 掩膜显示路径必须复制后裁剪，禁止原地修改 state 数据

- 日期：2026-08-16
- 背景：田5 25顶点紧贴圈选在 DECISION-020 修复后仍持续失败（23:46/00:09/00:47 三次报错完全一致：`runs=3 bands=1 skip_detail obs=43/29 span=0.044/0.03`，输入掩膜像素=1892007）。journal 观测点证明：推理提交 `inference mask commit: nonzero=4467084 admitted=True`（外扩掩膜正确提交），但 3 秒后掩膜处理收到 `mask process input: nonzero=1892007`（田块多边形内像素）。同一 state.mask_raw 被原地裁剪。
- 根因：推理完成后 UI 自动显示掩膜（`_mask_overlay_dirty` 重绘 → `_show_mask_overlay`/`_show_path` → `_draw_mask_evidence_layers`），其中 `_normalise_mask_for_overlay` 对 uint8 数组返回原引用（不复制），随后 `clip_mask_to_polygon_stripes(mask, 田块多边形)` 按条带 `dst=result[y0:y1]` **原地**把多边形外像素置 0——直接污染 `state.mask_raw`（446万→189万），掩膜处理随后在田块内 blob 上失败。源码脚本复现（不走 UI 显示路径）因此始终正常，exe 每次都复现。
- 决策：
  1. `_draw_mask_evidence_layers` 内全部 `clip_mask_to_polygon_stripes` 调用（主掩膜、headland、uncertain 证据层）改为传入 `.copy()`，显示裁剪只作用于副本。
  2. `clip_mask_to_polygon_stripes` 保持原地语义（省内存条带实现），row_geometry 内部调用（raw/rebuilt/_output_layer 均为局部副本）不受影响。
  3. 观测日志保留：`inference mask commit: nonzero=... admitted=...`、`mask process input: nonzero=... shape=... offset=... field_pts=...`。
- 经验教训：凡函数以"省内存"为名原地写回输入数组，调用方若持有 state 引用必须显式传副本；显示层永远不得修改模型数据。
- 代码锚点：`pyside6_app.main_window.MainWindow._draw_mask_evidence_layers`（`.copy()` 三处）、`row_geometry.clip_mask_to_polygon_stripes`（原地语义不变）。
- 测试锚点：全量 577 passed / 1 skipped 无回归。

## DECISION-022：分离种植区以主要区域内部主轴补充全局行向候选

- 日期：2026-08-16
- 背景：田5返青期12 m的有效推理掩膜包含多个空间分离的大区域。全局 PCA 被区域质心连线偏至 122.3°，而最大三个区域内部主轴稳定在约 108°；原有只搜索全局粗角 ±12° 的实现无法进入真实方向，横向投影退化为单个 43.8 m 宽的连续区间并报“仅检测到 1 条稳定种植带”。
- 决策：`estimate_row_angle` 保留全局 PCA 候选，同时从面积足够大的前六个外轮廓提取内部二阶矩主轴；各候选在既有 ±12°、49 点横向投影周期性搜索中竞争，最高周期性评分的角度为最终行向。候选按 3° 去重，小碎片不进入候选集。
- 不变量：不得通过降低 `profile_threshold`、`merge_gap_m` 或虚构行带绕过失败；只在有效掩膜已有清晰横向周期结构而全局布局偏置方向时恢复处理。最终带多边形、中心线和路径仍须接受原田界裁剪与足迹验证。
- 实测：诊断包 `识别002.zip` 的 4,519,691 px 掩膜由 122.26°/1 条稳定带恢复为 107.92°/10 条重构带（6宽、4窄）；未放宽处理阈值。
- 缓存：`MASK_REGULARIZATION_VERSION` 升至 18，版本 17 及更早处理产物必须重跑。
- 测试锚点：`tests/test_mask_strategy.py::test_row_angle_estimation_uses_component_axes_when_global_layout_is_skewed`。

## DECISION-023：小范围圈选分离分析上下文、前台显示与作业范围

- 日期：2026-08-16
- 背景：`识别003.zip` 为分蘖期12 m小范围圈选。推理掩膜有6,547,620 px；若掩膜处理前硬裁到圈内，重构退化为3个候选、1条有效带。保留2 m上下文仍失败，保留6 m上下文恢复为107.98°、10条基础带（6宽4窄）。圈外像素因此是方向与宽窄关系分析证据，不是圈外作业目标。
- 决策：原圈选多边形定义最终作业范围；推理与掩膜分析可使用默认6 m外扩上下文。`mask_raw` 保留该上下文供后台计算；前台主掩膜、田头和不确定层均在显示专用副本上裁回原圈选；行带轮廓、中心线、语义输出层及路径继续裁回原圈选。
- 模型状态：成功的新推理提交必须设置 `cache_model_matches_active_engine=true`，因为新掩膜已由当前已加载权重产生；历史缓存不匹配只允许阻止旧掩膜复用，不得阻止刚完成的新掩膜处理。
- 不变量：不得为小范围任务执行无边界的整幅正射图推理；上下文缓冲受控且不改变田块面积、显示范围或规划范围。显示裁剪禁止原地修改 `state.mask_raw` 或 `mask_result` 数组。自动规划前使用 `selection-work-band-filter-v1`：候选中心线的法向余量必须至少容纳半个割幅；位于选择区最外侧且宽度相对同组中位数显著异常、又未达到明确等距区阈值的孤立候选保留为掩膜证据，但不自动生成作业线。排除理由写入 `layout.selection_work_band_filter`，不得静默删除识别带。
- 路径缓存：`PATH_PLANNING_VERSION` 升至 `footprint-path-v28`，旧版六候选路径不能复用。
- 实测规划：`识别003.zip` 的6个宽带候选过滤为中央4条作业带，生成4条作业线、3个转弯，履带越界率由4.4%降为0.0%；路线仍因语义支撑区外33.8%被安全门禁拒绝，未绕过机器执行约束。
- 测试锚点：`test_mask_overlay_hides_analysis_context_outside_selected_field`、`test_source_tif_inference_streams_oversized_crop_at_source_tile_size`、`test_selection_filter_keeps_only_four_complete_work_bands`。

## DECISION-024：作业带确定后生成全量中心线，由左侧勾选并统一连接掉头

- 日期：2026-08-17
- 背景：作业带确定、候选中心线、操作员采用的作业线和掉头拓扑是不同阶段；算法上必须先根据掩膜和农机作业需求确定作业带，再从作业带生成中心线。界面不新增作业带选择按钮，避免把算法前置步骤误写成新的操作流程。
- 决策：掩膜处理提交后，系统先依据最终宽度、方向、连续性和田界证据确定可用于规划的作业带，并形成最终作业线几何；随后由左侧任务清单的“生成中心线”步骤生成整田全部候选中心线目录 `operator-centerline-catalog-v1`，按主方向法向位置稳定编号 `centerline-NNN`。完成后进入独立的“选择中心线”步骤逐线显示一个复选框，默认全选；取消勾选立即隐藏该中心线，并且该线不得进入正式路径。操作员选完后设置起点/可选终点与卸粮点，再点击“路径规划”，规划器才对当前选中集合统一排序、确定方向、共享田头墙并连接掉头；不提供画布作业带直选模式。
- 评价口径：显式选区的规划目标掩膜由选中中心线按本次割台宽度栅格化，并与 `planning_support_mask` 相交；`planned_target_geometric_coverage_pct` 以该选区为分母。`processed_mask`、原始识别掩膜和全田作物碾压指标仍保留原语义，不得把未选区域静默删除为无作物。
- 状态边界：`centerline_catalog`、`selected_centerline_ids` 和 `centerline_selection_report` 是规范状态，必须进入系统撤销、项目缓存和路径 provenance；`work_band_*` 仅作为历史缓存/布局兼容别名。重新推理或重做掩膜必须清空目录与选择；只重新规划保留选择；缓存恢复必须使目录、复选框状态、画布高亮和正式规划使用同一 ID。空选区失败关闭，不能生成只有掉头段的路线。
- 路径缓存：`PATH_PLANNING_VERSION` 升至 `footprint-path-v31`，避免旧画布作业带选择或未经过最终田界过滤的路径 provenance 被误认为新的中心线勾选结果。
- 不变量：所有可规划作业带必须在中心线目录生成前确定；所有候选中心线必须在任何操作员选择之前生成；最终目录还必须对整条中心线执行已确认田块边界覆盖检查，田外孤立候选不得进入目录；取消勾选的中心线不绘制、不栅格化覆盖目标、不参与转弯连接；区域边界只影响直线生成证据，不产生分区转弯。田界过滤结果写入 `layout.centerline_field_boundary_filter`。
- 代码锚点：`path_planner.prepare_operator_centerline_plan`、`path_planner._apply_operator_centerline_selection`、`pyside6_app.workers.CenterlineCatalogWorker`、`TaskPanel.update_centerline_selection`、`CenterlineSelectionPanel`、`MainWindow._on_centerline_selection_changed`、`MainWindow._draw_centerline_overlay`。
- 测试锚点：`tests/test_work_band_selection.py` 和 `tests/test_align_logic.py`（全量中心线目录、独立生成/选择步骤、左侧逐线复选框、取消后隐藏/失效、状态失效链）以及 `tests/test_mixed_region_strategy.py`。

## DECISION-025：每个逻辑操作独立持久化并可跨重启逐步撤销

- 日期：2026-09-01
- 背景：原有 `undo_stack`/`redo_stack` 能在单次运行中逐步撤销，但项目磁盘缓存只写入最近的 `project_state.json` 和当前掩膜产物。程序退出后内存历史丢失，重新打开项目只能恢复最终状态，不能可靠回到“路径前、优化掩膜前、AI 掩膜前、田块圈选前”等相邻阶段。
- 决策：最近项目状态与操作历史分开保存。每次田块/禁行区圈选、模型或掩膜阶段提交、中心线生成与选择、服务点变更、路径规划及人工路线编辑等逻辑操作，在修改前生成一个具有稳定 `history_id` 的状态快照；项目历史使用有序索引保存撤销/重做栈，每个快照写入独立 JSON。快照内的原始掩膜、处理后掩膜和语义层等数值数组按内容 SHA-256 写入独立 NPZ，并由不同快照共享相同数组产物，避免重复复制大数组。
- 不变量：
  1. 一次用户可感知的逻辑操作只增加一个撤销事务；拖拽中的连续鼠标移动不得形成大量中间撤销步，释放时保存最终状态。新操作必须清空重做分支。
  2. 历史与最近状态分别落盘，并共享同一次保存的进程会话 ID、递增 revision 和项目 generation；延迟到达的旧异步写入不得覆盖同项目的较新状态或历史。恢复项目时先恢复经当前 provenance 校验的最近状态，再安装与其 revision 一致的历史栈。撤销一次只回到紧邻快照，不得直接退回初始状态。
  3. 历史必须绑定源 GeoTIFF 的规范路径、大小、时间戳和 SHA-256；源影像变化时整组历史失败关闭。数组加载必须复核 dtype、shape 和内容哈希；某个快照或数组损坏时，只恢复该损坏点之后仍连续可验证的较新历史，不跨过缺口猜测旧状态。
  4. 系统快照必须同步包含田块边界、原始/优化掩膜及其 provenance、中心线目录与选择、服务点、农机参数、路径几何、验证与导出状态。跨重启恢复的历史路径几何仅作历史状态，`auto_path_valid`、`simulation_done` 和 `export_done` 必须失效，正式使用前重新验证。
  5. 历史保留最近 50 个撤销和 50 个重做状态；被索引淘汰的快照与不再引用的数组产物可从该项目专属历史目录清理，不得触碰源 GeoTIFF、模型、正式导出或其他项目缓存。
  6. 历史数组视为不可变证据；显示裁剪或算法修改必须先复制，不能原地修改历史或当前 `state` 共享数组。
- 存储契约：`project_state.json` 保存最新轻量状态；`history/index.json` 保存顺序和源影像身份；`history/snapshots/<history_id>.json` 保存独立状态；`history/artifacts/<sha256>.npz` 保存去重数组。以上均位于 `%LOCALAPPDATA%/reUI/cache/<project>/`。
- 代码锚点：`state.AppState.capture_system_snapshot`、`state.AppState.push_history`、`state.AppState.undo`、`state.AppState.redo`、`cache._write_project_history`、`cache.load_project_history`、`pyside6_app.workers.CacheRestoreWorker`、`pyside6_app.main_window.MainWindow._persist_completed_history_checkpoint`、`MainWindow._restore_cache_payload`。
- 测试锚点：`tests/test_system_history.py`（掩膜阶段独立快照、跨重启逐步撤销、源影像变更拒绝恢复、窗口历史栈安装）以及既有系统/路线撤销回归。

## DECISION-026：人工编辑路径校验分离硬错误与软警告

- 日期：2026-09-08
- 背景：DECISION-019 实现的人工编辑功能允许用户从零手画路径或修改自动规划结果。原有验证逻辑对所有问题（田界越界、碾压率略高、覆盖率略低）一视同仁，只要存在任何 issue 就标记 `valid=False` 并阻断导出。但用户在手动编辑路径时，某些问题是硬约束（如田界越界、禁行区冲突、曲率不足），某些只是优化建议（如碾压率比推荐值高 5%、覆盖率比目标低 3%）。强制阻断会破坏"以用户意图为主、系统只做提示"的人工编辑交互原则。
- 决策：将路径验证问题分为硬错误（severity="error"）和软警告（severity="warning"）。人工编辑模式（`manual_edit_mode=True`）下，只有硬错误阻断导出，软警告以黄色提示显示但不影响 `valid` 标记。自动规划模式保持原有行为（所有问题均为 error）。
- 硬错误（error，阻断导出）：
  - 田界越界（`track_outside_field_pct` 超限）
  - 禁行区冲突（`track_forbidden_overlap_pct` / `machine_forbidden_overlap_pct` 超限）
  - 曲率不足（人工转弯局部曲率半径 < 最小转弯半径）
  - 坐标异常（路径点非有限值、必需指标缺失）
  - 作业线交叉（`non_adjacent_work_line_crossing`）
- 软警告（warning，仅提示）：
  - 碾压率略高（`track_core_overlap_pct` 超限，但非田界/禁行区问题）
  - 覆盖率略低（`planned_target_geometric_coverage_pct` 低于目标）
  - 支撑区略不足（`track_outside_support_pct` / `track_uncertain_overlap_pct` 超限）
- 不变量：
  1. `footprint_planner.assess_footprint_hard_constraints` 新增 `manual_edit_mode: bool = False` 参数；开启时将非关键指标超限问题标记为 `severity="warning"` 而非 `"error"`。
  2. `validate_footprints` 从 `config["manual_edit_mode"]` 读取该标志并传递给 `assess_footprint_hard_constraints`。
  3. `MainWindow._validate_manual_route` 在调用 `validate_path` 前设置 `config["manual_edit_mode"] = True`，并只检查 `errors` 列表判断 `valid` 状态；`warnings` 列表追加到状态描述但不改变 `auto_path_valid`。
  4. 返回的 `validation` 结果包含：`issues`（完整列表，每项含 `description` 和 `severity`）、`errors`（仅 error）、`warnings`（仅 warning）。
  5. UI 显示时，红色图标+红色文字显示 errors，黄色图标+黄色文字显示 warnings。
  6. 自动规划模式（`manual_edit_mode=False`）保持原有行为：所有超限问题均为 error，保持科研门禁严格。
- 代码锚点：`footprint_planner.assess_footprint_hard_constraints`（新增 `manual_edit_mode` 参数与 severity 分类逻辑）、`footprint_planner.validate_footprints`（读取 `config["manual_edit_mode"]` 并传递）、`pyside6_app.main_window.MainWindow._validate_manual_route`（设置 `manual_edit_mode=True`、分离 errors/warnings、只用 errors 判断 valid）。
- 测试锚点：`tests/test_route_edit_validation.py`（待新增：manual_edit_mode 下 warnings 不阻断 valid、errors 仍阻断、自动规划模式行为不变）。
- 允许的变更方式：可调整具体哪些指标属于硬错误/软警告，但必须保持"田界越界/禁行区冲突/曲率不足永远是硬错误"原则，不得让用户绕过物理安全约束。

<!-- DECISION-026: manual route edit validation separates hard errors from soft warnings. -->

## DECISION-027：行向候选近平分必须以下游条带证据决胜，重建退化必须显式失败

- 日期：2026-10-09
- 背景：真实缓存案例（5.27 田5 三模式 12m）中 `estimate_row_angle` 的直方图能量评分在正确方向（≈108.0°）与错误方向（≈112.8°）之间形成近平分（差距约 1%，沙箱复跑 0.9%~1.9%），错误方向胜出时横向剖面塌陷为 2/12 条带（IoU 0.35）且无任何门槛拦截；外扩像素或工作分辨率的微小变化即可在两个方向之间翻转（刀锋敏感）。沙箱验证还表明单纯"横剖 run 数"判据会被假周期骗过（错误角度可获得 5~6 条假条带），因此决胜判据必须使用"组合条带数"（宽度在 0.25–4.0 m 的横向 run 数，排除宽等距块与碎片）。
- 决策：行向估计保留原候选与评分机制，另增近平分决胜——top1 与 top2 得分差低于 `angle_tiebreak_margin_ratio`（默认 0.05）时，取前 4 个角度家族（去重 2°），每家族在 ±1.5°（步长 0.5°）内以 `_comb_run_quality` 扫描；仅当挑战家族最优组合条带数 ≥ 当前家族 + max(2, 当前 25%) 时才切换行向，否则维持评分冠军。另设重建退化门槛：候选带数 ≥8 且重建带数不足 40% 候选时抛出 `种植带重建疑似退化`（附 skip_detail），不得静默输出塌陷几何。全过程写入 `diagnostics.angle_diagnostics` 与 `diagnostics.rebuild_ratio`。
- 不变量：
  1. 清晰胜出（得分差 ≥ 5%）的行向结果必须与旧版逐位一致；决胜逻辑仅在近平分时生效。
  2. 决胜切换必须保守：挑战家族证据未达优势门槛时不得切换；切换后行向必须等于该家族扫描最优点。
  3. 退化门槛必须失败关闭且携带可诊断明细；`degenerate_candidates_min`/`degenerate_rebuild_ratio_min` 可调但不得删除。
  4. `angle_diagnostics`（候选排名、家族扫描表、决策与理由）必须随掩膜诊断持久化，供论文复现与审计。
- 缓存：`MASK_REGULARIZATION_VERSION` 升至 `19`，版本 18 及更早处理产物必须重跑。
- 代码锚点：`row_geometry._rank_row_angles`、`row_geometry._comb_run_quality`、`row_geometry._resolve_row_angle`、`row_geometry._degenerate_rebuild_reason`、`row_geometry.regularize_crop_mask`。
- 测试锚点：`tests/test_mask_strategy.py`（排名包装等价、组合条带计数、近平分切换/维持、退化门槛）、`tests/test_research_integrity.py`（版本断言）。

<!-- DECISION-027: near-tied row angles resolve by transverse comb evidence; degenerate rebuilds fail closed. -->

## DECISION-028：人工参考线作为中心线评价的外部基准（sidecar 存储、离线评价）

- 日期：2026-10-09
- 背景：论文承诺对"宽行带中心线提取"给出量化评价（方向角偏差与横向偏差），但源码此前只有掩膜层四级消融（`paper_experiments.py`），没有"提取中心线 vs 人工参考线"的评价链路。已核验文献（Ou 2025、Yun 2025、肖延伟、张艳超、董静等）范式高度一致：人工绘制参考线为基准，统计方向角偏差与横向偏差（沿参考线采样到提取线的最短距离，报平均与最大）。参考线属于人工研究证据，不是作业状态。
- 决策：新增两块能力并分层落位——① 桌面端"参考线标注"研究工具（工具菜单）：沿宽行带中心逐点添加桩点（建议桩距约 5 m、每带 10-16 桩），双击收线后最小二乘拟合并即时保存；② 离线评价工具 `src/centerline_evaluation.py`：读取参考线 sidecar 与项目缓存，重跑当前掩膜处理，输出逐线指标表、人工重复性（第 2 遍）、叠加图/箱线图与 zip 评价包。两者共享纯几何模块 `src/reference_lines.py`（拟合、采样、偏差度量、payload 校验）。
- 不变量：
  1. 参考线以 sidecar `reference_lines.json` 保存在项目缓存目录，schema 为 `centerline-reference-lines-v1`；不进入 `project_state.json`、撤销历史、缓存恢复校验与导出闸门，删除该文件不得影响项目恢复或路径导出。
  2. 坐标口径为全幅显示坐标（等同掩膜局部坐标 + `mask_offset`）；评价前必须转换为掩膜局部坐标再与提取中心线比较。
  3. 指标口径：方向角偏差 Eθ 为两条直线方向差（180° 取小）；横向偏差 Elat 为沿参考线按 1 m 等间隔采样到提取中心线（折线）最短距离的平均/最大（cm）；另报重叠率与采样数。不引入文献无背书的指标（P95、组合加权精度 A%、实车跟踪偏差等）。
  4. 每带两遍参考线之间的同口径偏差作为人工判读噪声底，与算法偏差同表报告；参考线落膜覆盖率低时只提示复核、不阻断保存。
  5. 标注模式与田块圈选/禁行区绘制互斥；双击第二下仅收线、不新增桩点；保存失败必须显式报错。
- 代码锚点：`reference_lines`（纯几何与契约）、`centerline_evaluation`（离线评价与评价包）、`cache.reference_lines_path/load_reference_lines/save_reference_lines/project_cache_dir`、`pyside6_app.main_window.MainWindow._set_reference_mode/_handle_reference_click/_finish_reference_line/_draw_reference_lines_overlay/_on_export_reference_lines`、`pyside6_app.image_view.DISPLAY_LAYER_KEYS` 的 `reference_lines` 图层。
- 测试锚点：`tests/test_centerline_evaluation.py`（拟合/度量口径、payload 校验、匹配、评价包与图形）、`tests/test_reference_lines_ui.py`（模式互斥、加桩点与双击收线、右键取消、两遍并存、图层绘制）。

<!-- DECISION-028: manual reference lines as external ground truth for centerline evaluation; sidecar storage and offline evaluation package. -->

## DECISION-029：行带级三项指标由独立离线评测模块复算（band-eval-v1）

- 日期：2026-10-13
- 背景：专利代理批注 #18 要求明确"行带识别完整率／行带分类正确率／中心线可用率"三项指标的获取方式，材料历史口径只有名称；系统此前没有任何对应统计代码（全仓检索完整率/正确率/可用率零命中）。三项指标需要可复算实现，供论文评价章节与答复材料引用。
- 决策：新增独立离线工具 `src/band_evaluation.py`（版本标识 band-eval-v1），纯只读复算：`centerline_evaluation.load_project_inputs` 重建缓存环境（tif 路径自动重映射）→ `process_mask` → 镜像 `pyside6_app.workers` 的 path_planning 配置快照与语义掩膜注入 → `path_planner.prepare_operator_centerline_plan`（与 plan_path 同一步"生成全部候选中心线"）→ 逐条中心线 `footprint_planner.validate_footprints`（同函数同阈值）→ 输出评价包（summary.json / per_band.csv / per_truth.csv / README.md / overlay.png）。指标分三口径：①主口径中心线可用率＝生成作业路径的宽行带中心线数 ÷ 提线（宽行带）总数（全自动）；②严格口径＝叠加逐线履带足迹门禁无问题（越界/支撑区/不确定区/禁行区）；③通道包含判据对照列＝按相邻行带边界配对推导净宽并计算履带接地投影余量（冠层级实测，与 `machine_recommendation._corridor_geometry` 同口径）。完整率/正确率以 labelme/AnyLabeling 多边形真值（标签宽行带/窄行带或 wide/narrow，IoU≥0.5 贪心匹配）计算；无真值时如实记"待实测"。
- 不变量：
  1. 本模块不改动工程代码与缓存内作业数据：不写 `project_state.json`、`mask_*.npz` 等状态文件，不影响导出闸门与任何规划行为；评价包默认写入缓存目录下独立子目录 `band_eval/<时间戳>`（`--out` 可指定），不落工程目录。
  2. 指标数值必须使用与规划器相同的函数与阈值口径（`validate_footprints`、validation_profile 取自 path_planning 配置）；不得另行发明近似公式。
  3. 机具几何统一取仓库默认（实机实测 2.0/0.35/1.3）；不将缓存内历史 harvester_params 快照注入评测状态。
  4. "主口径/严格口径/通道包含判据"三个数字必须同时输出并各自注明口径；不得把严格门禁数字当作唯一可用率对外表述（系统对门禁问题的处理是记录＋人工确认＋正式导出闸门，研究导出不阻断）。
  5. 无真值时必须显式输出"待实测（未提供真值）"，不得猜测或补造完整率/正确率。
- 代码锚点：`band_evaluation`（管线复算、三口径指标、真值匹配、评价包写出）、`centerline_evaluation.load_project_inputs`（共享缓存/tif 重建口径）、`path_planner.prepare_operator_centerline_plan`、`footprint_planner.validate_footprints`、`machine_recommendation._corridor_geometry`（通道净宽同口径）。
- 测试锚点：`tests/test_band_evaluation.py`（通道几何余量数学、多边形 IoU 与真值匹配、标签归一化）。

<!-- DECISION-029: band-level three metrics computed by an offline evaluation module (band-eval-v1); read-only re-run with the planner's own validation functions. -->

## DECISION-030：粘连带瓶颈切割（主流程内局部几何切割，默认开启）

- 日期：2026-10-13
- 背景：交底书 S3 承诺"对粘连较严重、剖面仍连通的相邻行带，可进一步采用沿法向的线性结构元执行定向开运算以断开细桥接，并以距离变换分析粘连体的宽度瓶颈，将距离值低的位置切开"。此前主流程只有横向占据剖面分离；旧版 `mask_processor.cut_adhesions`（定向开运算 + 全局距离变换 0.3×max 切割）会整体侵蚀所有带宽，只能留在兼容回退层。六个真实案例缓存中不存在真实两带合并（疑似宽块均为等距区或"带+窄附件"结构），用合成粘连桥完成能力验证。
- 决策：在 `row_geometry.regularize_crop_mask` 主流程内新增局部、按需的瓶颈切割 `_split_adhesion_runs`：只对宽度在 [1.35, 2.6] m 的疑似合并剖面 run，用距离变换宽度包络找鞍点，满足全部条件才在鞍点处按 0.06 m 切口一分为二；未触发 run 逐位保留。报告挂载于 `mask_result.diagnostics.adhesion_split`。
- 不变量：
  1. 三层保护必须同时生效且可配置：宽度窗 [1.35, 2.6] m、鞍点/平台比 ≤ 0.90（默认）、切口两侧各 ∈ [0.45, 1.5] m；不得为切割调整 `profile_threshold` 或绕过保护条件。
  2. 未触发 run 与正常案例必须逐位零改动（六案例安全对照逐位一致；合成粘连桥 19→20 带恢复）。
  3. 本切割只改剖面 run 区间，不得改掩膜数组本体；候选与切割明细必须随掩膜诊断持久化（论文复现与审计）。
  4. 全填充通道（无几何腰）的切口位置天然欠定、子带宽度为近似值；不得据其对外声明精确带宽恢复。
- 缓存：`MASK_REGULARIZATION_VERSION` 升至 `20`，版本 19 及更早处理产物必须重跑。
- 代码锚点：`row_geometry._adhesion_width_profile`、`row_geometry._split_adhesion_runs`、`row_geometry.regularize_crop_mask`。
- 测试锚点：`tests/test_mask_strategy.py`（合成瓶颈分割、保护条件拦截、禁用开关）、`tests/test_research_integrity.py`（版本断言）。

<!-- DECISION-030: adhesion bottleneck split inside the main row-band pipeline; local geometric cut only for suspected merged runs, zero change otherwise. -->

## DECISION-031：演示动画视觉层与足迹校验模型分离（harvester-visual-v1）

- 日期：2026-10-19
- 背景：动画原先直接用 `planar_geometry_m` 的抽象足迹绘制蓝色机体；该足迹的机身长度取 `max(轴距, 接地长)=1.8 m`，服务的是足迹/碾压校验，不是整机外形。把整机真实外形塞进 `planar_geometry_m` 会改变足迹、转弯与碾压评价口径。
- 决策：新增 `pyside6_app.harvester_visual.HarvesterItem(QGraphicsItem)`，局部坐标以米为单位（+x 前进、+y 机组右侧、原点在履带接地中心），通过 `setPos/setRotation/setScale(px_per_m)` 放置；`_render_anim_frame` 只替换视觉层，位姿、分段速度、转弯圆弧、进度跳转仍由 `AnimationEngine` 提供。已割带用 `QGraphicsPathItem` 沿 work 段绘制，笔宽=割幅×`px_per_m`，终点按割刀前伸量延伸到割刀位置。
- 机型修正与视觉修正（2026-10-19）：实机为星光 **4LZZ-4.5Z** 再生稻低碾压联合收获样机（双 Z），不是普通 4LZ-4.5Z；此前按普通机型建模，型号与后部比例都不对。初版粮箱 2.68 m 一直拖到机尾，按实机照片重画为：总长 3.57 m、后悬 0.80 m、粮箱 1.05 m、后部发动机/清选舱 0.55 m。上述后悬与总长为照片估算，待实机测量确认；`harvester_params` 的割幅、履带、轨距、接地长保留实测值。
- 不变量：
  1. `planar_geometry_m`、`footprint_planner` 与碾压/足迹校验口径不因视觉层改动；`_draw_harvester_pose` 保留为校验用抽象位姿与测试锚点。
  2. 视觉尺寸只取 `harvester_params` 实测值（割幅 2.0、割刀前伸 1.35、履带 0.35、轨距 1.30、轴距 1.80、接地长 1.70、转弯半径 0.90）与按照片比例推导的机身包络；不得把官网通用型号参数冒充实机实测。
  3. 视觉层不参与导出、路径校验或任何规划数值；停止/异常/完成动画时从场景移除，不污染工程图。
  4. 严格正交俯视，不使用透视或等轴测。
- 代码锚点：`pyside6_app.harvester_visual.HarvesterItem`、`pyside6_app.main_window.MainWindow._update_harvester_item/_update_cut_swath/_dispose_harvester_item/_render_anim_frame`。
- 测试锚点：`tests/test_harvester_visual.py`（包围盒覆盖整机、离屏绘制非空、旋转后包围盒方向）、`tests/test_animation_controls.py`（动画控制与失败回滚不回归）。

## DECISION-032：界面图标改为统一矢量绘制，配色收敛为「一主色 + 状态色」（ui-icons-v1）

- 日期：2026-10-20
- 背景：顶栏撤回/恢复用 `↩`/`↪`、设置用 `⚙`、主题用 `☀`/`☾`，任务卡片用 `▸`/`▾`/`✓`，测量用 `•`/`/`/`□`，缩放用 `＋`/`−`。这些字符的字形、粗细、基线随字体变化，无法做严格镜像，放大后是手绘感；同时 13 个任务各带一种专属色（绿、棕、红、橄榄），等于没有强调色。
- 决策：新增 `pyside6_app.ui_icons`，在 20 单位网格上用 QPainter 以 1.7 px 圆头描边绘制全部界面图标，按主题色渲染并支持禁用态；撤回与恢复共用一条路径水平镜像。配色改为：主操作统一 `accent`，次级操作为透明底 + `border_light` 描边，任务状态方块按状态取 green/accent/red/border_light。
- 不变量：
  1. 界面不再用字符或 emoji 充当图标；新增图标一律进 `ui_icons`，不在业务代码里写图形字符。
  2. 图标颜色必须来自 `COLORS`，不得硬编码十六进制，保证深/浅主题都跟随。
  3. 任务主按钮不再使用任务专属色；`TASK_DEFS` 的 `color` 仅保留为历史字段，不再驱动按钮。
  4. 次级按钮（回退）不得用高饱和填充色，避免与主操作争夺注意力。
- 代码锚点：`pyside6_app.ui_icons`（`undo_icon`/`redo_icon`/`theme_icon`/`settings_icon`/`chevron_icon`/`check_icon`/`point_icon`/`line_icon`/`area_icon`/`plus_icon`/`minus_icon`）、`top_bar.TopToolbar`、`panels.TaskRow._update_badge/_secondary_button_style`、`panels.StepCard`、`image_view.ImageViewerGroup`。
- 测试锚点：`tests/test_align_logic.py`（任务面板与视图控件不回归）、`tests/test_system_history.py`（撤回/恢复可用态）。

## DECISION-033：界面图标采用 Lucide 网格，配色对齐 ZoneDeck / Codex++ 参考（ui-icons-v2）

- 日期：2026-10-20
- 背景：第一版图标在 20 单位网格上自行画曲线，撤回/恢复看起来仍是手绘；同时深色主题用青蓝、浅色主题用系统蓝，两套主题没有统一的设计语言。
- 决策：图标统一到 Lucide 的 24×24 网格、2.0px 圆头描边（`undo-2`/`redo-2`/`moon`/`sliders-horizontal`/`chevron`/`check`/`plus`/`minus`/`search`/`refresh-cw`/`trash-2`/`info`），撤回与恢复由同一条路径水平镜像；深色主题取 ZoneDeck 的蓝黑底 + 薄荷绿强调，浅色主题取 Codex++ 的近白底 + 靛蓝强调；新增 `on_accent` 令牌承载强调色底上的文字色。
- 不变量：
  1. 图标必须在 24 单位网格上绘制，线宽 2.0px、圆头圆角，不得再出现字体字符或 emoji。
  2. 撤回/恢复必须由同一路径镜像产生，不得分别手画。
  3. 每套主题只允许一个强调色；状态色（成功/失败/警告）不得用于装饰。
  4. 强调色底上的文字必须用 `on_accent`，不得用 `btn_default`。
- 代码锚点：`pyside6_app.ui_icons`、`pyside6_app.styles.THEMES`、`panels.StepCard.refresh_state/set_active`、`panels.TaskRow._update_badge/_secondary_button_style`。
- 测试锚点：`tests/test_align_logic.py`、`tests/test_system_history.py`。

## DECISION-034：侧栏卡片采用「状态竖条 + 可提亮高亮面」的自绘卡片（step-card-v2）

- 日期：2026-10-20
- 背景：参考实现（Codex++ / ZoneDeck）的侧栏列表项由两层构成——透明的行本体 + `::before` 高亮面；悬停与选中复用同一层，只切换 background token（`row-surface` = 文字色 3.5%，`row-selected` = 强调色 8.5%），过渡 160–180 ms，选中项另有 outline 与左侧竖条。Qt 没有伪元素，原先用 `setStyleSheet` 的 `border-left` 无法同时表达悬停浮起与圆角裁剪。
- 决策：`StepCard` 改为在 `paintEvent` 中自绘三层（圆角高亮面、1 px 描边、3 px 状态竖条），悬停强度 `_hover` 用 160 ms 补间；状态由 `_active` / `_done` 决定竖条与底色。
- 不变量：
  1. 高亮面必须用圆角 `QPainterPath` 裁剪，不得靠 `setStyleSheet` 的 `border-radius` 间接实现。
  2. 悬停与选中必须复用同一层填充，只换颜色，不得各画一套。
  3. 悬停时长固定 160 ms，与参考实现的 `opacity 160ms` 对齐。
  4. `paintEvent` 中访问的实例属性必须有默认值（`_done`/`_active`/`_hover`），自绘异常会导致进程级崩溃而非 Python 异常。
- 代码锚点：`panels.StepCard.paintEvent/_chrome/_set_hover`、`panels.StepCard.set_active`、`ui_motion.tween_value`。
- 测试锚点：`tests/test_align_logic.py`（手风琴展开、卡片宽度与不重叠）。

## DECISION-035：动效分三层——短过渡、少量过冲、循环提示（motion-v1）

- 日期：2026-10-20
- 背景：参考实现（Codex++ WebView2 / ZoneDeck Tauri）的"灵动"不是单一动画，而是三层规则：①绝大多数交互为 100–260 ms 过渡；②只有容器展开与箭头旋转带过冲（`cubic-bezier(.34,1.56,.64,1)`，`.22s`）；③运行态用无限循环提示（状态点呼吸 1.2 s、进度高光 1.8 s、完成光束 1.6 s 仅一次）。
- 决策：reUI 动效按同三层实现，集中在 `pyside6_app/ui_motion.py`：
  - 过渡层：`BACK_OUT`/`SOFT_OUT`/`EASE_IN_OUT` 三条自解贝塞尔曲线，时长常量 `FAST_MS=120 / BASE_MS=220 / SLOW_MS=300`。
  - 过冲层：只用于 `StepCard` 手风琴与 `ChevronToggle` 旋转，其他位置一律单调缓动。
  - 循环层：`BreathingDot`（1.2 s 呼吸）与 `ShimmerBar`（1.8 s 高光扫过）只在"进行中"启动，静止状态必须停表。
- 不变量：
  1. 循环动画必须有明确的关闭路径，禁止在空闲态常驻定时器。
  2. 状态点四态语义固定：`idle` 灰、`busy` 强调色呼吸、`ready` 绿、`error` 红，不得新增第五种。
  3. 自绘控件的 `paintEvent` 不得抛异常；Qt 绘制回调异常会导致进程级崩溃，需为所有实例属性设默认值。
  4. 子控件必须加入父控件自身的布局；加到第二个独立布局会落在父矩形之外并被裁剪。
  5. 折叠卡片的汇总状态只统计带 `_status_key` 的真实任务行，并同时接受 `done=True` 与 `status="success"` 两种完成标记。
- 代码锚点：`ui_motion.BreathingDot/ShimmerBar/cubic_bezier_curve/tween_value`、`panels.TaskRow._update_badge`、`panels.StepCard.collapsed_status_dot`、`main_window.set_status_text`。

## DECISION-036：工作流步骤按整链校验，选填项以行名标注（chain-v1）

- 日期：2026-10-20
- 背景：审计 12 组缺失前置的状态组合，发现四处缺陷——①`segment`/`process` 只校验自身标志位，上游被删后仍可显示 done；②`process` 不检查 `inference_done`；③`centerline_catalog` 不跟随掩膜，旧目录会让下游按过期作业线规划；④`plan` 的锁定条件不覆盖影像/田块/识别/掩膜缺失。
- 决策：
  1. `derive_task_statuses` 改为整链校验：`field_done = image_loaded ∧ boundary`，`inference_done = image_loaded ∧ flag ∧ mask_raw`，`process_done = inference_done ∧ field_done ∧ flag ∧ mask_result`；`process_done` 为假时 `centerline_items` 与 `selected_centerlines` 一律视为空。
  2. 每个被锁定的步骤必须给出原因文本（先导入影像／先圈选田块／先完成作物识别／先完成掩膜优化／先生成作业线／先选择作业线／先确认高级设置／先设置并确认起点）。
  3. 选填项固定为 `exit`、`unload`（`OPTIONAL_TASKS`）。选填项缺失时状态为 done（不阻塞下游），但被修改时必须触发 `_invalidate_analysis_from("path")`。
  4. 选填标识只允许写在行名（`终点（选填）` / `卸粮点（选填）` / `起点（必选）`）+ 行名强调色 + 状态文字 + tooltip；**禁止**再加独立小标签控件（320 px 紧凑面板下会裁断行名）。
  5. 卡片完成瞬间允许触发一次 `CompletionBeam`（1600 ms、仅一次），生成中允许 `BobIcon` 循环（920 ms）；两者都必须有明确的停止路径。
- 代码锚点：`pyside6_app/task_state.derive_task_statuses/OPTIONAL_TASKS/TaskStatus.optional`、`panels.TASK_DEFS[].optional`、`panels.TaskRow.set_status/refresh_theme`、`ui_motion.CompletionBeam/BobIcon`。
- 测试锚点：`tests/test_task_chain.py`（10 条，含"done 蕴含前置全 done"的顺序不变量）。

## DECISION-037：动效只保留交互反馈，移除完成扫光（motion-v2）

- 日期：2026-10-20
- 背景：`CompletionBeam`（1600 ms 旋转渐变扫光）在卡片完成时播放，但一旦 `_active` 变假而控件仍可见，旋转的渐变条会以斜向半透明块的形式残留在卡片上，观感极差且被误认为"平行四边形阴影"；用户明确表示**不需要光效，只要交互反馈**。
- 决策：删除 `CompletionBeam` 及其在 `StepCard` 中的触发、几何同步与单次燃烧逻辑，完成状态只由「绿色对勾 + 状态点」表达。
- 保留的交互反馈：悬停浮起（160 ms）、按钮按下内缩回弹（`SpringButton`）、箭头旋转过冲（`ChevronToggle`）、手风琴展开过冲、状态点呼吸（1.2 s）、进度条高光（1.8 s）。
- 不变量：
  1. 卡片不得叠加任何会遮挡或改变底色的覆盖层；状态只用对勾、状态点、3 px 竖条和描边表达。
  2. 任何一次性动画结束必须 `hide()` 自身，不允许以"不可见状态仍可见控件"的方式收尾。
  3. 明令禁止装饰性光效（扫光、霓虹、呼吸光晕），只允许表达"进行中/完成/失败"的功能性反馈。
- 代码锚点：`panels.StepCard.refresh_state/_chrome/paintEvent`、`ui_motion`（`BreathingDot`/`ShimmerBar`/`SpringButton`/`ChevronToggle`）。

## DECISION-038：步骤完成判定为「影像 + 自身产物」，失效由动作触发（chain-v2）

- 日期：2026-10-20
- 背景：DECISION-036 曾把 `field_done` 写进 `inference_done` / `process_done` 的五重条件，导致只要田块边界缺省（缓存部分恢复、异步恢复窗口），掩膜、农机参数、作业线就会被整体降级，用户看到"步骤突然都退了"。而缓存白名单（`_restore_cache_payload`）本来就把 `field_boundary` 与 `inference_done`、`mask_processed` 成对恢复，正常项目不会出现"有掩膜无边界"。
- 决策：
  1. `inference_done = image_loaded ∧ inference_done ∧ mask_raw`；`process_done = inference_done ∧ mask_processed ∧ mask_result`。**不再要求田块边界**。
  2. 田块不是识别的硬前置：`segment` 在"已圈选或已有掩膜"时可操作；边界缺失只让 `draw_field` 退回可重选，不连坐下游。
  3. 失效由**动作**触发：重画/删除田块调用 `_finish_field_polygon → _invalidate_analysis_from("inference")` 清空整条下游；状态机不得从"边界为空"推断掩膜过期。
  4. `plan` 的锁链不再包含田块（识别已产出掩膜即可继续）；其余锁链保持严格：掩膜→参数确认→作业线→选线→起点锁定。
- 代价与取舍：若真是"边界被外部清空但掩膜仍在"的损坏存档，界面会允许在旧掩膜上继续规划。这是有意的取舍——用可恢复的宽容换掉不可恢复的"整套降级"，且这种存档本身不可能由本程序产生。
- 测试锚点：`tests/test_task_chain.py::test_mask_derived_steps_survive_a_partially_restored_cache`、`::test_deleting_field_polygon_invalidates_every_downstream_step`、`tests/test_align_logic.py::test_cache_restore_reenables_confirmed_parameters_after_processed_mask`（断言恢复为 done）。

## DECISION-039：进度只由真实任务驱动；卡片展开改为内容区高度动画（motion-v3）

- 日期：2026-10-20
- 缺陷一（进度）：`_on_progress` 末尾有 `pct < 30 → segment / < 70 → process / else plan` 三段**按百分比猜步骤**的兜底。任何一次进度事件（含启动时加载模型）都会把"作物识别"标成"处理中 25%"，即使用户还没圈选田块、没点过处理。任务行状态必须来自 `state` 的 `inference_running` / `mask_processing_running` / `plan_running` / 中心线 token；模型加载等非流水线阶段只更新状态栏与进度条。
- 缺陷二（展开）：旧实现只动画卡片自身的 `minimumHeight`，内容区瞬间 `setVisible(True)`，导致内容先撑开再回缩、卡片高度挤压相邻卡片。
- 决策（按 `qt-collapsible-section`，Qt Wiki 的经典 Section 实现）：
  1. 内容区垂直 `Fixed`、初始 `maximumHeight=0`、默认隐藏；
  2. 展开/收起用 `QParallelAnimationGroup` 三个并行动画：内容区 `maximumHeight` 0↔内容高度、卡片 `minimumHeight`/`maximumHeight` 折叠高度↔折叠+内容高度；
  3. 表头固定在布局第一行，内容在第二行，**标题相对位置全程不变**；
  4. 时长 220 ms、`OutCubic`。内容高度本身在变，不再叠加过冲（会让底部先空一块再收回）。
- 不变量：
  1. 卡片高度只能通过 `minimumHeight` 与 `maximumHeight` 成对设置，禁止只设其中一个。
  2. 展开动画期间表头不得移动；任何让表头位移的实现视为缺陷。
  3. 任务行进度只允许由真实运行标志驱动，禁止从百分比区间推断当前步骤。
- 代码锚点：`panels.StepCard._collapsed_height/_content_height/_apply_expanded_state/_animate_expansion/_on_expand_finished`、`main_window._on_progress`。
- 测试锚点：`tests/test_align_logic.py::test_planning_card_fully_expands_in_scroll_area_without_overlap`（等待动画结束后再量几何）。

## DECISION-040：行高随内容自适应、步骤推进自动换焦点、字号与 hinting 调整（layout-v1）

- 日期：2026-10-20
- 缺陷一（进度条压按钮）：`TaskRow` 行高写死 56 px，但"标题 16 + 状态/按钮 30 + 进度条 4 + 间距边距"需要 66 px；进度条显示时被挤到按钮矩形上。改为 `_sync_row_height()`：布局 `invalidate/activate` 后取 `sizeHint().height()`，`max(56, hint)` 作最小高度；进度条可见性每次变化都调用。
- 缺陷二（步骤不自动切换）：`_refresh_cards` 里自动展开受 `_manual_step` / `_user_collapsed` 抑制，用户点过一次卡片后就再也不自动推进。改为**步骤推进（`first` 变化）时无条件清除这两个标记**，收起旧卡片、展开新卡片，符合"识别完成→自动进入下一步"。
- 缺陷三（字体发虚）：`app_entry` 设了 `QFont.PreferAntialias`，在 Windows 上会退回灰度抗锯齿，中文小字号笔画发糊；且全局最小字号 10–11 px 偏小。改为 `setHintingPreference(PreferFullHinting)`（渲染交回 DirectWrite/ClearType），并把全部 QSS 字号整体上调一档（10→11、11→12、12→13、13→14、15→16、16→17、18→19）。
- 不变量：
  1. 任何含可变高度子控件（进度条等）的行，行高必须由布局 `sizeHint` 驱动，不得写死常量。
  2. 步骤推进必须自动切换展开焦点；只有"用户在当前步骤内手动开合"才允许保持用户选择。
  3. 字号调整后必须验证无横向滚动（`_scroll.horizontalScrollBar().maximum() == 0`）且无标签/按钮越界。
- 代码锚点：`panels.TaskRow._sync_row_height/_update_badge`、`panels.TaskPanel._refresh_cards`、`app_entry` 字体初始化、`styles.py` 全局 QSS 字号。

## DECISION-041：田块边界必须是简单环，闭合重复点击自动修复而非静默失败（field-boundary-geometry-v1）

- 日期：2026-10-21
- 背景：现场项目 `result.tif` 在"生成作业线"步骤报 **无有效中心线**，日志只有 `正在生成全部候选中心线… / 当前没有可供选择的有效中心线`。实测缓存（`%LOCALAPPDATA%\reUI\cache\4cf071d37373`）：`wide_bands=15`、`all_bands=30`、`main_angle=1.9076`、掩膜 `5129×3769` 全部正常，`_prepare_work_line_layout` 也正常生成 **15 条**作业线，但 `centerline_field_boundary_filter` 把这 15 条**全部**判为 `centerline_outside_field_boundary`。
- 根因：`field_boundary` 有 6 个顶点，其中第 5、6 点距首点仅 29.47 px（两次闭合点击追加的顶点），环自相交（shapely `Self-intersection[1591.76 345.15]`、`is_simple=False`）。`segment_within_field_boundary` 在 `not field.is_valid` 时一律 `return False`，于是田界门禁对任何折线都答"越界"。判定与作业线几何无关：这 15 条线的 `covers` 在修复后的环上全部为 True。
- 结论：田块边界在进入任何门禁前必须归一化为简单环。`row_geometry.sanitize_field_polygon` 去除连续重复顶点，对自相交环取 `make_valid` 的最大连通多边形，**仅当被丢弃面积 ≤ 2%** 才接受（闭合双点的 sliver 通常 < 1e-4）；`field_polygon_local` 统一走该修复，圈选完成、缓存恢复、生成作业线三处都写入日志。
- 不变量：
  1. 不得用"放宽田界判定"或"跳过田界检查"掩盖无效边界；修复对象是边界本身，门禁语义不变。
  2. 真实交叉（画成"8"字，丢弃面积 > 2%）必须拒绝并提示重画，不得猜测哪一瓣是本意——猜错会静默规划出另一块田。
  3. 空目录必须报出真实原因：`layout.centerline_hard_reasons` 区分"田块边界几何无效"与"作业线在田界外"，不得只显示"无有效中心线"。
  4. 修复证据写入 `layout.field_boundary_geometry`（`repaired`/`reason`/`dropped_area_ratio`/`duplicate_vertex_count`），属于可审计状态，不改变田块面积口径以外的任何语义。
- 实测（同一真实缓存）：修复前 `work_lines=0 / catalog_items=0`，修复后 `work_lines=15 / catalog_items=15`、`kept_line_count=15`；被丢弃面积为 0.0008%（最大片 9361921.3 px² vs 总 9361995.9 px²）。
- 代码锚点：`row_geometry.sanitize_field_polygon(_report)`、`row_geometry.field_polygon_local(_report)`、`path_planner.segment_field_boundary_status`、`path_planner._centerline_hard_reasons`、`MainWindow._finish_field_polygon/_normalise_field_boundary/_restore_cache_payload`。
- 测试锚点：`tests/test_path_boundary.py::test_field_boundary_closing_loop_is_repaired_instead_of_rejecting_everything`、`::test_genuine_field_boundary_crossing_still_fails_closed`、`::test_invalid_field_outline_reports_its_own_reason`。

## DECISION-042：套圈转弯的诊断标签必须报拓扑名，不报底层几何生成器名

- 日期：2026-10-21
- 现象：`state.turn_strategy` 默认已是 `contour`，日志首行也打印 `转弯=contour`，但每条转弯却打印"策略=弓形转弯, 原因=用户指定"，被误判为"套圈没生效、退回了弓形"。
- 根因：`TURN_STRATEGIES["contour"] = _turn_bow`——套圈的转弯几何由弓形生成器实现（DECISION-016 的"共享田头墙横向切线"）；`selected_strategy` 又把 `contour` 改写成 `bow` 存入 `turn_decisions[-1]["strategy"]`，日志按该字段查 `TURN_STRATEGY_LABELS`，于是把拓扑名换成了几何名。
- 修正：日志按 `requested_strategy == "contour"` 或 `ordering_strategy == "contour"` 判定，输出 `套圈转弯（两半交错，共享田头墙横向切线）`、原因 `套圈收割拓扑`；`TURN_STRATEGY_LABELS["contour"]` 由 `套圈收割` 改为 `套圈转弯`。
- 不变量：
  1. 面向人的诊断字段必须表达**用户选择的拓扑**；底层复用哪个几何生成器属于实现细节，不得出现在策略标签里。
  2. `turn_decisions` 必须同时保留 `requested_strategy`、`ordering_strategy`、`effective_strategy` 三个字段，禁止只留 `strategy`。
- 口径确认（2026-10-21，用户选定）：保留两半交错拓扑。跨度 14–16 m 是"每条作业线跨半田去对面那条"的必然结果，不是普通行距，也不是弓形退化；不做相邻顺序紧凑掉头，不做闭合轮廓螺旋套圈。
- 代码锚点：`path_planner.TURN_STRATEGY_LABELS`、`path_planner` 转弯日志段、`_contour_order_work_lines`。

## DECISION-043：工作流面板的六处交互缺陷（步骤停留、标注遮挡、勾选卡顿、选填二级折叠）

- 日期：2026-10-21
- 现象（用户截图）：①掩膜优化运行时进度条压住「处理强度」标签；②生成作业线后「选择作业线」卡片刚展开就被推到下一步，来不及看；③14 条作业线的编号在端头叠成一片、看不出线从哪到哪；④勾选/取消作业线明显卡顿；⑤起点已确认却「路径规划」不可用，且没有任何说明；⑥终点、卸粮点与必选的起点平铺，看着像也必须填。
- 根因：
  1. `StepCard` 展开时把内容区 `maximumHeight` 固定成展开瞬间的高度；进度条是运行中才出现、行会变高，但卡片不重算，布局把进度条挤到标签上。
  2. 生成后的自动全选写的是 `selection_origin="default_all"`，而状态机把「有选中」直接当成完成，`_refresh_cards` 立刻把焦点交给下一步。
  3. 所有编号都画在 `points[0] + (8, 8)`：14 条平行线的端点在同一排，标签必然互相压字。
  4. 每次勾选都 `_save_system_undo`（系统快照）+ `update_centerline_selection`（销毁重建 14 行）+ 13 行/10 张卡片重写 `setStyleSheet`；实测单次约 32 ms，连点就成卡顿。
  5. 状态机把「高级设置（农机参数确认）」当成规划前置门槛；但高级设置按惯例是可动可不动、用默认值即可，用户被逼着先"确认"才能规划。
  6. 选填行与必选行同层同样式。
- 修正：
  1. 进度条位置**常驻**：`ShimmerBar` 始终占 4 px，空闲时 `set_idle(True)` 不绘制（用户要求"预留出进度条的位置"），行高与卡片高度从头到尾不变；`TaskRow._sync_row_height` 仍会在行高变化时回调 `refresh_content_geometry`，供选填分组、路线编辑工具行等其他可见性变化使用。
  2. `derive_task_statuses`：生成后的报告写 `selection_confirmed=False`，`centerline_select` 保持 `available`、动作为「确认」；勾选/取消不会把已确认改回未确认，确认动作走新信号 `centerline_selection_confirm_requested`（点行仍只展开卡片，只有点按钮才确认）。
  3. 编号做成**压在作业线上、但绝不断线**的标注（用户要求"一定要在线上，否则分不清对应哪条"，且明确"不要断线，不能影响用户判断"）：**不用实心色块**（用户否掉"大黑块"），也不在编号处切开折线（试过 AutoCAD `DIMGAP` 式"线断、数字嵌缺口"，被用户否掉：线一断就判断不出这条线从哪到哪）；最终做法是**线整条画完（一条线的所有折点一次画完，不做缺口）+ 编号写全称「中心线 N」（用户要求：只写数字信息量不够）+ 白字 1.5 px 深色描边**（Mapbox `text-halo`／QGIS label buffer 的写法，描边代替底色块）。文字中心对到各线中点（投影区间 50%），随线旋转；碰撞判定用分离轴（SAT）判真实矩形，并**先把矩形沿自身两轴外扩 2 px 再判**（只判不重叠会让编号紧贴成一堵字墙）；缩放后防抖重画、放大时逐步补齐（实测 fit 7 → zoom1 13 → zoom2/3 14）；每条线两个端点各画一个圆点。不要用富文本 span 做底色（`QGraphicsTextItem` 会按文档默认页宽排版，实测 boundingRect 宽 1118 px）。
  4. 勾选走 `save_centerline_selection_undo`（只快照选择相关 6 个字段）、`sync_centerline_selection`（只改勾选状态与计数）、落盘用 400 ms 单次定时器合并；`TaskRow.set_status` 与 `StepCard.refresh_state` 在状态未变时直接返回（换主题时 `force=True`）。
  5. 取消该门槛：`ready_to_plan` 不再要求 `harvester_params_confirmed`（默认参数由 `_effective_harvester_params` 从配置补齐，参数来源在导出/校验阶段才把关）；「高级设置」标记为 optional、状态显示「默认参数（可修改）」，卡片回到「路径规划」之后；plan 行仍把 blocker 直接写成状态文字（例如「先设置并确认起点」）。
  6. 新增 `OptionalRowGroup`：终点/卸粮点折进「选填：终点与卸粮点」二级分组，默认收起、表头暗色、右侧摘要「均未设置 / 终点已设」。
  7. 浮岛式卡片留白（用户要求"预留足够空隙，不然看着累"）：卡片间距 4→10 px、卡片内边距 7/6→12/10、内容行距 2→8、行内边距 8/4→10/6；横向仍不允许出现滚动条（既有测试兜底）。
- 不变量：
  1. 卡片内任何子控件高度变化都必须让卡片按 `sizeHint` 重算高度，不得依赖展开瞬间的快照高度。
  2. 系统默认全选是**建议**不是确认；只有报告里 `selection_confirmed=True`（用户点过确认）才算完成该步骤。
  3. 「选择作业线」的整行点击只展开卡片，确认只由行内按钮触发（`TaskRow.clicked` 与 `action_clicked` 分流）。
  4. 可选步骤不得成为必选门禁：`params`（高级设置）只影响参数来源标注，不阻塞规划；卡片顺序体现依赖，锁定原因在目标行就地可见。
  5. 作业线标注优先保证「不叠字」：同一横截线对齐、旋转、SAT 碰撞过滤、按间距自动降字号；被过滤掉的标签宁可少画（fit 视图 7 条等距），放大后逐步补齐（13 → 14 条），不得叠画。
  6. 高频交互路径（勾选、切换行状态）禁止抓系统快照、重建行控件、重写未变化的样式表。
- 代码锚点：`panels.TaskRow._sync_row_height/_notify_card_height_change`、`panels.StepCard._on_expand_finished/refresh_state(force)`、`panels.OptionalRowGroup`、`panels.TaskPanel._on_task_action/_sync_optional_summary`、`task_state.derive_task_statuses`、`state.AppState.save_centerline_selection_undo`、`ui_motion.ShimmerBar.set_idle`、`MainWindow._confirm_centerline_selection/_schedule_centerline_selection_persist/_aligned_label_anchors/_overlay_text_polygon/_polygons_overlap/_on_view_scale_changed`、`ImageViewer.draw_text(rotation=)`。
- 测试锚点：`tests/test_align_logic.py::test_mask_progress_bar_does_not_cover_strength_label`、`::test_optional_points_live_in_a_collapsed_secondary_group`、`::test_plan_row_shows_why_it_is_still_pending`、`::test_workflow_separates_centerlines_from_path_planning_and_advanced_settings`；`tests/test_task_row_click.py::test_action_button_click_is_reported_separately_from_row_click`、`::test_centerline_selection_action_requests_confirmation`；`tests/test_work_band_selection.py::test_default_all_selection_keeps_the_selection_step_open`、`::test_confirm_centerline_selection_advances_the_workflow`、`::test_checkbox_toggle_reuses_rows_and_keeps_undo_snapshot_light`、`::test_unchecked_centerline_is_hidden_from_canvas_overlay`（端点契约）、`::test_centerline_label_is_a_halo_badge_centered_on_its_line`（编号压在线上、用描边而非底色块）。

## DECISION-044：产品改名 reUI → ALIGN（2026-10-22）

- 背景：用户要求换成"像 TRAE / workbuddy / ChatGPT 那样"的短名，且缩写展开后要能组成一个有正面含义的词（对标 GOAT = Greatest Of All Time）。
- 选定：**ALIGN = Autonomous Line-guided Intelligent Guidance & Navigation**。`ALIGN` 本身即"对齐"，直指课题核心"对行"；五个字母可连读，像成熟软件名。
- 落点：程序名与窗口标题（`ALIGN 农机作业规划系统`）、EXE（`align.exe`）、安装包（`align-Setup-*.exe`）、便携包（`align-Windows-x64-*.zip`）、构件脚本（`packaging/align.spec`、`packaging/align_setup.iss`、`packaging/align.ico`）、环境变量前缀（`REUI_*` → `ALIGN_*`）、GitHub 仓库名。
- 数据迁移：数据目录改为 `%LOCALAPPDATA%\ALIGN`；`runtime_paths.user_data_root()` 在目标目录不存在时把旧的 `%LOCALAPPDATA%\reUI` 整体 `rename`（失败则 `copytree`）过来，只搬一次，用户既有项目与缓存不丢。`REUI_USER_DATA_DIR` 保留为环境变量兼容别名。
- 历史记录不改写：`CHANGELOG.md` 旧条目与本文 043 之前的条目继续使用旧名，属"当时的事实"。
- 不变量：
  1. 迁移只在目标目录不存在时发生，且绝不删除旧目录内容（rename 失败退化为复制）。
  2. 测试统一通过 `ALIGN_USER_DATA_DIR` 指向临时目录，不触发迁移。
  3. 安装包 AppId GUID 保持不变，改名后的安装包可以直接覆盖升级旧版本。
- 代码锚点：`runtime_paths.APP_DATA_NAME/_migrate_legacy_data_root/user_data_root`。
- 测试锚点：`tests/test_runtime_packaging.py::test_version_has_one_runtime_source`、`::test_build_spec_keeps_exe_at_release_root`。


## DECISION-045：GPU 支持改为「应用内按需下载运行时」（2026-10-22）

- 背景：用户质疑「为什么不能在系统内一键配置 CPU/GPU，不然用户怎么知道怎么配置」。原先只有 CPU 版发布包，GPU 要自己建环境，普通用户无法操作。
- 约束：
  1. PyInstaller 冻结后没有 pip/python，应用内不能装包，只能替换或外挂一个运行时目录。
  2. CUDA 版 torch 解压 4.3 GB、wheel 2.86 GB，超过 GitHub Release 单文件 2 GiB 上限，塞不进安装包；两个安装包（761 MB / 1.7 GB）会让下载页更难选。
  3. 打包时 torch 是以包目录收集进 `_internal`（不是冻结进 PYZ），因此 `sys.path` 优先级对它有效。
- 方案：发布包保持单个 CPU 版；「设置 → 运行设备」检测显卡并在用户点击后下载 CUDA 运行时到 `%LOCALAPPDATA%\ALIGN\gpu-runtime`，启动时 `device_runtime.activate()` 把它插到 `sys.path[0]`，`import torch` 即拿到 CUDA 版；移除目录自动退回 CPU。
- 不变量：
  1. 挂载必须发生在任何 `torch` 导入之前（`frozen_main.py` 在 `from app_entry import run_app` 之前调用），`torch` 已在 `sys.modules` 时 `activate()` 返回 False，不假装成功。
  2. GPU 运行时与内置 torch 严格同版本（torch 2.9.1 / torchvision 0.24.1），版本不一致会 `_C.pyd` ABI 崩。
  3. 安装过程失败/取消/磁盘不足必须清理半成品，避免下次启动挂载残缺 torch（`install_gpu_runtime` 的 `except BaseException` 分支 `rmtree`）。
  4. 没有 NVIDIA 显卡时不给「启用」按钮，文案明确写「不需要做任何设置」，不制造焦虑。
  5. 下载走镜像回退（上海交大 → PyTorch 官方），每个 wheel 校验官方 sha256。
- 代码锚点：`src/device_runtime.py`（`activate/install_gpu_runtime/remove_gpu_runtime/detect_nvidia_gpu/torch_status`）、`src/frozen_main.py`、`src/main.py`、`src/app_entry.py::describe_device`、`src/pyside6_app/dialogs.py::SettingsDialog._device_tab/_apply_device_state/_start_gpu_install`。
- 测试锚点：`tests/test_device_runtime.py`（19 条：目录/挂载幂等/已导入不挂载/移除/显卡解析/状态降级/安装解包/失败清理/取消/磁盘不足/版本对齐/设置页三个状态分支）。

# 数据与溯源契约

## 坐标约定

- 显示和路径编辑使用 GeoTIFF 降采样预览像素坐标。
- `GeoUtils` 使用与预览尺寸对应的仿射变换，将像素中心转换为源 CRS，再转换为 WGS84。
- GeoJSON、CSV、KML 和 JSON 使用带字段名或标准规定的 `lon, lat` 顺序。
- `$PATH` 候选路径按当前兼容格式写入 `lat, lon`；该顺序必须与目标终端协议逐项核对，不能由其他导出格式推断。
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
- `headland_mask`：方向和位置证据支持的田头作物候选，用于田头碾压统计和全田真实作物碾压并集，不等同于低质量主体；与主体重叠的像素只计一次。
- `headland_corridor_mask`：仅在田块端部、跨越足够多行且具有连续横向证据时重建的完整田头走廊；其语义优先于纵向规则化主体。
- `uncertain_residual_mask`：不能可靠归属到主体或田头的原始残差，只用于显示和中性场地支撑。
- `planning_support_mask`：场地支撑并集，只用于越界/支撑判断，不得替代 `processed_mask` 计算作物覆盖或碾压。
- `forbidden_regions`：用户在显示坐标中确认的禁行多边形；规划时按掩膜偏移栅格化，不得由不确定残差自动生成。

这些数组与处理后主体保存在同一个原子 NPZ 产物中；项目 JSON 只保存轻量元数据。

## 路径段语义

- `work`：由作物中心线支持、启用割刀几何覆盖计算的作业段。
- `turn_approach`：从作物端点到校正转弯锚点的田头直行段，不计入割台作物覆盖。
- `turn`：满足策略几何和最小转弯半径的前进转弯段。
- `turn_reverse` / `turn_aux`：包含倒车或辅助折返动作的转弯段。
- `track_outside_field_pct`：履带扫掠超出田块硬边界的比例。
- `track_outside_support_pct`：履带扫掠离开语义支撑区的比例，不能替代田界指标。
- `track_uncertain_overlap_pct`：履带扫掠经过未确认残差的比例，仅在提供该图层时有效。
- `track_forbidden_overlap_pct`：履带扫掠进入人工确认禁行区的比例，仅在提供该图层时有效。
- `machine_outside_field_pct`：二维刚体模型中机身平面包络扫掠越出田块硬边界的比例；它不等同于履带接地越界率，也不等同于真实 CAD 外廓越界率。
- `machine_forbidden_overlap_pct`：二维刚体模型中机身平面包络扫掠进入人工确认禁行区的比例；它不替代履带禁行区指标。
- `footprint_model=planar_rigid_footprint_v2`：路径点表示 `track_contact_center`；左右履带由宽度、中心距和接地长度构成有向矩形。割刀由横向有效割幅线段表示，其纵向位置必须来自实测 `cutter_forward_offset_m`。演示和验证必须调用同一几何定义；`cutter_visual` 仅用于 UI 中显示有厚度的割台外观，不参与覆盖率、碾压率或足迹校验。该模型仍不包含土壤压实、打滑、坡度、悬挂、惯性、GNSS 天线偏移或真实 CAD 外廓。
- `rolling_body_pct`：履带扫掠与 `processed_mask` 主体重叠面积 / 主体作物面积。
- `rolling_headland_pct`：履带扫掠与 `headland_mask` 重叠面积 / 田头作物面积；缺少田头层时为 `null`。
- `rolling_canopy_pct`：履带扫掠与 `processed_mask | headland_mask` 去重并集的重叠面积 / 全部已确认作物面积；作为界面和论文中的“预计碾压率”，不得加入农业价值权重。
- `physical_crop_rolling_area_m2`：履带扫掠压过的全部已确认作物物理面积。
- `agronomic_damage_cost_m2`：`track_body_overlap_area_m2 + headland_damage_weight * track_headland_overlap_area_m2`；仅用于候选路径排序，不得称为碾压率。
- `headland_damage_weight`：田头单位物理碾压面积相对主体的规划权重，范围 `(0,1]`，无田间标定时必须使用 `1.0`。
- `rolling_core_pct`：履带扫掠与作物核心区重叠面积 / 作物核心区总面积；只作为详细风险指标。
- `track_core_overlap_pct`：履带扫掠中落在作物核心区的面积 / 履带总扫掠面积；用于路径安全门限，不得称为预计碾压率。

碾压模型只使用左右履带矩形扫掠，不把机身或割刀计入碾压。割刀线只参与主体作物掩膜几何覆盖率。通过硬约束的候选路径先按独立的 `agronomic_damage_cost_m2` 排序，再比较不加权的 `rolling_canopy_pct`、语义风险、核心区重叠、几何覆盖率和长度。`rolling_crop_pct` 已废止，不得导出或作为兼容别名继续产生第二种“碾压率”定义。无掩膜区域不得仅凭分割结果自动视为确认无作物道路。

- `crop_mask_geometric_coverage_pct`：割刀有效割幅线段沿全部 `work` 段扫掠后，与处理后作物主体掩膜的交集面积 / 作物主体掩膜面积。
- `planned_target_geometric_coverage_pct`：同一割刀扫掠区域对规划目标掩膜的几何覆盖比例，是路径有效性的默认覆盖门限。
- `detected_mask_geometric_coverage_pct`：同一割刀扫掠区域对原始识别掩膜的几何覆盖比例，只用于对照诊断。

上述三个指标均不代表实际收获率、产量或割台损失。缺少实测 `cutter_forward_offset_m` 时不得生成几何覆盖率或路径；系统配置中该字段默认 `null`。

验证和演示必须调用 `path_pose` 中同一个按弧长插值、航向窗口和倒车方向算法。`footprint_pose_heading_window_m` 随验证结果记录；作业、转弯、进出场速度来自农机配置，播放倍率只改变观看快慢，不改变路径几何或足迹指标。

任何米制掩膜、路径和足迹函数缺少有效地理配准或掩膜状态时必须失败关闭，不得回退为约定像素比例。农机平面几何所需字段必须完整传入，不得在几何模块内部补造缺失尺寸。

视觉作物端点不得直接冒充物理转弯锚点；新增 `turn_approach` 必须保留在框选田块内并记录校正距离。

## 导出清单

`<route>.<ext>.manifest.json` 至少包含：

- 应用版本和 Git 提交号。
- 内部路径参考点 `track_contact_center` 与足迹模型版本；它不等同于目标终端的 GNSS 天线参考点。
- 输出文件 SHA-256、点数和坐标系。
- 源 GeoTIFF 完整 SHA-256、尺寸、波段、NoData、CRS、仿射和范围。
- 模型 SHA-256、模型注册信息。
- 三阶段完整证据、验证指标、农机参数和规划因素。
- Python 和核心依赖版本。
- 研究查看或受控田间试验用途限制、路径状态、详细检查项和已确认禁行区。

路径文件与清单必须作为一个归档单元保存。清单生成失败时，路径文件会被删除。

## 掩膜诊断包

掩膜处理完成后可导出 ZIP 诊断包，其中必须包含：

- `layers.npz`：原始掩膜、规则化主体、主体残差、残差合并结果、补缝新增、支撑约束移除、最终主体、田头、不确定残差和规划支撑等图层。
- `components.csv` / `components.json`：每个连通区域的面积、方向、主体距离、接触比例、端部外伸、最近行带和复核标志。
- `overview.png`：使用固定颜色区分处理阶段，并用编号标出需复核区域。
- `manifest.json`：算法版本、米制分辨率、处理阈值、各层像素数和复核区域总数。

诊断标志只用于定位异常，不得自动删除作物主体；算法调整必须结合原始影像和真实田块证据。

田头走廊修复必须满足端部位置、跨行连续性、原始田头种子占比和最大主体切入深度四项约束。走廊确认后，主体掩膜、行带轮廓和中心线必须同步裁剪；被移出的原始像素转入田头层，不得丢弃。内部横向结构和稀疏端部碎片不得自动升级为田头走廊。

## `$PATH` 候选路径契约

- 每个路径点当前写为 `$PATH,route_id,index,total,lat,lon,offset_x,offset_y,status*`。
- `index` 从 `0` 开始；坐标系为 `EPSG:4326`；坐标字段顺序为纬度、经度。
- 状态码当前约定为：普通转弯 `0`、作业 `1`、进田 `2`、出田 `3`、倒车或辅助转弯 `4`。
- 文件末尾写入配置中的 `$INFM` 记录。
- 以上只是当前软件兼容格式，不代表目标收获机已经验证这些字段、状态码、索引和结束记录。
- manifest 必须记录 `candidate_machine_path=true`、`terminal_protocol_validated=false` 及完整 `path_protocol`。

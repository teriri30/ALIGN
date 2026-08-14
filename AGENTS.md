# reUI Agent Entry Point

本项目是科研可审计原型，不是通用农机自动驾驶产品。任何 AI 或开发者修改代码前必须依次阅读：

1. `AI_RULES.md`
2. `docs/DECISIONS.md`
3. `docs/DATA_CONTRACTS.md`
4. 与修改模块相关的 `DECISION-*` 代码锚点和不变量测试

修改前必须列出受影响的 `DECISION-*` 编号。修改科研链路时，不得只改实现而不核对对应测试。

默认验证环境：

```powershell
D:\zhl\anaconda\envs\game\python.exe -m compileall -q .
D:\zhl\anaconda\envs\game\python.exe -m pytest -q
```

Git Bash / Hermes 环境使用 `bash scripts/test.sh -q`，脚本会清理继承的 `PYTHONPATH` 并支持用 `REUI_PYTHON` 覆盖解释器路径。

本文件只负责入口。完整规则以 `AI_RULES.md` 和 `docs/DECISIONS.md` 为准。

# reUI 长期项目事实

- 当前可访问的目标仓库为 `E:\zhlonly\project\reUI`，是 PySide6 农用路径规划系统；用户迁移指令中的 `D:\zhlonly\project\reUI` 当前不存在，若两者同时出现必须先核验是否为同一工作树，不得自动创建或同步替代目录。
- 论文主线为 UAV + YOLO11-seg + 宽窄行中心线对齐；关键组件包括 YOLO11-seg、SAHI 和 TIF 地理参考。
- `mask_processor.py`：PCA+FFT 两阶段方向检测与掩膜正则化，处理碎片聚合、黏连切割和断裂修复。
- `row_geometry.py`：`BandInstance`/`Corridor` 及行带排序配对、宽/窄行分类、中心线平滑和通行宽度建模。
- `footprint_planner.py`：履带足迹几何模型参数为 `track_width=0.40m`、`track_gauge=1.60m`、`cutter_width=2.15m`、`wheelbase=1.80m`、`turn_radius=0.90m`；模型名为 `planar_rigid_footprint_v2`。
- `path_planner.py`：Bow、Semicircle、Pear、Fishtail 四类转弯策略，Gaussian kernel 路径平滑，使用 `pyproj` 导出 WGS84；`path_planner.py` 同时处理田块边界/田头通行廊道，`inference.py` 使用 YOLO11 + SAHI。
- 履带碾压评价是基于足迹几何模型的理论估算，包括理论碾压率、重复碾压率和田头转弯碾压面积；没有真实田间记录时，不得表述为实测作物损伤、产量影响或已完成田间验证。

# 路径规划不变量

- `contour` 是作业线排序策略，不是几何生成器；默认策略不能无依据退回 `bow`。
- 只有宽行生成作业线，窄行不直接生成作业线；等距宽区必须按割幅拆分覆盖，不能整块排除。
- 宽窄行分类使用 `profile_width_m`/法向剖面宽度，不使用 bbox `width_m`；不要启用会强行交替宽窄行的 alternation prior。
- 缓存 JSON 中的 `wide_bands`/`all_bands` 可能被截断；需要完整行带几何时必须重新运行 `process_mask`，并恢复 `state.mask_raw`。
- 显示坐标到地理坐标的 affine 缩放必须按当前显示比例乘以 `SCALE`，不能误除；涉及 ppm、掩膜正则化或路径距离时先验证坐标变换。
- 全链路复现优先使用真实 GeoTIFF、真实模型和真实缓存状态；合成数据只能辅助定位，不能绕过田块边界、行带截断、坐标变换或转弯分支。

# 修改与调试纪律

- 修改前依次阅读 `AI_RULES.md`、`docs/DECISIONS.md`、`docs/DATA_CONTRACTS.md` 以及相关 `DECISION-*` 锚点和不变量测试，并列出受影响的决策编号。
- 先建立能针对用户症状变红、修复后变绿的紧反馈回路；完成根因调查后再修改，一次只验证一个假设，不做无关重构。
- PySide6 布局验证必须构造真实窗口，执行 `show()` 和 `app.processEvents()` 后再量几何；重点检查 QScrollArea、折叠卡片、父子布局最小高度和滚动条，不能用未 show 的窗口或预先 resize 成 `sizeHint()` 制造假绿。
- UI、worker、缓存、坐标空间、撤销/恢复和导出是同一条状态链；项目切换要清理旧状态，延迟 worker 结果要校验项目身份，手工改线要使旧验证失效，机器格式导出要从当前路径和当前证据重新检查 readiness/provenance。
- readiness 中的布尔值必须严格检查 `value is True`，数值必须存在且为有限数；不得把字符串 `"false"`、`NaN`、无穷大或陈旧摘要当作有效证据。
- 原生崩溃或无 traceback 时，优先用目标解释器、真实输入和独立子进程复现，记录退出码、`faulthandler`、模块路径和环境污染；不要把最后一条普通日志直接当成根因。
- 临时探针、测试脚本、截图和输出写入系统临时目录或明确的临时目录，用完删除；不得把一次性脚本、缓存和生成物留在仓库中。

# 交付验证

- 每次修改影响系统运行行为的源码、配置、模型装载或打包文件后，交付前必须运行 `build_windows.bat D:\zhl\anaconda\envs\game\python.exe`，同步生成仓库根目录的 `reUI.exe`、`_internal`、`version.json` 和 `dist\reUI`；不能只交付源码而声称系统已更新。
- EXE 同步必须实际回读验证：确认根目录与 `dist\reUI\reUI.exe` 的 SHA-256 一致、文件时间晚于本次源码修改，并运行 `dist\reUI\reUI.exe --self-test`；构建、测试或自检失败时，必须明确报告 EXE 未同步。
- 交付时报告实际修改文件、决策锚点、聚焦测试与完整测试；明确区分 code-green、目标运行时验证、真实数据验证和田间证据。
- 代码、文档、缓存和导出产物的状态必须可追溯；不要把通过单元测试写成真实农机执行已经验证。
- 修改后回读关键配置、参数、测试结果和输出路径；遇到 D/E 路径差异、缺失真实数据、缺失模型或外部环境不可用时，如实报告“待核验/需补充”，不得编造结果。

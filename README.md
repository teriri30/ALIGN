# reUI 智能农机规划系统

`reUI` 是面向宽窄行再生稻无人自主收获收割机对行低碾压作业的科研系统。系统实现 GeoTIFF 导入、田块圈选、实例分割、掩膜规整与种植带重建、作业带确定、作业中心线生成、农机约束路径规划、路径验证、终端路径导出和 WGS84 坐标导出。

## 普通用户使用（Windows 便携版）

普通用户请从 [下载最新 Windows 便携版](https://github.com/teriri30/reUI/releases/latest) 获取 ZIP。完整解压后，双击最外层的 `reUI.exe`；不要把 EXE 单独移出 `_internal` 文件夹。便携版已携带 Python 运行库、PySide6、CPU 模型推理依赖、GeoTIFF/坐标依赖和内置模型，不需要安装 Python、Conda、pip、CUDA 或项目依赖；因此下载到普通 Windows 电脑即可运行。开发者本地 CUDA 环境仍保留，GPU 版可按本地构建流程单独生成。

GitHub 的 `Code → Download ZIP` 是源码包，只适合开发者，不包含可直接运行的 EXE。普通用户应下载 Release 中的 `reUI-Windows-x64-<version>.zip`，解压后即可运行整个便携项目。

维护者发布新版本时，可以推送 `v*` 标签，或在 GitHub Actions 的 `build-release` 中填写 `release_tag` 手动发布；工作流会先完成测试、构建和 `reUI.exe --self-test`，通过后才上传 Release。

程序配置、缓存、日志和用户模型保存在 `%LOCALAPPDATA%\\reUI`，更新程序不会覆盖这些数据。开发者运行 `update.bat` 后，脚本会先确认源码分支已推送，再询问是否创建 `v*` 标签；标签触发 Actions 构建并发布完整 Release ZIP。它不是普通用户的 EXE 更新器。

作为科研软件，模型结果用于记录当前路径的几何指标，具体口径仍是下文所述的理论几何估算。项目数据也不是只保存“最后一个画面”：系统同时保存最近完成状态和最近 50 步逐操作历史；田块圈选、原始掩膜、优化后掩膜、中心线选择、服务点和路径等状态可在重新打开项目后继续逐步撤销/重做。大掩膜按内容哈希独立存储并在多个历史步骤间复用，源 GeoTIFF 发生变化或历史产物校验失败时会拒绝恢复不可信历史。

## 目录结构

发布后的项目根目录直接包含 `reUI.exe` 和 `_internal/`；应用源码统一位于
`src/`，PySide6 界面位于 `src/pyside6_app/`，测试、构建脚本、文档和模型数据
分别位于 `tests/`、`packaging/`、`scripts/`、`docs/`、`data/` 和 `developer/`。

掩膜处理统一使用“标准”档位；界面不再提供轻量、增强或强力切换，旧配置中的这些名称会自动归一为标准。

## 项目定位

<!-- DECISION-008: 科研适用范围与指标含义。修改前阅读 docs/DECISIONS.md。 -->

- 用途：面向无人自主收获收割机的论文方法验证、路径生成、导航终端导入、田间自主作业试验和结果记录。
- 农机尺寸建议基于当前影像中的行带宽度、田头空间和已填写的机具参数计算。
- 当前工具链还包括禁行区确认、起点/终点与卸粮点、路线编辑、混合种植区域处理、农机兼容 `$PATH` 测试文件和掩膜诊断包。
- 论文实验链覆盖种植带感知、作业带确定、中心线生成、路径生成、终端导入和田间自主执行评价；转弯策略是已实现的固定路径生成方式，只有形成独立对照试验时才报告策略差异，不默认宣称策略优化。

## 运行环境

已验证环境为 `D:\zhl\anaconda\envs\game`：Python 3.10.20、PyTorch 2.9.1+cu128、CUDA 12.8。完整版本见 `developer/environment.game.yml` 和 `developer/requirements/requirements.txt`。

```powershell
D:\zhl\anaconda\envs\game\python.exe src\main.py
```

## 验证

```powershell
D:\zhl\anaconda\envs\game\python.exe -m compileall -q .
D:\zhl\anaconda\envs\game\python.exe -m pytest -q
D:\zhl\anaconda\envs\game\python.exe -m pip check
```

在 Git Bash、Hermes 或其他可能注入 Python 环境变量的终端中，使用项目脚本以隔离 `PYTHONPATH`：

```bash
bash scripts/test.sh -q
# 或指定解释器：REUI_PYTHON=/d/zhl/anaconda/envs/game/python.exe bash scripts/test.sh -q
```

导出后可独立验证路径文件和阶段证据；加入 `--verify-sources` 会重新哈希源 GeoTIFF 和模型：

```powershell
D:\zhl\anaconda\envs\game\python.exe src\integrity_check.py path_result.csv.manifest.json --verify-sources
```

## 数据安全规则

1. 没有有效 CRS、仿射变换或米制尺度时，禁止掩膜处理、路径规划和地理导出。
2. 推理分块失败时整次推理失败，不把失败分块当作背景。
3. 模型、配置、掩膜、农机参数、调头策略或服务点变化时，下游结果失效。
4. 路径状态显示当前几何与足迹校验结果；理论碾压率、重叠率和覆盖率用于结果记录，不作为科研导出的硬门禁。
5. 正式导出同时生成 `.manifest.json`，记录源影像、模型、参数、算法版本、运行环境和阶段指纹。
6. 手工修改会立即清除旧自动规划证据；可用“验证人工路线”按当前掩膜、田界、禁行区和农机足迹重算几何指标。科研导出保留来源与校验记录，不要求机器就绪记录。

## 输入与输出

- 输入：带有效 CRS 和仿射变换的 GeoTIFF、可信 `.pt` 分割模型、农机几何参数。
- 输出：GeoJSON、CSV、KML、JSON 或 `$PATH`，坐标统一为 WGS84 经度/纬度。
- 每个正式输出都应与同名 `.manifest.json` 一起归档。

## 默认工作流

算法依赖、界面步骤、论文环节和田间证据的统一定义见 [研究、算法、界面与田间验证统一流程](docs/RESEARCH_SYSTEM_WORKFLOW.md)。

- 掩膜处理后，系统先依据宽度、方向、连续性和田界证据确定作业带：宽窄行区按四行宽行带确定作业带，等距区按割幅确定作业带，窄行不直接生成作业线。
- 在上述作业带确定结果基础上，左侧任务清单的“生成中心线”步骤生成整田候选中心线；随后进入“选择中心线”步骤逐条勾选，取消勾选的线会隐藏且不参与规划，最后设置起点/终点并点击“路径规划”统一连接掉头。
- 弓形、半圆形、梨形和鱼尾形是四种已实现的固定转弯策略；系统校验所选转弯几何，不搜索田头偏移，也不进行转弯路径优化。
- GeoJSON、CSV、KML、JSON 和 `$PATH` 均可作为科研结果导出，并同步保存参数、坐标和校验记录。
- “农机兼容测试文件”仅用于核对目标终端的 `$PATH` 文本格式和直线化转弯投影；用于田间自主作业的路径还必须在目标无人收获收割机终端中导入，并以实际执行轨迹和作业后调查作为试验证据。

当前足迹验证采用 `planar_rigid_footprint_v2` 二维平面模型。路径参考点为两条履带接地区域的几何中心；预计碾压率按有限长度有向矩形扫掠计算，作物掩膜几何覆盖按具有实测前置距离的横向有效割幅线段扫掠计算。割刀前置距离、履带宽度和中心距等参数应按实测值填写。模型结果用于记录当前路径的理论几何指标，不能替代作业后实际碾压调查；土壤压实、打滑、坡度、悬挂、惯性、GNSS 天线偏移和完整车身外廓可作为田间试验或扩展模型评价因素。

## 文档

- [AI 修改规则](AI_RULES.md)
- [研究、算法、界面与田间验证统一流程](docs/RESEARCH_SYSTEM_WORKFLOW.md)
- [关键设计决策](docs/DECISIONS.md)
- [科研验证协议](docs/SCIENTIFIC_VALIDATION.md)
- [数据与溯源契约](docs/DATA_CONTRACTS.md)
- [使用说明](docs/USER_MANUAL.md)

## 田间自主作业试验资料归档

- 如使用 GCP 或 RTK，应归档其测量文件和精度说明，用于量化 GeoTIFF 的绝对定位误差；未使用时应明确只评价影像坐标下的内部一致性。
- 模型注册表可补充训练数据版本、类别映射和训练预处理，以便复现实例分割训练。
- 应归档目标农机终端格式核对、路径导入结果、实际自主执行轨迹、人工接管或中断记录，用于校核导出结果和自主作业效果。
- 田间试验应记录作业、转弯、进出场速度、作业时间、实际轨迹及碾压调查；软件演示按 GeoTIFF 米制距离和名义速度播放，只能用于观察路径顺序、转弯连接和理论足迹变化。

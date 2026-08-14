# reUI 智能农机规划系统

`reUI` 是面向宽窄行再生稻低碾压收获路径规划的科研软件。系统实现 GeoTIFF 导入、田块圈选、实例分割、行带正则化、农机约束路径规划、路径验证、模拟和 WGS84 路径导出。

## 普通用户使用（Windows 便携版）

普通用户使用发布文件夹根目录的 `reUI.exe`，完整保留旁边的 `_internal` 文件夹后双击启动。源码仓库和运行包是两种交付物；直接下载 GitHub 的 `Code → Download ZIP` 不会包含本地构建产生的 EXE。便携版已携带 Python 运行库、PySide6、模型推理依赖、GeoTIFF/坐标依赖和内置模型，不需要安装 Python、Conda、pip 或项目依赖。使用 GPU 推理时仍需要电脑安装兼容的 NVIDIA 显卡驱动。

程序配置、缓存、日志和用户模型保存在 `%LOCALAPPDATA%\\reUI`，更新程序不会覆盖这些数据。`update.bat` 是开发者同步源代码到 GitHub 的工具，不是普通用户的 EXE 更新器。

## 目录结构

发布后的项目根目录直接包含 `reUI.exe` 和 `_internal/`；应用源码统一位于
`src/`，PySide6 界面位于 `src/pyside6_app/`，测试、构建脚本、文档和模型数据
分别位于 `tests/`、`packaging/`、`scripts/`、`docs/`、`data/` 和 `developer/`。

掩膜处理统一使用“标准”档位；界面不再提供轻量、增强或强力切换，旧配置中的这些名称会自动归一为标准。

## 项目定位

<!-- DECISION-008: 科研适用范围与指标含义。修改前阅读 docs/DECISIONS.md。 -->

- 用途：论文方法验证、内部实验和田间试验准备。
- 农机尺寸建议基于当前输入的行带几何与配置参数计算；坡度、土壤承载力、重心等因素可在后续试验中补充。

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
4. 路径必须通过作物掩膜几何覆盖率、履带重叠、越界和交叉检查后才能正式导出。
5. 正式导出同时生成 `.manifest.json`，记录源影像、模型、参数、算法版本、运行环境和阶段指纹。
6. 手工修改会立即清除旧自动规划证据；可用“验证人工路线”按当前掩膜、田界、禁行区和农机足迹重算几何指标。人工来源不会冒充自动规划 provenance；正式科研/机器导出继续受来源与 readiness 门禁约束，兼容测试文件只能作为明确标记的预览。

## 输入与输出

- 输入：带有效 CRS 和仿射变换的 GeoTIFF、可信 `.pt` 分割模型、农机几何参数。
- 输出：GeoJSON、CSV、KML、JSON 或 `$PATH`，坐标统一为 WGS84 经度/纬度。
- 每个正式输出都应与同名 `.manifest.json` 一起归档。

## 使用模式

- `田间试验`：默认模式，使用足迹优化和田间试验门限。
- `快速预览`：用于论文对照、算法排错和快速查看。
- `高级校验`：在田间试验门限基础上增加终端格式、车辆标定和实际轨迹记录检查。

当前足迹验证采用 `planar_rigid_footprint_v2` 二维平面模型。路径参考点为两条履带接地区域的几何中心；预计碾压率按有限长度有向矩形扫掠计算，作物掩膜几何覆盖按具有实测前置距离的横向有效割幅线段扫掠计算。割刀前置距离、履带宽度和中心距等参数应按实测值填写。模型结果用于比较不同路径方案的几何差异；土壤压实、打滑、坡度、悬挂、惯性、GNSS 天线偏移和完整车身外廓可作为扩展评价因素。

## 文档

- [AI 修改规则](AI_RULES.md)
- [关键设计决策](docs/DECISIONS.md)
- [科研验证协议](docs/SCIENTIFIC_VALIDATION.md)
- [数据与溯源契约](docs/DATA_CONTRACTS.md)

## 建议补充的试验数据

- GCP 或 RTK 实测可用于量化 GeoTIFF 的绝对定位误差。
- 模型注册表可补充训练数据版本、类别映射和训练预处理，以便复现实例分割训练。
- 可记录目标农机终端格式与实际执行轨迹，用于校核导出结果。
- 农机演示按 GeoTIFF 米制距离和名义速度播放，适合观察路径顺序、转弯连接和足迹变化；打滑、加减速、坡度、土壤、惯性和控制器延迟可在实测阶段一并记录。

# reUI 智能农机规划系统

`reUI` 是面向宽窄行再生稻低碾压收获路径规划的科研原型与辅助决策平台。系统实现 GeoTIFF 导入、田块圈选、实例分割、行带正则化、农机约束路径规划、路径验证、模拟和 WGS84 路径导出。

## 项目定位

<!-- DECISION-008: 科研原型能力边界。修改前阅读 docs/DECISIONS.md。 -->

- 适用：论文方法验证、内部实验、受控条件下的田间试验准备。
- 不适用：未经人工复核直接驱动农机、通用地形安全判断或商业化无人作业。
- 当前农机尺寸建议是启发式候选，不包含 DEM、土壤承载力、重心和侧翻模型。

## 运行环境

已验证环境为 `D:\zhl\anaconda\envs\game`：Python 3.10.20、PyTorch 2.9.1+cu128、CUDA 12.8。完整版本见 `environment.game.yml` 和 `requirements.txt`。

```powershell
D:\zhl\anaconda\envs\game\python.exe pyside6_main.py
```

## 验证

```powershell
D:\zhl\anaconda\envs\game\python.exe -m compileall -q .
D:\zhl\anaconda\envs\game\python.exe -m pytest -q
D:\zhl\anaconda\envs\game\python.exe -m pip check
```

导出后可独立验证路径文件和阶段证据；加入 `--verify-sources` 会重新哈希源 GeoTIFF 和模型：

```powershell
D:\zhl\anaconda\envs\game\python.exe integrity_check.py path_result.csv.manifest.json --verify-sources
```

## 数据安全规则

1. 没有有效 CRS、仿射变换或米制尺度时，禁止掩膜处理、路径规划和地理导出。
2. 推理分块失败时整次推理失败，不把失败分块当作背景。
3. 模型、配置、掩膜、农机参数、调头策略或服务点变化时，下游结果失效。
4. 路径必须通过作物掩膜几何覆盖率、履带重叠、越界和交叉检查后才能正式导出。
5. 正式导出同时生成 `.manifest.json`，记录源影像、模型、参数、算法版本、运行环境和阶段指纹。
6. 手工修改路线后必须重新规划和验证，未经验证的路线禁止导出。

## 输入与输出

- 输入：带有效 CRS 和仿射变换的 GeoTIFF、可信 `.pt` 分割模型、农机几何参数。
- 输出：GeoJSON、CSV、KML、JSON 或 `$PATH`，坐标统一为 WGS84 经度/纬度。
- 每个正式输出都应与同名 `.manifest.json` 一起归档。

## 使用模式

- `田间试验（推荐）`：默认模式，使用足迹优化和田间试验门限；生成后仍需人工检查影像位置、禁行区和路线。
- `快速预览`：用于论文对照、算法排错和快速查看。
- `高级校验`：在田间试验门限基础上继续检查终端格式、车辆标定和实际轨迹证据。

当前足迹验证采用 `planar_rigid_footprint_v2` 二维平面模型。路径参考点为两条履带接地区域的几何中心；履带碾压按有限长度有向矩形扫掠计算，作物掩膜几何覆盖按具有实测前置距离的横向有效割幅线段扫掠计算。缺少割刀前置距离时规划失败关闭。该覆盖指标不等于实际收获率；模型也不包含土壤压实、打滑、坡度、悬挂、惯性、GNSS 天线偏移或真实 CAD 外廓。

## 文档

- [AI 修改规则](AI_RULES.md)
- [关键设计决策](docs/DECISIONS.md)
- [科研验证协议](docs/SCIENTIFIC_VALIDATION.md)
- [数据与溯源契约](docs/DATA_CONTRACTS.md)

## 已知限制

- 没有 GCP 或 RTK 实测时，系统只能保证内部坐标链路一致，不能证明 GeoTIFF 的绝对定位误差。
- 模型注册表中尚未记录训练数据版本、类别映射和训练预处理，补齐前不能完整复现实例分割模型训练。
- 尚未完成目标农机终端协议和真实执行轨迹闭环验证。
- 农机演示按 GeoTIFF 米制距离和名义速度播放，但仍是路径可视化；不包含打滑、加减速、坡度、土壤、惯性和控制器延迟，不能称为实车 1:1 仿真。

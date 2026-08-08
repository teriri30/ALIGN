# AI 修改规则

## 基本原则

本项目采用 DAT（Decision Anchor + Invariant Test）规则。凡是为了数据安全、科研复现、缓存失效、坐标准确性或路径可用性而保留的行为，必须同时具备：

1. `docs/DECISIONS.md` 中的决策记录。
2. 生产代码中的 `DECISION-*` 锚点。
3. 至少一个带相同编号的不变量测试。

不能仅凭“代码可以更短”“逻辑看起来重复”删除决策约束。若确需改变决策，必须在同一次修改中更新决策正文、代码锚点和测试，并说明风险变化。

## 修改前

1. 阅读 `README.md`、`docs/DECISIONS.md`、`docs/DATA_CONTRACTS.md` 和 `docs/SCIENTIFIC_VALIDATION.md`。
2. 搜索受影响模块中的 `DECISION-*`。
3. 在计划或说明中列出本次影响的决策编号；没有影响则明确写“无”。
4. 运行相关测试，确认修改前基线。
5. 工作区已有改动默认属于用户，不得回退或覆盖。

## 禁止行为

1. 禁止新增绕过 `_geo_export_payload()` 的正式地理路径导出入口。
2. 禁止在 CRS、仿射或米制尺度不可靠时继续米制处理或经纬度导出。
3. 禁止把失败推理分块当作背景或继续提交部分成功结果。
4. 禁止模型、源影像、配置、掩膜、农机参数或服务点变化后继续复用旧结果。
5. 禁止删除 provenance、SHA-256、manifest 或缓存失败关闭机制。
6. 禁止让 UI 预览图替代原始 GeoTIFF 作为正式推理输入。
7. 禁止未经说明改变 RGB 波段、NoData 或位深归一化规则。
8. 禁止把本项目描述成可直接驱动农机的成熟生产系统。
9. 禁止在没有更新或确认不变量测试的情况下改变关键科研链路。

## 重点模块

修改以下文件时必须先搜索其决策锚点：

- `geo.py`
- `inference.py`
- `raster_preprocessing.py`
- `mask_processor.py`
- `row_geometry.py`
- `legacy_route_coverage.py`
- `path_planner.py`
- `pyside6_app/main_window.py`
- `pyside6_app/workers.py`
- `provenance.py`
- `cache.py`
- `integrity_check.py`

## 新决策规则

- 新的系统级安全或科研不变量使用下一个未占用的 `DECISION-NNN` 编号。
- 普通实现细节、局部重构和显而易见的代码不创建决策编号。
- 决策正文必须包含：结论、原因、不变量、代码锚点、测试锚点、允许的变更方式。
- 测试应验证行为，不应只检查注释字符串存在。`test_decision_anchors_current.py` 只负责检查三类锚点没有断链，不能替代行为测试。

## 修改完成

提交或交付说明必须写明：

- 改了什么以及为什么。
- 影响了哪些 `DECISION-*`。
- 哪些不变量保持不变。
- 运行了哪些测试。
- 仍未验证的外部条件，例如 GCP、RTK、农机终端或田间闭环。

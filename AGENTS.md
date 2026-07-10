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

reUI Windows 便携版

发布目录就是本项目文件夹：文件夹最外层已经放置 `reUI.exe`，不要再套一层目录。

1. 完整解压此压缩包。
2. 双击最外层的 reUI.exe。
3. 不要单独移动 reUI.exe；它需要旁边的 _internal 文件夹。

程序配置、缓存、日志和用户模型位于：
%LOCALAPPDATA%\reUI

运行包由项目维护者发布在 GitHub Release：
https://github.com/teriri30/reUI/releases/latest
请下载并完整解压 Release 中的 reUI-Windows-x64-<version>.zip；不要用 GitHub 的
“Code -> Download ZIP”源码包代替本运行包。

开发者同步源码可以运行仓库根目录的 `update.bat`；它会在确认后推送源码和
`v*` 发布标签，由 GitHub Actions 在远端构建 Release ZIP，不会在本机自动重建
本目录。若要本机验证，单独运行 `build_windows.bat`。

GitHub Release 便携包使用 CPU 版 PyTorch，以便把完整运行库放进一个可下载
的 ZIP 并兼容普通 Windows 电脑；不需要安装 Python、Conda、CUDA 或 NVIDIA
驱动。开发者本地环境仍可使用 CUDA 版 PyTorch；若要发布 GPU 版，请另行构建
并确认 Release 资产大小，不要把本目录中的 CUDA DLL 随意删掉。

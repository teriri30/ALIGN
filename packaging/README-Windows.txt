reUI 农机作业规划系统 便携版
================================

怎么用：
1. 把压缩包完整解压出来（别只把 reUI.exe 拖出来，它需要旁边的 _internal 文件夹）。
2. 双击最外层的 reUI.exe。

更喜欢“装一下”？到 https://github.com/teriri30/reUI/releases/latest 下载
reUI-Setup-*.exe，双击一路下一步即可，装在当前用户目录、不需要管理员权限。

数据保存在 %LOCALAPPDATA%\reUI（配置、缓存、日志、自己训的模型）。
升级、换包、卸载都不会动这个目录。

系统要求：Windows 10 / 11，64 位。不需要安装 Python、Conda、CUDA 或驱动。

开发者：源码 https://github.com/teriri30/reUI ，本机构建用 build_windows.bat，
发布物由 GitHub Actions 在打 tag 时自动生成。

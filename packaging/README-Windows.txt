reUI Windows 便携版

发布目录就是本项目文件夹：文件夹最外层已经放置 `reUI.exe`，不要再套一层目录。

1. 完整解压此压缩包。
2. 双击最外层的 reUI.exe。
3. 不要单独移动 reUI.exe；它需要旁边的 _internal 文件夹。

程序配置、缓存、日志和用户模型位于：
%LOCALAPPDATA%\reUI

源码更新由项目维护者发布新的 GitHub Release。请不要用 GitHub 的
“Code -> Download ZIP”源码包代替本运行包。

开发者同步源码可以运行仓库根目录的 `update.bat`；它只负责 Git 提交和推送，
不会自动重建本目录。代码变更后请重新运行 `build_windows.bat`，它会把新的
运行包整理回项目根目录；然后把项目根目录整体打包。

当前便携构建来自 CUDA 版 PyTorch。目标电脑需要兼容的 NVIDIA 驱动；若只做
CPU 推理，应另外构建 CPU 版运行包，不能把本目录中的 DLL 单独删掉。

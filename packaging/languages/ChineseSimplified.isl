; reUI 安装向导的简体中文文案（Inno Setup 6）
;
; 官方 Inno Setup 不自带中文，这里只翻译用户会看到的主要文案；
; 没翻译的条目会自动回落到 Default.isl 的英文，不影响安装。
; 名称必须与 Inno 的 Default.isl 完全一致，写错只会告警并被忽略。

[LangOptions]
LanguageName=简体中文
LanguageID=$0804
LanguageCodePage=0

[Messages]
ClickNext=点「下一步」继续，或点「取消」退出安装。
WizardSelectDir=选择安装位置
WizardSelectTasks=选择附加任务
WizardReady=准备安装
WizardPreparing=正在准备安装
WizardInstalling=正在安装
WizardUninstalling=卸载状态
SetupAppTitle=安装程序
SetupWindowTitle=%1 安装程序
UninstallAppTitle=卸载
UninstallAppFullTitle=卸载 %1
InformationTitle=提示
ConfirmTitle=确认
ErrorTitle=错误
ButtonNext=下一步(&N) >
ButtonBack=< 上一步(&B)
ButtonCancel=取消
ButtonInstall=安装(&I)
ButtonFinish=完成(&F)
ButtonBrowse=浏览(&R)...
ButtonOK=确定
ButtonYes=是(&Y)
ButtonNo=否(&N)
WelcomeLabel1=欢迎使用 [name] 安装向导
WelcomeLabel2=现在会把 [name/ver] 装到你的电脑上。%n%n建议先关掉其他程序和正在编辑的文件。
SelectDirLabel3=安装程序会把 [name] 装到下面的文件夹。
SelectDirBrowseLabel=点「下一步」继续；想换个位置就点「浏览」。
DiskSpaceMBLabel=至少需要 [mb] MB 可用空间。
SelectTasksDesc=选择附加任务
SelectTasksLabel2=选择安装程序要做的附加任务，然后点「下一步」。
ReadyLabel1=准备安装 [name]。
ReadyLabel2a=点「安装」开始；想改设置就点「上一步」。
ReadyLabel2b=点「安装」开始。
PreparingDesc=安装程序正在准备安装 [name]，请稍等。
InstallingLabel=正在安装 [name]，请稍等...
FinishedHeadingLabel=[name] 安装完成
FinishedLabel=安装完成。桌面和开始菜单里已经有 [name] 的图标，双击就能用。
FinishedLabelNoIcons=安装完成，点「完成」退出。
ClickFinish=点「完成」退出安装程序。
StatusCreateIcons=正在创建快捷方式...
StatusCreateRegistryEntries=正在写入注册表...
StatusUninstalling=正在卸载 %1...
ConfirmUninstall=确定要卸载 %1 吗？%n%n程序文件会被删除，但你的数据（%LOCALAPPDATA%\ALIGN）会保留。
UninstalledAll=%1 已卸载完成。
UninstallStatusLabel=正在卸载 %1，请稍等...

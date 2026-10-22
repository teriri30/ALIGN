; reUI Windows 安装包（Inno Setup 6）
; 由 build-release.yml 调用：
;   ISCC.exe /DMyAppVersion=0.1.3 packaging\reUI_setup.iss
; 输入：dist\reUI（PyInstaller 产物，含 reUI.exe 与 _internal\）
; 输出：installer\reUI-Setup-<version>.exe
;
; 设计取舍：
; - PrivilegesRequired=lowest：装到 %LOCALAPPDATA%\Programs\reUI，普通用户双击就能装，
;   不弹 UAC，也不需要管理员权限（实验室电脑常见无管理员场景）。
; - 不做开机自启、不改注册表关联，卸载只删程序目录；用户数据在 %LOCALAPPDATA%\reUI，
;   升级和卸载都不动它。

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

#define MyAppName "reUI 农机作业规划系统"
#define MyAppShortName "reUI"
#define MyAppPublisher "安徽农业大学 周黄磊"
#define MyAppExeName "reUI.exe"
#define MyAppId "{{7C1D6A54-9E3B-4C2F-9C77-5A61B0D2E4F1}"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}.0
DefaultDirName={localappdata}\Programs\{#MyAppShortName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableWelcomePage=no
PrivilegesRequired=lowest
OutputDir=..\installer
OutputBaseFilename=reUI-Setup-{#MyAppVersion}
SetupIconFile=reUI.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
; 不写 ArchitecturesAllowed：x64compatible 需要 Inno 6.3+，CI 上的版本不一定够新；
; 目标环境本就是 Windows 10/11 64 位，没必要为此冒编译失败的风险。
CloseApplications=yes
RestartApplications=no
AllowNoIcons=yes
MinVersion=10.0

[Languages]
; 官方 Inno 不带中文，这里用自带的简体中文文案（缺失条目自动回落到英文）
Name: "zh"; MessagesFile: "compiler:Default.isl,languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: checkedonce

[Files]
Source: "..\dist\reUI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppShortName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppShortName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动 reUI"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

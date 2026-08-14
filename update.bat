@echo off
chcp 936 >nul
setlocal EnableExtensions EnableDelayedExpansion

title GitHub 通用更新工具

echo ==========================================
echo         GitHub 一键更新工具（通用版）
echo ==========================================
echo.
echo 用法：拖拽任何 Git 项目文件夹到本工具，自动同步到 GitHub
echo.

set "FOLDER=%~1"
if "!FOLDER!"=="" set "FOLDER=%cd%"
echo 当前目录: !FOLDER!
echo.
if "!FOLDER!"=="%cd%" (
    echo 直接回车 = 更新当前目录的项目
    echo 拖拽文件夹 = 更新拖入的项目
    set /p "FOLDER=请选择 [回车使用当前目录]: "
)
set "FOLDER=!FOLDER:"=!"
if "!FOLDER!"=="" set "FOLDER=%cd%"

if not exist "!FOLDER!" (
    echo [错误] 路径不存在: !FOLDER!
    pause
    exit /b 1
)

cd /d "!FOLDER!"

if not exist ".git" (
    echo.
    echo [提示] 这个文件夹还不是 Git 仓库
    echo        请先初始化仓库并配置 GitHub 远程地址
    echo.
    pause
    exit /b 1
)

git remote -v | findstr "github.com" >nul
if errorlevel 1 (
    echo.
    echo [提示] 未检测到 GitHub 远程仓库
    echo        请先配置名为 origin 的 GitHub 远程地址
    echo.
    pause
    exit /b 1
)

set "REMOTE="
for /f "tokens=*" %%a in ('git remote get-url origin 2^>nul') do set "REMOTE=%%a"
if "!REMOTE!"=="" (
    echo.
    echo [错误] 无法读取 origin 远程地址
    echo.
    pause
    exit /b 1
)

echo.
echo [信息] 项目: !FOLDER!
echo [信息] 远程: !REMOTE!
echo.

git status --short | findstr . >nul
if errorlevel 1 (
    echo [信息] 工作区没有未提交改动，将继续检查并推送已有提交
) else (
    echo 当前检测到的改动：
    echo ------------------------------------------
    git status --short
    echo ------------------------------------------
    echo.

    set /p "MSG=请输入更新说明（本次更新做了什么）: "
    if "!MSG!"=="" set "MSG=update"

    echo.
    echo [1/3] 添加文件...
    git add -A
    if errorlevel 1 (
        echo [错误] git add 失败
        pause
        exit /b 1
    )

    echo [2/3] 提交...
    git commit -m "!MSG!"
    if errorlevel 1 (
        echo.
        echo [错误] 提交失败，请检查：
        echo   1. git config user.name / user.email 是否已设置
        echo   2. 更新说明里是否包含感叹号等特殊字符
        echo   3. 是否有 pre-commit 检查未通过
        echo.
        pause
        exit /b 1
    )
)

echo.
echo [3/3] 推送到 GitHub...
set "PUSH_OK=0"
git push -u origin HEAD
if not errorlevel 1 set "PUSH_OK=1"

if "!PUSH_OK!"=="0" (
    echo !REMOTE! | findstr /b /c:"git@github.com:" >nul
    if not errorlevel 1 (
        echo.
        echo [提示] SSH 推送失败，可能是网络阻止了 SSH 端口 22
        set "RETRY="
        set /p "RETRY=是否改用 HTTPS 远程地址并重试？[Y/n]: "
        if "!RETRY!"=="" set "RETRY=Y"
        if /i "!RETRY!"=="Y" (
            set "HTTPS_REMOTE=!REMOTE:git@github.com:=https://github.com/!"
            git remote set-url origin "!HTTPS_REMOTE!"
            if not errorlevel 1 (
                set "REMOTE=!HTTPS_REMOTE!"
                echo [信息] 已改用: !REMOTE!
                git push -u origin HEAD
                if not errorlevel 1 set "PUSH_OK=1"
            )
        )
    )
)

if "!PUSH_OK!"=="0" (
    echo !REMOTE! | findstr /b /c:"https://" >nul
    if not errorlevel 1 (
        echo.
        echo [提示] HTTPS 推送失败，请确认网络、代理和 GitHub 登录凭据
        set "RETRY="
        set /p "RETRY=是否按当前 HTTPS 地址再试一次？[y/N]: "
        if /i "!RETRY!"=="Y" (
            git push -u origin HEAD
            if not errorlevel 1 set "PUSH_OK=1"
        )
    )
)

if "!PUSH_OK!"=="1" (
    echo.
    echo ==========================================
    echo   更新成功：本地提交已经同步到 GitHub
    echo ==========================================
    echo   远程仓库地址: !REMOTE!
    echo.
    git status --short --branch
) else (
    echo.
    echo [错误] 推送失败，GitHub 尚未更新
    echo [安全] 本地提交仍然保留，没有被删除或回退
    echo [重试] 网络或凭据恢复后运行: git push -u origin HEAD
)

echo.
pause

if "!PUSH_OK!"=="1" exit /b 0
exit /b 1

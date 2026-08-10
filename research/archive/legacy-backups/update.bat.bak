@echo off
setlocal enabledelayedexpansion

title GitHub 通用更新工具

echo ==========================================
echo         GitHub 一键更新工具（通用版）
echo ==========================================
echo.
echo 用法：拖拽任何 Git 项目文件夹到下方，自动同步到 GitHub
echo.

:: ---- 获取项目路径 ----
set "FOLDER=%cd%"
echo 当前目录: !FOLDER!
echo.
echo 1) 直接回车 = 更新当前目录的项目
echo 2) 拖拽文件夹 = 更新拖入的项目
set /p "FOLDER=请选择 [回车使用当前目录]: "
set "FOLDER=!FOLDER:"=!"
if "!FOLDER!"=="" set "FOLDER=%cd%"

if not exist "!FOLDER!" (
    echo [错误] 路径不存在: !FOLDER!
    pause
    exit /b 1
)

cd /d "!FOLDER!"

:: ---- 检查是否为 Git 仓库 ----
if not exist ".git" (
    echo.
    echo [提示] 这个文件夹还不是 Git 仓库
    echo        请先使用 upload.bat 初始化并上传
    echo.
    pause
    exit /b 1
)

:: ---- 检查远程仓库 ----
git remote -v | findstr "github.com" >nul
if errorlevel 1 (
    echo.
    echo [提示] 未检测到 GitHub 远程仓库
    echo        请先使用 upload.bat 初始化
    echo.
    pause
    exit /b 1
)

:: ---- 获取远程仓库地址（用于显示） ----
for /f "tokens=*" %%a in ('git remote get-url origin 2^>nul') do set "REMOTE=%%a"
echo.
echo [信息] 项目: !FOLDER!
echo [信息] 远程: !REMOTE!
echo.

:: ---- 检查是否有改动 ----
git status --short | findstr . >nul
if errorlevel 1 (
    echo [提示] 没有检测到任何改动，无需更新
    echo.
    pause
    exit /b 0
)

echo 当前检测到的改动：
echo ------------------------------------------
git status --short
echo ------------------------------------------
echo.

:: ---- 获取更新说明 ----
set /p "MSG=请输入更新说明（描述这次改了啥）："
if "!MSG!"=="" set "MSG=update"

echo.
echo [1/3] 添加文件...
git add -A

echo [2/3] 提交...
git commit -m "!MSG!"

:: ---- 判断远程是 HTTPS 还是 SSH ----
echo.
echo [3/3] 推送到 GitHub...
echo !REMOTE! | findstr "https://" >nul
if errorlevel 1 (
    :: SSH 地址
    git push
) else (
    :: HTTPS 地址 - 先检查代理
    git push 2>&1
    if errorlevel 1 (
        echo.
        echo HTTPS 推送失败，常见原因：
        echo   1. 网络被墙 → 使用 SSH（推荐）
        echo      git remote set-url origin git@github.com:用户名/仓库名.git
        echo   2. 需要代理 → 设置代理
        echo      git config --global http.proxy http://127.0.0.1:7890
        echo.
        set /p "RETRY=重试？(Y/N): "
        if /i "!RETRY!"=="Y" git push
    )
)

if errorlevel 1 (
    echo.
    echo [错误] 推送失败，请检查网络
) else (
    echo.
    echo ==========================================
    echo   更新成功！代码已同步到 GitHub
    echo ==========================================
    echo   远程仓库地址: !REMOTE!
)

echo.
pause

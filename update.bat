@echo off
setlocal enabledelayedexpansion

title GitHub 通用更新工具

echo ==========================================
echo         GitHub 一键更新工具（通用版）
echo ==========================================
echo.
echo 用法：拖拽任何 Git 项目文件夹到下方，自动同步到 GitHub
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
    echo        请先使用 upload.bat 初始化并上传
    echo.
    pause
    exit /b 1
)

git remote -v | findstr "github.com" >nul
if errorlevel 1 (
    echo.
    echo [提示] 未检测到 GitHub 远程仓库
    echo        请先使用 upload.bat 初始化
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('git remote get-url origin 2^>nul') do set "REMOTE=%%a"
echo.
echo [信息] 项目: !FOLDER!
echo [信息] 远程: !REMOTE!
echo.

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

set /p "MSG=请输入更新说明（描述这次改了啥）："
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
    echo [错误] 提交失败！请检查：
    echo   1. git config user.name / user.email 是否已配置
    echo   2. 更新说明里不能包含引号和特殊符号
    echo   3. 是否有 pre-commit 钩子拦截
    echo.
    pause
    exit /b 1
)

echo.
echo [3/3] 推送到 GitHub...
set "PUSH_OK=0"
echo !REMOTE! | findstr "https://" >nul
if errorlevel 1 (
    git push
    if not errorlevel 1 set "PUSH_OK=1"
) else (
    git push
    if errorlevel 1 (
        echo.
        echo HTTPS 推送失败，常见原因：
        echo   1. 网络被墙 - 改用 SSH 推荐
        echo      git remote set-url origin git@github.com:用户名/仓库名.git
        echo   2. 需要代理 - 设置代理
        echo      git config --global http.proxy http://127.0.0.1:7890
        echo.
        set /p "RETRY=重试？（Y/N）: "
        if /i "!RETRY!"=="Y" (
            git push
            if not errorlevel 1 set "PUSH_OK=1"
        )
    ) else (
        set "PUSH_OK=1"
    )
)

if "!PUSH_OK!"=="1" (
    echo.
    echo ==========================================
    echo   更新成功！代码已同步到 GitHub
    echo ==========================================
    echo   远程仓库地址: !REMOTE!
) else (
    echo.
    echo [错误] 推送失败，GitHub 未更新！
    echo   手动重试: 在该目录运行 git push
)

echo.
pause

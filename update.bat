@echo off
chcp 936 >nul
setlocal EnableExtensions EnableDelayedExpansion

title GitHub 一键同步（源码 + 便携版）

rem Sync source commits and optionally publish a complete Windows portable
rem package through the build-release GitHub Action.  The EXE and _internal
rem folder are intentionally ignored by Git; the Release ZIP is built after
rem this script pushes a v* tag.

set "FOLDER=%~1"
if "!FOLDER!"=="" set "FOLDER=%cd%"
set "FOLDER=!FOLDER:"=!"

if not exist "!FOLDER!\" (
    echo [错误] 文件夹不存在: !FOLDER!
    pause
    exit /b 1
)

cd /d "!FOLDER!"
for /f "delims=" %%A in ('git rev-parse --show-toplevel 2^>nul') do set "REPO_ROOT=%%A"
if not defined REPO_ROOT (
    echo [错误] 选择的文件夹不是 Git 仓库。
    pause
    exit /b 1
)
cd /d "!REPO_ROOT!"

for /f "delims=" %%A in ('git remote get-url origin 2^>nul') do if not defined REMOTE set "REMOTE=%%A"
if not defined REMOTE (
    echo [错误] 未配置 origin 远程地址。
    pause
    exit /b 1
)
echo "!REMOTE!" | findstr /I "github.com" >nul
if errorlevel 1 (
    echo [错误] origin 不是 GitHub 远程地址: !REMOTE!
    pause
    exit /b 1
)

for /f "delims=" %%A in ('git symbolic-ref --short -q HEAD 2^>nul') do set "BRANCH=%%A"
if not defined BRANCH (
    echo [错误] 当前是 detached HEAD；为安全起见不创建发布标签。
    pause
    exit /b 1
)

set "WEB_URL=!REMOTE!"
if /I "!WEB_URL:~0,15!"=="git@github.com:" set "WEB_URL=!WEB_URL:git@github.com:=https://github.com/!"
set "WEB_URL=!WEB_URL:.git=!"

echo ==========================================
echo       GitHub 源码 + 便携版同步
echo ==========================================
echo 项目: !REPO_ROOT!
echo 分支: !BRANCH!
echo 远程: !REMOTE!
echo.

set "DIRTY="
for /f "delims=" %%A in ('git status --porcelain 2^>nul') do set "DIRTY=1"
if defined DIRTY (
    echo [检查] 以下改动将被提交:
    echo ------------------------------------------
    git status --short
    echo ------------------------------------------
    set "COMMIT_OK="
    set /p "COMMIT_OK=提交这些改动并同步吗？[Y/n]: "
    if "!COMMIT_OK!"=="" set "COMMIT_OK=Y"
    if /I "!COMMIT_OK!"=="Y" (
        set "MSG=%REUI_COMMIT_MESSAGE%"
        if "!MSG!"=="" set /p "MSG=提交说明 [update]: "
        if "!MSG!"=="" set "MSG=update"
        git add -A
        if errorlevel 1 goto :commit_failed
        git commit -m "!MSG!"
        if errorlevel 1 goto :commit_failed
    ) else (
        echo [提示] 已取消提交；本地改动已保留。
        set "SKIP_RELEASE=1"
    )
)

echo.
echo [1/3] 推送源码提交到 GitHub...
call :push_branch
if errorlevel 1 goto :push_failed

set "LOCAL_SHA="
for /f "delims=" %%A in ('git rev-parse HEAD 2^>nul') do set "LOCAL_SHA=%%A"
set "REMOTE_SHA="
for /f "tokens=1" %%A in ('git ls-remote origin "refs/heads/!BRANCH!" 2^>nul') do if not defined REMOTE_SHA set "REMOTE_SHA=%%A"
if not defined REMOTE_SHA (
    echo [错误] 推送命令成功，但无法回读远程分支 SHA，未确认同步结果。
    goto :push_failed
)
if /I not "!LOCAL_SHA!"=="!REMOTE_SHA!" (
    echo [错误] 远程分支 SHA 与本地 HEAD 不一致。
    echo         本地: !LOCAL_SHA!
    echo         远程: !REMOTE_SHA!
    goto :push_failed
)
echo [完成] 源码已确认同步: !LOCAL_SHA!

if defined SKIP_RELEASE goto :finish

echo.
echo GitHub Actions 将构建完整便携环境:
echo   reUI.exe + _internal + bundled models, self-test, Release ZIP
set "PUBLISH_RELEASE=%REUI_PUBLISH_RELEASE%"
if "!PUBLISH_RELEASE!"=="" set /p "PUBLISH_RELEASE=现在创建 v* 标签并发布便携版吗？[Y/n]: "
if "!PUBLISH_RELEASE!"=="" set "PUBLISH_RELEASE=Y"
if /I not "!PUBLISH_RELEASE!"=="Y" goto :finish

set "RELEASE_TAG=%REUI_RELEASE_TAG%"
if "!RELEASE_TAG!"=="" set /p "RELEASE_TAG=发布标签（例如 v0.1.0；直接回车使用 v0.1.0）: "
if "!RELEASE_TAG!"=="" set "RELEASE_TAG=v0.1.0"
echo(!RELEASE_TAG!| findstr /R /B /C:"v[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*" >nul
if errorlevel 1 (
    echo [错误] 标签必须类似 v0.1.0: !RELEASE_TAG!
    goto :release_failed
)

set "EXISTING_TAG_SHA="
for /f "tokens=1" %%A in ('git ls-remote origin "refs/tags/!RELEASE_TAG!" 2^>nul') do if not defined EXISTING_TAG_SHA set "EXISTING_TAG_SHA=%%A"
if defined EXISTING_TAG_SHA (
    if /I "!EXISTING_TAG_SHA!"=="!LOCAL_SHA!" (
        echo [提示] 标签已存在且指向当前提交，跳过重复发布。
        goto :release_link
    )
    echo [错误] 远程标签 !RELEASE_TAG! 已指向其他提交。
    echo         请更换版本号，不覆盖历史发布。
    goto :release_failed
)

git tag "!RELEASE_TAG!"
if errorlevel 1 goto :release_failed
echo.
echo [2/3] 推送发布标签 !RELEASE_TAG! ...
git push origin "!RELEASE_TAG!"
if errorlevel 1 goto :release_failed

set "TAG_SHA="
for /f "tokens=1" %%A in ('git ls-remote origin "refs/tags/!RELEASE_TAG!" 2^>nul') do if not defined TAG_SHA set "TAG_SHA=%%A"
if not defined TAG_SHA (
    echo [错误] 无法回读远程标签 SHA。
    goto :release_failed
)
if /I not "!TAG_SHA!"=="!LOCAL_SHA!" (
    echo [错误] 远程标签 SHA 与本地 HEAD 不一致。
    goto :release_failed
)
echo [完成] 发布标签已确认同步: !TAG_SHA!

:release_link
echo.
echo [3/3] GitHub Actions 正在构建便携环境。
echo        构建: !WEB_URL!/actions/workflows/build-release.yml
echo        下载: !WEB_URL!/releases/latest
echo.
echo Actions 通过检查和自检后，新的 ZIP 才会出现在 Releases 页面。
goto :finish

:commit_failed
echo [错误] 提交失败；未继续推送或创建发布标签。
echo         请检查 Git 身份、钩子和冲突。
goto :finish_failed

:push_failed
echo [错误] 未确认源码已到远端；未继续创建发布标签。
echo         本地提交已保留，可稍后重试。
goto :finish_failed

:release_failed
echo [错误] 发布标签步骤失败；源码提交仍按上面的确认结果保留。
echo         请检查 GitHub 凭据或更换版本标签。
goto :finish_failed

:finish
echo ==========================================
echo 同步脚本结束。
echo ==========================================
pause
exit /b 0

:finish_failed
echo.
pause
exit /b 1

:push_branch
git push -u origin HEAD
if not errorlevel 1 exit /b 0
echo [提示] SSH 推送失败。
echo "!REMOTE!" | findstr /B /C:"git@github.com:" >nul
if errorlevel 1 exit /b 1
set "SWITCH_HTTPS="
set /p "SWITCH_HTTPS=是否切换 origin 为 HTTPS 后重试？[Y/n]: "
if "!SWITCH_HTTPS!"=="" set "SWITCH_HTTPS=Y"
if /I not "!SWITCH_HTTPS!"=="Y" exit /b 1
set "HTTPS_REMOTE=!REMOTE:git@github.com:=https://github.com/!"
git remote set-url origin "!HTTPS_REMOTE!"
if errorlevel 1 exit /b 1
set "REMOTE=!HTTPS_REMOTE!"
git push -u origin HEAD
if errorlevel 1 exit /b 1
exit /b 0

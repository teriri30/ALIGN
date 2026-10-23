@echo off
setlocal

rem Build the portable Windows package from the selected Python environment.
rem Usage: build_windows.bat D:\zhl\anaconda\envs\game\python.exe

set "PYTHON=%~1"
if "%PYTHON%"=="" set "PYTHON=python"
set "VERSION=%ALIGN_VERSION%"
if "%VERSION%"=="" (
  rem 版本号以 src/version.py 为唯一来源；cmd 的 for /f 处理带引号路径会碎，先落临时文件再读。
  "%PYTHON%" -c "import sys;sys.path.insert(0,'src');from version import APP_VERSION;print(APP_VERSION)" > "%TEMP%/align_version.txt" 2>nul
  set /p VERSION=<"%TEMP%/align_version.txt"
  del "%TEMP%/align_version.txt" >nul 2>nul
)
if "%VERSION%"=="" set "VERSION=0.1.0"
set "CHANNEL=%ALIGN_CHANNEL%"
if "%CHANNEL%"=="" set "CHANNEL=stable"

echo [1/6] Checking build interpreter: %PYTHON%
"%PYTHON%" -c "import sys; print(sys.executable); print(sys.version)"
if errorlevel 1 exit /b 1

echo [2/6] Installing build dependency
"%PYTHON%" -m pip install -r developer\requirements\requirements-build.txt
if errorlevel 1 exit /b 1

echo [3/6] Generating build metadata
"%PYTHON%" scripts\generate_build_info.py --version "%VERSION%" --channel "%CHANNEL%"
if errorlevel 1 exit /b 1

echo [4/6] Running static checks and tests
rem Compile only project-owned Python trees.  The repository may contain a
rem developer .venv or pytest/build scratch directories created by another
rem interpreter; compileall . would incorrectly treat those as release code.
for %%D in (src scripts tests) do (
  "%PYTHON%" -m compileall -q "%%D"
  if errorlevel 1 exit /b 1
)
rem Do not reuse a fixed temp directory: Windows ACLs or a previous pytest
rem worker can leave it undeletable, turning every test into a setup error.
rem An explicit ALIGN_PYTEST_BASETEMP is supported for CI; otherwise use a
rem unique directory under the active TEMP and clean only that exact path.
set "BUILD_PYTEST_BASETEMP=%ALIGN_PYTEST_BASETEMP%"
set "BUILD_PYTEST_BASETEMP_OWNED=0"
if "%BUILD_PYTEST_BASETEMP%"=="" (
  set "BUILD_PYTEST_BASETEMP=%TEMP%\align-pytest-%RANDOM%-%RANDOM%"
  set "BUILD_PYTEST_BASETEMP_OWNED=1"
)
if exist "%BUILD_PYTEST_BASETEMP%" rmdir /S /Q "%BUILD_PYTEST_BASETEMP%"
mkdir "%BUILD_PYTEST_BASETEMP%"
if errorlevel 1 exit /b 1
"%PYTHON%" -m pytest -q -p no:cov --basetemp "%BUILD_PYTEST_BASETEMP%"
set "PYTEST_EXIT=%ERRORLEVEL%"
if "%BUILD_PYTEST_BASETEMP_OWNED%"=="1" if exist "%BUILD_PYTEST_BASETEMP%" rmdir /S /Q "%BUILD_PYTEST_BASETEMP%"
if not "%PYTEST_EXIT%"=="0" exit /b %PYTEST_EXIT%

echo [5/6] Building onedir bundle
rem PyInstaller resolves transitive native dependencies through PATH.  Keep
rem host tools (for example Codex/Poppler, GIS, or another Qt installation)
rem out of that search so their same-named DLLs cannot enter the bundle.
set "BUILD_ORIGINAL_PATH=%PATH%"
set "BUILD_ORIGINAL_YOLO_AUTOINSTALL=%YOLO_AUTOINSTALL%"
for %%I in ("%PYTHON%") do set "BUILD_PYTHON_DIR=%%~dpI"
set "BUILD_PYINSTALLER_PATH=%BUILD_PYTHON_DIR%;%BUILD_PYTHON_DIR%Scripts;%BUILD_PYTHON_DIR%Library\bin;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0"
set "PATH=%BUILD_PYINSTALLER_PATH%"
set "YOLO_AUTOINSTALL=false"
"%PYTHON%" -m PyInstaller packaging\align.spec --noconfirm --clean
set "PYINSTALLER_EXIT=%ERRORLEVEL%"
set "PATH=%BUILD_ORIGINAL_PATH%"
set "YOLO_AUTOINSTALL=%BUILD_ORIGINAL_YOLO_AUTOINSTALL%"
if not "%PYINSTALLER_EXIT%"=="0" exit /b %PYINSTALLER_EXIT%

echo [6/6] Running frozen self-test and preparing portable folder
set "SELF_TEST_DIR=%TEMP%\ALIGN-build-self-test-%RANDOM%"
if exist "%SELF_TEST_DIR%" rmdir /S /Q "%SELF_TEST_DIR%"
mkdir "%SELF_TEST_DIR%"
if errorlevel 1 exit /b 1
set "ALIGN_USER_DATA_DIR=%SELF_TEST_DIR%"
dist\align\align.exe --self-test
set "SELF_TEST_EXIT=%ERRORLEVEL%"
set "ALIGN_USER_DATA_DIR="
rmdir /S /Q "%SELF_TEST_DIR%"
if not "%SELF_TEST_EXIT%"=="0" exit /b %SELF_TEST_EXIT%
copy /Y build_info.json dist\align\version.json >nul
copy /Y packaging\README-Windows.txt dist\align\README-Windows.txt >nul
echo Publishing portable files to the repository root...
if exist align.exe del /F /Q align.exe
if exist _internal rmdir /S /Q _internal
if exist README-Windows.txt del /F /Q README-Windows.txt
if exist version.json del /F /Q version.json
rem 改名老账：reUI 时期的产物不在上面的删除范围内，留着就会有人双击到旧版本
rem （2026-10-23 实际发生过：设置/主题按钮还是小尺寸）。
if exist reUI.exe del /F /Q reUI.exe
if exist reUI-Windows-x64.zip del /F /Q reUI-Windows-x64.zip
if exist disteUI rmdir /S /Q disteUI
if exist buildeUI rmdir /S /Q buildeUI
robocopy "dist\align" "." /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
if errorlevel 8 exit /b 1
rem Verify the published root EXE against the staging copy; the [verify]
rem lines are part of the build log evidence kept after cleanup.
rem Hash with the build interpreter, not PowerShell: Get-FileHash needs module
rem autoloading, and a host with a broken PSModulePath returned two empty
rem strings for a mismatched pair, so the comparison silently passed.
"%PYTHON%" -c "import hashlib,sys;a=hashlib.sha256(open('align.exe','rb').read()).hexdigest();b=hashlib.sha256(open(r'dist\align\align.exe','rb').read()).hexdigest();print('[verify] root EXE  sha256: '+a);print('[verify] dist copy sha256: '+b);sys.exit(0 if a==b else 1)"
if errorlevel 1 (
  echo [ERROR] Published EXE and dist staging copy differ; dist kept for inspection.
  exit /b 1
)
echo [verify] root EXE matches dist staging copy.
if /I not "%ALIGN_MAKE_ARCHIVE%"=="1" goto cleanup_outputs
tar.exe -a -c -f "align-Windows-x64-v%VERSION%.zip" -C dist\align .
if errorlevel 1 exit /b 1
certutil -hashfile "align-Windows-x64-v%VERSION%.zip" SHA256 > "align-Windows-x64-v%VERSION%.zip.sha256"
echo Archive created: align-Windows-x64-v%VERSION%.zip
:cleanup_outputs
rem Rebuildable outputs (build\, dist\) are regenerated by every build; they
rem are removed by default.  Set ALIGN_CLEAN_DIST=0 to keep them.
if /I "%ALIGN_CLEAN_DIST%"=="0" (
  echo [keep] ALIGN_CLEAN_DIST=0: build\ and dist\ retained.
  goto folder_ready
)
echo Removing rebuildable outputs: build\ dist\ ...
if exist build rmdir /S /Q build
if exist dist rmdir /S /Q dist
if exist build echo [WARN] build\ still present; check for locked files.
if exist dist echo [WARN] dist\ still present; check for locked files.
echo [clean] build\ and dist\ cleanup finished.
:folder_ready
echo Build complete: repository root (align.exe is at the folder root)
endlocal

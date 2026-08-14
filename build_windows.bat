@echo off
setlocal

rem Build the portable Windows package from the selected Python environment.
rem Usage: build_windows.bat D:\zhl\anaconda\envs\game\python.exe

set "PYTHON=%~1"
if "%PYTHON%"=="" set "PYTHON=python"
set "VERSION=%REUI_VERSION%"
if "%VERSION%"=="" set "VERSION=0.1.0"
set "CHANNEL=%REUI_CHANNEL%"
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
"%PYTHON%" -m pytest -q
if errorlevel 1 exit /b 1

echo [5/6] Building onedir bundle
"%PYTHON%" -m PyInstaller packaging\reUI.spec --noconfirm --clean
if errorlevel 1 exit /b 1

echo [6/6] Running frozen self-test and preparing portable folder
set "SELF_TEST_DIR=%TEMP%\reUI-build-self-test-%RANDOM%"
if exist "%SELF_TEST_DIR%" rmdir /S /Q "%SELF_TEST_DIR%"
mkdir "%SELF_TEST_DIR%"
if errorlevel 1 exit /b 1
set "REUI_USER_DATA_DIR=%SELF_TEST_DIR%"
dist\reUI\reUI.exe --self-test
set "SELF_TEST_EXIT=%ERRORLEVEL%"
set "REUI_USER_DATA_DIR="
rmdir /S /Q "%SELF_TEST_DIR%"
if not "%SELF_TEST_EXIT%"=="0" exit /b %SELF_TEST_EXIT%
copy /Y build_info.json dist\reUI\version.json >nul
copy /Y packaging\README-Windows.txt dist\reUI\README-Windows.txt >nul
echo Publishing portable files to the repository root...
if exist reUI.exe del /F /Q reUI.exe
if exist _internal rmdir /S /Q _internal
if exist README-Windows.txt del /F /Q README-Windows.txt
if exist version.json del /F /Q version.json
robocopy "dist\reUI" "." /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
if errorlevel 8 exit /b 1
if /I not "%REUI_MAKE_ARCHIVE%"=="1" goto folder_ready
tar.exe -a -c -f "reUI-Windows-x64-v%VERSION%.zip" -C dist\reUI .
if errorlevel 1 exit /b 1
certutil -hashfile "reUI-Windows-x64-v%VERSION%.zip" SHA256 > "reUI-Windows-x64-v%VERSION%.zip.sha256"
echo Archive created: reUI-Windows-x64-v%VERSION%.zip
:folder_ready
echo Build complete: repository root (reUI.exe is at the folder root)
endlocal

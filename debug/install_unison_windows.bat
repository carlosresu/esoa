@echo off
setlocal enabledelayedexpansion

set "REPO_NAME=esoa"
set "LOCAL_ROOT=%USERPROFILE%\github_repos\%REPO_NAME%"
set "OD_ROOT=%OneDrive%\GitHubRepositories\%REPO_NAME%"
set "UNISON_DIR=%USERPROFILE%\.unison"
set "PROFILE_NAME=esoa"
set "PROFILE_PATH=%UNISON_DIR%\%PROFILE_NAME%.prf"
set "USER_BIN=%USERPROFILE%\bin"
set "WATCH_CMD=%USER_BIN%\unison_esoa_watch.cmd"
set "TASK_NAME=Unison ESOA Watch"

echo === Unison ESOA Install (Windows) ===
echo.

if "%OneDrive%"=="" (
  echo OneDrive environment variable not found. Open OneDrive once or set the path manually.
  exit /b 1
)

call :ensure_dir "%LOCAL_ROOT%"
call :ensure_dir "%OD_ROOT%"
call :ensure_dir "%UNISON_DIR%"
call :ensure_dir "%USER_BIN%"

call :ensure_unison
if errorlevel 1 exit /b 1

call :compute_extended_paths
if "%LOCAL_EXT%"=="" (
  echo Failed to compute extended paths.
  exit /b 1
)

echo Reminder: make sure OneDrive folder is fully local.
echo In File Explorer: right-click "%OD_ROOT%" ^> "Always keep on this device".
echo.

call :write_profile
if errorlevel 1 exit /b 1

echo Running initial sync...
unison %PROFILE_NAME% -ui text

echo.
echo Watch script created at: %WATCH_CMD%
call :write_watch

choice /M "Start continuous watch sync now?"
if errorlevel 2 goto ask_schedule
start "" cmd.exe /c "\"%WATCH_CMD%\""

:ask_schedule
choice /M "Auto-start watch sync on login via Task Scheduler?"
if errorlevel 2 goto done
schtasks /Create /TN "%TASK_NAME%" /TR "\"%WATCH_CMD%\"" /SC ONLOGON /RL LIMITED /F >nul 2>&1
if errorlevel 1 (
  echo Task Scheduler registration failed (maybe no rights).
  echo Fallback: add the watch script to Startup (Win+R, type: shell:startup).
) else (
  echo Scheduled task registered: %TASK_NAME%
)

:done
echo.
echo Done. Two-way Unison sync is set up for FULL roots between:
echo   %LOCAL_ROOT%
echo   %OD_ROOT%
echo Profile: %PROFILE_PATH%
goto :eof

:ensure_dir
if not exist "%~1" mkdir "%~1" >nul 2>&1
exit /b 0

:ensure_unison
where unison >nul 2>&1
if %errorlevel%==0 exit /b 0
set "ans=Y"
set /p ans=Unison not found. Install using Scoop? (recommended) [Y/n]:
if /i "%ans%"=="n" goto try_choco
where scoop >nul 2>&1
if %errorlevel%==0 (
  scoop install unison
  where unison >nul 2>&1 && exit /b 0
  echo Scoop install failed; please install Unison manually.
  exit /b 1
)
echo Scoop not found. Install Scoop first or install Unison manually.
exit /b 1

:try_choco
set "ans2=N"
set /p ans2=Install Unison using Chocolatey instead? [y/N]:
if /i "%ans2%"=="y" (
  where choco >nul 2>&1
  if %errorlevel%==0 (
    choco install unison -y
    where unison >nul 2>&1 && exit /b 0
    echo Chocolatey install failed; please install Unison manually.
    exit /b 1
  )
  echo Chocolatey not found; install Unison manually.
  exit /b 1
)
echo Unison is required. Install it and re-run.
exit /b 1

:compute_extended_paths
for /f "tokens=1,2 delims=|" %%A in ('
  powershell -NoProfile -Command "function Ext([string]$p){$f=[IO.Path]::GetFullPath($p); if($f.StartsWith('\\?\')){$f=$f.Substring(4)}; if($f.StartsWith('\')){'//?/UNC'+($f.Substring(1) -replace '\\','/')} else {'//?/' + ($f -replace '\\','/')}}; Write-Output ((Ext '%LOCAL_ROOT%') + '|' + (Ext '%OD_ROOT%'))"
') do (
  set "LOCAL_EXT=%%A"
  set "OD_EXT=%%B"
)
exit /b 0

:write_profile
> "%PROFILE_PATH%" (
  echo root = "%LOCAL_EXT%"
  echo root = "%OD_EXT%"
  echo.
  echo auto = true
  echo batch = true
  echo fastcheck = true
  echo times = true
  echo maxthreads = 1
  echo prefer = newer
  echo confirmbigdel = true
  echo.
  echo ignore = Name .*
  echo ignore = Name .git
  echo ignore = Name .git/
  echo ignore = Name __pycache__
  echo ignore = Name node_modules
  echo ignore = Name .venv
  echo ignore = Name *.tmp
  echo ignore = Name Thumbs.db
  echo ignore = Name desktop.ini
)
exit /b 0

:write_watch
> "%WATCH_CMD%" (
  echo @echo off
  echo unison %PROFILE_NAME% -repeat watch -ui text -log -logfile %TEMP%\unison_esoa.log
)
exit /b 0

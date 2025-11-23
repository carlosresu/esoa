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

echo === Unison ESOA Uninstall (OneDrive-safe) ===
echo NOTE: This script will NOT delete anything inside:
echo   %OD_ROOT%
echo.

echo Removing scheduled task (if present)...
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

if exist "%WATCH_CMD%" (
  set "ans=Y"
  set /p ans=Delete watch script at "%WATCH_CMD%"? [Y/n]:
  if /i not "%ans%"=="n" del /f "%WATCH_CMD%"
) else (
  echo Watch script not found.
)

if exist "%PROFILE_PATH%" (
  set "ans2=Y"
  set /p ans2=Delete Unison profile at "%PROFILE_PATH%"? [Y/n]:
  if /i not "%ans2%"=="n" del /f "%PROFILE_PATH%"
) else (
  echo Unison profile not found.
)

echo.
set "ans3=N"
set /p ans3=Delete LOCAL root %LOCAL_ROOT% and all its contents? [y/N]:
if /i "%ans3%"=="y" (
  if exist "%LOCAL_ROOT%" (
    rmdir /s /q "%LOCAL_ROOT%"
    echo Deleted %LOCAL_ROOT%
  ) else (
    echo Local root not found.
  )
) else (
  echo Leaving local root untouched.
)

echo.
set "ans4=N"
set /p ans4=Uninstall Unison package now? [y/N]:
if /i "%ans4%"=="y" (
  where scoop >nul 2>&1
  if %errorlevel%==0 (
    scoop uninstall unison
  ) else (
    where choco >nul 2>&1
    if %errorlevel%==0 (
      choco uninstall unison -y
    ) else (
      echo Scoop/Chocolatey not found. Uninstall Unison manually if needed.
    )
  )
)

echo.
echo OneDrive location left untouched:
echo   %OD_ROOT%
echo Done.

$ErrorActionPreference = "Stop"

function Prompt-YesNo($q, $defaultYes=$true) {
  $suffix = if ($defaultYes) { "[Y/n]" } else { "[y/N]" }
  $ans = Read-Host "$q $suffix"
  if ([string]::IsNullOrWhiteSpace($ans)) { return $defaultYes }
  return $ans.Trim().ToLower().StartsWith("y")
}

# Use extended-length paths so Unison can traverse OneDrive cloud roots without
# tripping on the Cloud Files reparse point ("Invalid argument" canonical name error).
function Convert-ToExtendedPath($path) {
  $full = [System.IO.Path]::GetFullPath($path)
  # Strip any existing extended prefix to avoid duplicating //?/ when re-running.
  $full = $full -replace '^\\\\\?\\', '' -replace '^//\?/', ''
  $extended = if ($full.StartsWith("\\")) { "\\\\?\\UNC" + $full.Substring(1) } else { "\\\\?\\" + $full }
  # Unison prefers forward-slash extended paths: //?/C:/Users/...
  $asSlashes = $extended -replace "\\","/"
  return $asSlashes -replace '^//\?/', '//?/'
}

# --- Roots (mac-equivalent) ---
$RepoName      = "esoa"
$LocalRootBase = Join-Path $env:USERPROFILE "github_repos"
$LocalRoot     = Join-Path $LocalRootBase $RepoName

# Prefer real OneDrive location from environment to avoid path mismatches.
if ([string]::IsNullOrWhiteSpace($env:OneDrive)) {
  throw "OneDrive environment variable not found. Open OneDrive once or set `$OneDriveRoot manually."
}
$OneDriveRootBase = Join-Path $env:OneDrive "GitHubRepositories"
$OneDriveRoot     = Join-Path $OneDriveRootBase $RepoName

# --- Unison config locations ---
$UnisonDir   = Join-Path $env:USERPROFILE ".unison"
$ProfileName = "esoa"
$ProfilePath = Join-Path $UnisonDir "$ProfileName.prf"
$UserBin     = Join-Path $env:USERPROFILE "bin"
$WatchCmd    = Join-Path $UserBin "unison_esoa_watch.cmd"
$TaskName    = "Unison ESOA Watch"

Write-Host "=== Unison ESOA Install (Windows) ===`n"

# --- Ensure Unison installed ---
if (-not (Get-Command unison -ErrorAction SilentlyContinue)) {
  Write-Host "Unison not found in PATH."
  if (Prompt-YesNo "Install Unison using Scoop? (recommended)" $true) {
    if (-not (Get-Command scoop -ErrorAction SilentlyContinue)) {
      if (Prompt-YesNo "Scoop not found. Install Scoop now?" $true) {
        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
        irm get.scoop.sh | iex
      } else { throw "Scoop required to install Unison." }
    }
    scoop install unison
  } elseif (Prompt-YesNo "Install Unison using Chocolatey instead?" $true) {
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
      if (Prompt-YesNo "Chocolatey not found. Install Chocolatey now? (requires admin)" $false) {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
        iex ((New-Object System.Net.WebClient).DownloadString("https://chocolatey.org/install.ps1"))
      } else { throw "Chocolatey required to install Unison." }
    }
    choco install unison -y
  } else {
    throw "Unison is required. Install it and re-run."
  }

  # refresh PATH for this session
  if (-not (Get-Command unison -ErrorAction SilentlyContinue)) {
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","User")
  }
  if (-not (Get-Command unison -ErrorAction SilentlyContinue)) {
    throw "Unison still not found after install. Restart PowerShell and re-run."
  }
}

# --- Create required folders (local + OneDrive) ---
New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OneDriveRoot | Out-Null
New-Item -ItemType Directory -Force -Path $UnisonDir | Out-Null
New-Item -ItemType Directory -Force -Path $UserBin | Out-Null

# --- OneDrive hydration warning ---
if (-not (Test-Path $OneDriveRoot)) {
  throw "OneDrive root does not exist: $OneDriveRoot"
}
Write-Host "`nReminder: make sure OneDrive folder is fully local."
Write-Host "In File Explorer: right-click '$OneDriveRoot' -> 'Always keep on this device'.`n"

# --- Write Unison profile using extended-length paths for OneDrive compatibility ---
$LocalRootU    = Convert-ToExtendedPath $LocalRoot
$OneDriveRootU = Convert-ToExtendedPath $OneDriveRoot

@"
root = "$LocalRootU"
root = "$OneDriveRootU"

auto = true
batch = true
fastcheck = true
times = true
maxthreads = 1
prefer = newer
confirmbigdel = true

# mac profile only ignores dotfiles; keep that plus a few safe dev ignores
ignore = Name .*
ignore = Name .git
ignore = Name .git/
ignore = Name __pycache__
ignore = Name node_modules
ignore = Name .venv
ignore = Name *.tmp
ignore = Name Thumbs.db
ignore = Name desktop.ini
"@ | Set-Content -Encoding UTF8 -Path $ProfilePath

# --- Initial sync ---
Write-Host "`nRunning initial sync..."
unison $ProfileName -ui text

# --- Watch script ---
@"
@echo off
unison $ProfileName -repeat watch -ui text -log -logfile %TEMP%\unison_esoa.log
"@ | Set-Content -Encoding ASCII -Path $WatchCmd

Write-Host "`nWatch script created at: $WatchCmd"

if (Prompt-YesNo "Start continuous watch sync now?" $true) {
  Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$WatchCmd`"" -WindowStyle Normal
}

# --- Auto-start on login ---
if (Prompt-YesNo "Auto-start watch sync on login via Task Scheduler?" $true) {
  try {
    $Action  = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$WatchCmd`""
    $Trigger = New-ScheduledTaskTrigger -AtLogOn

    Register-ScheduledTask `
      -TaskName $TaskName `
      -Action $Action `
      -Trigger $Trigger `
      -Force `
      -ErrorAction Stop | Out-Null

    Write-Host "Scheduled task registered: $TaskName"
  } catch {
    Write-Warning "Task Scheduler registration failed (no admin rights)."
    Write-Host "Fallback: add watch script to Startup folder:"
    Write-Host "  Win+R -> shell:startup"
    Write-Host "  Copy this file there:"
    Write-Host "  $WatchCmd"
  }
}

Write-Host "`nDone. Two-way Unison sync is set up for FULL roots between:"
Write-Host "  $LocalRoot"
Write-Host "  $OneDriveRoot"
Write-Host "Profile: $ProfilePath"

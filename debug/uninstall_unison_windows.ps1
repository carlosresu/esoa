$ErrorActionPreference = "Stop"

function Prompt-YesNo($q, $defaultYes=$true) {
  $suffix = if ($defaultYes) { "[Y/n]" } else { "[y/N]" }
  $ans = Read-Host "$q $suffix"
  if ([string]::IsNullOrWhiteSpace($ans)) { return $defaultYes }
  return $ans.Trim().ToLower().StartsWith("y")
}

# --- paths matching the WINDOWS mac-equivalent installer ---
$RepoName        = "esoa"
$LocalRootBase   = Join-Path $env:USERPROFILE "github_repos"
$LocalRoot       = Join-Path $LocalRootBase $RepoName

if ([string]::IsNullOrWhiteSpace($env:OneDrive)) {
  throw "OneDrive environment variable not found. Open OneDrive once or set `$OneDriveRoot manually."
}
$OneDriveRootBase = Join-Path $env:OneDrive "GitHubRepositories"
$OneDriveRoot     = Join-Path $OneDriveRootBase $RepoName

$UnisonDir   = Join-Path $env:USERPROFILE ".unison"
$ProfileName = "esoa"
$ProfilePath = Join-Path $UnisonDir "$ProfileName.prf"
$UserBin     = Join-Path $env:USERPROFILE "bin"
$WatchCmd    = Join-Path $UserBin "unison_esoa_watch.cmd"
$TaskName    = "Unison ESOA Watch"

Write-Host "=== Unison ESOA Uninstall (OneDrive-safe) ===`n"
Write-Host "NOTE: This script will NOT delete anything inside:"
Write-Host "  $OneDriveRoot`n"

# --- 1) Stop running watch sync (best-effort) ---
Write-Host "Stopping any running Unison watch processes..."
try {
  Get-CimInstance Win32_Process -Filter "Name='unison.exe'" |
    Where-Object { $_.CommandLine -match "-repeat\s+watch" -and $_.CommandLine -match $ProfileName } |
    ForEach-Object {
      Write-Host "  Killing PID $($_.ProcessId): $($_.CommandLine)"
      Stop-Process -Id $_.ProcessId -Force
    }
} catch {
  Write-Warning "  Could not enumerate/stop watch processes. You may close them manually."
}

# --- 2) Remove scheduled task if it exists ---
Write-Host "`nRemoving scheduled task '$TaskName' if present..."
try {
  $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if ($null -ne $existingTask) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "  Removed scheduled task."
  } else {
    Write-Host "  No scheduled task found."
  }
} catch {
  Write-Warning "  Failed to remove scheduled task (maybe no rights)."
}

# --- 3) Delete watch CMD script ---
Write-Host "`nRemoving watch script..."
if (Test-Path $WatchCmd) {
  if (Prompt-YesNo "Delete watch script at $WatchCmd ?" $true) {
    Remove-Item -Force $WatchCmd
    Write-Host "  Deleted $WatchCmd"
  }
} else {
  Write-Host "  Watch script not found."
}

# --- 4) Delete Unison profile ---
Write-Host "`nRemoving Unison profile..."
if (Test-Path $ProfilePath) {
  if (Prompt-YesNo "Delete Unison profile at $ProfilePath ?" $true) {
    Remove-Item -Force $ProfilePath
    Write-Host "  Deleted $ProfilePath"
  }
} else {
  Write-Host "  Profile not found."
}

# --- 5) Optional: remove ONLY local root (never OneDrive) ---
Write-Host "`nOptional cleanup of LOCAL root only:"
if (Test-Path $LocalRoot) {
  if (Prompt-YesNo "Delete local root folder $LocalRoot and all its contents?" $false) {
    Remove-Item -Recurse -Force $LocalRoot
    Write-Host "  Deleted $LocalRoot"
  } else {
    Write-Host "  Leaving local root untouched."
  }
} else {
  Write-Host "  Local root not found."
}

# --- 6) NEVER touch OneDriveRoot folders ---
Write-Host "`nOneDrive location left untouched:"
Write-Host "  $OneDriveRoot"

# --- 7) Optionally uninstall Unison ---
Write-Host "`nOptional: uninstall Unison itself."

$hasScoop  = Get-Command scoop -ErrorAction SilentlyContinue
$hasChoco  = Get-Command choco -ErrorAction SilentlyContinue
$hasUnison = Get-Command unison -ErrorAction SilentlyContinue

if ($hasUnison) {
  if (Prompt-YesNo "Uninstall Unison package now?" $false) {
    if ($hasScoop) {
      try { scoop uninstall unison; Write-Host "  Unison uninstalled via Scoop." }
      catch { Write-Warning "  Scoop uninstall failed." }
    }
    if ($hasChoco) {
      try { choco uninstall unison -y; Write-Host "  Unison uninstalled via Chocolatey." }
      catch { Write-Warning "  Chocolatey uninstall failed." }
    }
    if (-not $hasScoop -and -not $hasChoco) {
      Write-Warning "  Scoop/Chocolatey not found. Uninstall Unison manually if needed."
    }
  }
} else {
  Write-Host "  Unison not on PATH — skipping package uninstall."
}

Write-Host "`nDone. GitHub Repos Unison setup removed (OneDrive safe)."

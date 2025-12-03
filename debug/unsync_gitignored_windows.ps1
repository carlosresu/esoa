<# 
Teardown for OneDrive symlinks (username-agnostic).

Defaults:
  Repo root:      %USERPROFILE%\github_repos\pids-drg-esoa
  OneDrive root:  %OneDrive%\GitIgnored\pids-drg-esoa
Overrides (optional):
  $env:ESOA_REPO_ROOT
  $env:ESOA_ONEDRIVE_ROOT

Behavior:
- If repo folder is a symlink:
    - deletes symlink only
    - if copyBack=true: recreates real folder + copies OneDrive contents back
    - if copyBack=false: leaves NO folder in repo

Auto-elevates to Admin if needed and WAITs for completion.
#>

# ---------------------------
# Auto-elevate to Admin (wait for completion)
# ---------------------------
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[ELEVATE] Relaunching as administrator..."
    $argLine = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($args.Count -gt 0) {
        $argLine += " " + ($args -join " ")
    }

    try {
        Start-Process powershell.exe -Verb RunAs -ArgumentList $argLine -Wait | Out-Null
        Write-Host "[ELEVATE] Elevated run finished."
    } catch {
        Write-Error "Admin elevation cancelled or failed."
    }
    return
}

# ---------------------------
# Main script
# ---------------------------
$ErrorActionPreference = "Stop"

$userHome = $env:USERPROFILE
$oneDrive = if ($env:OneDrive) { $env:OneDrive } else { Join-Path $userHome "OneDrive" }

$repoRoot = if ($env:ESOA_REPO_ROOT) { $env:ESOA_REPO_ROOT } else { Join-Path $userHome "github_repos\pids-drg-esoa" }
$odRoot   = if ($env:ESOA_ONEDRIVE_ROOT) { $env:ESOA_ONEDRIVE_ROOT } else { Join-Path $oneDrive "GitIgnored\pids-drg-esoa" }

$folders  = @("raw", "outputs", "inputs")
$copyBack = $false   # <- set true if you want real folders restored with contents

function Is-Symlink($path) {
    if (-not (Test-Path $path)) { return $false }
    $item = Get-Item $path -Force
    return ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
}

foreach ($name in $folders) {
    $repoPath = Join-Path $repoRoot $name
    $odPath   = Join-Path $odRoot   $name

    if ( (Test-Path $repoPath) -and (Is-Symlink $repoPath) ) {
        Write-Host "[UNLINK] Removing symlink repo/$name"
        Remove-Item $repoPath -Force

        if ($copyBack -and (Test-Path $odPath)) {
            Write-Host "[RESTORE] Restoring real repo/$name and copying contents back"
            New-Item -ItemType Directory -Path $repoPath | Out-Null
            Copy-Item (Join-Path $odPath "*") $repoPath -Recurse -Force
        } else {
            Write-Host "[CLEAN] Leaving no repo/$name folder (symlink removed)."
        }
    }
    elseif (Test-Path $repoPath) {
        # Not a symlink; user asked to rid repo of these folders.
        Write-Host "[REMOVE] repo/$name exists but is not a symlink; removing to clean repo."
        Remove-Item $repoPath -Recurse -Force
    }
    else {
        Write-Host "[SKIP] repo/$name missing."
    }
}

Write-Host "`nDone. Repo cleaned (no raw/outputs/inputs dirs or symlinks)."

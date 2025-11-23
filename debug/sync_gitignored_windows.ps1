<# 
OneDrive symlink setup for raw/outputs/inputs (username-agnostic).

Defaults:
  Repo root:      %USERPROFILE%\github_repos\esoa
  OneDrive root:  %OneDrive%\GitIgnored\esoa
Overrides (optional):
  $env:ESOA_REPO_ROOT
  $env:ESOA_ONEDRIVE_ROOT

Behavior:
- Ensures OneDrive has raw/outputs/inputs dirs.
- If repo folder exists and is NOT a symlink:
    - moves its contents into OneDrive (merge-skip on collisions)
    - removes repo folder
- Creates directory symlink in repo -> OneDrive folder.

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

$repoRoot = if ($env:ESOA_REPO_ROOT) { $env:ESOA_REPO_ROOT } else { Join-Path $userHome "github_repos\esoa" }
$odRoot   = if ($env:ESOA_ONEDRIVE_ROOT) { $env:ESOA_ONEDRIVE_ROOT } else { Join-Path $oneDrive "GitIgnored\esoa" }

$folders = @("raw", "outputs", "inputs")

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

function Is-Symlink($path) {
    if (-not (Test-Path $path)) { return $false }
    $item = Get-Item $path -Force
    return ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
}

Ensure-Dir $repoRoot
Ensure-Dir $odRoot

foreach ($name in $folders) {
    $repoPath = Join-Path $repoRoot $name
    $odPath   = Join-Path $odRoot   $name

    Ensure-Dir $odPath

    if (Test-Path $repoPath) {
        if (Is-Symlink $repoPath) {
            Write-Host "[OK] Repo $name already a symlink. Skipping move."
        } else {
            Write-Host "[MOVE] Moving existing repo/$name contents to OneDrive..."
            Get-ChildItem $repoPath -Force | ForEach-Object {
                $dest = Join-Path $odPath $_.Name
                if (Test-Path $dest) {
                    Write-Host "  [MERGE] $($_.Name) exists in OneDrive. Skipping."
                } else {
                    Move-Item $_.FullName $odPath
                }
            }
            Remove-Item $repoPath -Recurse -Force
        }
    }

    if (-not (Test-Path $repoPath)) {
        Write-Host "[LINK] Creating symlink repo/$name -> OneDrive/$name"
        New-Item -ItemType SymbolicLink -Path $repoPath -Target $odPath | Out-Null
    }
}

Write-Host "`nDone. Symlinks:"
foreach ($name in $folders) {
    $repoPath = Join-Path $repoRoot $name
    cmd /c "dir /AL `"$repoPath`""
}

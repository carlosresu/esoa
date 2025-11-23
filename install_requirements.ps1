$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSCommandPath
$TargetPython = "3.12.10"
$VenvDir = Join-Path $RepoRoot ".venv"

function Write-Info {
    param($Message)
    Write-Host "[setup] $Message"
}

function Write-Warn {
    param($Message)
    Write-Warning $Message
}

function Run-Cmd {
    param(
        [string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    $joined = if ($Args) { $Args -join " " } else { "" }
    Write-Info ("Running: {0} {1}" -f $Command, $joined)
    & $Command @Args
}

function Resolve-RscriptPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    if (-not (Test-Path $Path -PathType Leaf)) { return $null }
    try {
        return (Resolve-Path $Path).Path
    } catch {
        return $null
    }
}

function Ensure-Submodules {
    if ((Test-Path (Join-Path $RepoRoot ".git")) -and (Test-Path (Join-Path $RepoRoot ".gitmodules"))) {
        $git = Get-Command git -ErrorAction SilentlyContinue
        if ($git) {
            try {
                Run-Cmd $git.Path @("submodule", "update", "--init", "--recursive")
            } catch {
                Write-Warn "Submodule update failed: $_"
            }
        }
    }
}

function Pyenv-Candidates {
    $userHome = [Environment]::GetFolderPath("UserProfile")
    @(
        (Get-Command pyenv -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path),
        (Join-Path $userHome ".pyenv\pyenv-win\bin\pyenv.cmd"),
        (Join-Path $userHome ".pyenv\pyenv-win\pyenv-win\bin\pyenv.cmd"),
        (Join-Path $userHome ".pyenv\pyenv-win\bin\pyenv.exe"),
        (Join-Path $userHome ".pyenv\pyenv-win\pyenv-win\bin\pyenv.exe")
    ) | Where-Object { $_ -and (Test-Path $_) }
}

function Install-PyenvWin {
    Write-Info "Installing pyenv-win..."
    $tmp = Join-Path $env:TEMP ("pyenv-win-install-{0}.ps1" -f ([guid]::NewGuid()))
    Invoke-WebRequest -UseBasicParsing -Uri "https://pyenv-win.github.io/pyenv-win/install.ps1" -OutFile $tmp
    powershell -NoProfile -ExecutionPolicy Bypass -File $tmp
    Remove-Item -Force $tmp -ErrorAction SilentlyContinue
}

function Ensure-Pyenv {
    $pyenv = Pyenv-Candidates | Select-Object -First 1
    if (-not $pyenv) {
        try {
            Install-PyenvWin
        } catch {
            Write-Warn "pyenv-win installation failed: $_"
        }
        $pyenv = Pyenv-Candidates | Select-Object -First 1
    }
    if ($pyenv) {
        $root = Split-Path -Parent (Split-Path -Parent $pyenv)
        $env:PYENV = $root
        $env:PYENV_ROOT = $root
        $env:PATH = "$root\pyenv-win\bin;$root\pyenv-win\shims;$env:PATH"
    }
    return $pyenv
}

function Ensure-PythonVersion {
    param([string]$PyenvPath)
    if (-not $PyenvPath) {
        Write-Warn "pyenv not found; ensure Python $TargetPython is installed manually."
        return
    }
    $versions = (& $PyenvPath "versions" "--bare" 2>$null) -join "`n"
    if ($versions -notmatch [regex]::Escape($TargetPython)) {
        Run-Cmd $PyenvPath @("install", "-s", $TargetPython)
    } else {
        Write-Info "Python $TargetPython already installed via pyenv."
    }
}

function Ensure-Venv {
    param([string]$PyenvPath)
    if (Test-Path $VenvDir) { return }
    if ($PyenvPath) {
        $env:PYENV_VERSION = $TargetPython
        Run-Cmd $PyenvPath @("exec", "python", "-m", "venv", $VenvDir)
    } else {
        Run-Cmd "python" @("-m", "venv", $VenvDir)
    }
}

function Get-VenvPython {
    return Join-Path $VenvDir "Scripts\python.exe"
}

function Find-Rscript {
    $onPath = Get-Command Rscript -ErrorAction SilentlyContinue
    if ($onPath) {
        $resolved = Resolve-RscriptPath $onPath.Path
        if ($resolved -and ($resolved.ToLower().EndsWith("rscript.exe"))) { return $resolved }
    }
    $candidates = @()
    foreach ($base in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($base) {
            $candidates += Get-ChildItem -Path (Join-Path $base "R") -Filter "Rscript.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
        }
    }
    if ($candidates) {
        $resolved = Resolve-RscriptPath (($candidates | Sort-Object)[-1])
        if ($resolved -and ($resolved.ToLower().EndsWith("rscript.exe"))) { return $resolved }
    }
    return $null
}

function Ensure-R {
    $existing = Find-Rscript
    if ($existing) {
        $env:PATH = "$(Split-Path -Parent $existing);$env:PATH"
        Write-Info "Rscript found at $existing"
        return $existing
    }

    Write-Info "Installing R via winget/choco..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if ($winget) {
        try {
            Run-Cmd $winget.Path @("install", "-e", "--id", "RProject.R", "--accept-package-agreements", "--accept-source-agreements", "--silent")
            $existing = Find-Rscript
            if ($existing) {
                $env:PATH = "$(Split-Path -Parent $existing);$env:PATH"
                Write-Info "Rscript installed at $existing"
                return $existing
            }
        } catch {
            Write-Warn "winget install failed: $_"
        }
    }
    if ($choco) {
        try {
            Run-Cmd $choco.Path @("install", "r.project", "-y")
            $existing = Find-Rscript
            if ($existing) {
                $env:PATH = "$(Split-Path -Parent $existing);$env:PATH"
                Write-Info "Rscript installed at $existing"
                return $existing
            }
        } catch {
            Write-Warn "Chocolatey install failed: $_"
        }
    }
    Write-Warn "R could not be installed automatically; please install it manually."
    return $null
}

function Install-RPackages {
    param([string]$RscriptPath)
    $rPath = if ($RscriptPath) { Resolve-RscriptPath $RscriptPath } else { Find-Rscript }
    if (-not $rPath) {
        Write-Warn "Skipping R package install because Rscript was not found."
        return
    }
    $expr = @"
pkgs <- c(
  "arrow","data.table","dplyr","furrr","future","future.apply","httr2","memoise",
  "pacman","purrr","readr","rvest","stringr","tibble","xml2","remotes","languageserver"
)
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) {
  install.packages(missing, repos="https://cloud.r-project.org")
}
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos="https://cloud.r-project.org")
}
if (!requireNamespace("dbdataset", quietly = TRUE)) {
  remotes::install_github("interstellar-Consultation-Services/dbdataset", quiet = TRUE, upgrade = "never")
}
"@
    Write-Info "Installing required R packages..."
    Run-Cmd $rPath @("-e", $expr)
}

function Install-PipDeps {
    $venvPy = Get-VenvPython
    if (-not (Test-Path $venvPy)) {
        Write-Warn "Virtualenv Python not found; skipping pip install."
        return
    }
    Run-Cmd $venvPy @("-m", "pip", "install", "--upgrade", "pip")
    $reqPath = Join-Path $RepoRoot "requirements.txt"
    if (Test-Path $reqPath) {
        Run-Cmd $venvPy @("-m", "pip", "install", "-r", $reqPath)
    } else {
        Write-Warn "requirements.txt not found; skipping main dependency install."
    }
    $scraperInstaller = Join-Path $RepoRoot "dependencies\fda_ph_scraper\install_requirements.py"
    if (Test-Path $scraperInstaller) {
        Run-Cmd $venvPy @($scraperInstaller)
    }
}

Write-Info "Bootstrapping environment on Windows..."
Ensure-Submodules
$pyenv = Ensure-Pyenv
Ensure-PythonVersion -PyenvPath $pyenv
Ensure-Venv -PyenvPath $pyenv
$rscriptPath = Ensure-R
# Resolve Rscript after installation in case PATH changed; fall back to the path returned by Ensure-R.
$resolvedRscript = Find-Rscript
if (-not $resolvedRscript -and $rscriptPath) {
    $resolvedRscript = Resolve-RscriptPath $rscriptPath
}
if ($resolvedRscript) {
    Install-RPackages -RscriptPath $resolvedRscript
} else {
    Write-Warn "Rscript not available; skipped R package installation."
}
Install-PipDeps
Write-Info "Setup complete. Activate with .\.venv\Scripts\Activate.ps1"

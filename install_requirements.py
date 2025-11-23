#!/usr/bin/env python3
"""Bootstrap Python (pyenv/pyenv-win), R, submodules, and pip deps."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

TARGET_PYTHON = "3.12.10"
VENV_DIR = Path(__file__).resolve().parent / ".venv"
REQUIREMENTS_PATH = Path(__file__).resolve().parent / "requirements.txt"


def run_command(cmd: List[str], env: Optional[dict] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command with basic logging."""
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=check, env=env)


def detect_platform() -> str:
    sysname = platform.system().lower()
    if "windows" in sysname:
        return "windows"
    if "darwin" in sysname:
        return "mac"
    if "linux" in sysname:
        return "linux"
    return "unknown"


def ensure_submodules() -> None:
    """Keep submodules present so downstream scripts run."""
    if not (Path(".git").exists() and Path(".gitmodules").exists()):
        return
    git = shutil.which("git")
    if not git:
        print("git not found; skipping submodule update.")
        return
    try:
        run_command([git, "submodule", "update", "--init", "--recursive"])
    except subprocess.CalledProcessError as exc:
        print(f"Submodule update failed (exit {exc.returncode}); continue with caution.")


def _pyenv_candidates(platform_name: str) -> List[str]:
    home = Path.home()
    candidates: List[str] = []
    if platform_name == "windows":
        candidates.extend(
            [
                shutil.which("pyenv"),
                str(home / ".pyenv" / "pyenv-win" / "bin" / "pyenv.cmd"),
                str(home / ".pyenv" / "pyenv-win" / "bin" / "pyenv.exe"),
            ]
        )
    else:
        candidates.extend([shutil.which("pyenv"), str(home / ".pyenv" / "bin" / "pyenv")])
    return [c for c in candidates if c]


def install_pyenv_win() -> bool:
    """Install pyenv-win via the official PowerShell installer."""
    installer = "https://pyenv-win.github.io/pyenv-win/install.ps1"
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        f"Invoke-WebRequest -UseBasicParsing -Uri '{installer}' -OutFile $env:TEMP\\pyenv-win-install.ps1; "
        "& $env:TEMP\\pyenv-win-install.ps1",
    ]
    try:
        run_command(cmd)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"pyenv-win installation failed: {exc}")
        return False


def install_pyenv_unix() -> bool:
    """Install pyenv via the upstream installer script (curl | bash)."""
    bash = shutil.which("bash")
    curl = shutil.which("curl")
    if not bash or not curl:
        print("bash/curl missing; cannot auto-install pyenv.")
        return False
    cmd = [bash, "-c", "curl -fsSL https://pyenv.run | bash"]
    try:
        run_command(cmd)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"pyenv installation failed: {exc}")
        return False


def ensure_pyenv(platform_name: str) -> Optional[str]:
    """Ensure pyenv/pyenv-win is available; return the resolved executable path."""
    for candidate in _pyenv_candidates(platform_name):
        if candidate and Path(candidate).exists():
            return candidate

    installed = False
    if platform_name == "windows":
        installed = install_pyenv_win()
    elif platform_name in {"mac", "linux"}:
        installed = install_pyenv_unix()

    if not installed:
        print("pyenv was not installed automatically; continuing with current Python.")
        return None

    # Refresh PATH for the current process so subsequent calls can find pyenv.
    home = Path.home()
    extra_paths: Iterable[Path]
    if platform_name == "windows":
        extra_paths = [
            home / ".pyenv" / "pyenv-win" / "bin",
            home / ".pyenv" / "pyenv-win" / "shims",
        ]
    else:
        extra_paths = [home / ".pyenv" / "bin"]
    for p in extra_paths:
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

    for candidate in _pyenv_candidates(platform_name):
        if candidate and Path(candidate).exists():
            return candidate

    print("pyenv still not found after installation attempt.")
    return None


def pyenv_exec(pyenv_cmd: str, version: str, args: List[str]) -> None:
    """Run a pyenv-managed command with PYENV_VERSION set."""
    env = os.environ.copy()
    env["PYENV_VERSION"] = version
    run_command([pyenv_cmd, "exec", *args], env=env)


def ensure_python_version(pyenv_cmd: str, version: str) -> None:
    """Install the requested Python version through pyenv/pyenv-win."""
    try:
        output = subprocess.check_output([pyenv_cmd, "versions", "--bare"], text=True)
    except subprocess.CalledProcessError:
        output = ""
    versions = {line.strip() for line in output.splitlines() if line.strip()}
    if version not in versions:
        run_command([pyenv_cmd, "install", "-s", version])
    else:
        print(f"Python {version} already available via pyenv.")


def ensure_virtualenv(pyenv_cmd: Optional[str], platform_name: str, version: str) -> Path:
    """Create or reuse the project virtualenv."""
    if VENV_DIR.is_dir():
        return VENV_DIR

    if pyenv_cmd:
        pyenv_exec(pyenv_cmd, version, ["python", "-m", "venv", str(VENV_DIR)])
    else:
        run_command([sys.executable, "-m", "venv", str(VENV_DIR)])

    return VENV_DIR


def venv_python_path(platform_name: str) -> Path:
    if platform_name == "windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_pip_requirements(venv_python: Path) -> None:
    if not REQUIREMENTS_PATH.exists():
        print("requirements.txt not found; skipping pip install.", file=sys.stderr)
        return
    run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)])


def find_rscript(platform_name: str) -> Optional[Path]:
    """Locate Rscript even if it's not on PATH."""
    on_path = shutil.which("Rscript")
    if on_path:
        return Path(on_path)

    if platform_name == "windows":
        paths: List[Path] = []
        bases = []
        pf = os.environ.get("ProgramFiles")
        if pf:
            bases.append(Path(pf))
        pfx = os.environ.get("ProgramFiles(x86)")
        if pfx:
            bases.append(Path(pfx))
        for base in bases:
            for pattern in ("R/R-*/bin/x64/Rscript.exe", "R/R-*/bin/Rscript.exe"):
                for candidate in base.glob(pattern):
                    if candidate.is_file():
                        paths.append(candidate)
        if paths:
            return sorted(paths)[-1]

    if platform_name == "mac":
        mac_path = Path("/Library/Frameworks/R.framework/Resources/bin/Rscript")
        if mac_path.is_file():
            return mac_path

    for candidate in (Path("/usr/bin/Rscript"), Path("/usr/local/bin/Rscript")):
        if candidate.is_file():
            return candidate

    return None


def ensure_r(platform_name: str) -> None:
    """Install R so Rscript is available for WHO/DrugBank runners."""
    existing = find_rscript(platform_name)
    if existing:
        os.environ["PATH"] = f"{existing.parent}{os.pathsep}{os.environ.get('PATH', '')}"
        print(f"Rscript already available at {existing}.")
        return

    success = False
    if platform_name == "windows":
        winget = shutil.which("winget")
        choco = shutil.which("choco")
        if winget:
            try:
                run_command(
                    [
                        winget,
                        "install",
                        "-e",
                        "--id",
                        "RProject.R",
                        "--accept-package-agreements",
                        "--accept-source-agreements",
                        "--silent",
                    ]
                )
                success = True
            except subprocess.CalledProcessError as exc:
                print(f"winget R install failed (exit {exc.returncode}); trying Chocolatey.")
        if not success and choco:
            try:
                run_command(["choco", "install", "r.project", "-y"])
                success = True
            except subprocess.CalledProcessError as exc:
                print(f"Chocolatey R install failed (exit {exc.returncode}).")
    elif platform_name == "mac":
        brew = shutil.which("brew")
        if brew:
            try:
                run_command([brew, "install", "--cask", "r"])
                success = True
            except subprocess.CalledProcessError:
                try:
                    run_command([brew, "install", "r"])
                    success = True
                except subprocess.CalledProcessError as exc:
                    print(f"Homebrew R install failed (exit {exc.returncode}).")
    elif platform_name == "linux":
        if shutil.which("apt-get"):
            try:
                run_command(["sudo", "apt-get", "update"])
                run_command(["sudo", "apt-get", "install", "-y", "r-base"])
                success = True
            except subprocess.CalledProcessError as exc:
                print(f"apt-get R install failed (exit {exc.returncode}).")
        elif shutil.which("dnf"):
            try:
                run_command(["sudo", "dnf", "install", "-y", "R"])
                success = True
            except subprocess.CalledProcessError as exc:
                print(f"dnf R install failed (exit {exc.returncode}).")
        elif shutil.which("yum"):
            try:
                run_command(["sudo", "yum", "install", "-y", "R"])
                success = True
            except subprocess.CalledProcessError as exc:
                print(f"yum R install failed (exit {exc.returncode}).")
        elif shutil.which("pacman"):
            try:
                run_command(["sudo", "pacman", "-S", "--noconfirm", "r"])
                success = True
            except subprocess.CalledProcessError as exc:
                print(f"pacman R install failed (exit {exc.returncode}).")

    if not success:
        print("R could not be installed automatically; please install it manually.")
        return

    located = find_rscript(platform_name)
    if located:
        os.environ["PATH"] = f"{located.parent}{os.pathsep}{os.environ.get('PATH', '')}"
        print(f"Rscript installed at {located}.")
    else:
        print("R installation attempted but Rscript is still missing; check your PATH.")


def main() -> None:
    platform_name = detect_platform()
    print(f"Detected platform: {platform_name}")

    ensure_submodules()

    pyenv_cmd = ensure_pyenv(platform_name)
    if pyenv_cmd:
        ensure_python_version(pyenv_cmd, TARGET_PYTHON)
    else:
        print("Proceeding without pyenv; using current interpreter for dependency install.")

    ensure_r(platform_name)

    venv_path = ensure_virtualenv(pyenv_cmd, platform_name, TARGET_PYTHON)
    venv_python = venv_python_path(platform_name)
    ensure_pip_requirements(venv_python)

    print("Environment bootstrap complete.")


if __name__ == "__main__":
    main()

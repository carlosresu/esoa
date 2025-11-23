# Environment Setup

Use the platform-specific installer at the repository root to provision everything required (Python, R, virtualenv, and submodule dependencies).

- **Windows:** run `.\install_requirements.ps1` from the repo root in PowerShell.
- **macOS/Linux:** run `./install_requirements.sh` from the repo root in a shell.

What the installers do:
- Update git submodules so dependency trees are present.
- Install pyenv/pyenv-win if missing, then install Python 3.12.10.
- Create or reuse `.venv` in the repo root and install `requirements.txt`.
- Install the FDA PH scraper requirements via its bundled installer.
- Install R (via winget/choco on Windows; Homebrew or common package managers on macOS/Linux) so the R helpers can run.

After the installer completes, activate the virtualenv before running any Python tools:

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source ./.venv/bin/activate
```

Then run the pipeline utilities normally (e.g., `python run_all.py`).

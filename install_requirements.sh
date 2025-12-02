#!/usr/bin/env bash
# Cross-platform bootstrapper for macOS/Linux: installs pyenv, Python 3.12.12, R, and Python deps in .venv.
set -euo pipefail

TARGET_PYTHON="3.12.12"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

log() { printf '[setup] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_submodules() {
  if [ -d "${REPO_ROOT}/.git" ] && [ -f "${REPO_ROOT}/.gitmodules" ] && have_cmd git; then
    log "Updating git submodules..."
    git -C "${REPO_ROOT}" submodule update --init --recursive
  fi
}

ensure_pyenv_paths() {
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
  export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
  if have_cmd pyenv; then
    eval "$(pyenv init -)" >/dev/null 2>&1 || true
  fi
}

install_pyenv() {
  ensure_pyenv_paths
  if have_cmd pyenv; then
    return 0
  fi

  log "Installing pyenv..."
  if have_cmd brew; then
    brew update && brew install pyenv
  else
    if have_cmd curl; then
      curl -fsSL https://pyenv.run | bash
    else
      warn "curl not available; install curl or pyenv manually."
      return 1
    fi
  fi
  ensure_pyenv_paths
  if ! have_cmd pyenv; then
    warn "pyenv installation did not succeed; please install pyenv manually."
    return 1
  fi
}

ensure_python() {
  if ! have_cmd pyenv; then
    install_pyenv || warn "Proceeding without pyenv; ensure Python ${TARGET_PYTHON} is available."
  fi

  if have_cmd pyenv; then
    if ! pyenv versions --bare | grep -qx "${TARGET_PYTHON}"; then
      log "Installing Python ${TARGET_PYTHON} via pyenv..."
      pyenv install -s "${TARGET_PYTHON}"
    else
      log "Python ${TARGET_PYTHON} already available via pyenv."
    fi
  fi
}

ensure_venv() {
  local py_cmd
  if [ -x "${VENV_DIR}/bin/python" ]; then
    return 0
  fi

  if have_cmd pyenv && pyenv versions --bare | grep -qx "${TARGET_PYTHON}"; then
    log "Creating virtualenv with pyenv ${TARGET_PYTHON}..."
    PYENV_VERSION="${TARGET_PYTHON}" pyenv exec python -m venv "${VENV_DIR}"
  elif have_cmd python3; then
    log "Creating virtualenv with system python3..."
    python3 -m venv "${VENV_DIR}"
  else
    warn "No Python interpreter available to create a virtualenv."
    return 1
  fi
}

install_r() {
  if have_cmd Rscript; then
    log "Rscript detected at $(command -v Rscript)"
    return 0
  fi

  log "Installing R..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    if have_cmd brew; then
      brew install --cask r || brew install r || true
    else
      warn "Homebrew not available; install R manually from https://cran.r-project.org/."
      return 1
    fi
  elif [[ "$OSTYPE" == "linux"* ]]; then
    if have_cmd apt-get; then
      sudo apt-get update && sudo apt-get install -y r-base || true
    elif have_cmd dnf; then
      sudo dnf install -y R || true
    elif have_cmd yum; then
      sudo yum install -y R || true
    elif have_cmd pacman; then
      sudo pacman -Sy --noconfirm r || true
    else
      warn "No supported package manager found; install R manually."
      return 1
    fi
  else
    warn "Unrecognized platform ($OSTYPE); install R manually."
    return 1
  fi

  if have_cmd Rscript; then
    log "Rscript installed at $(command -v Rscript)"
  else
    warn "R installation attempted but Rscript is still missing; please install R manually."
  fi
}

install_r_packages() {
  if ! have_cmd Rscript; then
    warn "Skipping R package install because Rscript is unavailable."
    return
  fi
  log "Installing required R packages..."
  Rscript -e 'pkgs <- c("arrow","data.table","dplyr","furrr","future","future.apply","httr2","memoise","pacman","purrr","readr","rvest","stringr","tibble","xml2","remotes","languageserver"); missing <- setdiff(pkgs, rownames(installed.packages())); if (length(missing)) install.packages(missing, repos="https://cloud.r-project.org"); if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", repos="https://cloud.r-project.org"); if (!requireNamespace("dbdataset", quietly = TRUE)) remotes::install_github("interstellar-Consultation-Services/dbdataset", quiet = TRUE, upgrade = "never")'
}

pip_install() {
  local py="${VENV_DIR}/bin/python"
  if [ ! -x "$py" ]; then
    warn "Virtualenv Python not found; cannot install requirements."
    return 1
  fi
  log "Upgrading pip and installing requirements..."
  "$py" -m pip install --upgrade pip
  "$py" -m pip install -r "${REPO_ROOT}/requirements.txt"
  if [ -f "${REPO_ROOT}/dependencies/fda_ph_scraper/install_requirements.py" ]; then
    log "Installing FDA PH scraper dependencies..."
    "$py" "${REPO_ROOT}/dependencies/fda_ph_scraper/install_requirements.py"
  fi
}

main() {
  log "Bootstrapping environment for macOS/Linux..."
  ensure_submodules
  install_pyenv || true
  ensure_pyenv_paths
  ensure_python
  ensure_venv
  install_r || true
  install_r_packages || true
  pip_install
  log "Setup complete. Activate the virtualenv with 'source ${VENV_DIR}/bin/activate'."
}

main "$@"

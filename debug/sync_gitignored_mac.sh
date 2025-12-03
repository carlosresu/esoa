#!/usr/bin/env bash
set -euo pipefail

# Username-agnostic OneDrive symlink setup for raw/outputs/inputs.
#
# Defaults:
#   Repo root:     "$HOME/github_repos/pids-drg-esoa"
#   OneDrive root: "$HOME/Library/CloudStorage/OneDrive-Personal/GitIgnored/pids-drg-esoa"
# Overrides (optional):
#   ESOA_REPO_ROOT
#   ESOA_ONEDRIVE_ROOT
#
# Behavior:
# - Ensures OneDrive has raw/outputs/inputs dirs.
# - If repo folder exists and is NOT a symlink:
#     - moves its contents into OneDrive (merge-skip on collisions)
#     - removes repo folder
# - Creates directory symlink in repo -> OneDrive folder.

REPO_ROOT="${ESOA_REPO_ROOT:-$HOME/github_repos/pids-drg-esoa}"
OD_ROOT="${ESOA_ONEDRIVE_ROOT:-$HOME/Library/CloudStorage/OneDrive-Personal/GitIgnored/pids-drg-esoa}"
FOLDERS=("raw" "outputs" "inputs")

mkdir -p "$REPO_ROOT" "$OD_ROOT"

is_symlink() { [[ -L "$1" ]]; }

for name in "${FOLDERS[@]}"; do
  repo_path="$REPO_ROOT/$name"
  od_path="$OD_ROOT/$name"

  mkdir -p "$od_path"

  if [[ -e "$repo_path" ]]; then
    if is_symlink "$repo_path"; then
      echo "[OK] Repo $name already a symlink. Skipping move."
    else
      echo "[MOVE] Moving existing repo/$name contents to OneDrive..."
      shopt -s dotglob nullglob
      for item in "$repo_path"/*; do
        base="$(basename "$item")"
        if [[ -e "$od_path/$base" ]]; then
          echo "  [MERGE] $base exists in OneDrive. Skipping."
        else
          mv "$item" "$od_path/"
        fi
      done
      shopt -u dotglob nullglob
      rm -rf "$repo_path"
    fi
  fi

  if [[ ! -e "$repo_path" ]]; then
    echo "[LINK] Creating symlink repo/$name -> OneDrive/$name"
    ln -s "$od_path" "$repo_path"
  fi
done

echo ""
echo "Done. Symlinks:"
ls -l "$REPO_ROOT" | egrep "raw|outputs|inputs" || true

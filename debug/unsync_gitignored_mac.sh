#!/usr/bin/env bash
set -euo pipefail

# Username-agnostic teardown for OneDrive symlinks.

REPO_ROOT="${ESOA_REPO_ROOT:-$HOME/github_repos/pids-drg-esoa}"
OD_ROOT="${ESOA_ONEDRIVE_ROOT:-$HOME/Library/CloudStorage/OneDrive-Personal/GitIgnored/pids-drg-esoa}"
FOLDERS=("raw" "outputs" "inputs")

is_symlink() { [[ -L "$1" ]]; }

for name in "${FOLDERS[@]}"; do
  repo_path="$REPO_ROOT/$name"
  od_path="$OD_ROOT/$name"

  if is_symlink "$repo_path"; then
    echo "[RM] Removing symlink repo/$name"
    rm "$repo_path"
  elif [[ -e "$repo_path" ]]; then
    echo "[SKIP] repo/$name exists but is not a symlink; leaving as-is."
  else
    echo "[SKIP] repo/$name does not exist."
  fi
done

echo ""
echo "Done. Repo restored."

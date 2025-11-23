#!/usr/bin/env bash
set -euo pipefail

# Username-agnostic teardown for OneDrive symlinks.

REPO_ROOT="${ESOA_REPO_ROOT:-$HOME/github_repos/esoa}"
OD_ROOT="${ESOA_ONEDRIVE_ROOT:-$HOME/Library/CloudStorage/OneDrive-Personal/GitIgnored/esoa}"
FOLDERS=("raw" "outputs" "inputs")
COPY_BACK=true  # set false for empty restored dirs

is_symlink() { [[ -L "$1" ]]; }

for name in "${FOLDERS[@]}"; do
  repo_path="$REPO_ROOT/$name"
  od_path="$OD_ROOT/$name"

  if [[ -e "$repo_path" && is_symlink "$repo_path" ]]; then
    echo "[UNLINK] Removing symlink repo/$name"
    rm "$repo_path"
    mkdir -p "$repo_path"

    if $COPY_BACK && [[ -d "$od_path" ]]; then
      echo "[COPY] Copying OneDrive/$name contents back into repo/$name"
      cp -a "$od_path/." "$repo_path/"
    fi
  else
    echo "[SKIP] repo/$name not a symlink (or missing)."
  fi
done

echo ""
echo "Done. Repo restored."

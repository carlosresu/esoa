#!/usr/bin/env bash
set -euo pipefail

# Username-agnostic teardown for OneDrive symlinks.

REPO_ROOT="${ESOA_REPO_ROOT:-$HOME/github_repos/esoa}"
OD_ROOT="${ESOA_ONEDRIVE_ROOT:-$HOME/Library/CloudStorage/OneDrive-Personal/GitIgnored/esoa}"
FOLDERS=("raw" "outputs" "inputs")
# By default we only remove the symlink so data stays in OneDrive; set COPY_BACK=true to restore files locally.
COPY_BACK=${COPY_BACK:-false}

is_symlink() { [[ -L "$1" ]]; }

for name in "${FOLDERS[@]}"; do
  repo_path="$REPO_ROOT/$name"
  od_path="$OD_ROOT/$name"

  if [[ -e "$repo_path" ]] && is_symlink "$repo_path"; then
    echo "[UNLINK] Removing symlink repo/$name"
    rm "$repo_path"
    mkdir -p "$repo_path"

    if [[ "$COPY_BACK" == "true" && -d "$od_path" ]]; then
      echo "[COPY] Copying OneDrive/$name contents back into repo/$name"
      cp -a "$od_path/." "$repo_path/"
    else
      echo "[KEEP] Leaving contents only in OneDrive/$name (COPY_BACK=false)."
    fi
  else
    echo "[SKIP] repo/$name not a symlink (or missing)."
  fi
done

echo ""
echo "Done. Repo restored."

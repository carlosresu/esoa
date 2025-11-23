#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="$HOME/github_repos"
OD_BASE="$HOME/Library/CloudStorage"

# Pick the first OneDrive container (GitHubRepositories may be missing).
find_onedrive_root() {
  for candidate in "$OD_BASE"/OneDrive*; do
    [ -d "$candidate" ] || continue
    printf '%s\n' "$candidate"
    return 0
  done
  return 1
}

OD_CONTAINER="$(find_onedrive_root || true)"
if [ -z "$OD_CONTAINER" ]; then
  echo "Could not find OneDrive root under $OD_BASE (looked for OneDrive*)."
  exit 1
fi
OD_ROOT="$OD_CONTAINER/GitHubRepositories"

SERVICE_LABEL="com.github-repos-unison"
PLIST="$HOME/Library/LaunchAgents/${SERVICE_LABEL}.plist"
PRF_DIR="$HOME/.unison"
PRF_FILE="$PRF_DIR/github_repos.prf"

echo "=== Unison GitHub Repos Uninstall (OneDrive-safe) ==="
echo "NOTE: This script will NOT delete anything inside:"
echo "  $OD_ROOT"
echo

# Stop + unload LaunchAgent if present
if launchctl list | grep -q "$SERVICE_LABEL"; then
  echo "Stopping LaunchAgent $SERVICE_LABEL ..."
  launchctl stop "$SERVICE_LABEL" 2>/dev/null || true
fi

if [ -f "$PLIST" ]; then
  echo "Unloading plist: $PLIST"
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
else
  echo "Plist not found, skipping."
fi

# Remove profile
if [ -f "$PRF_FILE" ]; then
  echo "Removing Unison profile: $PRF_FILE"
  rm -f "$PRF_FILE"
else
  echo "Profile not found, skipping."
fi

# Optional delete local root only
echo
read -r -p "Delete LOCAL root $LOCAL_ROOT and all its contents? [y/N] " ans
ans="${ans:-N}"
if [[ "$ans" =~ ^[Yy]$ ]]; then
  if [ -d "$LOCAL_ROOT" ]; then
    rm -rf "$LOCAL_ROOT"
    echo "Deleted $LOCAL_ROOT"
  else
    echo "Local root not found."
  fi
else
  echo "Leaving local root untouched."
fi

echo
echo "OneDrive root left untouched:"
echo "  $OD_ROOT"
echo
echo "Done."

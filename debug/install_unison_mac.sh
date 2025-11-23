#!/usr/bin/env bash
set -euo pipefail

# --- Roots (macOS) ---
LOCAL_ROOT="$HOME/github_repos"
OD_BASE="$HOME/Library/CloudStorage"

# Pick the first OneDrive container (GitHubRepositories will be created if missing).
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

# --- Unison / LaunchAgent paths ---
SERVICE_LABEL="com.github-repos-unison"
PLIST="$HOME/Library/LaunchAgents/${SERVICE_LABEL}.plist"
PRF_DIR="$HOME/.unison"
PRF_FILE="$PRF_DIR/github_repos.prf"
UNISON_BIN="/opt/homebrew/bin/unison"

echo "=== Unison GitHub Repos Install (macOS) ==="
echo

# --- Preflight ---
if [ ! -x "$UNISON_BIN" ]; then
  echo "Unison not found at $UNISON_BIN"
  echo "Install with: brew install unison"
  exit 1
fi

mkdir -p "$LOCAL_ROOT"
mkdir -p "$OD_ROOT"
mkdir -p "$PRF_DIR"
mkdir -p "$(dirname "$PLIST")"

# remove old agent/profile (safe)
launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
rm -f "$PRF_FILE"

# --- Write Unison profile ---
cat > "$PRF_FILE" <<PRF
root = "$LOCAL_ROOT"
root = "$OD_ROOT"
auto = true
batch = true
fastcheck = true
times = true
maxthreads = 1
prefer = newer
confirmbigdel = true

ignore = Name .*
ignore = Name .git
ignore = Name .git/
ignore = Name __pycache__
ignore = Name node_modules
ignore = Name .venv
ignore = Name *.tmp
ignore = Name .DS_Store
PRF

# --- Write LaunchAgent plist ---
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$SERVICE_LABEL</string>

  <key>ProgramArguments</key>
  <array>
    <string>$UNISON_BIN</string>
    <string>github_repos</string>
    <string>-ui</string>
    <string>text</string>
    <string>-repeat</string>
    <string>watch</string>
    <string>-log</string>
    <string>-logfile</string>
    <string>/tmp/unison_github_repos.log</string>
  </array>

  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>

  <key>StandardOutPath</key><string>/tmp/unison_github_repos.out</string>
  <key>StandardErrorPath</key><string>/tmp/unison_github_repos.err</string>
</dict>
</plist>
PLIST

# --- Load agent + start watch ---
launchctl load "$PLIST"
launchctl start "$SERVICE_LABEL"

# --- Initial sync (foreground) ---
"$UNISON_BIN" github_repos -ui text

echo
echo "Done. Two-way Unison sync is set up for FULL roots between:"
echo "  $LOCAL_ROOT"
echo "  $OD_ROOT"
echo "Profile: $PRF_FILE"
echo "LaunchAgent: $PLIST"

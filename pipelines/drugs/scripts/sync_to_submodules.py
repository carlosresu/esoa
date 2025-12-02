"""
Sync shared scripts to submodules.

This module copies unified constants and shared utilities to submodule directories
so they can be used when running submodule code independently.

Usage:
    # Import at top of any run_*.py script to auto-sync:
    from scripts.sync_to_submodules import sync_all
    sync_all()

    # Or run directly:
    python -m scripts.sync_to_submodules
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

# Get project root (esoa/)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEPENDENCIES_DIR = PROJECT_ROOT / "dependencies"

# Source files to sync
SHARED_SCRIPTS_DIR = SCRIPT_DIR

# Files to copy to ALL submodules under input/
# (unified_constants.py includes all needed functions)
UNIVERSAL_FILES = [
    "unified_constants.py",
]

# Submodule-specific overrides (if any submodule needs different target dir)
SUBMODULE_SPECIFIC_FILES = {
    # All submodules use input/ directory by default
}


def get_submodule_dirs() -> List[Path]:
    """Get all submodule directories under dependencies/."""
    if not DEPENDENCIES_DIR.exists():
        return []
    return [d for d in DEPENDENCIES_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]


def copy_file_if_changed(src: Path, dst: Path) -> bool:
    """
    Copy file only if it doesn't exist or content has changed.
    Returns True if file was copied.
    """
    if not src.exists():
        print(f"  [WARN] Source not found: {src}")
        return False
    
    # Create parent directory if needed
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if copy needed
    if dst.exists():
        src_content = src.read_bytes()
        dst_content = dst.read_bytes()
        if src_content == dst_content:
            return False  # No change
    
    # Copy file
    shutil.copy2(src, dst)
    return True


def sync_universal_files() -> List[Tuple[str, str]]:
    """Copy universal files to all submodules."""
    synced = []
    
    for submodule_dir in get_submodule_dirs():
        input_dir = submodule_dir / "input"  # singular for consistency
        
        for filename in UNIVERSAL_FILES:
            src = SHARED_SCRIPTS_DIR / filename
            dst = input_dir / filename
            
            if copy_file_if_changed(src, dst):
                synced.append((str(src.relative_to(PROJECT_ROOT)), 
                              str(dst.relative_to(PROJECT_ROOT))))
    
    return synced


def sync_submodule_specific_files() -> List[Tuple[str, str]]:
    """Copy submodule-specific files."""
    synced = []
    
    for submodule_name, targets in SUBMODULE_SPECIFIC_FILES.items():
        submodule_dir = DEPENDENCIES_DIR / submodule_name
        
        if not submodule_dir.exists():
            continue
        
        for target_subdir, files in targets.items():
            target_dir = submodule_dir / target_subdir
            
            for filename in files:
                src = SHARED_SCRIPTS_DIR / filename
                dst = target_dir / filename
                
                if copy_file_if_changed(src, dst):
                    synced.append((str(src.relative_to(PROJECT_ROOT)), 
                                  str(dst.relative_to(PROJECT_ROOT))))
    
    return synced


def sync_all(verbose: bool = False) -> List[Tuple[str, str]]:
    """
    Sync all shared files to submodules.
    Returns list of (src, dst) pairs that were copied.
    """
    all_synced = []
    
    # Sync universal files
    all_synced.extend(sync_universal_files())
    
    # Sync submodule-specific files
    all_synced.extend(sync_submodule_specific_files())
    
    if verbose and all_synced:
        print(f"[sync] Synced {len(all_synced)} file(s) to submodules:")
        for src, dst in all_synced:
            print(f"  {src} → {dst}")
    
    return all_synced


def check_sync_status() -> dict:
    """Check which files are out of sync (for diagnostics)."""
    status = {"synced": [], "outdated": [], "missing_src": [], "missing_dst": []}
    
    for submodule_dir in get_submodule_dirs():
        input_dir = submodule_dir / "input"  # singular for consistency
        
        for filename in UNIVERSAL_FILES:
            src = SHARED_SCRIPTS_DIR / filename
            dst = input_dir / filename
            
            if not src.exists():
                status["missing_src"].append(str(src.relative_to(PROJECT_ROOT)))
            elif not dst.exists():
                status["missing_dst"].append(str(dst.relative_to(PROJECT_ROOT)))
            elif src.read_bytes() != dst.read_bytes():
                status["outdated"].append(str(dst.relative_to(PROJECT_ROOT)))
            else:
                status["synced"].append(str(dst.relative_to(PROJECT_ROOT)))
    
    return status


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync shared scripts to submodules")
    parser.add_argument("--check", action="store_true", help="Check sync status only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if args.check:
        status = check_sync_status()
        print("Sync status:")
        print(f"  Synced: {len(status['synced'])}")
        print(f"  Outdated: {len(status['outdated'])}")
        if status['outdated']:
            for f in status['outdated']:
                print(f"    - {f}")
        print(f"  Missing src: {len(status['missing_src'])}")
        print(f"  Missing dst: {len(status['missing_dst'])}")
    else:
        synced = sync_all(verbose=True)
        if not synced:
            print("[sync] All files already in sync")

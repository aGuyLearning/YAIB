"""Utilities for cleaning up per-run cache and preproc directories."""

import argparse
import logging
import shutil
from pathlib import Path

CACHE_SUBDIRS = ("cache", "preproc")


def clean_run_cache(run_dir: Path) -> None:
    """Delete cache/ and preproc/ inside a single run directory."""
    for sub in CACHE_SUBDIRS:
        target = run_dir / sub
        if target.exists():
            shutil.rmtree(target)
            logging.info(f"Deleted {target}")


def clean_all_caches(log_root: Path, dry_run: bool = False) -> int:
    """Walk log_root recursively, delete every cache/ and preproc/ directory.

    Returns the number of directories deleted (or that would be deleted in dry-run mode).
    """
    deleted = 0
    for sub in CACHE_SUBDIRS:
        for target in sorted(log_root.rglob(sub)):
            if not target.is_dir():
                continue
            if dry_run:
                print(f"[dry-run] Would delete {target}")
            else:
                shutil.rmtree(target)
                print(f"Deleted {target}")
            deleted += 1
    if deleted == 0:
        print("No cache directories found.")
    else:
        action = "Would delete" if dry_run else "Deleted"
        print(f"\n{action} {deleted} director{'y' if deleted == 1 else 'ies'}.")
    return deleted


def main():
    parser = argparse.ArgumentParser(description="Clean up YAIB per-run cache directories.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Root log directory to scan.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be deleted.")
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        parser.error(f"Directory does not exist: {args.log_dir}")

    clean_all_caches(args.log_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

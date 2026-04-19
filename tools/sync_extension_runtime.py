from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RUNTIME_DIR = ROOT / "shared" / "runtime" / "local_image_runtime"
EXTENSION_ROOTS = {
    "sd15": ROOT / "extensions" / "sd15",
    "sdxl-base": ROOT / "extensions" / "sdxl-base",
    "flux-schnell": ROOT / "extensions" / "flux-schnell",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync the canonical shared runtime from shared/runtime/local_image_runtime "
            "into each child extension vendored runtime directory."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify whether vendored runtime copies are already in sync.",
    )
    parser.add_argument(
        "--extension",
        dest="extensions",
        action="append",
        choices=sorted(EXTENSION_ROOTS),
        help="Restrict sync/check to one or more child extensions.",
    )
    return parser.parse_args()


def selected_extensions(explicit: list[str] | None) -> tuple[str, ...]:
    if explicit:
        return tuple(explicit)
    return tuple(sorted(EXTENSION_ROOTS))


def target_runtime_dir(extension_id: str) -> Path:
    return EXTENSION_ROOTS[extension_id] / "src" / "local_image_runtime"


def compare_runtime_dirs(source_dir: Path, target_dir: Path) -> tuple[bool, list[str]]:
    differences: list[str] = []
    if not target_dir.exists():
        return False, [f"missing target directory: {target_dir}"]

    comparison = filecmp.dircmp(source_dir, target_dir, ignore=["__pycache__"])
    differences.extend(sorted(f"left-only:{name}" for name in comparison.left_only))
    differences.extend(sorted(f"right-only:{name}" for name in comparison.right_only))
    differences.extend(sorted(f"diff:{name}" for name in comparison.diff_files))
    differences.extend(sorted(f"funny:{name}" for name in comparison.funny_files))

    for subdir in sorted(comparison.common_dirs):
        same, nested = compare_runtime_dirs(source_dir / subdir, target_dir / subdir)
        if not same:
            differences.extend(nested)

    return not differences, differences


def sync_runtime_dir(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(
        source_dir,
        target_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def main() -> int:
    args = parse_args()
    extension_ids = selected_extensions(args.extensions)
    failures: list[str] = []

    if not CANONICAL_RUNTIME_DIR.exists():
        raise SystemExit(f"Canonical runtime directory not found: {CANONICAL_RUNTIME_DIR}")

    print(f"Canonical source: {CANONICAL_RUNTIME_DIR}")
    for extension_id in extension_ids:
        target_dir = target_runtime_dir(extension_id)
        if args.check:
            same, differences = compare_runtime_dirs(CANONICAL_RUNTIME_DIR, target_dir)
            if same:
                print(f"[ok] {extension_id}: {target_dir}")
                continue
            failures.append(extension_id)
            print(f"[out-of-sync] {extension_id}: {target_dir}")
            for difference in differences:
                print(f"  - {difference}")
            continue

        sync_runtime_dir(CANONICAL_RUNTIME_DIR, target_dir)
        print(f"[synced] {extension_id}: {target_dir}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

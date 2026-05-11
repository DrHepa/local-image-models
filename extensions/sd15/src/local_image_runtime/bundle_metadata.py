from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


BUNDLE_VERSION = "0.1.1"


@dataclass(frozen=True)
class BundleVersionPolicyResult:
    ready: bool
    current_version: str | None
    next_version_policy: str | None
    diagnostics: tuple[str, ...]


def load_manifest_version(manifest_path: str | Path) -> str:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Manifest '{manifest_path}' is missing a non-empty version field.")
    return version.strip()


def load_manifest_versions(manifest_paths: dict[str, str | Path]) -> dict[str, str]:
    return {
        extension_id: load_manifest_version(manifest_path)
        for extension_id, manifest_path in sorted(manifest_paths.items())
    }


def evaluate_bundle_version_policy(
    *,
    manifest_versions: dict[str, str],
    package_versions: dict[str, str] | None = None,
    next_version_policy: str | None,
) -> BundleVersionPolicyResult:
    """Require synchronized current versions and an explicit next-version policy."""

    package_versions = package_versions or {}
    all_versions = {
        f"manifest:{name}": version.strip()
        for name, version in manifest_versions.items()
        if isinstance(version, str) and version.strip()
    }
    all_versions.update(
        {
            f"package:{name}": version.strip()
            for name, version in package_versions.items()
            if isinstance(version, str) and version.strip()
        }
    )

    diagnostics: list[str] = []
    unique_versions = sorted(set(all_versions.values()))
    current_version = unique_versions[0] if len(unique_versions) == 1 else None
    if len(all_versions) != len(manifest_versions) + len(package_versions) or not all_versions:
        diagnostics.append("version policy cannot be evaluated with missing version metadata.")
    if len(unique_versions) != 1:
        diagnostics.append(f"version drift detected across bundle metadata: {all_versions}.")
    elif current_version != BUNDLE_VERSION:
        diagnostics.append(
            f"bundle version source is {BUNDLE_VERSION}, but manifests/packages report {current_version}."
        )

    normalized_policy = next_version_policy.strip() if isinstance(next_version_policy, str) else ""
    if not normalized_policy:
        diagnostics.append("missing next-version policy blocks GitHub install testing.")

    return BundleVersionPolicyResult(
        ready=not diagnostics,
        current_version=current_version,
        next_version_policy=normalized_policy or None,
        diagnostics=tuple(diagnostics),
    )

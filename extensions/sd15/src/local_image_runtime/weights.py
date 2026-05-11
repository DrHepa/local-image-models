from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
from typing import Any, Protocol

from .descriptors import get_extension_descriptor, get_node_weight_specs, get_optional_feature_specs


MODELS_DIR_ENV_VARS = (
    "LOCAL_IMAGE_MODELS_DIR",
    "MODELS_DIR",
    "MODLY_MODELS_DIR",
)

FLUX_SCHNELL_EXTENSION_ID = "flux-schnell"
FLUX_SCHNELL_TEXT_TO_IMAGE_NODE_ID = "text-to-image"
FLUX_SCHNELL_HF_URL = "https://huggingface.co/black-forest-labs/FLUX.1-schnell"
FLUX_SCHNELL_HF_GATED_GUIDANCE = (
    f"Visit {FLUX_SCHNELL_HF_URL}, log in with the same Hugging Face account/token used by "
    "Modly, accept the model conditions and share contact information if requested, then "
    "retry the download in Modly."
)


class SnapshotDownloader(Protocol):
    def snapshot_download(
        self,
        *,
        repo_id: str,
        local_dir: Path,
        allow_patterns: tuple[str, ...] | None = None,
        revision: str | None = None,
    ) -> Path:
        ...


class HuggingFaceSnapshotDownloader:
    def snapshot_download(
        self,
        *,
        repo_id: str,
        local_dir: Path,
        allow_patterns: tuple[str, ...] | None = None,
        revision: str | None = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {"repo_id": repo_id, "local_dir": str(local_dir)}
        if allow_patterns is not None:
            kwargs["allow_patterns"] = list(allow_patterns)
        if revision:
            kwargs["revision"] = revision
        return Path(snapshot_download(**kwargs))


class FluxWeightDownloadError(RuntimeError):
    """Base error for Flux Schnell weight acquisition failures."""


class FluxWeightAuthError(FluxWeightDownloadError):
    """Raised when Hugging Face authentication or gated access blocks download."""


class FluxWeightNetworkError(FluxWeightDownloadError):
    """Raised when a network failure interrupts weight acquisition."""


class FluxWeightDiskError(FluxWeightDownloadError):
    """Raised when local disk access or capacity blocks weight acquisition."""


class FluxWeightPartialDownloadError(FluxWeightDownloadError):
    """Raised when the downloader returns without the required check file."""


class OptionalFeatureWeightDownloadError(RuntimeError):
    """Base error for optional feature weight acquisition failures."""


class OptionalFeatureWeightPartialDownloadError(OptionalFeatureWeightDownloadError):
    """Raised when an optional feature download misses its required check file."""


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _nearest_existing_ancestor(path: Path) -> Path | None:
    current = path
    while True:
        if current.exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def resolve_models_dir(models_dir: str | Path | None = None) -> dict[str, Any]:
    if models_dir is not None:
        resolved = Path(models_dir).expanduser().resolve()
        return {
            "models_dir": resolved,
            "source": "argument",
            "diagnostics": (),
        }

    for env_name in MODELS_DIR_ENV_VARS:
        raw_value = os.environ.get(env_name)
        if isinstance(raw_value, str) and raw_value.strip():
            return {
                "models_dir": Path(raw_value.strip()).expanduser().resolve(),
                "source": f"env:{env_name}",
                "diagnostics": (),
            }

    env_list = ", ".join(MODELS_DIR_ENV_VARS)
    return {
        "models_dir": None,
        "source": None,
        "diagnostics": (
            "modelsDir is not configured for node-scoped weights. "
            f"Set one of: {env_list}.",
        ),
    }


def extension_models_dir(models_dir: Path, extension_id: str) -> Path:
    return models_dir / extension_id


def node_models_dir(models_dir: Path, extension_id: str, node_id: str) -> Path:
    return extension_models_dir(models_dir, extension_id) / node_id


def download_check_path(
    models_dir: Path, extension_id: str, node_id: str, download_check: str
) -> Path:
    return node_models_dir(models_dir, extension_id, node_id) / download_check


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    direct_status = getattr(exc, "status_code", None)
    return direct_status if isinstance(direct_status, int) else None


def _is_network_exception(exc: Exception) -> bool:
    network_exception_names = {
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "NetworkError",
        "OfflineModeIsEnabled",
    }
    return isinstance(exc, TimeoutError) or any(
        cls.__name__ in network_exception_names for cls in type(exc).mro()
    )


def _exception_search_text(exc: Exception) -> str:
    values = [str(exc), type(exc).__name__]
    response = getattr(exc, "response", None)
    if response is not None:
        values.extend(
            str(value)
            for value in (
                getattr(response, "text", ""),
                getattr(response, "reason", ""),
                getattr(response, "url", ""),
                response,
            )
            if value
        )
    return " ".join(values).casefold()


def _is_likely_flux_hf_gated_access_error(exc: Exception) -> bool:
    if isinstance(exc, PermissionError) or _http_status_code(exc) in {401, 403}:
        return True

    text = _exception_search_text(exc)
    gated_markers = (
        "gated",
        "access denied",
        "unauthorized",
        "terms",
        "token",
        "contact information",
        "repo not found",
        "repository not found",
    )
    return any(marker in text for marker in gated_markers)


def _with_flux_hf_gated_guidance(base_message: str) -> str:
    return f"{base_message} {FLUX_SCHNELL_HF_GATED_GUIDANCE}"


def _map_flux_download_exception(exc: Exception, *, target_dir: Path) -> FluxWeightDownloadError:
    if _is_likely_flux_hf_gated_access_error(exc):
        return FluxWeightAuthError(
            _with_flux_hf_gated_guidance(
                "Flux Schnell weight download failed because Hugging Face authentication or gated "
                "model access was denied."
            )
        )

    if isinstance(exc, OSError) and getattr(exc, "errno", None) in {
        errno.ENOSPC,
        errno.EDQUOT,
        errno.EACCES,
        errno.EROFS,
    }:
        return FluxWeightDiskError(
            f"Flux Schnell weight download failed because disk access or capacity blocked writes "
            f"under '{target_dir}'."
        )

    if _is_network_exception(exc):
        return FluxWeightNetworkError(
            "Flux Schnell weight download failed because the Hugging Face request could not "
            "complete over the network. Check connectivity and retry."
        )

    return FluxWeightDownloadError(f"Flux Schnell weight download failed: {exc}")


def acquire_flux_schnell_weights(
    *,
    models_dir: str | Path,
    downloader: SnapshotDownloader | None = None,
) -> dict[str, Any]:
    node_specs = get_node_weight_specs(FLUX_SCHNELL_EXTENSION_ID)
    node_spec = node_specs[FLUX_SCHNELL_TEXT_TO_IMAGE_NODE_ID]
    repo_id = node_spec["hf_repo"]
    download_check = node_spec["download_check"]
    root = Path(models_dir).expanduser().resolve()
    target_dir = node_models_dir(
        root,
        FLUX_SCHNELL_EXTENSION_ID,
        FLUX_SCHNELL_TEXT_TO_IMAGE_NODE_ID,
    )
    check_path = target_dir / download_check

    if check_path.exists():
        return {
            "status": "ready",
            "extension_id": FLUX_SCHNELL_EXTENSION_ID,
            "node_id": FLUX_SCHNELL_TEXT_TO_IMAGE_NODE_ID,
            "hf_repo": repo_id,
            "model_dir": str(target_dir),
            "check_path": str(check_path),
            "downloaded": False,
        }

    active_downloader = downloader or HuggingFaceSnapshotDownloader()
    try:
        active_downloader.snapshot_download(repo_id=repo_id, local_dir=target_dir)
    except Exception as exc:
        raise _map_flux_download_exception(exc, target_dir=target_dir) from exc

    if not check_path.exists():
        raise FluxWeightPartialDownloadError(
            f"Flux Schnell weight download appears partial: required download_check "
            f"'{check_path}' is missing after snapshot download."
        )

    return {
        "status": "ready",
        "extension_id": FLUX_SCHNELL_EXTENSION_ID,
        "node_id": FLUX_SCHNELL_TEXT_TO_IMAGE_NODE_ID,
        "hf_repo": repo_id,
        "model_dir": str(target_dir),
        "check_path": str(check_path),
        "downloaded": True,
    }


def acquire_optional_feature_weights(
    extension_id: str,
    feature_id: str,
    models_dir: str | Path,
    *,
    downloader: SnapshotDownloader | None = None,
) -> dict[str, Any]:
    optional_specs = get_optional_feature_specs(extension_id)
    feature_spec = optional_specs.get(feature_id)
    if feature_spec is None:
        raise ValueError(f"Unknown optional feature '{feature_id}' for extension '{extension_id}'.")
    if not feature_spec.get("supported", True):
        reason = feature_spec.get("unsupported_reason") or f"Optional feature '{feature_id}' is not supported."
        raise ValueError(str(reason))

    repo_id = feature_spec["hf_repo"]
    revision = str(feature_spec.get("revision", "")).strip() or None
    download_check = feature_spec["download_check"]
    required_files = tuple(feature_spec.get("required_files", (download_check,)))
    allow_patterns = tuple(feature_spec.get("allow_patterns", required_files))
    root = Path(models_dir).expanduser().resolve()
    target_dir = extension_models_dir(root, extension_id) / "optional" / feature_id
    check_path = target_dir / download_check
    missing_files = tuple(relative_path for relative_path in required_files if not (target_dir / relative_path).exists())

    if not missing_files:
        return {
            "status": "ready",
            "extension_id": extension_id,
            "feature_id": feature_id,
            "node_id": feature_spec["node_id"],
            "hf_repo": repo_id,
            "model_dir": str(target_dir),
            "check_path": str(check_path),
            "required_files": required_files,
            "missing_files": (),
            "downloaded": False,
            "revision": revision or "unknown",
        }

    active_downloader = downloader or HuggingFaceSnapshotDownloader()
    try:
        download_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir": target_dir,
            "allow_patterns": allow_patterns,
        }
        if revision is not None:
            download_kwargs["revision"] = revision
        active_downloader.snapshot_download(**download_kwargs)
    except Exception as exc:
        raise OptionalFeatureWeightDownloadError(
            f"Optional feature '{feature_id}' weight download failed for extension '{extension_id}': {exc}"
        ) from exc

    missing_files = tuple(relative_path for relative_path in required_files if not (target_dir / relative_path).exists())
    if missing_files:
        raise OptionalFeatureWeightPartialDownloadError(
            f"Optional feature '{feature_id}' weight download appears partial: required files "
            f"{', '.join(str(target_dir / relative_path) for relative_path in missing_files)} "
            "are missing after snapshot download."
        )

    return {
        "status": "ready",
        "extension_id": extension_id,
        "feature_id": feature_id,
        "node_id": feature_spec["node_id"],
        "hf_repo": repo_id,
        "model_dir": str(target_dir),
        "check_path": str(check_path),
        "required_files": required_files,
        "missing_files": (),
        "downloaded": True,
        "revision": revision or "unknown",
    }


def collect_optional_feature_asset_identity(
    extension_id: str,
    feature_id: str,
    *,
    models_dir: str | Path | None = None,
    extension_model_dir: str | Path | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    optional_specs = get_optional_feature_specs(extension_id)
    feature_spec = optional_specs.get(feature_id)
    if feature_spec is None:
        raise ValueError(f"Unknown optional feature '{feature_id}' for extension '{extension_id}'.")
    if not feature_spec.get("supported", True):
        return _blocked_optional_feature_asset_identity(
            extension_id=extension_id,
            feature_id=feature_id,
            feature_spec=feature_spec,
            models_dir=models_dir,
            extension_model_dir=extension_model_dir,
        )

    if extension_model_dir is not None:
        feature_root = Path(extension_model_dir).expanduser().resolve() / "optional" / feature_id
    elif models_dir is not None:
        root = Path(models_dir).expanduser().resolve()
        feature_root = extension_models_dir(root, extension_id) / "optional" / feature_id
    else:
        raise ValueError("models_dir or extension_model_dir is required to collect optional feature asset identity.")
    required_files = tuple(feature_spec.get("required_files", (feature_spec["download_check"],)))
    repo = str(feature_spec["hf_repo"])
    spec_revision = str(feature_spec.get("revision", "")).strip()
    resolved_revision = revision.strip() if isinstance(revision, str) and revision.strip() else spec_revision or "unknown"

    assets: list[dict[str, Any]] = []
    missing_files: list[str] = []
    missing_paths: list[str] = []
    for relative_path in required_files:
        asset_path = feature_root / relative_path
        if not asset_path.exists():
            missing_files.append(relative_path)
            missing_paths.append(str(asset_path))
            continue
        content = asset_path.read_bytes()
        assets.append(
            {
                "relative_path": relative_path,
                "path": str(asset_path),
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "repo": repo,
                "revision": resolved_revision,
            }
        )

    diagnostics: list[str] = []
    if missing_paths:
        diagnostics.append(
            f"Missing local optional assets for '{extension_id}/{feature_id}': {', '.join(missing_paths)}. "
            "Run Install/Repair to acquire supported assets; no download was attempted during this local identity check."
        )

    return {
        "status": "ready" if not missing_files else "missing",
        "extension_id": extension_id,
        "feature_id": feature_id,
        "repo": repo,
        "revision": resolved_revision,
        "model_dir": str(feature_root),
        "required_files": required_files,
        "allow_patterns": tuple(feature_spec.get("allow_patterns", required_files)),
        "missing_files": tuple(missing_files),
        "missing_paths": tuple(missing_paths),
        "assets": tuple(assets),
        "diagnostics": tuple(diagnostics),
        "local_files_only": True,
    }


def _blocked_optional_feature_asset_identity(
    *,
    extension_id: str,
    feature_id: str,
    feature_spec: dict[str, Any],
    models_dir: str | Path | None,
    extension_model_dir: str | Path | None,
) -> dict[str, Any]:
    if extension_model_dir is not None:
        feature_root = Path(extension_model_dir).expanduser().resolve() / "optional" / feature_id
    elif models_dir is not None:
        feature_root = extension_models_dir(Path(models_dir).expanduser().resolve(), extension_id) / "optional" / feature_id
    else:
        feature_root = Path(f"<modelsDir>/{extension_id}/optional/{feature_id}")

    reason = feature_spec.get("unsupported_reason") or f"Optional feature '{feature_id}' is not supported."
    repo = str(feature_spec.get("hf_repo", "")).strip() or "unknown"
    required_files = tuple(feature_spec.get("required_files", ()))
    allow_patterns = tuple(feature_spec.get("allow_patterns", ()))
    diagnostics = [
        str(reason),
        (
            f"Optional asset identity for '{extension_id}/{feature_id}' is blocked before model load: "
            "exact repo/source revision, required_files, allow_patterns, adapter file, image encoder files, "
            "sizes, and SHA256 hashes are not fully discovered."
        ),
    ]
    return {
        "status": "blocked",
        "extension_id": extension_id,
        "feature_id": feature_id,
        "repo": repo,
        "revision": "unknown",
        "model_dir": str(feature_root),
        "required_files": required_files,
        "allow_patterns": allow_patterns,
        "missing_files": required_files,
        "missing_paths": (),
        "assets": (),
        "diagnostics": tuple(diagnostics),
        "local_files_only": True,
    }


def evaluate_extension_weights(
    extension_id: str,
    *,
    models_dir: str | Path | None = None,
    legacy_models_dir: str | Path | None = None,
    source_label: str | None = None,
) -> dict[str, Any]:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        raise ValueError(f"Unknown extension id '{extension_id}'.")

    resolved_models = resolve_models_dir(models_dir)
    resolved_models_dir = resolved_models["models_dir"]
    node_specs = get_node_weight_specs(extension_id)
    legacy_extension_dir = None
    if legacy_models_dir is not None:
        legacy_extension_dir = Path(legacy_models_dir).expanduser().resolve() / extension_id

    diagnostics = list(resolved_models["diagnostics"])
    nodes: dict[str, dict[str, Any]] = {}
    optional_features: dict[str, dict[str, Any]] = {}
    ready_node_count = 0

    if resolved_models_dir is not None:
        if resolved_models_dir.exists() and not resolved_models_dir.is_dir():
            diagnostics.append(
                f"Configured modelsDir path '{resolved_models_dir}' exists but is not a directory."
            )
        else:
            writable_probe = (
                resolved_models_dir
                if resolved_models_dir.exists()
                else _nearest_existing_ancestor(resolved_models_dir)
            )
            if writable_probe is None or not os.access(writable_probe, os.W_OK):
                diagnostics.append(
                    f"Configured modelsDir path '{resolved_models_dir}' is not writable from existing ancestor "
                    f"'{writable_probe or resolved_models_dir}'."
                )

    for node_id in descriptor.supported_nodes:
        node_spec = node_specs.get(node_id, {})
        hf_repo = str(node_spec.get("hf_repo", "")).strip()
        download_check = str(node_spec.get("download_check", "")).strip()
        node_root = (
            node_models_dir(resolved_models_dir, extension_id, node_id)
            if resolved_models_dir is not None
            else None
        )
        check_path = (
            download_check_path(resolved_models_dir, extension_id, node_id, download_check)
            if resolved_models_dir is not None and download_check
            else None
        )

        node_diagnostics: list[str] = []
        status = "missing"
        if not hf_repo:
            status = "invalid"
            node_diagnostics.append(
                f"Missing hf_repo metadata for '{extension_id}/{node_id}'."
            )
        if not download_check:
            status = "invalid"
            node_diagnostics.append(
                f"Missing download_check metadata for '{extension_id}/{node_id}'."
            )

        if resolved_models_dir is None:
            expected = f"<modelsDir>/{extension_id}/{node_id}/{download_check or '<download_check>'}"
            node_diagnostics.append(
                f"Cannot evaluate weights for '{extension_id}/{node_id}' without modelsDir. "
                f"Expected '{expected}'."
            )
            status = "unconfigured" if status != "invalid" else status
        elif check_path is not None and check_path.exists():
            status = "ready"
            ready_node_count += 1
        elif check_path is not None:
            node_diagnostics.append(
                f"Missing download_check '{download_check}' for '{extension_id}/{node_id}' at '{check_path}'."
            )

        diagnostics.extend(node_diagnostics)
        nodes[node_id] = {
            "node_id": node_id,
            "status": status,
            "ready": status == "ready",
            "hf_repo": hf_repo,
            "download_check": download_check,
            "model_dir": str(node_root) if node_root is not None else None,
            "check_path": str(check_path) if check_path is not None else None,
            "diagnostics": node_diagnostics,
        }

    for feature_id, spec in get_optional_feature_specs(extension_id).items():
        download_check = spec["download_check"]
        required_files = tuple(spec.get("required_files", (download_check,)))
        feature_root = extension_models_dir(resolved_models_dir, extension_id) / "optional" / feature_id if resolved_models_dir is not None else None
        check_path = feature_root / download_check if feature_root is not None and download_check else None
        missing_files = (
            tuple(relative_path for relative_path in required_files if not (feature_root / relative_path).exists())
            if feature_root is not None
            else required_files
        )
        feature_diagnostics: list[str] = []
        status = "missing"
        if not spec.get("supported", True):
            status = "unsupported"
            missing_files = ()
            reason = spec.get("unsupported_reason") or f"Optional {spec['label']} is not supported."
            feature_diagnostics.append(str(reason))
        elif resolved_models_dir is None:
            status = "unconfigured"
            feature_diagnostics.append(
                f"Optional {spec['label']} readiness cannot be evaluated without modelsDir."
            )
        elif not missing_files:
            status = "ready"
        else:
            feature_diagnostics.append(
                f"Optional {spec['label']} is missing for '{extension_id}/{spec['node_id']}'. "
                f"Expected IP-Adapter required files {', '.join(missing_files)} under '{feature_root}'. "
                "Run Install/Repair to acquire supported optional assets; readiness checks do not download."
            )
        diagnostics.extend(feature_diagnostics)
        optional_features[feature_id] = {
            "feature_id": feature_id,
            "label": spec["label"],
            "family": spec["family"],
            "node_id": spec["node_id"],
            "status": status,
            "ready": status == "ready",
            "hf_repo": spec["hf_repo"],
            "download_check": download_check,
            "required_files": required_files,
            "missing_files": missing_files,
            "model_dir": str(feature_root) if feature_root is not None else None,
            "check_path": str(check_path) if check_path is not None else None,
            "diagnostics": feature_diagnostics,
        }

    overall_status = "ready"
    if resolved_models_dir is None:
        overall_status = "unconfigured"
    elif ready_node_count != len(nodes):
        overall_status = "missing"

    return {
        "status": overall_status,
        "models_dir": str(resolved_models_dir) if resolved_models_dir is not None else None,
        "source": source_label or resolved_models["source"],
        "extension_dir": (
            str(extension_models_dir(resolved_models_dir, extension_id))
            if resolved_models_dir is not None
            else None
        ),
        "ready_node_count": ready_node_count,
        "total_node_count": len(nodes),
        "nodes": nodes,
        "optional_features": optional_features,
        "diagnostics": _unique_strings(diagnostics),
        "legacy": {
            "model_dir": str(legacy_extension_dir) if legacy_extension_dir is not None else None,
            "exists": legacy_extension_dir.exists() if legacy_extension_dir is not None else False,
        },
    }


__all__ = [
    "FLUX_SCHNELL_EXTENSION_ID",
    "FLUX_SCHNELL_TEXT_TO_IMAGE_NODE_ID",
    "FluxWeightAuthError",
    "FluxWeightDiskError",
    "FluxWeightDownloadError",
    "FluxWeightNetworkError",
    "FluxWeightPartialDownloadError",
    "HuggingFaceSnapshotDownloader",
    "MODELS_DIR_ENV_VARS",
    "OptionalFeatureWeightDownloadError",
    "OptionalFeatureWeightPartialDownloadError",
    "SnapshotDownloader",
    "acquire_optional_feature_weights",
    "acquire_flux_schnell_weights",
    "collect_optional_feature_asset_identity",
    "download_check_path",
    "evaluate_extension_weights",
    "extension_models_dir",
    "node_models_dir",
    "resolve_models_dir",
]

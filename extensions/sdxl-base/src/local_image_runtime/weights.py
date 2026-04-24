from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any, Protocol

from .descriptors import get_extension_descriptor, get_node_weight_specs


MODELS_DIR_ENV_VARS = (
    "LOCAL_IMAGE_MODELS_DIR",
    "MODELS_DIR",
    "MODLY_MODELS_DIR",
)

FLUX_SCHNELL_EXTENSION_ID = "flux-schnell"
FLUX_SCHNELL_TEXT_TO_IMAGE_NODE_ID = "text-to-image"


class SnapshotDownloader(Protocol):
    def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
        ...


class HuggingFaceSnapshotDownloader:
    def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=repo_id, local_dir=str(local_dir)))


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


def _map_flux_download_exception(exc: Exception, *, target_dir: Path) -> FluxWeightDownloadError:
    status_code = _http_status_code(exc)
    if isinstance(exc, PermissionError) or status_code in {401, 403}:
        return FluxWeightAuthError(
            "Flux Schnell weight download failed because Hugging Face authentication or gated "
            "model access was denied. Configure an authorized Hugging Face token and retry."
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


def evaluate_extension_weights(
    extension_id: str,
    *,
    models_dir: str | Path | None = None,
    legacy_models_dir: str | Path | None = None,
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

    overall_status = "ready"
    if resolved_models_dir is None:
        overall_status = "unconfigured"
    elif ready_node_count != len(nodes):
        overall_status = "missing"

    return {
        "status": overall_status,
        "models_dir": str(resolved_models_dir) if resolved_models_dir is not None else None,
        "source": resolved_models["source"],
        "extension_dir": (
            str(extension_models_dir(resolved_models_dir, extension_id))
            if resolved_models_dir is not None
            else None
        ),
        "ready_node_count": ready_node_count,
        "total_node_count": len(nodes),
        "nodes": nodes,
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
    "SnapshotDownloader",
    "acquire_flux_schnell_weights",
    "download_check_path",
    "evaluate_extension_weights",
    "extension_models_dir",
    "node_models_dir",
    "resolve_models_dir",
]

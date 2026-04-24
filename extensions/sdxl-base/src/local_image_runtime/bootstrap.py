from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .descriptors import (
    extension_metadata_map,
    get_extension_descriptor,
    missing_required_paths,
    registered_extension_ids,
    resolve_extension_id,
)
from .weights import evaluate_extension_weights, resolve_models_dir


RUNTIME_DIRNAME = ".local-image-runtime"
EXTENSION_ROOT_OVERRIDE_ENV = "LOCAL_IMAGE_RUNTIME_EXTENSION_ROOT"
_UNSET = object()
EXTENSION_STATUS_NOT_INSTALLED = "not_installed"
EXTENSION_STATUS_INSTALLING = "installing"
EXTENSION_STATUS_INSTALLED = "installed"
EXTENSION_STATUS_ERROR = "error"
EXTENSION_STATUSES = {
    EXTENSION_STATUS_NOT_INSTALLED,
    EXTENSION_STATUS_INSTALLING,
    EXTENSION_STATUS_INSTALLED,
    EXTENSION_STATUS_ERROR,
}
SETUP_STATUS_READY = "ready"
SETUP_STATUS_INSTALLING = "installing"
SETUP_STATUS_FAILED = "failed"
SETUP_STATUSES = {
    SETUP_STATUS_READY,
    SETUP_STATUS_INSTALLING,
    SETUP_STATUS_FAILED,
}
SetupStatus = str


class RuntimeStateError(RuntimeError):
    """Base error raised for invalid persisted runtime state."""


class CorruptStateFileError(RuntimeStateError):
    """Raised when a runtime state file cannot be decoded as JSON."""


class InvalidStateFileError(RuntimeStateError):
    """Raised when a runtime state file has an unexpected shape."""


class RuntimeOperationError(RuntimeError):
    """Base error for runtime CLI operations."""


class UnknownExtensionError(RuntimeOperationError):
    """Raised when an operation receives an unknown extension id."""


class InvalidExtensionSourceError(RuntimeOperationError):
    """Raised when a local extension source directory is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value.strip() else fallback


@dataclass(frozen=True)
class RuntimePaths:
    root_dir: Path
    runtime_dir: Path
    models_dir: Path
    cache_dir: Path
    outputs_dir: Path
    logs_dir: Path
    state_dir: Path
    bootstrap_state_file: Path
    models_state_file: Path


@dataclass(frozen=True)
class RuntimeSnapshot:
    paths: RuntimePaths
    bootstrap_state: dict[str, Any]
    extensions: dict[str, dict[str, Any]]
    legacy_models: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class InstalledExtensionResult:
    extension_id: str
    source_dir: Path
    destination_dir: Path
    required_paths: tuple[str, ...]
    snapshot: RuntimeSnapshot


def extension_root() -> Path:
    override = os.environ.get(EXTENSION_ROOT_OVERRIDE_ENV)
    if override:
        candidate = Path(override).expanduser().resolve()
        if (candidate / "manifest.json").exists():
            return candidate

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "manifest.json").exists():
            return parent
    return current.parents[2]


def runtime_paths() -> RuntimePaths:
    root_dir = extension_root()
    runtime_dir = root_dir / RUNTIME_DIRNAME
    state_dir = runtime_dir / "state"
    return RuntimePaths(
        root_dir=root_dir,
        runtime_dir=runtime_dir,
        models_dir=runtime_dir / "models",
        cache_dir=runtime_dir / "cache",
        outputs_dir=runtime_dir / "outputs",
        logs_dir=runtime_dir / "logs",
        state_dir=state_dir,
        bootstrap_state_file=state_dir / "bootstrap-state.json",
        models_state_file=state_dir / "models-state.json",
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise CorruptStateFileError(
            f"State file '{path}' is corrupt and could not be parsed as JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise InvalidStateFileError(
            f"State file '{path}' must contain a JSON object at the top level."
        )

    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_bootstrap_state(paths: RuntimePaths) -> dict[str, Any]:
    timestamp = _utc_now()
    return {
        "status": "ready",
        "runtime_dir": str(paths.runtime_dir),
        "created_at": timestamp,
        "updated_at": timestamp,
        "version": 2,
    }


def _normalize_bootstrap_state(
    raw_state: dict[str, Any] | None, paths: RuntimePaths
) -> dict[str, Any]:
    default_state = _default_bootstrap_state(paths)
    if raw_state is None:
        return default_state
    if not isinstance(raw_state.get("status", "ready"), str):
        raise InvalidStateFileError("Bootstrap state field 'status' must be a string.")

    normalized = dict(raw_state)
    normalized["status"] = "ready"
    normalized["runtime_dir"] = str(paths.runtime_dir)
    normalized["version"] = max(
        2,
        _coerce_version(normalized.get("version", default_state["version"]), field_name="version"),
    )
    normalized["created_at"] = _normalize_timestamp(
        normalized.get("created_at"), default_state["created_at"]
    )
    normalized["updated_at"] = _normalize_timestamp(
        normalized.get("updated_at"), normalized["created_at"]
    )
    return normalized


def _coerce_version(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidStateFileError(f"State field '{field_name}' must be an integer.") from exc


def _normalize_extension_status(value: Any) -> str:
    if isinstance(value, str) and value in EXTENSION_STATUSES:
        return value
    return EXTENSION_STATUS_NOT_INSTALLED


def detect_platform() -> dict[str, str]:
    return {
        "system": platform.system().lower(),
        "machine": platform.machine().lower(),
    }


def expected_venv_python(ext_dir: str | Path) -> Path:
    ext_root = Path(ext_dir).expanduser()
    if platform.system().lower() == "windows":
        return ext_root / "venv" / "Scripts" / "python.exe"
    return ext_root / "venv" / "bin" / "python"


def _normalize_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _normalize_platform_info(value: Any) -> dict[str, str]:
    platform_info = detect_platform()
    if not isinstance(value, dict):
        return platform_info
    system = _normalize_string(value.get("system"))
    machine = _normalize_string(value.get("machine"))
    if system is not None:
        platform_info["system"] = system
    if machine is not None:
        platform_info["machine"] = machine
    return platform_info


def _normalize_setup_step(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = _normalize_string(value.get("name")) or "unknown"
    status = _normalize_string(value.get("status")) or "unknown"
    detail = _normalize_string(value.get("detail"))
    return {"name": name, "status": status, "detail": detail}


def _normalize_setup_steps(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    normalized: list[dict[str, Any]] = []
    for item in value:
        step = _normalize_setup_step(item)
        if step is not None:
            normalized.append(step)
    return tuple(normalized)


def _smoke_test_runtime_imports(venv_python: Path, imports: tuple[str, ...]) -> tuple[bool, str | None]:
    filtered_imports = tuple(item for item in imports if isinstance(item, str) and item.strip())
    if not filtered_imports:
        return True, None

    command = [
        str(venv_python),
        "-c",
        (
            "import importlib\n"
            f"modules = {list(filtered_imports)!r}\n"
            "for name in modules:\n"
            "    importlib.import_module(name)\n"
            "print(', '.join(modules))\n"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return False, f"Could not execute import smoke test with '{venv_python}': {exc}"
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        return False, detail

    detail = completed.stdout.strip()
    return True, detail or ", ".join(filtered_imports)


def _evaluate_setup_readiness(
    *, extension_id: str, ext_dir: str | None, python_exe: str | None
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    ext_root = Path(ext_dir).expanduser() if ext_dir is not None else None
    python_path = Path(python_exe).expanduser() if python_exe is not None else None
    venv_python = str(expected_venv_python(ext_root)) if ext_root is not None else None
    descriptor = get_extension_descriptor(extension_id)
    readiness_imports = descriptor.readiness_imports if descriptor is not None else ()

    if ext_root is None:
        steps.append(
            {
                "name": "validate_ext_dir",
                "status": "failed",
                "detail": "Missing ext_dir in persisted setup state.",
            }
        )
        diagnostics.append("Missing ext_dir for setup readiness evaluation.")
    elif not ext_root.exists():
        steps.append(
            {
                "name": "validate_ext_dir",
                "status": "failed",
                "detail": f"Install root '{ext_root}' does not exist.",
            }
        )
        diagnostics.append(f"Resolved ext_dir does not exist: {ext_root}")
    else:
        steps.append(
            {
                "name": "validate_ext_dir",
                "status": "ok",
                "detail": f"Install root resolved to '{ext_root}'.",
            }
        )

    if python_path is None:
        steps.append(
            {
                "name": "validate_python_exe",
                "status": "failed",
                "detail": "Missing python_exe in persisted setup state.",
            }
        )
        diagnostics.append("Missing python_exe for setup readiness evaluation.")
    elif not python_path.exists():
        steps.append(
            {
                "name": "validate_python_exe",
                "status": "failed",
                "detail": f"Configured python executable '{python_path}' does not exist.",
            }
        )
        diagnostics.append(f"Configured python_exe does not exist: {python_path}")
    else:
        steps.append(
            {
                "name": "validate_python_exe",
                "status": "ok",
                "detail": f"Installer python resolved to '{python_path}'.",
            }
        )

    if ext_root is None:
        steps.append(
            {
                "name": "verify_venv_python",
                "status": "failed",
                "detail": "Cannot resolve venv path without ext_dir.",
            }
        )
    else:
        venv_path = expected_venv_python(ext_root)
        if not venv_path.exists():
            steps.append(
                {
                    "name": "verify_venv_python",
                    "status": "failed",
                    "detail": f"Expected virtualenv interpreter at '{venv_path}'.",
                }
            )
            diagnostics.append(f"Missing virtualenv interpreter: {venv_path}")
        else:
            steps.append(
                {
                    "name": "verify_venv_python",
                    "status": "ok",
                    "detail": f"Virtualenv interpreter is present at '{venv_path}'.",
                }
            )
            imports_ok, import_detail = _smoke_test_runtime_imports(
                venv_path, readiness_imports
            )
            if imports_ok:
                steps.append(
                    {
                        "name": "verify_runtime_imports",
                        "status": "ok",
                        "detail": "Runtime imports succeeded: "
                        + (import_detail or ", ".join(readiness_imports)),
                    }
                )
            else:
                steps.append(
                    {
                        "name": "verify_runtime_imports",
                        "status": "failed",
                        "detail": import_detail
                        or "Runtime import smoke test failed inside the extension venv.",
                    }
                )
                diagnostics.append(
                    "Runtime import smoke test failed"
                    + (f": {import_detail}" if import_detail else ".")
                )

    status = SETUP_STATUS_READY if not diagnostics else SETUP_STATUS_FAILED
    return {
        "status": status,
        "venv_python": venv_python,
        "steps": tuple(steps),
        "diagnostics": tuple(diagnostics),
    }


def _normalize_setup_state(extension_id: str, current: Any, timestamp: str) -> dict[str, Any]:
    payload = current if isinstance(current, dict) else {}
    ext_dir = _normalize_string(payload.get("ext_dir"))
    python_exe = _normalize_string(payload.get("python_exe"))
    platform_info = _normalize_platform_info(payload.get("platform"))
    evaluated = _evaluate_setup_readiness(
        extension_id=extension_id,
        ext_dir=ext_dir,
        python_exe=python_exe,
    )

    status = payload.get("status")
    if status in {SETUP_STATUS_INSTALLING, SETUP_STATUS_FAILED}:
        steps = _normalize_setup_steps(payload.get("steps")) or evaluated["steps"]
        diagnostics = _normalize_string_list(payload.get("diagnostics")) or evaluated["diagnostics"]
        venv_python = _normalize_string(payload.get("venv_python")) or evaluated["venv_python"]
    else:
        status = evaluated["status"]
        steps = evaluated["steps"]
        diagnostics = evaluated["diagnostics"]
        venv_python = evaluated["venv_python"]

    return {
        "status": status if status in SETUP_STATUSES else SETUP_STATUS_FAILED,
        "ext_dir": ext_dir,
        "python_exe": python_exe,
        "venv_python": venv_python,
        "platform": platform_info,
        "steps": list(steps),
        "diagnostics": list(diagnostics),
        "updated_at": _normalize_timestamp(payload.get("updated_at"), timestamp),
    }


def _reconcile_extension_lifecycle_status(*, current_status: Any, setup_status: Any) -> str:
    normalized_current_status = _normalize_extension_status(current_status)
    if setup_status == SETUP_STATUS_READY:
        return EXTENSION_STATUS_INSTALLED
    if setup_status == SETUP_STATUS_INSTALLING:
        return EXTENSION_STATUS_INSTALLING
    if setup_status == SETUP_STATUS_FAILED:
        return EXTENSION_STATUS_ERROR
    return normalized_current_status


def _normalize_extension_state(
    extension_id: str,
    existing: dict[str, Any] | None,
    paths: RuntimePaths,
    timestamp: str,
) -> dict[str, Any]:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        raise InvalidStateFileError(
            f"Cannot normalize unknown registered extension '{extension_id}'."
        )

    current = existing if isinstance(existing, dict) else {}
    setup_state = _normalize_setup_state(extension_id, current.get("setup"), timestamp)
    reconciled_status = _reconcile_extension_lifecycle_status(
        current_status=current.get("status"),
        setup_status=setup_state["status"],
    )
    installed_at = current.get("installed_at")
    if reconciled_status == EXTENSION_STATUS_INSTALLED:
        installed_at = _normalize_timestamp(installed_at, setup_state["updated_at"])
    else:
        installed_at = None

    error = current.get("error")
    if reconciled_status == EXTENSION_STATUS_ERROR:
        diagnostics = setup_state["diagnostics"]
        error = diagnostics[0] if diagnostics else _normalize_string(error)
    else:
        error = None

    weight_state = evaluate_extension_weights(
        extension_id,
        models_dir=paths.models_dir,
        legacy_models_dir=paths.models_dir,
    )
    return {
        "id": extension_id,
        "status": reconciled_status,
        "readiness": setup_state["status"],
        "weights_readiness": weight_state["status"],
        "label": descriptor.label,
        "tier": descriptor.tier,
        "family": descriptor.family,
        "dependency_family": descriptor.dependency_family,
        "backend": descriptor.backend,
        "hf_repo": descriptor.hf_repo,
        "readiness_imports": list(descriptor.readiness_imports),
        "model_dir": str(paths.models_dir / extension_id),
        "weights_dir": weight_state["extension_dir"],
        "weights": weight_state,
        "venv_python": setup_state["venv_python"],
        "setup": setup_state,
        "installed_at": installed_at,
        "error": error,
        "updated_at": _normalize_timestamp(current.get("updated_at"), timestamp),
    }


def _default_models_state(paths: RuntimePaths) -> dict[str, Any]:
    timestamp = _utc_now()
    return {
        "version": 2,
        "updated_at": timestamp,
        "extensions": {
            extension_id: _normalize_extension_state(extension_id, None, paths, timestamp)
            for extension_id in extension_metadata_map()
        },
        "legacy_models": {},
    }


def _collect_legacy_models(
    raw_legacy_models: Any,
) -> dict[str, dict[str, Any]]:
    if raw_legacy_models is None:
        raw_legacy_models = {}
    if not isinstance(raw_legacy_models, dict):
        raise InvalidStateFileError("Models state field 'legacy_models' must be an object.")

    legacy_models: dict[str, dict[str, Any]] = {}
    for model_id, payload in raw_legacy_models.items():
        if isinstance(payload, dict):
            legacy_models[model_id] = dict(payload)
    return legacy_models


def _legacy_model_residue(
    model_id: str,
    payload: dict[str, Any],
    *,
    migrated_to_extension_id: str | None = None,
) -> dict[str, Any]:
    residue = dict(payload)
    residue.setdefault("legacy_model_id", model_id)
    if migrated_to_extension_id is not None:
        residue["migrated_to_extension_id"] = migrated_to_extension_id
    return residue


def _normalize_models_state(
    raw_state: dict[str, Any] | None, paths: RuntimePaths
) -> dict[str, Any]:
    timestamp = _utc_now()
    default_state = _default_models_state(paths)
    if raw_state is None:
        return default_state

    version = _coerce_version(raw_state.get("version", 1), field_name="version")
    if version >= 2:
        raw_extensions = raw_state.get("extensions", {})
        if raw_extensions is None:
            raw_extensions = {}
        if not isinstance(raw_extensions, dict):
            raise InvalidStateFileError("Models state field 'extensions' must be an object.")

        normalized_extensions = {
            extension_id: _normalize_extension_state(
                extension_id, raw_extensions.get(extension_id), paths, timestamp
            )
            for extension_id in extension_metadata_map()
        }
        legacy_models = _collect_legacy_models(raw_state.get("legacy_models"))
    else:
        raw_models = raw_state.get("models", {})
        if raw_models is None:
            raw_models = {}
        if not isinstance(raw_models, dict):
            raise InvalidStateFileError("Models state field 'models' must be an object.")

        normalized_extensions = {}
        legacy_models: dict[str, dict[str, Any]] = {}
        for model_id, payload in raw_models.items():
            if not isinstance(payload, dict):
                continue
            extension_id = resolve_extension_id(model_id)
            legacy_models[model_id] = _legacy_model_residue(
                model_id,
                payload,
                migrated_to_extension_id=extension_id,
            )
            if extension_id is None:
                continue
            normalized_extensions[extension_id] = _normalize_extension_state(
                extension_id,
                payload,
                paths,
                timestamp,
            )

        for extension_id in registered_extension_ids():
            normalized_extensions.setdefault(
                extension_id,
                _normalize_extension_state(extension_id, None, paths, timestamp),
            )

        legacy_models.update(
            _collect_legacy_models(raw_state.get("legacy_models"))
        )

    normalized = dict(raw_state)
    normalized["version"] = 2
    normalized["updated_at"] = _normalize_timestamp(raw_state.get("updated_at"), timestamp)
    normalized["extensions"] = normalized_extensions
    normalized["legacy_models"] = legacy_models
    normalized.pop("models", None)
    return normalized


def _ensure_runtime_layout(paths: RuntimePaths) -> None:
    for directory in (
        paths.runtime_dir,
        paths.models_dir,
        paths.cache_dir,
        paths.outputs_dir,
        paths.logs_dir,
        paths.state_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def write_bootstrap_state(paths: RuntimePaths, state: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_bootstrap_state(state, paths)
    _write_json(paths.bootstrap_state_file, normalized)
    return normalized


def write_models_state(paths: RuntimePaths, state: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_models_state(state, paths)
    _write_json(paths.models_state_file, normalized)
    return normalized


def update_extension_lifecycle(
    snapshot: RuntimeSnapshot,
    extension_id: str,
    *,
    status: str,
    installed_at: str | None | object = _UNSET,
    error: str | None | object = _UNSET,
) -> RuntimeSnapshot:
    if get_extension_descriptor(extension_id) is None:
        raise InvalidStateFileError(
            f"Cannot update unregistered extension '{extension_id}'."
        )
    if status not in EXTENSION_STATUSES:
        raise InvalidStateFileError(
            f"Invalid extension status '{status}'. Allowed values: {sorted(EXTENSION_STATUSES)}."
        )

    current = dict(snapshot.extensions[extension_id])
    current["status"] = status
    current["model_dir"] = str(snapshot.paths.models_dir / extension_id)
    current["updated_at"] = _utc_now()
    if installed_at is not _UNSET:
        current["installed_at"] = installed_at
    elif status != EXTENSION_STATUS_INSTALLED:
        current["installed_at"] = None
    if error is not _UNSET:
        current["error"] = error
    elif status != EXTENSION_STATUS_ERROR:
        current["error"] = None

    models_state = {
        "version": 2,
        "updated_at": _utc_now(),
        "extensions": {**snapshot.extensions, extension_id: current},
        "legacy_models": snapshot.legacy_models,
    }
    normalized_models_state = write_models_state(snapshot.paths, models_state)
    bootstrap_state = write_bootstrap_state(snapshot.paths, snapshot.bootstrap_state)
    return RuntimeSnapshot(
        paths=snapshot.paths,
        bootstrap_state=bootstrap_state,
        extensions=normalized_models_state["extensions"],
        legacy_models=normalized_models_state["legacy_models"],
    )


def persist_extension_setup(
    snapshot: RuntimeSnapshot,
    extension_id: str,
    *,
    status: SetupStatus,
    ext_dir: str | None,
    python_exe: str | None,
    venv_python: str | None,
    steps: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    diagnostics: tuple[str, ...] | list[str],
    platform_info: dict[str, str] | None = None,
) -> RuntimeSnapshot:
    if get_extension_descriptor(extension_id) is None:
        raise InvalidStateFileError(
            f"Cannot update setup state for unregistered extension '{extension_id}'."
        )
    if status not in SETUP_STATUSES:
        raise InvalidStateFileError(
            f"Invalid setup status '{status}'. Allowed values: {sorted(SETUP_STATUSES)}."
        )

    current = dict(snapshot.extensions[extension_id])
    current_setup = current.get("setup") if isinstance(current.get("setup"), dict) else {}
    normalized_steps = [step for step in (_normalize_setup_step(item) for item in steps) if step is not None]
    normalized_diagnostics = [item.strip() for item in diagnostics if isinstance(item, str) and item.strip()]
    setup_state = {
        **current_setup,
        "status": status,
        "ext_dir": _normalize_string(ext_dir),
        "python_exe": _normalize_string(python_exe),
        "venv_python": _normalize_string(venv_python),
        "platform": _normalize_platform_info(platform_info),
        "steps": normalized_steps,
        "diagnostics": normalized_diagnostics,
        "updated_at": _utc_now(),
    }
    reconciled_status = _reconcile_extension_lifecycle_status(
        current_status=current.get("status"),
        setup_status=setup_state["status"],
    )
    current["setup"] = setup_state
    current["status"] = reconciled_status
    current["readiness"] = setup_state["status"]
    current["venv_python"] = setup_state["venv_python"]
    if reconciled_status == EXTENSION_STATUS_INSTALLED:
        current["installed_at"] = _normalize_timestamp(
            current.get("installed_at"),
            setup_state["updated_at"],
        )
        current["error"] = None
    elif reconciled_status == EXTENSION_STATUS_ERROR:
        current["installed_at"] = None
        current["error"] = normalized_diagnostics[0] if normalized_diagnostics else None
    else:
        current["installed_at"] = None
        current["error"] = None
    current["updated_at"] = _utc_now()

    models_state = {
        "version": 2,
        "updated_at": _utc_now(),
        "extensions": {**snapshot.extensions, extension_id: current},
        "legacy_models": snapshot.legacy_models,
    }
    normalized_models_state = write_models_state(snapshot.paths, models_state)
    bootstrap_state = write_bootstrap_state(snapshot.paths, snapshot.bootstrap_state)
    return RuntimeSnapshot(
        paths=snapshot.paths,
        bootstrap_state=bootstrap_state,
        extensions=normalized_models_state["extensions"],
        legacy_models=normalized_models_state["legacy_models"],
    )


def reevaluate_extension_setup(
    snapshot: RuntimeSnapshot,
    extension_id: str,
    *,
    ext_dir: str | None | object = _UNSET,
    python_exe: str | None | object = _UNSET,
    step_prefix: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    platform_info: dict[str, str] | None = None,
) -> RuntimeSnapshot:
    current = snapshot.extensions.get(extension_id, {})
    current_setup = current.get("setup") if isinstance(current.get("setup"), dict) else {}
    resolved_ext_dir = current_setup.get("ext_dir") if ext_dir is _UNSET else ext_dir
    resolved_python_exe = current_setup.get("python_exe") if python_exe is _UNSET else python_exe
    evaluated = _evaluate_setup_readiness(
        extension_id=extension_id,
        ext_dir=_normalize_string(resolved_ext_dir),
        python_exe=_normalize_string(resolved_python_exe),
    )
    return persist_extension_setup(
        snapshot,
        extension_id,
        status=evaluated["status"],
        ext_dir=_normalize_string(resolved_ext_dir),
        python_exe=_normalize_string(resolved_python_exe),
        venv_python=evaluated["venv_python"],
        steps=tuple(step_prefix) + evaluated["steps"],
        diagnostics=evaluated["diagnostics"],
        platform_info=platform_info,
    )


def mark_extension_installing(snapshot: RuntimeSnapshot, extension_id: str) -> RuntimeSnapshot:
    return update_extension_lifecycle(snapshot, extension_id, status=EXTENSION_STATUS_INSTALLING)


def mark_extension_installed(snapshot: RuntimeSnapshot, extension_id: str) -> RuntimeSnapshot:
    return update_extension_lifecycle(
        snapshot,
        extension_id,
        status=EXTENSION_STATUS_INSTALLED,
        installed_at=_utc_now(),
        error=None,
    )


def mark_extension_not_installed(snapshot: RuntimeSnapshot, extension_id: str) -> RuntimeSnapshot:
    return update_extension_lifecycle(
        snapshot,
        extension_id,
        status=EXTENSION_STATUS_NOT_INSTALLED,
        installed_at=None,
        error=None,
    )


def mark_extension_error(
    snapshot: RuntimeSnapshot, extension_id: str, message: str | None = None
) -> RuntimeSnapshot:
    return update_extension_lifecycle(
        snapshot,
        extension_id,
        status=EXTENSION_STATUS_ERROR,
        error=message,
    )


def bootstrap_runtime(extension_id: str | None = None) -> RuntimeSnapshot:
    if extension_id is not None and get_extension_descriptor(extension_id) is None:
        supported = ", ".join(sorted(registered_extension_ids()))
        raise UnknownExtensionError(
            f"Unknown extension '{extension_id}'. Registered extensions: {supported}."
        )

    paths = runtime_paths()
    _ensure_runtime_layout(paths)

    bootstrap_state = write_bootstrap_state(paths, _load_json(paths.bootstrap_state_file) or {})
    models_state = write_models_state(paths, _load_json(paths.models_state_file) or {})

    return RuntimeSnapshot(
        paths=paths,
        bootstrap_state=bootstrap_state,
        extensions=models_state["extensions"],
        legacy_models=models_state["legacy_models"],
    )


def get_extension_record(snapshot: RuntimeSnapshot, extension_id: str) -> dict[str, Any]:
    return snapshot.extensions.get(extension_id, {})


def extension_is_installed(snapshot: RuntimeSnapshot, extension_id: str) -> bool:
    return (
        get_extension_record(snapshot, extension_id).get("status")
        == EXTENSION_STATUS_INSTALLED
    )


def runtime_status(
    snapshot: RuntimeSnapshot, extension_id: str | None = None
) -> dict[str, Any]:
    selected_extension = None
    if extension_id is not None:
        selected_extension = snapshot.extensions.get(extension_id)
    models_dir_context = resolve_models_dir()
    return {
        "status": snapshot.bootstrap_state.get("status", "unknown"),
        "runtime_dir": str(snapshot.paths.runtime_dir),
        "models_dir": (
            str(models_dir_context["models_dir"])
            if models_dir_context["models_dir"] is not None
            else None
        ),
        "models_dir_source": models_dir_context["source"],
        "models_dir_diagnostics": list(models_dir_context["diagnostics"]),
        "runtime_models_dir": str(snapshot.paths.models_dir),
        "state_dir": str(snapshot.paths.state_dir),
        "bootstrap_version": snapshot.bootstrap_state.get("version"),
        "state_version": 2,
        "registered_extensions": sorted(snapshot.extensions),
        "extensions": [
            snapshot.extensions[current_extension_id]
            for current_extension_id in sorted(snapshot.extensions)
        ],
        "current_extension": selected_extension,
        "legacy_models": snapshot.legacy_models,
    }


def extension_runtime_status(snapshot: RuntimeSnapshot, extension_id: str) -> dict[str, Any]:
    if get_extension_descriptor(extension_id) is None:
        supported = ", ".join(sorted(registered_extension_ids()))
        raise UnknownExtensionError(
            f"Unknown extension '{extension_id}'. Registered extensions: {supported}."
        )

    current_extension = snapshot.extensions[extension_id]
    models_dir_context = resolve_models_dir()
    return {
        "status": snapshot.bootstrap_state.get("status", "unknown"),
        "runtime_dir": str(snapshot.paths.runtime_dir),
        "models_dir": (
            str(models_dir_context["models_dir"])
            if models_dir_context["models_dir"] is not None
            else None
        ),
        "models_dir_source": models_dir_context["source"],
        "models_dir_diagnostics": list(models_dir_context["diagnostics"]),
        "runtime_models_dir": str(snapshot.paths.models_dir),
        "state_dir": str(snapshot.paths.state_dir),
        "bootstrap_version": snapshot.bootstrap_state.get("version"),
        "state_version": 2,
        "extension": current_extension,
        "legacy_models": snapshot.legacy_models,
    }


def install_extension_from_local_dir(
    snapshot: RuntimeSnapshot,
    extension_id: str,
    source_dir: str | Path,
) -> InstalledExtensionResult:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        supported = ", ".join(sorted(extension_metadata_map()))
        raise UnknownExtensionError(
            f"Unknown extension '{extension_id}'. Registered extensions: {supported}."
        )

    source_path = Path(source_dir).expanduser()
    if not source_path.exists():
        raise InvalidExtensionSourceError(
            f"Source directory '{source_path}' does not exist."
        )
    if not source_path.is_dir():
        raise InvalidExtensionSourceError(
            f"Source directory '{source_path}' is not a directory."
        )

    missing_paths = missing_required_paths(extension_id, source_path)
    if missing_paths:
        raise InvalidExtensionSourceError(
            f"Extension '{extension_id}' source is missing required paths: {', '.join(missing_paths)}."
        )

    destination_dir = snapshot.paths.models_dir / extension_id
    temp_dir = snapshot.paths.models_dir / f".{extension_id}.tmp-{uuid.uuid4().hex}"
    installing_snapshot = mark_extension_installing(snapshot, extension_id)

    try:
        shutil.copytree(source_path, temp_dir)
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        temp_dir.replace(destination_dir)
        installed_snapshot = mark_extension_installed(installing_snapshot, extension_id)
    except Exception as exc:
        mark_extension_error(installing_snapshot, extension_id, str(exc))
        raise RuntimeOperationError(
            f"Failed to install extension '{extension_id}' from '{source_path}': {exc}"
        ) from exc
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return InstalledExtensionResult(
        extension_id=extension_id,
        source_dir=source_path.resolve(),
        destination_dir=destination_dir.resolve(),
        required_paths=descriptor.required_paths,
        snapshot=installed_snapshot,
    )

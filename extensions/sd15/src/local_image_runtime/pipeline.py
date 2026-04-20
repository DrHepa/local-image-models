from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .bootstrap import RuntimeSnapshot, extension_is_installed, get_extension_record
from .descriptors import get_extension_descriptor, registered_extension_ids


class DomainError(RuntimeError):
    """Base domain error for the local image extension."""


class ExtensionNotInstalledError(DomainError):
    """Raised when the requested extension is known but not installed."""


class BackendNotImplementedError(DomainError):
    """Raised when runtime validation passes but execution is not implemented yet."""


class RequestValidationError(DomainError):
    """Raised when an execution request fails domain validation."""


@dataclass(frozen=True)
class ExecutionRequest:
    node_id: str
    input: dict[str, Any]
    params: dict[str, Any]
    workspace_dir: str | None = None
    temp_dir: str | None = None


@dataclass(frozen=True)
class ValidatedPayload:
    prompt: str | None
    source_image_path: str | None
    numeric_params: dict[str, float | int]
    legacy_model_id: str | None


@dataclass(frozen=True)
class BackendJob:
    command: tuple[str, ...]
    payload: dict[str, Any]
    workspace_dir: Path


def _request_path_candidates(request: ExecutionRequest, raw_path: str) -> tuple[Path, ...]:
    path = Path(raw_path)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if request.workspace_dir:
            candidates.append(Path(request.workspace_dir) / path)
        candidates.append(path)
    return tuple(candidates)


def _validate_numeric_param(
    params: dict[str, Any],
    name: str,
    *,
    expected_type: type[int] | type[float],
    minimum: int | float | None = None,
    maximum: int | float | None = None,
) -> int | float | None:
    value = params.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestValidationError(f"Parameter '{name}' must be numeric when provided.")
    if expected_type is int and not isinstance(value, int):
        raise RequestValidationError(f"Parameter '{name}' must be an integer.")
    numeric_value = int(value) if expected_type is int else float(value)
    if minimum is not None and numeric_value < minimum:
        raise RequestValidationError(f"Parameter '{name}' must be >= {minimum}.")
    if maximum is not None and numeric_value > maximum:
        raise RequestValidationError(f"Parameter '{name}' must be <= {maximum}.")
    return numeric_value


def _require_supported_node(extension_id: str, node_id: str) -> None:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        supported = ", ".join(sorted(registered_extension_ids()))
        raise RequestValidationError(
            f"Unknown extension '{extension_id}'. Supported extensions: {supported}."
        )

    if node_id not in descriptor.supported_nodes:
        supported_nodes = ", ".join(descriptor.supported_nodes)
        raise RequestValidationError(
            f"Extension '{extension_id}' does not support node '{node_id}'. Supported nodes: {supported_nodes}."
        )


def _resolve_legacy_model_id(params: dict[str, Any], extension_id: str) -> str | None:
    if "model_id" not in params:
        return None

    legacy_model_id = params.get("model_id")
    if not isinstance(legacy_model_id, str) or not legacy_model_id.strip():
        raise RequestValidationError(
            "Legacy parameter 'model_id', when provided, must be a non-empty string."
        )

    normalized_model_id = legacy_model_id.strip()
    if normalized_model_id != extension_id:
        raise RequestValidationError(
            f"Legacy parameter 'model_id'='{normalized_model_id}' does not match fixed extension '{extension_id}'."
        )
    return normalized_model_id


def _validate_text_prompt(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{field_name} requires a non-empty string.")
    prompt = value.strip()
    if len(prompt) > 4000:
        raise RequestValidationError(f"{field_name} must be 4000 characters or fewer.")
    return prompt


def _validate_node_payload(
    request: ExecutionRequest, legacy_model_id: str | None
) -> ValidatedPayload:
    numeric_params = {
        name: value
        for name, value in {
            "steps": _validate_numeric_param(request.params, "steps", expected_type=int, minimum=1, maximum=150),
            "width": _validate_numeric_param(request.params, "width", expected_type=int, minimum=64, maximum=2048),
            "height": _validate_numeric_param(request.params, "height", expected_type=int, minimum=64, maximum=2048),
            "strength": _validate_numeric_param(request.params, "strength", expected_type=float, minimum=0.0, maximum=1.0),
            "guidance_scale": _validate_numeric_param(request.params, "guidance_scale", expected_type=float, minimum=0.0, maximum=50.0),
            "seed": _validate_numeric_param(request.params, "seed", expected_type=int, minimum=0),
        }.items()
        if value is not None
    }

    if request.node_id == "text-to-image":
        prompt = request.params.get("prompt")
        if prompt is None:
            prompt = request.input.get("text")
        return ValidatedPayload(
            prompt=_validate_text_prompt(prompt, field_name="text-to-image prompt"),
            source_image_path=None,
            numeric_params=numeric_params,
            legacy_model_id=legacy_model_id,
        )

    source_image_path = request.input.get("filePath")
    if not isinstance(source_image_path, str) or not source_image_path.strip():
        raise RequestValidationError(
            "image-to-image requires input.filePath pointing to the source image."
        )
    source_image_path = source_image_path.strip()
    if source_image_path.endswith(("/", "\\")):
        raise RequestValidationError(
            "image-to-image input.filePath must point to a file, not a directory."
        )

    resolved_source_path = None
    for candidate in _request_path_candidates(request, source_image_path):
        if candidate.exists() and candidate.is_file():
            resolved_source_path = str(candidate.resolve())
            break
    if resolved_source_path is None:
        raise RequestValidationError(
            "image-to-image input.filePath must point to an existing local file."
        )

    prompt = request.params.get("prompt")
    validated_prompt = None
    if prompt is not None:
        validated_prompt = _validate_text_prompt(
            prompt, field_name="image-to-image params.prompt"
        )

    if "strength" not in numeric_params:
        raise RequestValidationError(
            "image-to-image requires params.strength between 0.0 and 1.0."
        )

    return ValidatedPayload(
        prompt=validated_prompt,
        source_image_path=resolved_source_path,
        numeric_params=numeric_params,
        legacy_model_id=legacy_model_id,
    )


def _validate_optional_text_param(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _validate_text_prompt(value, field_name=field_name)


def _resolve_workspace_dir(effective_workspace_dir: str) -> Path:
    return Path(effective_workspace_dir).expanduser().resolve()


def _require_executable_venv_python(extension_record: dict[str, Any], *, extension_id: str) -> Path:
    raw_venv_python = extension_record.get("venv_python")
    if not isinstance(raw_venv_python, str) or not raw_venv_python.strip():
        raise DomainError(
            f"Missing executable venv_python for extension '{extension_id}'. Run setup install/repair first."
        )

    venv_python = Path(raw_venv_python.strip()).expanduser()
    if not venv_python.exists() or not venv_python.is_file() or not os.access(venv_python, os.X_OK):
        raise DomainError(
            f"Missing executable venv_python for extension '{extension_id}': {venv_python}"
        )
    return venv_python


def _build_backend_job(
    *,
    request: ExecutionRequest,
    extension_id: str,
    extension_record: dict[str, Any],
    payload_details: ValidatedPayload,
    effective_workspace_dir: str,
) -> BackendJob:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        supported = ", ".join(sorted(registered_extension_ids()))
        raise RequestValidationError(
            f"Unknown extension '{extension_id}'. Supported extensions: {supported}."
        )

    workspace_dir = _resolve_workspace_dir(effective_workspace_dir)
    output_path = workspace_dir / (
        f"generated-{extension_id}-{request.node_id}-{uuid4().hex}.png"
    )
    params = dict(request.params)
    params.pop("negative_prompt", None)

    payload = {
        "extension_id": extension_id,
        "family": descriptor.family,
        "node_id": request.node_id,
        "model_dir": extension_record.get("model_dir"),
        "workspace_dir": str(workspace_dir),
        "output_path": str(output_path),
        "prompt": payload_details.prompt,
        "negative_prompt": _validate_optional_text_param(
            request.params.get("negative_prompt"), field_name="negative_prompt"
        ),
        "source_image_path": payload_details.source_image_path,
        "params": params,
    }
    return BackendJob(
        command=(
            str(_require_executable_venv_python(extension_record, extension_id=extension_id)),
            "-m",
            "local_image_runtime.inference_runner",
        ),
        payload=payload,
        workspace_dir=workspace_dir,
    )


def _command_failure_detail(exc: subprocess.CalledProcessError | OSError) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        detail = (_extract_error_message_from_output(exc.stdout) or exc.stderr or exc.stdout or str(exc)).strip()
        return detail or str(exc)
    return str(exc)


def _extract_error_message_from_output(stdout: str | None) -> str | None:
    if not stdout:
        return None

    message: str | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "error":
            continue

        error_message = event.get("message")
        if isinstance(error_message, str) and error_message.strip():
            message = error_message.strip()
    return message


def _parse_backend_events(
    stdout: str,
    *,
    emit_progress: Callable[[int, str], None],
    emit_log: Callable[[str], None],
) -> dict[str, Any]:
    done_result: dict[str, Any] | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DomainError(f"Backend subprocess emitted invalid NDJSON: {exc}") from exc
        if not isinstance(event, dict):
            raise DomainError("Backend subprocess events must be JSON objects.")

        event_type = event.get("type")
        if event_type == "progress":
            percent = event.get("percent")
            label = event.get("label")
            if isinstance(percent, int) and isinstance(label, str) and label.strip():
                emit_progress(percent, label)
            continue
        if event_type == "log":
            message = event.get("message")
            if isinstance(message, str) and message.strip():
                emit_log(message)
            continue
        if event_type == "error":
            message = event.get("message")
            raise DomainError(
                message if isinstance(message, str) and message.strip() else "Backend subprocess reported an error."
            )
        if event_type == "done":
            result = event.get("result")
            if not isinstance(result, dict):
                raise DomainError("Backend subprocess done event must include a result object.")
            done_result = result
            continue

        raise DomainError(f"Backend subprocess emitted unsupported event type: {event_type!r}")

    if done_result is None:
        raise DomainError("Backend subprocess did not emit a done event.")
    return done_result


def _resolve_output_path_within_workspace(result: dict[str, Any], *, workspace_dir: Path) -> Path:
    output_path = result.get("output_path")
    if not isinstance(output_path, str) or not output_path.strip():
        raise DomainError("Backend subprocess result must include a non-empty 'output_path'.")

    candidate = Path(output_path.strip())
    if not candidate.is_absolute():
        candidate = workspace_dir / candidate
    resolved_candidate = candidate.expanduser().resolve()
    try:
        resolved_candidate.relative_to(workspace_dir)
    except ValueError as exc:
        raise DomainError(
            f"Backend subprocess output_path '{resolved_candidate}' is outside workspace_dir '{workspace_dir}'."
        ) from exc
    return resolved_candidate


def _run_backend_job(
    job: BackendJob,
    *,
    emit_progress: Callable[[int, str], None],
    emit_log: Callable[[str], None],
) -> dict[str, Any]:
    payload_json = json.dumps(job.payload)
    try:
        completed = subprocess.run(
            list(job.command),
            input=payload_json,
            text=True,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        child_error_message = (
            _extract_error_message_from_output(exc.stdout)
            if isinstance(exc, subprocess.CalledProcessError)
            else None
        )
        detail = _command_failure_detail(exc)
        if child_error_message is not None:
            raise DomainError(detail) from exc
        raise DomainError(f"Backend subprocess failed: {detail}") from exc

    result = _parse_backend_events(
        completed.stdout,
        emit_progress=emit_progress,
        emit_log=emit_log,
    )
    resolved_output_path = _resolve_output_path_within_workspace(result, workspace_dir=job.workspace_dir)
    resolved_result = dict(result)
    resolved_result["output_path"] = str(resolved_output_path)
    return resolved_result


def execute(
    request: ExecutionRequest,
    runtime: RuntimeSnapshot,
    extension_id: str,
    emit_progress: Callable[[int, str], None],
    emit_log: Callable[[str], None],
) -> dict[str, Any]:
    _require_supported_node(extension_id, request.node_id)
    effective_workspace_dir = request.workspace_dir or str(runtime.paths.outputs_dir)
    emit_progress(35, "validating-request")

    legacy_model_id = _resolve_legacy_model_id(request.params, extension_id)
    payload_details = _validate_node_payload(request, legacy_model_id)
    emit_log(
        f"Validated node '{request.node_id}' for extension '{extension_id}'"
        + (
            f" (legacy model_id '{legacy_model_id}')"
            if legacy_model_id is not None
            else ""
        )
        + f". Workspace: {effective_workspace_dir}."
    )
    emit_progress(55, "checking-extension")

    extension_record = get_extension_record(runtime, extension_id)
    if not extension_is_installed(runtime, extension_id):
        model_dir = extension_record.get("model_dir", str(runtime.paths.models_dir / extension_id))
        raise ExtensionNotInstalledError(
            f"Extension '{extension_id}' is registered but not installed. Expected runtime model directory: "
            f"{model_dir}. Run the extension-local setup install command for '{extension_id}' before "
            f"executing '{request.node_id}'."
        )

    emit_progress(75, "backend-dispatch")
    job = _build_backend_job(
        request=request,
        extension_id=extension_id,
        extension_record=extension_record,
        payload_details=payload_details,
        effective_workspace_dir=effective_workspace_dir,
    )
    emit_log(
        f"Dispatching backend family '{job.payload['family']}' for node '{request.node_id}' using venv '{job.command[0]}'."
    )
    return _run_backend_job(
        job,
        emit_progress=emit_progress,
        emit_log=emit_log,
    )

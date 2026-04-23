from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .bootstrap import RuntimeSnapshot, extension_is_installed, get_extension_record
from .descriptors import get_extension_descriptor, registered_extension_ids
from . import lifecycle
from .quality_policy import resolve_effective_params


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
    model_dir_override: str | None = None


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
    cwd: Path
    env: dict[str, str]


BACKEND_TOTAL_TIMEOUT_SECONDS = 1800.0
BACKEND_STAGE_IDLE_TIMEOUT_SECONDS = 300.0
BACKEND_TERMINATE_GRACE_SECONDS = 5.0
BACKEND_EVENT_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class BackendTimeoutConfig:
    total_seconds: float = BACKEND_TOTAL_TIMEOUT_SECONDS
    idle_seconds: float = BACKEND_STAGE_IDLE_TIMEOUT_SECONDS
    terminate_grace_seconds: float = BACKEND_TERMINATE_GRACE_SECONDS
    poll_seconds: float = BACKEND_EVENT_POLL_SECONDS


def _default_backend_timeout_config() -> BackendTimeoutConfig:
    return BackendTimeoutConfig()


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
    if isinstance(value, str) and not value.strip():
        return ""
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


def _derive_runtime_src_dir(*, venv_python: Path, extension_id: str) -> Path:
    lexical_python = venv_python.expanduser()
    parent_dir = lexical_python.parent
    layout_dir = parent_dir.name
    if layout_dir not in {"bin", "Scripts"}:
        raise DomainError(
            f"Invalid virtualenv layout for extension '{extension_id}': {venv_python}"
        )

    venv_dir = parent_dir.parent
    if venv_dir.name != "venv":
        raise DomainError(
            f"Invalid virtualenv layout for extension '{extension_id}': {venv_python}"
        )

    runtime_src_dir = venv_dir.parent / "src"
    if not runtime_src_dir.exists() or not runtime_src_dir.is_dir():
        raise DomainError(
            f"Missing vendored runtime src for extension '{extension_id}': {runtime_src_dir}"
        )
    return runtime_src_dir


def _build_backend_env(*, runtime_src_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    runtime_src = str(runtime_src_dir)
    env["PYTHONPATH"] = (
        f"{runtime_src}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else runtime_src
    )
    return env


def _resolve_backend_model_dir(
    request: ExecutionRequest,
    extension_record: dict[str, Any],
) -> Any:
    override = request.model_dir_override
    if isinstance(override, str):
        normalized_override = override.strip()
        if normalized_override:
            return normalized_override
    return extension_record.get("model_dir")


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
        "model_dir": _resolve_backend_model_dir(request, extension_record),
        "workspace_dir": str(workspace_dir),
        "output_path": str(output_path),
        "prompt": payload_details.prompt,
        "negative_prompt": _validate_optional_text_param(
            request.params.get("negative_prompt"), field_name="negative_prompt"
        ),
        "source_image_path": payload_details.source_image_path,
        "params": params,
    }
    venv_python = _require_executable_venv_python(extension_record, extension_id=extension_id)
    runtime_src_dir = _derive_runtime_src_dir(venv_python=venv_python, extension_id=extension_id)
    return BackendJob(
        command=(
            str(venv_python),
            "-m",
            "local_image_runtime.inference_runner",
        ),
        payload=payload,
        workspace_dir=workspace_dir,
        cwd=runtime_src_dir,
        env=_build_backend_env(runtime_src_dir=runtime_src_dir),
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
        if event_type == "memory":
            stage = event.get("stage")
            if isinstance(stage, str) and stage.strip():
                continue
            raise DomainError("Backend subprocess memory event must include a non-empty stage string.")

        raise DomainError(f"Backend subprocess emitted unsupported event type: {event_type!r}")

    if done_result is None:
        raise DomainError("Backend subprocess did not emit a done event.")
    return done_result


def _read_stream(
    name: str,
    stream: Any,
    sink: "queue.Queue[tuple[str, str, str | None]]",
) -> None:
    try:
        while True:
            line = stream.readline()
            if line == "":
                sink.put(("eof", name, None))
                return
            sink.put(("line", name, line))
    except Exception as exc:  # pragma: no cover - defensive bridge for reader threads
        sink.put(("reader-error", name, str(exc)))


def _parse_backend_event_line(
    raw_line: str,
    *,
    emit_progress: Callable[[int, str], None],
    emit_log: Callable[[str], None],
) -> tuple[str, dict[str, Any] | str] | None:
    line = raw_line.strip()
    if not line:
        return None

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
            normalized_label = label.strip()
            emit_progress(percent, normalized_label)
            return ("progress", normalized_label)
        return None
    if event_type == "log":
        message = event.get("message")
        if isinstance(message, str) and message.strip():
            normalized_message = message.strip()
            emit_log(normalized_message)
            return ("log", normalized_message)
        return None
    if event_type == "error":
        message = event.get("message")
        return (
            "error",
            message if isinstance(message, str) and message.strip() else "Backend subprocess reported an error.",
        )
    if event_type == "done":
        result = event.get("result")
        if not isinstance(result, dict):
            raise DomainError("Backend subprocess done event must include a result object.")
        return ("done", result)
    if event_type == "memory":
        stage = event.get("stage")
        if isinstance(stage, str) and stage.strip():
            return ("memory", stage.strip())
        raise DomainError("Backend subprocess memory event must include a non-empty stage string.")

    raise DomainError(f"Backend subprocess emitted unsupported event type: {event_type!r}")


def _stream_backend_events(
    process: subprocess.Popen[str],
    *,
    emit_progress: Callable[[int, str], None],
    emit_log: Callable[[str], None],
    timeout_config: BackendTimeoutConfig,
    monotonic: Callable[[], float],
) -> tuple[dict[str, Any] | None, str | None, str]:
    if process.stdout is None or process.stderr is None:
        raise DomainError("Backend subprocess did not expose stdout/stderr pipes.")

    event_queue: "queue.Queue[tuple[str, str, str | None]]" = queue.Queue()
    readers = [
        threading.Thread(target=_read_stream, args=("stdout", process.stdout, event_queue), daemon=True),
        threading.Thread(target=_read_stream, args=("stderr", process.stderr, event_queue), daemon=True),
    ]
    for reader in readers:
        reader.start()

    open_streams = {"stdout", "stderr"}
    done_result: dict[str, Any] | None = None
    child_error: str | None = None
    stderr_chunks: list[str] = []
    started_at = monotonic()
    last_activity_at = started_at
    current_stage: str | None = None

    while open_streams:
        now = monotonic()
        total_elapsed = now - started_at
        if total_elapsed >= timeout_config.total_seconds:
            raise DomainError(
                f"Backend subprocess hit total backend timeout after {timeout_config.total_seconds:.1f}s."
            )

        idle_elapsed = now - last_activity_at
        if idle_elapsed >= timeout_config.idle_seconds:
            stalled_stage = current_stage or "unknown-stage"
            raise DomainError(
                f"Backend subprocess hit idle backend timeout after {timeout_config.idle_seconds:.1f}s in stage '{stalled_stage}'."
            )

        wait_timeout = max(
            0.0,
            min(
                timeout_config.poll_seconds,
                timeout_config.total_seconds - total_elapsed,
                timeout_config.idle_seconds - idle_elapsed,
            ),
        )
        try:
            event_kind, stream_name, payload = event_queue.get(timeout=wait_timeout)
        except queue.Empty:
            continue
        if event_kind == "eof":
            open_streams.discard(stream_name)
            continue
        if event_kind == "reader-error":
            raise DomainError(
                f"Backend subprocess {stream_name} reader failed: {payload or 'unknown error'}"
            )
        if payload is None:
            continue

        if stream_name == "stderr":
            stderr_chunks.append(payload)
            continue

        parsed_event = _parse_backend_event_line(
            payload,
            emit_progress=emit_progress,
            emit_log=emit_log,
        )
        if parsed_event is None:
            continue

        event_type, event_payload = parsed_event
        if event_type == "progress":
            last_activity_at = monotonic()
            current_stage = str(event_payload)
            continue
        if event_type == "log":
            last_activity_at = monotonic()
            continue
        if event_type == "done":
            done_result = event_payload if isinstance(event_payload, dict) else None
            continue
        if event_type == "memory":
            last_activity_at = monotonic()
            current_stage = str(event_payload)
            continue
        child_error = str(event_payload)

    for reader in readers:
        reader.join(timeout=1)

    return done_result, child_error, "".join(stderr_chunks)


def _stop_backend_process(
    process: subprocess.Popen[str],
    *,
    terminate_grace_seconds: float = BACKEND_TERMINATE_GRACE_SECONDS,
) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=terminate_grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=terminate_grace_seconds)


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
    timeout_config: BackendTimeoutConfig | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    payload_json = json.dumps(job.payload)
    resolved_timeout_config = timeout_config or _default_backend_timeout_config()
    try:
        process = subprocess.Popen(
            list(job.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(job.cwd),
            env=job.env,
        )
    except OSError as exc:
        raise DomainError(f"Backend subprocess failed: {exc}") from exc

    if process.stdin is None:
        raise DomainError("Backend subprocess did not expose a stdin pipe.")

    try:
        process.stdin.write(payload_json + "\n")
        process.stdin.flush()
        process.stdin.close()

        result, child_error, stderr_output = _stream_backend_events(
            process,
            emit_progress=emit_progress,
            emit_log=emit_log,
            timeout_config=resolved_timeout_config,
            monotonic=monotonic,
        )
        returncode = process.wait()
    except Exception:
        _stop_backend_process(
            process,
            terminate_grace_seconds=resolved_timeout_config.terminate_grace_seconds,
        )
        raise

    if child_error is not None:
        raise DomainError(child_error)
    if returncode != 0:
        detail = stderr_output.strip() or f"exit code {returncode}"
        raise DomainError(f"Backend subprocess failed: {detail}")
    if result is None:
        raise DomainError("Backend subprocess did not emit a done event.")

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
    host_generation_steps = lifecycle.host_generation_steps()
    for percent, label in host_generation_steps[:1]:
        emit_progress(percent, label)

    legacy_model_id = _resolve_legacy_model_id(request.params, extension_id)
    effective_request = ExecutionRequest(
        node_id=request.node_id,
        input=dict(request.input),
        params=resolve_effective_params(
            extension_id=extension_id,
            node_id=request.node_id,
            params=request.params,
        ),
        workspace_dir=request.workspace_dir,
        temp_dir=request.temp_dir,
        model_dir_override=request.model_dir_override,
    )
    payload_details = _validate_node_payload(effective_request, legacy_model_id)
    emit_log(
        f"Validated node '{request.node_id}' for extension '{extension_id}'"
        + (
            f" (legacy model_id '{legacy_model_id}')"
            if legacy_model_id is not None
            else ""
        )
        + f". Workspace: {effective_workspace_dir}."
    )
    for percent, label in host_generation_steps[1:2]:
        emit_progress(percent, label)

    extension_record = get_extension_record(runtime, extension_id)
    if not extension_is_installed(runtime, extension_id):
        model_dir = extension_record.get("model_dir", str(runtime.paths.models_dir / extension_id))
        raise ExtensionNotInstalledError(
            f"Extension '{extension_id}' is registered but not installed. Expected runtime model directory: "
            f"{model_dir}. Run the extension-local setup install command for '{extension_id}' before "
            f"executing '{request.node_id}'."
        )

    for percent, label in host_generation_steps[2:]:
        emit_progress(percent, label)
    job = _build_backend_job(
        request=effective_request,
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

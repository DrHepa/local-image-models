from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
    raise BackendNotImplementedError(
        f"Generation backend is not implemented yet for node '{request.node_id}' with extension '{extension_id}'. "
        f"Validation passed and the runtime scaffold is ready. Next step: implement the backend adapter in "
        f"shared/runtime/local_image_runtime/pipeline.py and write outputs to {effective_workspace_dir}. "
        f"Validated request summary: prompt={'yes' if payload_details.prompt else 'no'}, "
        f"source_image={'yes' if payload_details.source_image_path else 'no'}, "
        f"legacy_model_id={'yes' if payload_details.legacy_model_id else 'no'}, "
        f"numeric_params={sorted(payload_details.numeric_params)}."
    )

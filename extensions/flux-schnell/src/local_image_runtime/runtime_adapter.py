from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO
from uuid import uuid4

from .bootstrap import (
    EXTENSION_ROOT_OVERRIDE_ENV,
    RuntimeSnapshot,
    bootstrap_runtime,
)
from .descriptors import get_extension_descriptor
from .pipeline import DomainError, ExecutionRequest, execute

try:  # pragma: no cover - exercised only when Modly is importable in-process
    from api.services.generators.base import BaseGenerator as ImportedBaseGenerator
except ImportError:  # pragma: no cover - local fallback for tests/direct use
    class ImportedBaseGenerator:
        def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
            self.model_dir = model_dir
            self.outputs_dir = outputs_dir
            self._model = None
            self._params_schema: list[dict[str, Any]] = []

        def unload(self) -> None:
            self._model = None

        def params_schema(self) -> list[dict[str, Any]]:
            return list(self._params_schema)


GeneratorEventEmitter = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class GeneratorExecution:
    extension_id: str
    runtime: RuntimeSnapshot
    request: ExecutionRequest


def emit_event(message_type: str, *, stream: TextIO | None = None, **payload: Any) -> None:
    output = stream or sys.stdout
    output.write(json.dumps({"type": message_type, **payload}) + "\n")
    output.flush()


def emit_error(message: str, *, extension_id: str | None = None, stream: TextIO | None = None) -> int:
    payload: dict[str, Any] = {"message": message}
    if extension_id is not None:
        payload["extension_id"] = extension_id
    emit_event("error", stream=stream, **payload)
    return 1


@contextmanager
def scoped_extension_root(runtime_root: str | None):
    if runtime_root is None:
        yield
        return

    previous = os.environ.get(EXTENSION_ROOT_OVERRIDE_ENV)
    os.environ[EXTENSION_ROOT_OVERRIDE_ENV] = runtime_root
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(EXTENSION_ROOT_OVERRIDE_ENV, None)
        else:
            os.environ[EXTENSION_ROOT_OVERRIDE_ENV] = previous


def read_payload(*, stream: TextIO | None = None) -> dict[str, Any]:
    input_stream = stream or sys.stdin
    raw = input_stream.readline()
    if not raw:
        raise DomainError(
            "Generator expected one JSON payload line on stdin, but stdin was empty."
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DomainError(f"Invalid JSON payload received by generator: {exc}") from exc

    if not isinstance(payload, dict):
        raise DomainError(
            "Generator payload must be a JSON object with input, params, and nodeId fields."
        )
    return payload


def build_execution_request(payload: dict[str, Any]) -> ExecutionRequest:
    input_payload = payload.get("input")
    params = payload.get("params")

    if input_payload is None or not isinstance(input_payload, dict):
        raise DomainError("Payload field 'input' is required and must be an object.")

    if params is None or not isinstance(params, dict):
        raise DomainError("Payload field 'params' is required and must be an object.")

    node_id = payload.get("nodeId") or input_payload.get("nodeId")
    if not isinstance(node_id, str) or not node_id.strip():
        raise DomainError(
            "Payload must include a non-empty nodeId for a node declared by this extension."
        )

    workspace_dir = payload.get("workspaceDir")
    temp_dir = payload.get("tempDir")

    return ExecutionRequest(
        node_id=node_id.strip(),
        input=input_payload,
        params=params,
        workspace_dir=workspace_dir if isinstance(workspace_dir, str) else None,
        temp_dir=temp_dir if isinstance(temp_dir, str) else None,
    )


def prepare_execution(
    *, extension_id: str, payload: dict[str, Any], runtime_root: str | None = None
) -> GeneratorExecution:
    with scoped_extension_root(runtime_root):
        runtime = bootstrap_runtime(extension_id=extension_id)
        request = build_execution_request(payload)
    return GeneratorExecution(
        extension_id=extension_id,
        runtime=runtime,
        request=request,
    )


def run_payload(
    payload: dict[str, Any], *, extension_id: str, runtime_root: str | None = None
) -> dict[str, Any]:
    execution = prepare_execution(
        extension_id=extension_id,
        payload=payload,
        runtime_root=runtime_root,
    )
    logs: list[str] = []
    progress_events: list[dict[str, Any]] = []
    result = execute(
        execution.request,
        execution.runtime,
        extension_id=extension_id,
        emit_progress=lambda percent, label: progress_events.append(
            {"percent": percent, "label": label}
        ),
        emit_log=logs.append,
    )
    return {
        "extension_id": extension_id,
        "result": result,
        "logs": logs,
        "progress": progress_events,
    }


def run_generator_main(
    *,
    extension_id: str,
    runtime_root: str | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    try:
        payload = read_payload(stream=stdin)
        emit_event(
            "progress",
            stream=stdout,
            extension_id=extension_id,
            percent=5,
            label="payload-received",
        )
        execution = prepare_execution(
            extension_id=extension_id,
            payload=payload,
            runtime_root=runtime_root,
        )
        emit_event(
            "log",
            stream=stdout,
            extension_id=extension_id,
            message=f"Runtime ready at {execution.runtime.paths.runtime_dir}",
        )
        emit_event(
            "progress",
            stream=stdout,
            extension_id=extension_id,
            percent=20,
            label="runtime-ready",
        )
        result = execute(
            execution.request,
            execution.runtime,
            extension_id=extension_id,
            emit_progress=lambda percent, label: emit_event(
                "progress",
                stream=stdout,
                extension_id=extension_id,
                percent=percent,
                label=label,
            ),
            emit_log=lambda message: emit_event(
                "log",
                stream=stdout,
                extension_id=extension_id,
                message=message,
            ),
        )
        emit_event(
            "done",
            stream=stdout,
            extension_id=extension_id,
            result=result,
        )
        return 0
    except DomainError as exc:
        return emit_error(str(exc), extension_id=extension_id, stream=stdout)
    except Exception as exc:  # pragma: no cover - defensive error boundary
        return emit_error(
            f"Unexpected generator failure: {exc}",
            extension_id=extension_id,
            stream=stdout,
        )


class RuntimeGeneratorAdapter:
    extension_id: str = ""
    runtime_root: str | None = None

    def run_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.extension_id:
            raise DomainError("RuntimeGeneratorAdapter requires a non-empty extension_id.")
        return run_payload(
            payload,
            extension_id=self.extension_id,
            runtime_root=self.runtime_root,
        )


def _load_manifest(runtime_root: str | None) -> dict[str, Any]:
    if runtime_root is None:
        raise DomainError("BaseGenerator runtime adapter requires a concrete runtime_root.")

    manifest_path = Path(runtime_root) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DomainError(f"Missing manifest.json for runtime adapter at '{manifest_path}'.") from exc
    except json.JSONDecodeError as exc:
        raise DomainError(f"Invalid manifest.json for runtime adapter at '{manifest_path}': {exc}") from exc

    if not isinstance(manifest, dict):
        raise DomainError(f"Manifest at '{manifest_path}' must be a JSON object.")
    return manifest


def _resolve_node_id(*, extension_id: str, model_dir: Path) -> str:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        raise DomainError(f"Unknown extension '{extension_id}' for BaseGenerator runtime adapter.")

    if model_dir.name in descriptor.supported_nodes:
        return model_dir.name

    if len(descriptor.supported_nodes) == 1:
        return descriptor.supported_nodes[0]

    supported_nodes = ", ".join(descriptor.supported_nodes)
    raise DomainError(
        f"Could not resolve node for extension '{extension_id}' from model_dir.name '{model_dir.name}'. "
        f"Supported nodes: {supported_nodes}."
    )


def _resolve_params_schema(*, manifest: dict[str, Any], node_id: str) -> list[dict[str, Any]]:
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list):
        return []

    for node in nodes:
        if isinstance(node, dict) and node.get("id") == node_id:
            params_schema = node.get("params_schema")
            if isinstance(params_schema, list):
                return [item for item in params_schema if isinstance(item, dict)]
            return []
    return []


def _coerce_generate_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise DomainError("BaseGenerator generate() expects params to be a dictionary.")
    return dict(params)


def _nested_input_payload(params: dict[str, Any]) -> dict[str, Any]:
    input_payload = params.get("input")
    if input_payload is None:
        return {}
    if not isinstance(input_payload, dict):
        raise DomainError("BaseGenerator generate() expects params.input to be an object when provided.")
    return dict(input_payload)


def _resolve_output_path(result: dict[str, Any], *, outputs_dir: Path) -> Path:
    output_path = result.get("output_path")
    if not isinstance(output_path, str) or not output_path.strip():
        raise DomainError("Generation result must include a non-empty 'output_path'.")

    candidate_path = Path(output_path.strip())
    if not candidate_path.is_absolute():
        candidate_path = outputs_dir / candidate_path

    resolved_outputs_dir = outputs_dir.resolve()
    resolved_candidate_path = candidate_path.resolve()
    try:
        resolved_candidate_path.relative_to(resolved_outputs_dir)
    except ValueError as exc:
        raise DomainError(
            "Generation result output_path "
            f"'{resolved_candidate_path}' is outside configured outputs_dir '{resolved_outputs_dir}'."
        ) from exc
    return resolved_candidate_path


def _raise_if_cancelled(cancel_event: Any | None) -> None:
    if cancel_event is None or not hasattr(cancel_event, "is_set"):
        return
    if cancel_event.is_set():
        raise DomainError("Generation cancelled before backend execution.")


class BaseGeneratorRuntimeAdapter(ImportedBaseGenerator):
    extension_id: str = ""
    runtime_root: str | None = None
    MODEL_ID: str = ""

    def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
        super().__init__(Path(model_dir), Path(outputs_dir))
        if not self.extension_id:
            raise DomainError("BaseGeneratorRuntimeAdapter requires a non-empty extension_id.")

        self.model_dir = Path(model_dir)
        self.outputs_dir = Path(outputs_dir)
        self.node_id = _resolve_node_id(extension_id=self.extension_id, model_dir=self.model_dir)
        self._runtime_snapshot: RuntimeSnapshot | None = None
        manifest = _load_manifest(self.runtime_root)
        self._params_schema = _resolve_params_schema(manifest=manifest, node_id=self.node_id)

    def load(self) -> None:
        if self._runtime_snapshot is not None:
            return

        with scoped_extension_root(self.runtime_root):
            self._runtime_snapshot = bootstrap_runtime(extension_id=self.extension_id)
        self._model = self._runtime_snapshot

    def unload(self) -> None:
        self._runtime_snapshot = None
        super().unload()

    def _loaded_runtime(self) -> RuntimeSnapshot:
        self.load()
        if self._runtime_snapshot is None:
            raise DomainError(
                f"Extension '{self.extension_id}' did not produce a runtime snapshot during load()."
            )
        return self._runtime_snapshot

    def _materialize_image_input(self, image_bytes: bytes) -> Path:
        if not image_bytes:
            raise DomainError("image-to-image generate() requires non-empty image_bytes.")
        inputs_dir = self.outputs_dir / ".modly-inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        image_path = inputs_dir / f"{uuid4().hex}.img"
        image_path.write_bytes(image_bytes)
        return image_path.resolve()

    def _build_generate_request(self, image_bytes: bytes, params: dict[str, Any]) -> ExecutionRequest:
        request_params = _coerce_generate_params(params)
        request_input = _nested_input_payload(request_params)

        if self.node_id == "text-to-image":
            prompt = request_params.get("prompt")
            if prompt is not None:
                request_params["prompt"] = prompt
            else:
                request_params.pop("prompt", None)
            return ExecutionRequest(
                node_id=self.node_id,
                input={"text": request_input.get("text")} if "text" in request_input else {},
                params=request_params,
                workspace_dir=str(self.outputs_dir),
            )

        if self.node_id == "image-to-image":
            materialized_input = self._materialize_image_input(image_bytes)
            return ExecutionRequest(
                node_id=self.node_id,
                input={"filePath": str(materialized_input)},
                params=request_params,
                workspace_dir=str(self.outputs_dir),
            )

        raise DomainError(
            f"BaseGenerator generate() does not support node '{self.node_id}' for extension '{self.extension_id}'."
        )

    def generate(
        self,
        image_bytes: bytes,
        params: dict[str, Any],
        progress_cb: Callable[[int, str], None] | None = None,
        cancel_event: Any | None = None,
    ) -> Path:
        runtime = self._loaded_runtime()
        request = self._build_generate_request(image_bytes, params)

        def emit_progress(percent: int, label: str) -> None:
            _raise_if_cancelled(cancel_event)
            if progress_cb is not None:
                progress_cb(percent, label)

        def emit_log(message: str) -> None:
            _raise_if_cancelled(cancel_event)

        _raise_if_cancelled(cancel_event)
        result = execute(
            request,
            runtime,
            extension_id=self.extension_id,
            emit_progress=emit_progress,
            emit_log=emit_log,
        )
        return _resolve_output_path(result, outputs_dir=self.outputs_dir)


def create_generator_class(
    name: str, *, extension_id: str, runtime_root: str | None = None
) -> type[BaseGeneratorRuntimeAdapter]:
    class _Generator(BaseGeneratorRuntimeAdapter):
        pass

    _Generator.__name__ = name
    _Generator.__qualname__ = name
    _Generator.extension_id = extension_id
    _Generator.runtime_root = runtime_root
    _Generator.MODEL_ID = extension_id
    return _Generator


__all__ = [
    "GeneratorExecution",
    "BaseGeneratorRuntimeAdapter",
    "RuntimeGeneratorAdapter",
    "_resolve_node_id",
    "build_execution_request",
    "create_generator_class",
    "emit_error",
    "emit_event",
    "prepare_execution",
    "read_payload",
    "run_generator_main",
    "run_payload",
]

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, TextIO

from .bootstrap import (
    EXTENSION_ROOT_OVERRIDE_ENV,
    RuntimeSnapshot,
    bootstrap_runtime,
)
from .pipeline import DomainError, ExecutionRequest, execute


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


def create_generator_class(
    name: str, *, extension_id: str, runtime_root: str | None = None
) -> type[RuntimeGeneratorAdapter]:
    class _Generator(RuntimeGeneratorAdapter):
        pass

    _Generator.__name__ = name
    _Generator.__qualname__ = name
    _Generator.extension_id = extension_id
    _Generator.runtime_root = runtime_root
    return _Generator


__all__ = [
    "GeneratorExecution",
    "RuntimeGeneratorAdapter",
    "build_execution_request",
    "create_generator_class",
    "emit_error",
    "emit_event",
    "prepare_execution",
    "read_payload",
    "run_generator_main",
    "run_payload",
]

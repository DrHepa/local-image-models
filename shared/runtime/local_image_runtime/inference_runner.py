from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

from .diffusers_memory import (
    apply_post_load_memory_optimizations,
    build_diffusers_load_attempts,
    collect_stage_memory_snapshot,
    is_retryable_diffusers_load_error,
    should_emit_memory_events,
)


class InferenceRunnerError(RuntimeError):
    """Raised when the child inference runner cannot execute a job."""


def _load_auto_text_pipeline():
    from diffusers import AutoPipelineForText2Image

    return AutoPipelineForText2Image


def _load_auto_image_pipeline():
    from diffusers import AutoPipelineForImage2Image

    return AutoPipelineForImage2Image


def _load_flux_pipeline():
    from diffusers import FluxPipeline

    return FluxPipeline


_PIPELINE_LOADERS: dict[tuple[str, str], Any] = {
    ("stable-diffusion", "text-to-image"): _load_auto_text_pipeline,
    ("stable-diffusion", "image-to-image"): _load_auto_image_pipeline,
    ("sdxl", "text-to-image"): _load_auto_text_pipeline,
    ("sdxl", "image-to-image"): _load_auto_image_pipeline,
    ("flux", "text-to-image"): _load_flux_pipeline,
}

_STAGE_PROGRESS = (
    (80, "loading-pipeline", "Loading inference pipeline."),
    (90, "running-inference", "Running inference."),
    (95, "saving-output", "Saving output image."),
)

_RUNNING_INFERENCE_HEARTBEAT_SECONDS = 15.0


def emit_event(event_type: str, *, stdout: TextIO | None = None, **payload: Any) -> None:
    stream = stdout or sys.stdout
    stream.write(json.dumps({"type": event_type, **payload}) + "\n")
    stream.flush()


def emit_error(message: str, *, stdout: TextIO | None = None) -> int:
    emit_event("error", stdout=stdout, message=message)
    return 1


def _emit_stage_event(label: str, *, stdout: TextIO | None = None) -> None:
    for percent, stage_label, message in _STAGE_PROGRESS:
        if stage_label != label:
            continue
        emit_event("progress", stdout=stdout, percent=percent, label=stage_label)
        emit_event("log", stdout=stdout, message=message)
        return
    raise InferenceRunnerError(f"Unsupported stage label '{label}'")


def _read_job(*, stdin: TextIO | None = None) -> dict[str, Any]:
    stream = stdin or sys.stdin
    raw = stream.readline()
    if not raw:
        raise InferenceRunnerError(
            "Inference runner expected one JSON job line on stdin, but stdin was empty."
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InferenceRunnerError("Invalid JSON job received by inference runner") from exc

    if not isinstance(payload, dict):
        raise InferenceRunnerError("Inference runner job must be a JSON object.")
    return payload


def _require_string_field(job: dict[str, Any], field_name: str) -> str:
    value = job.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise InferenceRunnerError(f"Inference runner job field '{field_name}' must be a non-empty string.")
    return value.strip()


def _resolve_loader(job: dict[str, Any]) -> Any:
    family = _require_string_field(job, "family")
    node_id = _require_string_field(job, "node_id")
    loader = _PIPELINE_LOADERS.get((family, node_id))
    if loader is None:
        raise InferenceRunnerError(
            f"Unsupported inference backend for family '{family}' and node '{node_id}'"
        )
    return loader


def _instantiate_pipeline(loader: Any, *, job: dict[str, Any], torch_module: Any | None = None) -> Any:
    model_dir = _require_string_field(job, "model_dir")
    family = _require_string_field(job, "family")
    node_id = _require_string_field(job, "node_id")
    extension_id = job.get("extension_id") if isinstance(job.get("extension_id"), str) else ""
    resolved_loader = loader() if callable(loader) and not hasattr(loader, "from_pretrained") else loader
    if hasattr(resolved_loader, "from_pretrained"):
        attempts = build_diffusers_load_attempts(
            extension_id=extension_id,
            family=family,
            node_id=node_id,
            torch_module=torch_module,
        )
        last_error: Exception | None = None
        for _, load_kwargs in attempts:
            try:
                return resolved_loader.from_pretrained(model_dir, **load_kwargs)
            except Exception as exc:
                last_error = exc
                if load_kwargs and is_retryable_diffusers_load_error(exc):
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise InferenceRunnerError("Inference loader did not provide any load attempts.")
    if callable(resolved_loader):
        return resolved_loader(model_dir)
    raise InferenceRunnerError("Inference loader must expose from_pretrained(model_dir) or be callable.")


def _load_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _resolve_execution_device(*, torch_module: Any | None = None) -> str:
    resolved_torch = torch_module if torch_module is not None else _load_torch()
    if resolved_torch is None:
        return "cpu"

    cuda = getattr(resolved_torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda"

    backends = getattr(resolved_torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        return "mps"

    return "cpu"


def _place_pipeline_on_device(pipeline: Any, *, execution_device: str) -> Any:
    move_to_device = getattr(pipeline, "to", None)
    if callable(move_to_device):
        placed_pipeline = move_to_device(execution_device)
        return placed_pipeline if placed_pipeline is not None else pipeline
    return pipeline


def _seeded_generator(
    params: dict[str, Any],
    *,
    execution_device: str,
    torch_module: Any | None = None,
) -> Any:
    seed = params.get("seed")
    if seed is None:
        return None

    resolved_torch = torch_module if torch_module is not None else _load_torch()
    if resolved_torch is None:
        return None

    return resolved_torch.Generator(device=execution_device).manual_seed(int(seed))


def _open_source_image(source_image_path: str):
    from PIL import Image

    return Image.open(source_image_path)


def _build_pipeline_kwargs(job: dict[str, Any], *, execution_device: str) -> dict[str, Any]:
    params = job.get("params")
    if not isinstance(params, dict):
        raise InferenceRunnerError("Inference runner job field 'params' must be an object.")

    kwargs: dict[str, Any] = {"prompt": job.get("prompt")}
    if "negative_prompt" in job:
        kwargs["negative_prompt"] = job.get("negative_prompt")

    numeric_fields = {
        "steps": "num_inference_steps",
        "width": "width",
        "height": "height",
        "guidance_scale": "guidance_scale",
    }
    for source_name, target_name in numeric_fields.items():
        if source_name in params:
            kwargs[target_name] = params[source_name]

    generator = _seeded_generator(params, execution_device=execution_device)
    if generator is not None:
        kwargs["generator"] = generator

    source_image_path = job.get("source_image_path")
    if source_image_path is not None:
        if not isinstance(source_image_path, str) or not source_image_path.strip():
            raise InferenceRunnerError(
                "Inference runner job field 'source_image_path' must be a non-empty string when provided."
            )
        kwargs["image"] = _open_source_image(source_image_path)
        if "strength" in params:
            kwargs["strength"] = params["strength"]

    return kwargs


def _run_pipeline_with_liveness(
    pipeline: Any,
    *,
    pipeline_kwargs: dict[str, Any],
    stdout: TextIO | None = None,
    heartbeat_interval_seconds: float | None = None,
) -> Any:
    resolved_heartbeat_interval = heartbeat_interval_seconds or _RUNNING_INFERENCE_HEARTBEAT_SECONDS
    stop_heartbeats = threading.Event()

    def emit_heartbeats() -> None:
        while not stop_heartbeats.wait(resolved_heartbeat_interval):
            emit_event("log", stdout=stdout, message="Running inference heartbeat.")

    heartbeat_thread = threading.Thread(target=emit_heartbeats, daemon=True)
    heartbeat_thread.start()
    try:
        return pipeline(**pipeline_kwargs)
    finally:
        stop_heartbeats.set()
        heartbeat_thread.join(timeout=max(resolved_heartbeat_interval, 1.0))


def _emit_memory_event_for_stage(
    stage: str,
    *,
    extension_id: str,
    torch_module: Any | None,
    stdout: TextIO | None = None,
) -> None:
    if not should_emit_memory_events(extension_id=extension_id):
        return
    emit_event("memory", stdout=stdout, **collect_stage_memory_snapshot(stage=stage, torch_module=torch_module))


def run_child_job(job: dict[str, Any], *, stdout: TextIO | None = None) -> dict[str, Any]:
    model_dir = _require_string_field(job, "model_dir")
    output_path = _require_string_field(job, "output_path")
    family = _require_string_field(job, "family")
    node_id = _require_string_field(job, "node_id")
    extension_id = job.get("extension_id") if isinstance(job.get("extension_id"), str) else ""
    loader = _resolve_loader(job)
    torch_module = _load_torch()
    execution_device = _resolve_execution_device()
    _emit_stage_event("loading-pipeline", stdout=stdout)
    pipeline = _place_pipeline_on_device(
        _instantiate_pipeline(loader, job=job, torch_module=torch_module),
        execution_device=execution_device,
    )
    apply_post_load_memory_optimizations(pipeline=pipeline, extension_id=extension_id)
    _emit_memory_event_for_stage(
        "loading-pipeline",
        extension_id=extension_id,
        torch_module=torch_module,
        stdout=stdout,
    )
    _emit_stage_event("running-inference", stdout=stdout)
    result = _run_pipeline_with_liveness(
        pipeline,
        pipeline_kwargs=_build_pipeline_kwargs(job, execution_device=execution_device),
        stdout=stdout,
    )
    _emit_memory_event_for_stage(
        "running-inference",
        extension_id=extension_id,
        torch_module=torch_module,
        stdout=stdout,
    )

    images = getattr(result, "images", None)
    if not isinstance(images, list) or not images:
        raise InferenceRunnerError("Inference pipeline did not return any images.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _emit_stage_event("saving-output", stdout=stdout)
    images[0].save(str(output_file))
    _emit_memory_event_for_stage(
        "saving-output",
        extension_id=extension_id,
        torch_module=torch_module,
        stdout=stdout,
    )

    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    return {
        "output_path": str(output_file),
        "metadata": {
            "family": family,
            "node_id": node_id,
            "seed": params.get("seed"),
            "negative_prompt_used": bool(job.get("negative_prompt")),
            "source_image_used": job.get("source_image_path") is not None,
        },
    }


def run_child_main(*, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    try:
        result = run_child_job(_read_job(stdin=stdin), stdout=stdout)
    except InferenceRunnerError as exc:
        return emit_error(str(exc), stdout=stdout)
    except Exception as exc:  # pragma: no cover - defensive boundary
        return emit_error(f"Unexpected inference runner failure: {exc}", stdout=stdout)

    emit_event("done", stdout=stdout, result=result)
    return 0


def main() -> int:
    return run_child_main()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())

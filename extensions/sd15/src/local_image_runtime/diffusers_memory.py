from __future__ import annotations

from typing import Any

try:
    import resource as _resource
except ModuleNotFoundError:  # pragma: no cover - exercised via Windows runtime.
    _resource = None


_OPTIMIZED_EXTENSION_IDS = {"sd15", "sdxl-base"}


def _resolved_float16(torch_module: Any | None) -> Any:
    if torch_module is None:
        return None
    return getattr(torch_module, "float16", None)


def build_diffusers_load_attempts(
    *,
    extension_id: str,
    family: str,
    node_id: str,
    torch_module: Any | None = None,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    del family, node_id
    if extension_id not in _OPTIMIZED_EXTENSION_IDS:
        return (("baseline", {}),)

    torch_dtype = _resolved_float16(torch_module)
    optimized_kwargs: dict[str, Any] = {
        "variant": "fp16",
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        optimized_kwargs["torch_dtype"] = torch_dtype

    return (
        ("optimized-fp16", dict(optimized_kwargs)),
        ("optimized-no-variant", {key: value for key, value in optimized_kwargs.items() if key != "variant"}),
        (
            "optimized-no-safetensors",
            {key: value for key, value in optimized_kwargs.items() if key not in {"variant", "use_safetensors"}},
        ),
        (
            "optimized-no-low-cpu-mem",
            {key: value for key, value in optimized_kwargs.items() if key not in {"variant", "use_safetensors", "low_cpu_mem_usage"}},
        ),
        ("baseline", {}),
    )


def is_retryable_diffusers_load_error(error: Exception) -> bool:
    message = str(error).lower()
    if isinstance(error, TypeError):
        return "unexpected keyword argument" in message
    if isinstance(error, ValueError):
        return "variant" in message or "fp16" in message
    if isinstance(error, OSError):
        return "variant" in message or "fp16" in message
    return False


def apply_post_load_memory_optimizations(*, pipeline: Any, extension_id: str) -> None:
    if extension_id not in _OPTIMIZED_EXTENSION_IDS:
        return

    enable_attention_slicing = getattr(pipeline, "enable_attention_slicing", None)
    if callable(enable_attention_slicing):
        enable_attention_slicing("auto")

    enable_vae_slicing = getattr(pipeline, "enable_vae_slicing", None)
    if callable(enable_vae_slicing):
        enable_vae_slicing()


def should_emit_memory_events(*, extension_id: str) -> bool:
    return extension_id in _OPTIMIZED_EXTENSION_IDS


def collect_stage_memory_snapshot(*, stage: str, torch_module: Any | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "stage": stage,
    }
    if _resource is not None:
        snapshot["rss_mib"] = round(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)

    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        memory_allocated = getattr(cuda, "memory_allocated", None)
        memory_reserved = getattr(cuda, "memory_reserved", None)
        if callable(memory_allocated):
            snapshot["cuda_allocated_mib"] = round(float(memory_allocated()) / (1024.0 * 1024.0), 2)
        if callable(memory_reserved):
            snapshot["cuda_reserved_mib"] = round(float(memory_reserved()) / (1024.0 * 1024.0), 2)

    return snapshot

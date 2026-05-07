from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .quality_policy import get_node_defaults, get_node_help


@dataclass(frozen=True)
class ExtensionDescriptor:
    extension_id: str
    label: str
    tier: str
    family: str
    dependency_family: str
    backend: str
    hf_repo: str
    description: str
    readiness_imports: tuple[str, ...] = field(default_factory=tuple)
    supported_nodes: tuple[str, ...] = field(default_factory=tuple)
    required_paths: tuple[str, ...] = field(default_factory=tuple)
    node_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_help: dict[str, dict[str, str]] = field(default_factory=dict)
    node_weight_specs: dict[str, dict[str, str]] = field(default_factory=dict)
    optional_feature_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    legacy_model_ids: tuple[str, ...] = field(default_factory=tuple)


EXTENSION_DESCRIPTORS = (
    ExtensionDescriptor(
        extension_id="sd15",
        label="Stable Diffusion 1.5",
        tier="baseline",
        family="stable-diffusion",
        dependency_family="sd15",
        backend="local-image-v1",
        hf_repo="runwayml/stable-diffusion-v1-5",
        description="Baseline latent diffusion model suited to lightweight local experiments.",
        readiness_imports=(
            "torch",
            "torchvision",
            "numpy",
            "PIL",
            "diffusers",
            "transformers",
            "accelerate",
            "huggingface_hub",
            "safetensors",
            "sentencepiece",
            "scipy",
        ),
        supported_nodes=("text-to-image", "image-to-image"),
        required_paths=(
            "model_index.json",
            "scheduler",
            "text_encoder",
            "tokenizer",
            "unet",
            "vae",
        ),
        node_defaults={
            "text-to-image": get_node_defaults("sd15", "text-to-image"),
            "image-to-image": get_node_defaults("sd15", "image-to-image"),
        },
        node_help={
            "text-to-image": get_node_help("sd15", "text-to-image"),
            "image-to-image": get_node_help("sd15", "image-to-image"),
        },
        node_weight_specs={
            "text-to-image": {
                "hf_repo": "runwayml/stable-diffusion-v1-5",
                "download_check": "model_index.json",
            },
            "image-to-image": {
                "hf_repo": "runwayml/stable-diffusion-v1-5",
                "download_check": "model_index.json",
            },
        },
        optional_feature_specs={
            "sd15_ip_adapter_style": {
                "label": "SD1.5 IP-Adapter style reference",
                "family": "stable-diffusion",
                "node_id": "image-to-image",
                "supported": False,
                "unsupported_reason": (
                    "SD1.5 style reference is not supported because compatible IP-Adapter adapter, "
                    "image encoder assets, and target-platform smoke evidence are not verified yet."
                ),
            },
            "sd15_controlnet_canny": {
                "label": "SD1.5 ControlNet Canny structural control",
                "family": "stable-diffusion",
                "node_id": "sd15-controlnet-canny-image-to-image",
                "supported": False,
                "unsupported_reason": (
                    "ControlNet Canny is planned as a separate per-control bundle node, not a base "
                    "image-to-image parameter. Preprocessor dependencies, model assets, platform gates, "
                    "and target-platform smoke evidence are not verified yet."
                ),
                "explicit_node_required": True,
                "node_strategy": "per-control explicit nodes",
                "dependency_group": "sd15_controlnet_preprocessors",
            },
        },
        legacy_model_ids=("sd15",),
    ),
    ExtensionDescriptor(
        extension_id="sdxl-base",
        label="SDXL Base",
        tier="quality",
        family="sdxl",
        dependency_family="sdxl-base",
        backend="local-image-v1",
        hf_repo="stabilityai/stable-diffusion-xl-base-1.0",
        description="Higher-capacity SDXL base model scaffold entry for local generation.",
        readiness_imports=(
            "torch",
            "torchvision",
            "numpy",
            "PIL",
            "diffusers",
            "transformers",
            "accelerate",
            "huggingface_hub",
            "safetensors",
            "sentencepiece",
            "scipy",
        ),
        supported_nodes=("text-to-image", "image-to-image"),
        required_paths=(
            "model_index.json",
            "scheduler",
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
            "unet",
            "vae",
        ),
        node_defaults={
            "text-to-image": get_node_defaults("sdxl-base", "text-to-image"),
            "image-to-image": get_node_defaults("sdxl-base", "image-to-image"),
        },
        node_help={
            "text-to-image": get_node_help("sdxl-base", "text-to-image"),
            "image-to-image": get_node_help("sdxl-base", "image-to-image"),
        },
        node_weight_specs={
            "text-to-image": {
                "hf_repo": "stabilityai/stable-diffusion-xl-base-1.0",
                "download_check": "model_index.json",
            },
            "image-to-image": {
                "hf_repo": "stabilityai/stable-diffusion-xl-base-1.0",
                "download_check": "model_index.json",
            },
        },
        optional_feature_specs={
            "sdxl_ip_adapter_style": {
                "label": "SDXL IP-Adapter style reference",
                "family": "sdxl",
                "node_id": "image-to-image",
                "hf_repo": "h94/IP-Adapter",
                "download_check": "sdxl_models/ip-adapter_sdxl.bin",
                "required_files": (
                    "sdxl_models/ip-adapter_sdxl.bin",
                    "sdxl_models/image_encoder/config.json",
                    "sdxl_models/image_encoder/model.safetensors",
                ),
                "allow_patterns": (
                    "sdxl_models/ip-adapter_sdxl.bin",
                    "sdxl_models/image_encoder/config.json",
                    "sdxl_models/image_encoder/model.safetensors",
                ),
            },
            "sdxl_controlnet_canny": {
                "label": "SDXL ControlNet Canny structural control",
                "family": "sdxl",
                "node_id": "sdxl-controlnet-canny-image-to-image",
                "supported": False,
                "unsupported_reason": (
                    "ControlNet Canny is planned as a separate per-control bundle node, not a base "
                    "image-to-image parameter. Preprocessor dependencies, model assets, platform gates, "
                    "and target-platform smoke evidence are not verified yet."
                ),
                "explicit_node_required": True,
                "node_strategy": "per-control explicit nodes",
                "dependency_group": "sdxl-base_controlnet_preprocessors",
            },
        },
        legacy_model_ids=("sdxl-base",),
    ),
    ExtensionDescriptor(
        extension_id="flux-schnell",
        label="FLUX Schnell",
        tier="fast",
        family="flux",
        dependency_family="flux-schnell",
        backend="local-image-v1",
        hf_repo="black-forest-labs/FLUX.1-schnell",
        description="Fast FLUX scaffold entry reserved for future backend integration.",
        readiness_imports=(
            "torch",
            "torchvision",
            "numpy",
            "PIL",
            "diffusers",
            "transformers",
            "accelerate",
            "huggingface_hub",
            "safetensors",
            "sentencepiece",
            "google.protobuf",
        ),
        supported_nodes=("text-to-image",),
        required_paths=(
            "model_index.json",
            "scheduler",
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
            "transformer",
            "vae",
        ),
        node_defaults={
            "text-to-image": get_node_defaults("flux-schnell", "text-to-image"),
        },
        node_help={
            "text-to-image": get_node_help("flux-schnell", "text-to-image"),
        },
        node_weight_specs={
            "text-to-image": {
                "hf_repo": "black-forest-labs/FLUX.1-schnell",
                "download_check": "model_index.json",
            },
        },
        legacy_model_ids=("flux-schnell",),
    ),
)

EXTENSION_DESCRIPTORS_BY_ID = {
    descriptor.extension_id: descriptor for descriptor in EXTENSION_DESCRIPTORS
}
LEGACY_MODEL_TO_EXTENSION_ID = {
    legacy_model_id: descriptor.extension_id
    for descriptor in EXTENSION_DESCRIPTORS
    for legacy_model_id in descriptor.legacy_model_ids
}


def extension_metadata_map() -> dict[str, dict[str, Any]]:
    return {
        descriptor.extension_id: asdict(descriptor)
        for descriptor in EXTENSION_DESCRIPTORS
    }


def registered_extension_ids() -> tuple[str, ...]:
    return tuple(EXTENSION_DESCRIPTORS_BY_ID)


def get_extension_descriptor(extension_id: str) -> ExtensionDescriptor | None:
    return EXTENSION_DESCRIPTORS_BY_ID.get(extension_id)


def resolve_extension_id(value: str) -> str | None:
    if value in EXTENSION_DESCRIPTORS_BY_ID:
        return value
    return LEGACY_MODEL_TO_EXTENSION_ID.get(value)


def get_extension_descriptor_by_legacy_model_id(model_id: str) -> ExtensionDescriptor | None:
    extension_id = LEGACY_MODEL_TO_EXTENSION_ID.get(model_id)
    if extension_id is None:
        return None
    return get_extension_descriptor(extension_id)


def get_node_weight_specs(extension_id: str) -> dict[str, dict[str, str]]:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        raise ValueError(f"Unknown extension id '{extension_id}'.")
    return {
        node_id: {
            "hf_repo": str(spec.get("hf_repo", "")).strip(),
            "download_check": str(spec.get("download_check", "")).strip(),
        }
        for node_id, spec in descriptor.node_weight_specs.items()
        if isinstance(node_id, str) and node_id.strip() and isinstance(spec, dict)
    }


def _coerce_feature_paths(spec: dict[str, Any], key: str, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    raw_paths = spec.get(key)
    if raw_paths is None:
        return fallback
    if isinstance(raw_paths, str):
        raw_iterable: tuple[Any, ...] = (raw_paths,)
    elif isinstance(raw_paths, (list, tuple)):
        raw_iterable = tuple(raw_paths)
    else:
        return fallback
    paths = tuple(str(path).strip() for path in raw_iterable if isinstance(path, str) and path.strip())
    return paths or fallback


def get_optional_feature_specs(extension_id: str) -> dict[str, dict[str, Any]]:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        raise ValueError(f"Unknown extension id '{extension_id}'.")
    specs: dict[str, dict[str, Any]] = {}
    for feature_id, spec in descriptor.optional_feature_specs.items():
        if not (isinstance(feature_id, str) and feature_id.strip() and isinstance(spec, dict)):
            continue
        download_check = str(spec.get("download_check", "")).strip()
        required_files = _coerce_feature_paths(
            spec,
            "required_files",
            fallback=(download_check,) if download_check else (),
        )
        allow_patterns = _coerce_feature_paths(spec, "allow_patterns", fallback=required_files)
        specs[feature_id] = {
            "label": str(spec.get("label", "")).strip(),
            "family": str(spec.get("family", "")).strip(),
            "node_id": str(spec.get("node_id", "")).strip(),
            "supported": bool(spec.get("supported", True)),
            "unsupported_reason": str(spec.get("unsupported_reason", "")).strip(),
            "explicit_node_required": bool(spec.get("explicit_node_required", False)),
            "node_strategy": str(spec.get("node_strategy", "")).strip(),
            "dependency_group": str(spec.get("dependency_group", "")).strip(),
            "hf_repo": str(spec.get("hf_repo", "")).strip(),
            "download_check": download_check,
            "required_files": required_files,
            "allow_patterns": allow_patterns,
        }
    return specs


def missing_required_paths(extension_id: str, source_dir: str | Path | Any) -> tuple[str, ...]:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        raise ValueError(f"Unknown extension id '{extension_id}'.")

    root = Path(source_dir)
    return tuple(path for path in descriptor.required_paths if not (root / path).exists())

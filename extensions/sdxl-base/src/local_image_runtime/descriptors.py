from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
            "text-to-image": {
                "width": 512,
                "height": 512,
                "steps": 30,
                "guidance_scale": 7.5,
            },
            "image-to-image": {
                "width": 512,
                "height": 512,
                "steps": 30,
                "guidance_scale": 7.5,
                "strength": 0.75,
            },
        },
        node_help={
            "text-to-image": {
                "prompt": "Prompt text to generate from.",
                "guidance_scale": "How strongly generation follows the prompt.",
            },
            "image-to-image": {
                "prompt": "Optional prompt describing the desired edit target.",
                "strength": "How strongly the source image should be transformed.",
            },
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
            "text-to-image": {
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 5.0,
            },
            "image-to-image": {
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 5.0,
                "strength": 0.7,
            },
        },
        node_help={
            "text-to-image": {
                "prompt": "Prompt text for SDXL base generation.",
                "guidance_scale": "Lower values are often enough for SDXL base.",
            },
            "image-to-image": {
                "prompt": "Optional text guidance for the edit pass.",
                "strength": "Controls how much the source image is changed.",
            },
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
            "text-to-image": {
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance_scale": 0.0,
            },
        },
        node_help={
            "text-to-image": {
                "prompt": "Prompt text for FLUX Schnell text-to-image generation.",
                "guidance_scale": "Reserved for backend compatibility; scaffold defaults to 0.0.",
            },
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


def missing_required_paths(extension_id: str, source_dir: str | Path | Any) -> tuple[str, ...]:
    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        raise ValueError(f"Unknown extension id '{extension_id}'.")

    root = Path(source_dir)
    return tuple(path for path in descriptor.required_paths if not (root / path).exists())

from __future__ import annotations

from typing import Any


_QUALITY_POLICY: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    "sd15": {
        "text-to-image": {
            "defaults": {
                "width": 512,
                "height": 512,
                "steps": 30,
                "guidance_scale": 7.5,
                "negative_prompt": "blurry, low quality, bad anatomy, deformed, extra digits",
            },
            "help": {
                "prompt": "Prompt text to generate from.",
                "negative_prompt": "Quality preset applied by default; set an empty string to disable it explicitly.",
                "guidance_scale": "How strongly generation follows the prompt.",
            },
        },
        "image-to-image": {
            "defaults": {
                "width": 512,
                "height": 512,
                "steps": 30,
                "guidance_scale": 7.5,
                "strength": 0.75,
                "negative_prompt": "blurry, low quality, bad anatomy, deformed, extra digits",
            },
            "help": {
                "prompt": "Optional prompt describing the desired edit target.",
                "negative_prompt": "Quality preset applied by default; set an empty string to disable it explicitly.",
                "strength": "How strongly the source image should be transformed.",
            },
        },
    },
    "sdxl-base": {
        "text-to-image": {
            "defaults": {
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 5.0,
                "negative_prompt": "blurry, low quality, distorted, artifacts",
            },
            "help": {
                "prompt": "Prompt text for SDXL base generation.",
                "negative_prompt": "Quality preset applied by default; set an empty string to disable it explicitly.",
                "guidance_scale": "Lower values are often enough for SDXL base.",
            },
        },
        "image-to-image": {
            "defaults": {
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 5.0,
                "strength": 0.7,
                "negative_prompt": "blurry, low quality, distorted, artifacts",
            },
            "help": {
                "prompt": "Optional text guidance for the edit pass.",
                "negative_prompt": "Quality preset applied by default; set an empty string to disable it explicitly.",
                "strength": "Controls how much the source image is changed.",
            },
        },
    },
}


def _node_policy(extension_id: str, node_id: str) -> dict[str, dict[str, Any]] | None:
    family_policy = _QUALITY_POLICY.get(extension_id)
    if family_policy is None:
        return None
    return family_policy.get(node_id)


def get_node_defaults(extension_id: str, node_id: str) -> dict[str, Any]:
    policy = _node_policy(extension_id, node_id)
    if policy is None:
        return {}
    defaults = policy.get("defaults", {})
    return dict(defaults) if isinstance(defaults, dict) else {}


def get_node_help(extension_id: str, node_id: str) -> dict[str, str]:
    policy = _node_policy(extension_id, node_id)
    if policy is None:
        return {}
    help_text = policy.get("help", {})
    return dict(help_text) if isinstance(help_text, dict) else {}


def resolve_effective_params(*, extension_id: str, node_id: str, params: dict[str, Any]) -> dict[str, Any]:
    resolved_params = get_node_defaults(extension_id, node_id)
    resolved_params.update(params)
    return resolved_params

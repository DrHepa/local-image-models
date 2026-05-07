from __future__ import annotations

from typing import Any, TypeVar


_ErrorT = TypeVar("_ErrorT", bound=Exception)


def effective_denoising_steps(*, steps: int, strength: float) -> int:
    return int(steps * strength)


def validate_image_to_image_effective_denoising_steps(
    params: dict[str, Any],
    *,
    error_type: type[_ErrorT],
) -> None:
    steps = params.get("steps")
    strength = params.get("strength")
    if not isinstance(steps, int) or isinstance(steps, bool):
        return
    if not isinstance(strength, (int, float)) or isinstance(strength, bool):
        return

    effective_steps = effective_denoising_steps(steps=steps, strength=float(strength))
    if effective_steps >= 1:
        return

    raise error_type(
        "image-to-image params.steps and params.strength must produce at least one denoising/effective step; "
        f"got steps={steps}, strength={float(strength):g}, effective_steps={effective_steps}."
    )

from __future__ import annotations

from typing import Final


ProgressStep = tuple[int, str]

_BOOTSTRAP_STEPS: Final[tuple[ProgressStep, ...]] = (
    (5, "payload-received"),
    (20, "runtime-ready"),
)

_HOST_GENERATION_STEPS: Final[tuple[ProgressStep, ...]] = (
    (35, "validating-request"),
    (55, "checking-extension"),
    (75, "backend-dispatch"),
)

_CHILD_GENERATION_STEPS: Final[tuple[ProgressStep, ...]] = (
    (80, "loading-pipeline"),
    (90, "running-inference"),
    (95, "saving-output"),
)

_STEP_LOG_MESSAGES: Final[dict[str, str]] = {
    "loading-pipeline": "Loading inference pipeline.",
    "running-inference": "Running inference.",
    "saving-output": "Saving output image.",
}

_CANONICAL_GENERATION_STEPS: Final[tuple[ProgressStep, ...]] = (
    *_HOST_GENERATION_STEPS,
    *_CHILD_GENERATION_STEPS,
)


def bootstrap_steps() -> tuple[ProgressStep, ...]:
    return _BOOTSTRAP_STEPS


def host_generation_steps() -> tuple[ProgressStep, ...]:
    return _HOST_GENERATION_STEPS


def child_generation_steps() -> tuple[ProgressStep, ...]:
    return _CHILD_GENERATION_STEPS


def canonical_generation_steps() -> tuple[ProgressStep, ...]:
    return _CANONICAL_GENERATION_STEPS


def step_log_message(label: str) -> str:
    normalized_label = label.strip()
    if normalized_label in _STEP_LOG_MESSAGES:
        return _STEP_LOG_MESSAGES[normalized_label]
    return normalized_label.replace("-", " ").capitalize() + "."

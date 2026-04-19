from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


LINUX_ARM64_MACHINES = frozenset({"aarch64", "arm64"})
SUPPORTED_SYSTEM = "linux"

_TORCH_WHEELS = {
    "cu124": {
        "cp39": {
            "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp39-cp39-linux_aarch64.whl#sha256=012887a6190e562cb266d2210052c5deb5113f520a46dc2beaa57d76144a0e9b",
            "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp39-cp39-linux_aarch64.whl#sha256=e25b4ac3c9eec3f789f1c5491331dfe236b5f06a1f406ea82fa59fed4fc6f71e",
        },
        "cp310": {
            "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp310-cp310-linux_aarch64.whl#sha256=d468d0eddc188aa3c1e417ec24ce615c48c0c3f592b0354d9d3b99837ef5faa6",
            "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp310-cp310-linux_aarch64.whl#sha256=38765e53653f93e529e329755992ddbea81091aacedb61ed053f6a14efb289e5",
        },
        "cp311": {
            "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp311-cp311-linux_aarch64.whl#sha256=e080353c245b752cd84122e4656261eee6d4323a37cfb7d13e0fffd847bae1a3",
            "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp311-cp311-linux_aarch64.whl#sha256=2c5350a08abe005a16c316ae961207a409d0e35df86240db5f77ec41345c82f3",
        },
        "cp312": {
            "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp312-cp312-linux_aarch64.whl#sha256=302041d457ee169fd925b53da283c13365c6de75c6bb3e84130774b10e2fbb39",
            "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp312-cp312-linux_aarch64.whl#sha256=3e3289e53d0cb5d1b7f55b3f5912f46a08293c6791585ba2fc32c12cded9f9af",
        },
    },
    "cu128": {
        "cp39": {
            "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp39-cp39-manylinux_2_28_aarch64.whl#sha256=2f155388b1200e08f3e901bb3487ff93ca6d63cde87c29b97bb6762a8f63b373",
            "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp39-cp39-manylinux_2_28_aarch64.whl#sha256=7a398fad02f4ac6b7d18bea9a08dc14163ffc5a368618f29ceb0e53dfa91f69e",
        },
        "cp310": {
            "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=b1f0cdd0720ad60536deb5baa427b782fd920dd4fcf72e244d32974caafa3b9e",
            "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=566224d7b4f00bc6366bed1d62f834ca80f8e57fe41e10e4a5636bfa3ffb984e",
        },
        "cp311": {
            "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=47c895bcab508769d129d717a4b916b10225ae3855723aeec8dff8efe5346207",
            "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=6be714bcdd8849549571f6acfaa2dfa9e00676f042bda517432745fb116f7904",
        },
        "cp312": {
            "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6bba7dca5d9a729f1e8e9befb98055498e551efaf5ed034824c168b560afc1ac",
            "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6e9752b48c1cdd7f6428bcd30c3d198b30ecea348d16afb651f95035e5252506",
        },
        "cp313": {
            "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp313-cp313-manylinux_2_28_aarch64.whl#sha256=633f35e8b1b1f640ef5f8a98dbd84f19b548222ce7ba8f017fe47ce6badc106a",
            "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp313-cp313-manylinux_2_28_aarch64.whl#sha256=e4d4d5a14225875d9bf8c5221d43d8be97786adc498659493799bdeff52c54cf",
        },
    },
}


@dataclass(frozen=True)
class DependencyInstallStep:
    name: str
    packages: tuple[str, ...]
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class DependencyPlan:
    extension_id: str
    dependency_family: str
    platform_system: str
    platform_machine: str
    python_tag: str
    cuda_variant: str
    shared_steps: tuple[DependencyInstallStep, ...]
    family_steps: tuple[DependencyInstallStep, ...]
    readiness_imports: tuple[str, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> str:
        return (
            f"{self.extension_id}: {self.platform_system}/{self.platform_machine}, "
            f"{self.python_tag}, {self.cuda_variant}"
        )


class DependencyPlanError(RuntimeError):
    """Raised when no honest dependency plan exists for the requested target."""


def _normalize_system(value: str | None) -> str:
    return (value or "").strip().lower()


def _normalize_machine(value: str | None) -> str:
    return (value or "").strip().lower()


def _normalize_cuda_digits(value: str | int | float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = str(int(value)) if isinstance(value, float) else str(value)
    elif isinstance(value, str):
        numeric = "".join(character for character in value if character.isdigit())
    else:
        return None
    return numeric or None


def python_tag_from_interpreter(python_exe: str | Path) -> str:
    resolved_python = Path(python_exe).expanduser()
    try:
        completed = subprocess.run(
            [
                str(resolved_python),
                "-c",
                "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise DependencyPlanError(
            f"Could not inspect python interpreter '{resolved_python}': {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise DependencyPlanError(
            f"Could not inspect python interpreter '{resolved_python}': {detail}"
        ) from exc

    python_tag = completed.stdout.strip()
    if not python_tag.startswith("cp"):
        raise DependencyPlanError(
            f"Unsupported python ABI reported by '{resolved_python}': {python_tag or '<empty>'}."
        )
    return python_tag


def _select_cuda_variant(cuda_version: str | int | float | None) -> str:
    normalized = _normalize_cuda_digits(cuda_version)
    if normalized is None:
        raise DependencyPlanError(
            "Linux ARM64 installation requires cuda_version so the installer can select a verified wheel set (cu124 or cu128)."
        )

    cuda_value = int(normalized)
    if cuda_value >= 128:
        return "cu128"
    if cuda_value >= 124:
        return "cu124"
    raise DependencyPlanError(
        f"Unsupported/unverified CUDA version '{cuda_version}' for Linux ARM64. Verified variants are cu124 and cu128 only."
    )


def _torch_step(cuda_variant: str, python_tag: str) -> DependencyInstallStep:
    wheel_map = _TORCH_WHEELS[cuda_variant].get(python_tag)
    if wheel_map is None:
        supported_tags = ", ".join(sorted(_TORCH_WHEELS[cuda_variant]))
        raise DependencyPlanError(
            f"No verified Linux ARM64 PyTorch wheels for {cuda_variant} and Python ABI '{python_tag}'. Supported ABI tags: {supported_tags}."
        )
    return DependencyInstallStep(
        name="install_shared_torch",
        packages=(wheel_map["torch"], wheel_map["torchvision"]),
        extra_args=("--no-cache-dir",),
    )


def _shared_runtime_steps(cuda_variant: str, python_tag: str) -> tuple[DependencyInstallStep, ...]:
    return (
        _torch_step(cuda_variant, python_tag),
        DependencyInstallStep(
            name="install_shared_runtime",
            packages=(
                "Pillow",
                "numpy",
                "huggingface_hub",
                "safetensors",
                "accelerate",
            ),
        ),
    )


def _family_steps(dependency_family: str) -> tuple[DependencyInstallStep, ...]:
    family_matrix = {
        "sd15": (
            DependencyInstallStep(
                name="install_family_dependencies",
                packages=(
                    "diffusers==0.35.1",
                    "transformers>=4.46,<5",
                    "sentencepiece",
                    "scipy",
                ),
            ),
        ),
        "sdxl-base": (
            DependencyInstallStep(
                name="install_family_dependencies",
                packages=(
                    "diffusers==0.35.1",
                    "transformers>=4.46,<5",
                    "sentencepiece",
                    "scipy",
                ),
            ),
        ),
        "flux-schnell": (
            DependencyInstallStep(
                name="install_family_dependencies",
                packages=(
                    "diffusers==0.35.1",
                    "transformers>=4.46,<5",
                    "sentencepiece",
                    "protobuf<6",
                ),
            ),
        ),
    }
    steps = family_matrix.get(dependency_family)
    if steps is None:
        supported = ", ".join(sorted(family_matrix))
        raise DependencyPlanError(
            f"Unknown dependency family '{dependency_family}'. Supported families: {supported}."
        )
    return steps


def resolve_dependency_plan(
    *,
    extension_id: str,
    dependency_family: str,
    readiness_imports: Iterable[str],
    platform_info: dict[str, str],
    python_tag: str,
    cuda_version: str | int | float | None,
) -> DependencyPlan:
    system = _normalize_system(platform_info.get("system"))
    machine = _normalize_machine(platform_info.get("machine"))
    if system != SUPPORTED_SYSTEM or machine not in LINUX_ARM64_MACHINES:
        raise DependencyPlanError(
            "Unsupported/unverified install target detected: "
            f"system='{system or 'unknown'}', machine='{machine or 'unknown'}'. "
            "This change currently guarantees installation ONLY on Linux ARM64 (linux + aarch64/arm64)."
        )

    cuda_variant = _select_cuda_variant(cuda_version)
    return DependencyPlan(
        extension_id=extension_id,
        dependency_family=dependency_family,
        platform_system=system,
        platform_machine=machine,
        python_tag=python_tag,
        cuda_variant=cuda_variant,
        shared_steps=_shared_runtime_steps(cuda_variant, python_tag),
        family_steps=_family_steps(dependency_family),
        readiness_imports=tuple(module for module in readiness_imports if isinstance(module, str) and module.strip()),
    )


def pip_install_command(
    *,
    venv_python: str | Path,
    packages: Iterable[str],
    extra_args: Iterable[str] = (),
) -> list[str]:
    resolved_packages = [package for package in packages if isinstance(package, str) and package.strip()]
    if not resolved_packages:
        raise DependencyPlanError("Cannot build pip install command without at least one package.")
    command = [str(Path(venv_python).expanduser()), "-m", "pip", "install", *extra_args, *resolved_packages]
    return command


def current_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"

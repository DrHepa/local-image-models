from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable


LINUX_ARM64_MACHINES = frozenset({"aarch64", "arm64"})
SUPPORTED_SYSTEM = "linux"
PLAN_STATE_VERIFIED = "verified"
PLAN_STATE_CANDIDATE_INSTALL = "candidate_install"
PLAN_STATE_SETUP_NEEDED = "setup_needed"
PLAN_STATE_UNSUPPORTED = "unsupported"
PLAN_STATE_UNVERIFIED = "unverified"
WINDOWS_AMD64_MACHINES = frozenset({"amd64", "x86_64"})
SD15_WINDOWS_PLATFORM_KEY = "windows-amd64"
SD15_WINDOWS_PYTHON_TAG = "cp312"
SD15_WINDOWS_CUDA_VARIANT = "cu128"
SD15_WINDOWS_TORCH_VERSION = "2.7.0"
SD15_WINDOWS_TORCHVISION_VERSION = "0.22.0"
SD15_WINDOWS_MODEL_REPO = "runwayml/stable-diffusion-v1-5"
_WINDOWS_PLAN_STATES = {
    "sd15": PLAN_STATE_CANDIDATE_INSTALL,
    "sdxl-base": PLAN_STATE_CANDIDATE_INSTALL,
    "flux-schnell": PLAN_STATE_UNSUPPORTED,
}
_PYPI_INDEX_URL = "https://pypi.org/simple"
_TORCH_EXTRA_INDEX_URLS = {
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu128": "https://download.pytorch.org/whl/cu128",
}

_SD15_WINDOWS_REQUIRED_EVIDENCE_FIELDS = (
    "extension_id",
    "status",
    "reviewed",
    "platform_key",
    "os_name",
    "os_version",
    "os_build",
    "machine",
    "python_version",
    "python_abi",
    "sysconfig_platform",
    "pip_version",
    "gpu_name",
    "nvidia_driver",
    "torch_cuda_available",
    "torch_version",
    "torchvision_version",
    "torch_cuda_version",
    "cuda_variant",
    "import_results",
    "model_layout",
    "model_repo",
    "model_load",
    "smoke_inference",
    "timestamp",
    "operator",
    "tool_version",
    "failure_diagnostics",
)

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
    plan_state: str = PLAN_STATE_VERIFIED
    platform_key: str = "linux-aarch64"
    platform_supported: bool = True
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

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


def normalize_platform_key(platform_info: dict[str, str]) -> str:
    system = _normalize_system(platform_info.get("system")) or "unknown"
    machine = _normalize_machine(platform_info.get("machine")) or "unknown"
    if system == "windows" and machine in WINDOWS_AMD64_MACHINES:
        machine = "amd64"
    return f"{system}-{machine}"


def _unsupported_plan_diagnostic(*, system: str, machine: str) -> str:
    return (
        "Unsupported/unverified install target detected: "
        f"system='{system or 'unknown'}', machine='{machine or 'unknown'}'. "
        "This change currently guarantees installation ONLY on Linux ARM64 (linux + aarch64/arm64)."
    )


def _windows_diagnostic(*, extension_id: str, plan_state: str) -> str:
    if plan_state == PLAN_STATE_CANDIDATE_INSTALL:
        if extension_id == "sdxl-base":
            return (
                "SDXL Windows dependency setup for 'sdxl-base' is candidate_install: first-pass, "
                "unverified dependency installation is enabled only for windows-amd64, Python cp312, "
                "and CUDA cu128 to gather real pip, import, model-load, text-to-image, and Preview Image "
                "evidence. This is not verified compatibility or runtime readiness."
            )
        return (
            f"Windows dependency setup for '{extension_id}' is candidate_install: first-pass, "
            "experimental SD15 Windows dependency installation is enabled only to gather real pip, "
            "import, model-load, and smoke-inference evidence. This is not verified compatibility "
            "or runtime readiness."
        )
    if plan_state == PLAN_STATE_SETUP_NEEDED:
        return (
            f"Windows dependency setup for '{extension_id}' is setup_needed: a candidate Windows GPU "
            "wheel/runtime matrix is visible, but setup stops before dependency installation until "
            "reviewed Windows evidence validates install, imports, model load, and smoke inference."
        )
    if plan_state == PLAN_STATE_UNVERIFIED:
        return (
            f"Windows dependency setup for '{extension_id}' is unverified: dependency installation is "
            "disabled until a Windows runtime matrix is validated."
        )
    return (
        f"Windows dependency setup for '{extension_id}' is unsupported by this bundle version. "
        "No dependency installation was attempted."
    )


def _sd15_windows_torch_step() -> DependencyInstallStep:
    return DependencyInstallStep(
        name="install_shared_torch",
        packages=(
            f"torch=={SD15_WINDOWS_TORCH_VERSION}",
            f"torchvision=={SD15_WINDOWS_TORCHVISION_VERSION}",
        ),
        extra_args=_torch_step_extra_args(SD15_WINDOWS_CUDA_VARIANT),
    )


def _sd15_windows_candidate_shared_steps() -> tuple[DependencyInstallStep, ...]:
    return (
        _sd15_windows_torch_step(),
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


def _default_sd15_windows_evidence_path() -> Path | None:
    candidate = Path(__file__).with_name("sd15-windows-evidence.json")
    return candidate if candidate.exists() else None


def _load_json_evidence(path: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    evidence_path = Path(path).expanduser()
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"Could not read SD15 Windows evidence '{evidence_path}': {exc}"
    except JSONDecodeError as exc:
        return None, f"SD15 Windows evidence is not valid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "SD15 Windows evidence must be a JSON object."
    return payload, None


def _dict_field_status(payload: dict[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        return None
    status = value.get("status")
    return status if isinstance(status, str) else None


def _validate_sd15_windows_evidence_payload(payload: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    diagnostics: list[str] = []
    for field_name in _SD15_WINDOWS_REQUIRED_EVIDENCE_FIELDS:
        if field_name not in payload:
            diagnostics.append(f"SD15 Windows evidence missing required evidence field '{field_name}'.")

    if "pip_freeze" not in payload and "pip_inspect" not in payload:
        diagnostics.append("SD15 Windows evidence must include pip_freeze or pip_inspect.")

    exact_values: dict[str, object] = {
        "extension_id": "sd15",
        "status": PLAN_STATE_VERIFIED,
        "reviewed": True,
        "platform_key": SD15_WINDOWS_PLATFORM_KEY,
        "python_abi": SD15_WINDOWS_PYTHON_TAG,
        "cuda_variant": SD15_WINDOWS_CUDA_VARIANT,
        "torch_version": SD15_WINDOWS_TORCH_VERSION,
        "torchvision_version": SD15_WINDOWS_TORCHVISION_VERSION,
        "model_repo": SD15_WINDOWS_MODEL_REPO,
        "torch_cuda_available": True,
    }
    for field_name, expected in exact_values.items():
        if payload.get(field_name) != expected:
            diagnostics.append(
                f"SD15 Windows evidence field '{field_name}' must be {expected!r}; got {payload.get(field_name)!r}."
            )

    if not str(payload.get("torch_cuda_version", "")).startswith("12.8"):
        diagnostics.append("SD15 Windows evidence field 'torch_cuda_version' must describe CUDA 12.8.")

    import_results = payload.get("import_results")
    if isinstance(import_results, dict):
        required_imports = ("torch", "torchvision", "diffusers", "transformers", "sentencepiece", "scipy")
        for module_name in required_imports:
            if import_results.get(module_name) != "ok":
                diagnostics.append(f"SD15 Windows evidence import_results.{module_name} must be 'ok'.")
    elif "import_results" in payload:
        diagnostics.append("SD15 Windows evidence field 'import_results' must be an object.")

    for field_name in ("model_load", "smoke_inference"):
        if field_name in payload and _dict_field_status(payload, field_name) != "ok":
            diagnostics.append(f"SD15 Windows evidence field '{field_name}.status' must be 'ok'.")

    model_layout = payload.get("model_layout")
    if isinstance(model_layout, dict):
        if model_layout.get("model_index.json") != "present":
            diagnostics.append("SD15 Windows evidence model_layout.model_index.json must be 'present'.")
    elif "model_layout" in payload:
        diagnostics.append("SD15 Windows evidence field 'model_layout' must be an object.")

    return not diagnostics, tuple(diagnostics)


def _validate_sd15_windows_evidence(evidence_path: str | Path | None) -> tuple[bool, tuple[str, ...]]:
    resolved_path = Path(evidence_path).expanduser() if evidence_path is not None else _default_sd15_windows_evidence_path()
    if resolved_path is None:
        return False, ("No reviewed SD15 Windows evidence artifact is bundled; candidate remains candidate_install.",)
    payload, load_error = _load_json_evidence(resolved_path)
    if load_error is not None:
        return False, (load_error,)
    assert payload is not None
    return _validate_sd15_windows_evidence_payload(payload)


def _sd15_windows_plan(
    *,
    readiness_imports: Iterable[str],
    machine: str,
    python_tag: str,
    evidence_path: str | Path | None,
) -> DependencyPlan:
    evidence_ok, evidence_diagnostics = (
        _validate_sd15_windows_evidence(evidence_path) if evidence_path is not None else (False, ())
    )
    plan_state = PLAN_STATE_VERIFIED if evidence_ok else PLAN_STATE_CANDIDATE_INSTALL
    diagnostics = (
        ()
        if evidence_ok
        else (
            _windows_diagnostic(extension_id="sd15", plan_state=plan_state),
            *evidence_diagnostics,
        )
    )
    return DependencyPlan(
        extension_id="sd15",
        dependency_family="sd15",
        platform_system="windows",
        platform_machine="amd64" if machine in WINDOWS_AMD64_MACHINES else machine,
        python_tag=python_tag,
        cuda_variant=SD15_WINDOWS_CUDA_VARIANT,
        shared_steps=_sd15_windows_candidate_shared_steps(),
        family_steps=_family_steps("sd15"),
        readiness_imports=tuple(module for module in readiness_imports if isinstance(module, str) and module.strip()),
        plan_state=plan_state,
        platform_key=SD15_WINDOWS_PLATFORM_KEY,
        platform_supported=evidence_ok,
        diagnostics=diagnostics,
    )


def _sdxl_windows_plan(
    *,
    readiness_imports: Iterable[str],
    machine: str,
    python_tag: str,
) -> DependencyPlan:
    return DependencyPlan(
        extension_id="sdxl-base",
        dependency_family="sdxl-base",
        platform_system="windows",
        platform_machine="amd64" if machine in WINDOWS_AMD64_MACHINES else machine,
        python_tag=python_tag,
        cuda_variant=SD15_WINDOWS_CUDA_VARIANT,
        shared_steps=_sd15_windows_candidate_shared_steps(),
        family_steps=_family_steps("sdxl-base"),
        readiness_imports=tuple(module for module in readiness_imports if isinstance(module, str) and module.strip()),
        plan_state=PLAN_STATE_CANDIDATE_INSTALL,
        platform_key=SD15_WINDOWS_PLATFORM_KEY,
        platform_supported=False,
        diagnostics=(_windows_diagnostic(extension_id="sdxl-base", plan_state=PLAN_STATE_CANDIDATE_INSTALL),),
    )


def _diagnostic_plan(
    *,
    extension_id: str,
    dependency_family: str,
    readiness_imports: Iterable[str],
    system: str,
    machine: str,
    python_tag: str,
    cuda_version: str | int | float | None,
    plan_state: str,
    diagnostics: tuple[str, ...],
) -> DependencyPlan:
    return DependencyPlan(
        extension_id=extension_id,
        dependency_family=dependency_family,
        platform_system=system,
        platform_machine=machine,
        python_tag=python_tag,
        cuda_variant=str(cuda_version or "unverified"),
        shared_steps=(),
        family_steps=(),
        readiness_imports=tuple(module for module in readiness_imports if isinstance(module, str) and module.strip()),
        plan_state=plan_state,
        platform_key=normalize_platform_key({"system": system, "machine": machine}),
        platform_supported=False,
        diagnostics=diagnostics,
    )


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
        extra_args=_torch_step_extra_args(cuda_variant),
    )


def _torch_step_extra_args(cuda_variant: str) -> tuple[str, ...]:
    return (
        "--index-url",
        _PYPI_INDEX_URL,
        "--extra-index-url",
        _TORCH_EXTRA_INDEX_URLS[cuda_variant],
        "--no-cache-dir",
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
    evidence_path: str | Path | None = None,
) -> DependencyPlan:
    system = _normalize_system(platform_info.get("system"))
    machine = _normalize_machine(platform_info.get("machine"))
    platform_key = normalize_platform_key({"system": system, "machine": machine})
    if system == "windows":
        if extension_id == "sd15" and platform_key == SD15_WINDOWS_PLATFORM_KEY:
            return _sd15_windows_plan(
                readiness_imports=readiness_imports,
                machine=machine,
                python_tag=python_tag,
                evidence_path=evidence_path,
            )
        windows_cuda_variant: str | None = None
        try:
            windows_cuda_variant = _select_cuda_variant(cuda_version)
        except DependencyPlanError:
            windows_cuda_variant = None
        if (
            extension_id == "sdxl-base"
            and dependency_family == "sdxl-base"
            and platform_key == SD15_WINDOWS_PLATFORM_KEY
            and windows_cuda_variant == SD15_WINDOWS_CUDA_VARIANT
        ):
            return _sdxl_windows_plan(
                readiness_imports=readiness_imports,
                machine=machine,
                python_tag=python_tag,
            )
        plan_state = _WINDOWS_PLAN_STATES.get(extension_id, PLAN_STATE_UNVERIFIED)
        return _diagnostic_plan(
            extension_id=extension_id,
            dependency_family=dependency_family,
            readiness_imports=readiness_imports,
            system=system,
            machine="amd64" if machine in WINDOWS_AMD64_MACHINES else machine,
            python_tag=python_tag,
            cuda_version=cuda_version,
            plan_state=plan_state,
            diagnostics=(_windows_diagnostic(extension_id=extension_id, plan_state=plan_state),),
        )

    if system != SUPPORTED_SYSTEM or machine not in LINUX_ARM64_MACHINES:
        return _diagnostic_plan(
            extension_id=extension_id,
            dependency_family=dependency_family,
            readiness_imports=readiness_imports,
            system=system,
            machine=machine,
            python_tag=python_tag,
            cuda_version=cuda_version,
            plan_state=PLAN_STATE_UNSUPPORTED,
            diagnostics=(_unsupported_plan_diagnostic(system=system, machine=machine),),
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
        plan_state=PLAN_STATE_VERIFIED,
        platform_key=platform_key,
        platform_supported=True,
        diagnostics=(),
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

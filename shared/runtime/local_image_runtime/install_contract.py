from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Sequence

from .bootstrap import (
    SETUP_STATUS_FAILED,
    SETUP_STATUS_INSTALLING,
    SETUP_STATUS_READY,
    SetupStatus,
    bootstrap_runtime,
    detect_platform,
    expected_venv_python,
    get_extension_record,
    persist_extension_setup,
    reevaluate_extension_setup,
)
from .dependencies import (
    DependencyInstallStep,
    DependencyPlan,
    DependencyPlanError,
    PLAN_STATE_CANDIDATE_INSTALL,
    PLAN_STATE_VERIFIED,
    _TORCH_EXTRA_INDEX_URLS,
    pip_install_command,
    python_tag_from_interpreter,
    resolve_dependency_plan,
)
from .descriptors import get_extension_descriptor


@dataclass(frozen=True)
class SetupPayload:
    python_exe: str
    ext_dir: str
    gpu_sm: str | None = None
    cuda_version: str | None = None


@dataclass(frozen=True)
class SetupStep:
    name: str
    status: str
    detail: str | None = None


@dataclass(frozen=True)
class SetupResult:
    status: SetupStatus
    extension_id: str
    venv_python: str | None
    platform: dict[str, str]
    steps: tuple[SetupStep, ...]
    diagnostics: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 0 if self.status != SETUP_STATUS_FAILED else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "extension_id": self.extension_id,
            "venv_python": self.venv_python,
            "platform": dict(self.platform),
            "steps": [asdict(step) for step in self.steps],
            "diagnostics": list(self.diagnostics),
        }


class SetupContractError(RuntimeError):
    """Raised when setup payload parsing or validation fails."""


def _non_empty_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_required_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise SetupContractError(
            f"Setup payload field '{field_name}' must be a non-empty string."
        )
    return value.strip()


def _coerce_optional_string(payload: dict[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise SetupContractError(
            f"Setup payload field '{field_name}' must be a string or number when provided."
        )
    if isinstance(value, (int, float)):
        stripped = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        return stripped or None
    if not isinstance(value, str):
        raise SetupContractError(
            f"Setup payload field '{field_name}' must be a string or number when provided."
        )
    stripped = value.strip()
    return stripped or None


def _step(name: str, status: str, detail: str | None) -> dict[str, str | None]:
    return {"name": name, "status": status, "detail": detail}


class SetupExecutionError(RuntimeError):
    def __init__(self, *, step_name: str, detail: str):
        super().__init__(detail)
        self.step_name = step_name
        self.detail = detail


def _command_failure_detail(exc: subprocess.CalledProcessError | OSError) -> str:
    if isinstance(exc, OSError):
        return str(exc)
    output = (exc.stderr or exc.stdout or str(exc)).strip()
    return output or str(exc)


def _run_checked(*, command: Sequence[str], step_name: str, cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            list(command),
            check=True,
            cwd=str(cwd) if cwd is not None else None,
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise SetupExecutionError(step_name=step_name, detail=_command_failure_detail(exc)) from exc


def _install_dependency_step(
    *, venv_python: Path, install_step: DependencyInstallStep, cwd: Path
) -> None:
    command = pip_install_command(
        venv_python=venv_python,
        packages=install_step.packages,
        extra_args=install_step.extra_args,
    )
    _run_checked(command=command, step_name=install_step.name, cwd=cwd)


def _persist_failed_result(
    *,
    snapshot: Any,
    extension_id: str,
    ext_dir: str | None,
    python_exe: str | None,
    steps: Sequence[dict[str, Any]],
    diagnostics: Sequence[str],
    platform_info: dict[str, str],
    setup_state: str | None = None,
    dependency_plan_state: str | None = None,
    platform_key: str | None = None,
    platform_supported: bool | None = None,
) -> SetupResult:
    failed_snapshot = persist_extension_setup(
        snapshot,
        extension_id,
        status=SETUP_STATUS_FAILED,
        ext_dir=ext_dir,
        python_exe=python_exe,
        venv_python=str(expected_venv_python(Path(ext_dir))) if ext_dir else None,
        steps=tuple(steps),
        diagnostics=tuple(diagnostics),
        platform_info=platform_info,
        setup_state=setup_state,
        dependency_plan_state=dependency_plan_state,
        platform_key=platform_key,
        platform_supported=platform_supported,
    )
    return _result_from_snapshot(extension_id, failed_snapshot)


def _detail_for_plan(plan: DependencyPlan) -> str:
    if plan.plan_state == PLAN_STATE_CANDIDATE_INSTALL:
        return (
            f"Selected first-pass experimental candidate dependency install for {plan.extension_id} "
            f"on {plan.platform_key}, Python {plan.python_tag}, {plan.cuda_variant}. "
            "This permits setup execution for evidence gathering only; it is not verified compatibility."
        )
    return (
        f"Selected verified dependency plan for {plan.platform_system}/{plan.platform_machine}, "
        f"Python {plan.python_tag}, {plan.cuda_variant}, family '{plan.dependency_family}'."
    )


def _candidate_install_allowed(plan: DependencyPlan) -> bool:
    return (
        plan.extension_id == "sd15"
        and plan.platform_key == "windows-amd64"
        and plan.platform_system == "windows"
        and plan.platform_machine == "amd64"
        and plan.cuda_variant == "cu128"
        and plan.plan_state == PLAN_STATE_CANDIDATE_INSTALL
        and bool((*plan.shared_steps, *plan.family_steps))
    )


def _torch_failure_diagnostic(*, exc: SetupExecutionError, plan: DependencyPlan | None) -> str | None:
    if exc.step_name != "install_shared_torch" or plan is None:
        return None
    torch_index_url = _TORCH_EXTRA_INDEX_URLS.get(plan.cuda_variant)
    if torch_index_url is None:
        return None
    return (
        f"install_shared_torch failed for cuda_variant '{plan.cuda_variant}' using PyTorch index "
        f"'{torch_index_url}'. Verify that the PyTorch index is reachable and compatible with "
        "the selected CUDA variant."
    )


def _payload_text(*, argv: Sequence[str] | None, stdin_text: str | None) -> str:
    stdin_payload = _non_empty_text(stdin_text)
    if stdin_payload is not None:
        return stdin_payload

    argv_items = list(argv or sys.argv[1:])
    if argv_items:
        return argv_items[0]

    raise SetupContractError(
        "Missing Modly setup payload. Provide one JSON object on stdin or argv[1]."
    )


def parse_setup_payload(
    *, argv: Sequence[str] | None = None, stdin_text: str | None = None
) -> SetupPayload:
    payload_text = _payload_text(argv=argv, stdin_text=stdin_text)
    try:
        payload = json.loads(payload_text)
    except JSONDecodeError as exc:
        raise SetupContractError(f"Setup payload is not valid JSON: {exc.msg}.") from exc

    if not isinstance(payload, dict):
        raise SetupContractError("Setup payload must be a JSON object.")

    return SetupPayload(
        python_exe=_coerce_required_string(payload, "python_exe"),
        ext_dir=_coerce_required_string(payload, "ext_dir"),
        gpu_sm=_coerce_optional_string(payload, "gpu_sm"),
        cuda_version=_coerce_optional_string(payload, "cuda_version"),
    )


def _result_from_snapshot(extension_id: str, snapshot: Any) -> SetupResult:
    record = get_extension_record(snapshot, extension_id)
    setup = record.get("setup", {}) if isinstance(record, dict) else {}
    raw_steps = setup.get("steps", ()) if isinstance(setup, dict) else ()
    steps = tuple(
        SetupStep(
            name=str(step.get("name", "unknown")),
            status=str(step.get("status", "unknown")),
            detail=step.get("detail") if isinstance(step.get("detail"), str) else None,
        )
        for step in raw_steps
        if isinstance(step, dict)
    )
    diagnostics = tuple(
        diagnostic for diagnostic in setup.get("diagnostics", ()) if isinstance(diagnostic, str)
    )
    platform_info = setup.get("platform", {}) if isinstance(setup, dict) else {}
    return SetupResult(
        status=setup.get("status", SETUP_STATUS_FAILED),
        extension_id=extension_id,
        venv_python=setup.get("venv_python"),
        platform={
            "system": str(platform_info.get("system", "unknown")),
            "machine": str(platform_info.get("machine", "unknown")),
        },
        steps=steps,
        diagnostics=diagnostics,
    )


def run_install_setup_contract(
    *, extension_id: str, argv: Sequence[str] | None = None, stdin_text: str | None = None
) -> SetupResult:
    snapshot = bootstrap_runtime(extension_id=extension_id)
    platform_info = detect_platform()

    try:
        payload = parse_setup_payload(argv=argv, stdin_text=stdin_text)
    except SetupContractError as exc:
        failed_snapshot = persist_extension_setup(
            snapshot,
            extension_id,
            status=SETUP_STATUS_FAILED,
            ext_dir=None,
            python_exe=None,
            venv_python=None,
            steps=(
                {
                    "name": "parse_payload",
                    "status": "failed",
                    "detail": str(exc),
                },
            ),
            diagnostics=(str(exc),),
            platform_info=platform_info,
        )
        return _result_from_snapshot(extension_id, failed_snapshot)

    descriptor = get_extension_descriptor(extension_id)
    if descriptor is None:
        return _persist_failed_result(
            snapshot=snapshot,
            extension_id=extension_id,
            ext_dir=None,
            python_exe=None,
            steps=(_step("resolve_descriptor", "failed", f"Unknown extension '{extension_id}'."),),
            diagnostics=(f"Unknown extension '{extension_id}'.",),
            platform_info=platform_info,
        )

    prefix_steps = [
        _step("parse_payload", "ok", "Parsed Modly JSON payload from stdin/argv."),
        _step(
            "validate_payload",
            "ok",
            "Validated required fields python_exe/ext_dir and optional gpu_sm/cuda_version.",
        ),
    ]
    ext_root = Path(payload.ext_dir).expanduser()
    installer_python = Path(payload.python_exe).expanduser()
    diagnostics: list[str] = []
    plan: DependencyPlan | None = None
    plan_venv_python = expected_venv_python(ext_root)

    try:
        if not installer_python.exists():
            raise SetupExecutionError(
                step_name="validate_python_exe",
                detail=f"Configured python executable '{installer_python}' does not exist.",
            )
        prefix_steps.append(
            _step(
                "validate_python_exe",
                "ok",
                f"Installer python resolved to '{installer_python}'.",
            )
        )

        python_tag = python_tag_from_interpreter(installer_python)
        plan = resolve_dependency_plan(
            extension_id=extension_id,
            dependency_family=descriptor.dependency_family,
            readiness_imports=descriptor.readiness_imports,
            platform_info=platform_info,
            python_tag=python_tag,
            cuda_version=payload.cuda_version,
        )
        if plan.plan_state != PLAN_STATE_VERIFIED and not _candidate_install_allowed(plan):
            plan_diagnostics = plan.diagnostics or (f"Dependency plan state is {plan.plan_state}.",)
            raise SetupExecutionError(
                step_name="validate_target",
                detail=" ".join(plan_diagnostics),
            )
        prefix_steps.append(_step("validate_target", "ok", _detail_for_plan(plan)))
    except (DependencyPlanError, SetupExecutionError) as exc:
        prefix_steps.append(_step("validate_target", "failed", str(exc)))
        diagnostics.append(str(exc))
        return _persist_failed_result(
            snapshot=snapshot,
            extension_id=extension_id,
            ext_dir=payload.ext_dir,
            python_exe=payload.python_exe,
            steps=prefix_steps,
            diagnostics=diagnostics,
            platform_info=platform_info,
            setup_state=plan.plan_state if plan is not None else None,
            dependency_plan_state=plan.plan_state if plan is not None else None,
            platform_key=plan.platform_key if plan is not None else None,
            platform_supported=plan.platform_supported if plan is not None else None,
        )

    installing_snapshot = persist_extension_setup(
        snapshot,
        extension_id,
        status=SETUP_STATUS_INSTALLING,
        ext_dir=payload.ext_dir,
        python_exe=payload.python_exe,
        venv_python=str(plan_venv_python),
        steps=tuple(prefix_steps)
        + (_step("install_dependencies", "installing", "Creating venv and installing runtime dependencies."),),
        diagnostics=(),
        platform_info=platform_info,
        setup_state=SETUP_STATUS_READY if plan.plan_state == PLAN_STATE_VERIFIED else plan.plan_state,
        dependency_plan_state=plan.plan_state,
        platform_key=plan.platform_key,
        platform_supported=plan.platform_supported,
    )

    execution_steps = list(prefix_steps)

    try:
        ext_root.mkdir(parents=True, exist_ok=True)
        execution_steps.append(
            _step("validate_ext_dir", "ok", f"Install root resolved to '{ext_root}'.")
        )

        _run_checked(
            command=[str(installer_python), "-m", "venv", str(ext_root / "venv")],
            step_name="create_venv",
            cwd=ext_root,
        )
        execution_steps.append(
            _step("create_venv", "ok", f"Created or reused virtualenv at '{ext_root / 'venv'}'.")
        )

        if not plan_venv_python.exists():
            raise SetupExecutionError(
                step_name="verify_venv_python",
                detail=f"Expected virtualenv interpreter at '{plan_venv_python}'.",
            )
        execution_steps.append(
            _step(
                "verify_venv_python",
                "ok",
                f"Virtualenv interpreter is present at '{plan_venv_python}'.",
            )
        )

        _run_checked(
            command=[
                str(plan_venv_python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
            step_name="upgrade_pip",
            cwd=ext_root,
        )
        execution_steps.append(
            _step(
                "upgrade_pip",
                "ok",
                "Upgraded pip/setuptools/wheel inside the extension virtualenv.",
            )
        )

        if plan is None:
            raise SetupExecutionError(
                step_name="validate_target",
                detail="Dependency plan was not resolved before installation started.",
            )

        for install_step in (*plan.shared_steps, *plan.family_steps):
            _install_dependency_step(
                venv_python=plan_venv_python,
                install_step=install_step,
                cwd=ext_root,
            )
            execution_steps.append(
                _step(
                    install_step.name,
                    "ok",
                    f"Installed {', '.join(install_step.packages)}.",
                )
            )

    except SetupExecutionError as exc:
        execution_steps.append(_step(exc.step_name, "failed", exc.detail))
        diagnostics.append(exc.detail)
        torch_diagnostic = _torch_failure_diagnostic(exc=exc, plan=plan)
        if torch_diagnostic is not None:
            diagnostics.append(torch_diagnostic)
        return _persist_failed_result(
            snapshot=installing_snapshot,
            extension_id=extension_id,
            ext_dir=payload.ext_dir,
            python_exe=payload.python_exe,
            steps=execution_steps,
            diagnostics=diagnostics,
            platform_info=platform_info,
        )

    final_snapshot = reevaluate_extension_setup(
        installing_snapshot,
        extension_id,
        ext_dir=payload.ext_dir,
        python_exe=payload.python_exe,
        step_prefix=tuple(execution_steps),
        platform_info=platform_info,
    )
    return _result_from_snapshot(extension_id, final_snapshot)

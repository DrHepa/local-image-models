from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "shared" / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from local_image_runtime import (  # noqa: E402
    bootstrap,
    dependencies,
    install_contract,
    pipeline,
    runtime_adapter,
    weights,
)
from local_image_runtime.dependencies import DependencyInstallStep, DependencyPlan  # noqa: E402


SUPPORTED_PLATFORM = {"system": "linux", "machine": "aarch64"}
UNSUPPORTED_PLATFORM = {"system": "linux", "machine": "x86_64"}
EXTENSION_IDS = ("sd15", "sdxl-base", "flux-schnell")


class RuntimeHarnessTests(unittest.TestCase):
    maxDiff = None

    def _canonical_runtime_file(self, relative_name: str) -> Path:
        return REPO_ROOT / "shared" / "runtime" / "local_image_runtime" / relative_name

    def _vendored_runtime_file(self, extension_id: str, relative_name: str) -> Path:
        return REPO_ROOT / "extensions" / extension_id / "src" / "local_image_runtime" / relative_name

    def _resolve_plan(self, *, python_tag: str, cuda_version: str) -> DependencyPlan:
        return dependencies.resolve_dependency_plan(
            extension_id="sd15",
            dependency_family="sd15",
            readiness_imports=(),
            platform_info=SUPPORTED_PLATFORM,
            python_tag=python_tag,
            cuda_version=cuda_version,
        )

    def _extension_manifest(self, extension_id: str) -> str:
        return (REPO_ROOT / "extensions" / extension_id / "manifest.json").read_text(encoding="utf-8")

    def _extension_manifest_data(self, extension_id: str) -> dict[str, object]:
        return json.loads(self._extension_manifest(extension_id))

    def _load_generator_class(self, extension_id: str) -> type[object]:
        manifest = self._extension_manifest_data(extension_id)
        generator_path = REPO_ROOT / "extensions" / extension_id / "generator.py"
        spec = spec_from_file_location(f"test_generator_{extension_id.replace('-', '_')}", generator_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        generator_class_name = manifest["generator_class"]
        self.assertIsInstance(generator_class_name, str)
        return getattr(module, generator_class_name)

    def _make_model_dir(self, extension_id: str, model_dir_name: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix=f"model-dir-{extension_id}-"))
        model_dir = root / extension_id / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _make_runtime_root(self, extension_id: str) -> Path:
        runtime_root = Path(tempfile.mkdtemp(prefix=f"local-image-{extension_id}-"))
        (runtime_root / "manifest.json").write_text(
            self._extension_manifest(extension_id),
            encoding="utf-8",
        )
        (runtime_root / "src").mkdir(parents=True, exist_ok=True)
        return runtime_root

    def _make_runtime_snapshot(
        self,
        *,
        outputs_dir: Path | None = None,
        models_dir: Path | None = None,
    ) -> SimpleNamespace:
        resolved_outputs_dir = outputs_dir or Path(tempfile.mkdtemp(prefix="runtime-outputs-"))
        resolved_models_dir = models_dir or Path(tempfile.mkdtemp(prefix="runtime-models-"))
        return SimpleNamespace(
            paths=SimpleNamespace(
                outputs_dir=resolved_outputs_dir,
                models_dir=resolved_models_dir,
            )
        )

    def _make_executable_python(self, root: Path) -> Path:
        python_path = root / "venv" / "bin" / "python"
        python_path.parent.mkdir(parents=True, exist_ok=True)
        python_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        python_path.chmod(0o755)
        return python_path

    def _make_windows_executable_python(self, root: Path) -> Path:
        python_path = root / "venv" / "Scripts" / "python.exe"
        python_path.parent.mkdir(parents=True, exist_ok=True)
        python_path.write_text("", encoding="utf-8")
        python_path.chmod(0o755)
        return python_path

    def _completed_process(self, *, stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["fake-child"], returncode=returncode, stdout=stdout, stderr="")

    def _make_backend_job(self, *, workspace_dir: Path) -> pipeline.BackendJob:
        return pipeline.BackendJob(
            command=(sys.executable, "-m", "local_image_runtime.inference_runner"),
            payload={
                "extension_id": "sd15",
                "family": "stable-diffusion",
                "node_id": "text-to-image",
                "workspace_dir": str(workspace_dir),
                "output_path": str(workspace_dir / "streamed-output.png"),
                "prompt": "stream me",
                "params": {"steps": 4},
            },
            workspace_dir=workspace_dir,
            cwd=workspace_dir,
            env={"PYTHONPATH": str(workspace_dir)},
        )

    class _FakePipeStream:
        def __init__(self, owner: "RuntimeHarnessTests._FakePopen", name: str, lines: list[str]) -> None:
            self._owner = owner
            self._name = name
            self._lines = list(lines)
            self.read_count = 0

        def set_lines(self, lines: list[str]) -> None:
            self._lines = list(lines)

        def readline(self) -> str:
            self.read_count += 1
            if self._lines:
                return self._lines.pop(0)
            self._owner.mark_stream_eof(self._name)
            return ""

        def __iter__(self):
            while True:
                line = self.readline()
                if line == "":
                    break
                yield line

    class _FakePipeStdin:
        def __init__(self, on_close: Callable[[str], None] | None = None) -> None:
            self.chunks: list[str] = []
            self.closed = False
            self._on_close = on_close

        def write(self, text: str) -> int:
            self.chunks.append(text)
            return len(text)

        def flush(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True
            if self._on_close is not None:
                self._on_close(self.value)

        @property
        def value(self) -> str:
            return "".join(self.chunks)

    class _FakePopen:
        def __init__(
            self,
            *,
            stdout_lines: list[str],
            stderr_lines: list[str],
            returncode: int = 0,
            on_stdin_close: Callable[[str], tuple[list[str], list[str], int]] | None = None,
            wait_timeout_after_terminate: bool = False,
        ) -> None:
            def handle_stdin_close(payload: str) -> None:
                if on_stdin_close is None:
                    return
                next_stdout, next_stderr, next_returncode = on_stdin_close(payload)
                self.stdout.set_lines(next_stdout)
                self.stderr.set_lines(next_stderr)
                self._expected_returncode = next_returncode

            self.stdin = RuntimeHarnessTests._FakePipeStdin(on_close=handle_stdin_close)
            self.stdout = RuntimeHarnessTests._FakePipeStream(self, "stdout", stdout_lines)
            self.stderr = RuntimeHarnessTests._FakePipeStream(self, "stderr", stderr_lines)
            self._expected_returncode = returncode
            self.returncode: int | None = None
            self.wait_called = False
            self.terminate_called = False
            self.kill_called = False
            self._eof = {"stdout": False, "stderr": False}
            self._wait_timeout_after_terminate = wait_timeout_after_terminate
            self._terminate_wait_timed_out = False

        def mark_stream_eof(self, name: str) -> None:
            self._eof[name] = True

        def poll(self) -> int | None:
            if all(self._eof.values()) and self.stdin.closed:
                return self._expected_returncode
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.wait_called = True
            if (
                timeout is not None
                and self.terminate_called
                and self._wait_timeout_after_terminate
                and not self._terminate_wait_timed_out
            ):
                self._terminate_wait_timed_out = True
                raise subprocess.TimeoutExpired(cmd=["fake-child"], timeout=timeout)
            self.returncode = self._expected_returncode
            return self.returncode

        def terminate(self) -> None:
            self.terminate_called = True
            self.returncode = -15

        def kill(self) -> None:
            self.kill_called = True
            self.returncode = -9

    class _ScriptedClock:
        def __init__(self, *, start: float = 0.0) -> None:
            self.now = start

        def monotonic(self) -> float:
            return self.now

    class _ScriptedQueue:
        EMPTY = object()

        def __init__(
            self,
            *,
            clock: "RuntimeHarnessTests._ScriptedClock",
            items: list[tuple[float, tuple[str, str, str | None] | object]],
        ) -> None:
            self._clock = clock
            self._items = list(items)

        def put(self, item: tuple[str, str, str | None]) -> None:
            return None

        def get(self, timeout: float | None = None) -> tuple[str, str, str | None]:
            if not self._items:
                raise AssertionError("Scripted queue exhausted before runtime finished.")
            next_time, item = self._items.pop(0)
            self._clock.now = next_time
            if item is self.EMPTY:
                raise pipeline.queue.Empty
            assert isinstance(item, tuple)
            return item

    def _run_real_runner_popen(
        self,
        *,
        loader_map: dict[tuple[str, str], object],
        source_image_token: object | None = None,
    ):
        import local_image_runtime.inference_runner as inference_runner

        def on_stdin_close(input_text: str) -> tuple[list[str], list[str], int]:
            stdout = StringIO()
            with patch.dict(inference_runner._PIPELINE_LOADERS, loader_map, clear=True), patch.object(
                inference_runner, "_seeded_generator", return_value="generator-token"
            ):
                if source_image_token is None:
                    exit_code = inference_runner.run_child_main(stdin=StringIO(input_text), stdout=stdout)
                else:
                    with patch.object(
                        inference_runner,
                        "_open_source_image",
                        return_value=source_image_token,
                    ):
                        exit_code = inference_runner.run_child_main(stdin=StringIO(input_text), stdout=stdout)
            return stdout.getvalue().splitlines(keepends=True), [], exit_code

        def popen_side_effect(command, *, stdin, stdout, stderr, text, bufsize, cwd, env):
            self.assertEqual(command[1:], ["-m", "local_image_runtime.inference_runner"])
            self.assertIs(stdin, subprocess.PIPE)
            self.assertIs(stdout, subprocess.PIPE)
            self.assertIs(stderr, subprocess.PIPE)
            self.assertTrue(text)
            self.assertEqual(bufsize, 1)
            self.assertIsInstance(cwd, str)
            self.assertIsInstance(env, dict)
            self.assertIn("PYTHONPATH", env)
            return self._FakePopen(
                stdout_lines=[],
                stderr_lines=[],
                on_stdin_close=on_stdin_close,
            )

        return popen_side_effect

    def _parse_ndjson_events(self, payload: str) -> list[dict[str, object]]:
        return [json.loads(line) for line in payload.splitlines() if line.strip()]

    def _make_real_runner_loader(
        self,
        *,
        marker: str,
        invocations: list[dict[str, object]],
    ) -> SimpleNamespace:
        class FakeImage:
            def __init__(self, image_marker: str) -> None:
                self.image_marker = image_marker

            def save(self, output_path: str) -> None:
                Path(output_path).write_bytes(f"generated:{self.image_marker}".encode("utf-8"))

        class FakePipeline:
            def __init__(self, *, pipeline_marker: str, model_dir: str) -> None:
                self.pipeline_marker = pipeline_marker
                self.model_dir = model_dir

            def __call__(self, **kwargs):
                invocations.append(
                    {
                        "marker": self.pipeline_marker,
                        "model_dir": self.model_dir,
                        "kwargs": kwargs,
                    }
                )
                return SimpleNamespace(images=[FakeImage(self.pipeline_marker)])

        return SimpleNamespace(
            from_pretrained=lambda model_dir: FakePipeline(pipeline_marker=marker, model_dir=model_dir)
        )

    def _run_real_runner_subprocess(
        self,
        *,
        loader_map: dict[tuple[str, str], object],
        source_image_token: object | None = None,
    ):
        import local_image_runtime.inference_runner as inference_runner

        def run_side_effect(command, *, input, text, capture_output, check, cwd, env):
            self.assertTrue(text)
            self.assertTrue(capture_output)
            self.assertTrue(check)
            self.assertIsInstance(cwd, str)
            self.assertIsInstance(env, dict)
            self.assertIn("PYTHONPATH", env)

            stdout = StringIO()
            with patch.dict(inference_runner._PIPELINE_LOADERS, loader_map, clear=True), patch.object(
                inference_runner, "_seeded_generator", return_value="generator-token"
            ):
                if source_image_token is None:
                    exit_code = inference_runner.run_child_main(stdin=StringIO(input), stdout=stdout)
                else:
                    with patch.object(
                        inference_runner,
                        "_open_source_image",
                        return_value=source_image_token,
                    ):
                        exit_code = inference_runner.run_child_main(stdin=StringIO(input), stdout=stdout)

            if exit_code != 0:
                raise subprocess.CalledProcessError(
                    returncode=exit_code,
                    cmd=command,
                    output=stdout.getvalue(),
                    stderr="",
                )
            return self._completed_process(stdout=stdout.getvalue())

        return run_side_effect

    def _payload(self, runtime_root: Path) -> str:
        return json.dumps(
            {
                "python_exe": sys.executable,
                "ext_dir": str(runtime_root),
                "gpu_sm": "90",
                "cuda_version": "12.4",
            }
        )

    def _generator_payload(self) -> StringIO:
        return StringIO(
            json.dumps(
                {
                    "nodeId": "text-to-image",
                    "input": {"text": "a lighthouse at dusk"},
                    "params": {
                        "prompt": "a lighthouse at dusk",
                        "steps": 4,
                        "width": 512,
                        "height": 512,
                        "guidance_scale": 7.5,
                        "seed": 42,
                    },
                }
            )
            + "\n"
        )

    def _fake_plan(self, extension_id: str) -> DependencyPlan:
        descriptor = bootstrap.get_extension_descriptor(extension_id)
        self.assertIsNotNone(descriptor)
        return DependencyPlan(
            extension_id=extension_id,
            dependency_family=descriptor.dependency_family,
            platform_system="linux",
            platform_machine="aarch64",
            python_tag="cp311",
            cuda_variant="cu124",
            shared_steps=(
                DependencyInstallStep(
                    name="install_shared_runtime",
                    packages=("Pillow", "numpy"),
                ),
            ),
            family_steps=(
                DependencyInstallStep(
                    name="install_family_dependencies",
                    packages=("diffusers==0.35.1",),
                ),
            ),
            readiness_imports=(),
        )

    def _fake_torch_plan(self, extension_id: str, *, python_tag: str, cuda_version: str) -> DependencyPlan:
        return self._resolve_plan(python_tag=python_tag, cuda_version=cuda_version)

    def _run_checked_side_effect(self, *, command, step_name, cwd=None):
        if step_name != "create_venv":
            return None
        subprocess.run(
            list(command),
            check=True,
            cwd=str(cwd) if cwd is not None else None,
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
            capture_output=True,
            text=True,
        )
        return None

    def _run_setup_success(
        self, extension_id: str, runtime_root: Path | None = None
    ) -> tuple[Path, install_contract.SetupResult]:
        runtime_root = runtime_root or self._make_runtime_root(extension_id)
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                )
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM)
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=self._fake_plan(extension_id))
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract._run_checked", side_effect=self._run_checked_side_effect)
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract._install_dependency_step", return_value=None)
            )
            stack.enter_context(
                patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports"))
            )
            result = install_contract.run_install_setup_contract(
                extension_id=extension_id,
                stdin_text=self._payload(runtime_root),
            )
        return runtime_root, result

    def test_repair_rerun_recovers_missing_venv_from_partial_install(self) -> None:
        extension_id = "sd15"
        runtime_root, initial_result = self._run_setup_success(extension_id)
        self.assertEqual(initial_result.status, bootstrap.SETUP_STATUS_READY)

        venv_dir = runtime_root / "venv"
        self.assertTrue(venv_dir.exists())
        shutil.rmtree(venv_dir)

        with patch.dict(
            os.environ,
            {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
            clear=False,
        ):
            partial_snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)

        partial_record = bootstrap.get_extension_record(partial_snapshot, extension_id)
        self.assertEqual(partial_record["setup"]["status"], bootstrap.SETUP_STATUS_FAILED)
        self.assertEqual(partial_record["status"], bootstrap.EXTENSION_STATUS_ERROR)
        self.assertEqual(
            partial_record["setup"]["steps"][-1]["name"],
            "verify_venv_python",
        )
        self.assertEqual(
            partial_record["setup"]["steps"][-1]["status"],
            "failed",
        )
        self.assertIn(
            "Missing virtualenv interpreter:",
            partial_record["error"],
        )

        _, repaired_result = self._run_setup_success(extension_id, runtime_root=runtime_root)
        self.assertEqual(repaired_result.status, bootstrap.SETUP_STATUS_READY)
        self.assertTrue((runtime_root / "venv" / "bin" / "python").exists())
        self.assertEqual(repaired_result.steps[-1].name, "verify_runtime_imports")
        self.assertEqual(repaired_result.steps[-1].status, "ok")

        with patch.dict(
            os.environ,
            {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
            clear=False,
        ), patch(
            "local_image_runtime.bootstrap._smoke_test_runtime_imports",
            return_value=(True, "stubbed imports"),
        ):
            repaired_snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)

        repaired_record = bootstrap.get_extension_record(repaired_snapshot, extension_id)
        self.assertEqual(repaired_record["setup"]["status"], bootstrap.SETUP_STATUS_READY)
        self.assertEqual(repaired_record["status"], bootstrap.EXTENSION_STATUS_INSTALLED)
        self.assertIsNone(repaired_record["error"])

    def test_setup_success_persists_ready_and_installed(self) -> None:
        for extension_id in EXTENSION_IDS:
            with self.subTest(extension_id=extension_id):
                runtime_root, result = self._run_setup_success(extension_id)
                self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
                self.assertTrue((runtime_root / "venv" / "bin" / "python").exists())

                with patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                ), patch(
                    "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                    return_value=(True, "stubbed imports"),
                ):
                    snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)

                record = bootstrap.get_extension_record(snapshot, extension_id)
                self.assertEqual(record["setup"]["status"], bootstrap.SETUP_STATUS_READY)
                self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_INSTALLED)
                self.assertEqual(record["readiness"], bootstrap.SETUP_STATUS_READY)
                self.assertEqual(record["venv_python"], str(runtime_root / "venv" / "bin" / "python"))

    def test_unsupported_target_persists_clear_diagnostics(self) -> None:
        for extension_id in EXTENSION_IDS:
            with self.subTest(extension_id=extension_id):
                runtime_root = self._make_runtime_root(extension_id)
                with patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                ), patch(
                    "local_image_runtime.install_contract.detect_platform",
                    return_value=UNSUPPORTED_PLATFORM,
                ):
                    result = install_contract.run_install_setup_contract(
                        extension_id=extension_id,
                        stdin_text=self._payload(runtime_root),
                    )

                self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
                diagnostics_text = " ".join(result.diagnostics)
                self.assertIn("system='linux'", diagnostics_text)
                self.assertIn("machine='x86_64'", diagnostics_text)
                self.assertIn("Linux ARM64", diagnostics_text)

                with patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                ):
                    snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)
                record = bootstrap.get_extension_record(snapshot, extension_id)
                self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_ERROR)
                self.assertIn("Unsupported/unverified install target detected", record["error"])

    def test_dependency_failure_persists_failing_step_and_diagnostic(self) -> None:
        extension_id = "sd15"
        runtime_root = self._make_runtime_root(extension_id)

        def fail_on_family_step(*, venv_python, install_step, cwd):
            if install_step.name == "install_family_dependencies":
                raise install_contract.SetupExecutionError(
                    step_name=install_step.name,
                    detail="pip install failed for diffusers==0.35.1",
                )
            return None

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                )
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM)
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=self._fake_plan(extension_id))
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract._run_checked", side_effect=self._run_checked_side_effect)
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract._install_dependency_step", side_effect=fail_on_family_step)
            )
            stack.enter_context(
                patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports"))
            )
            result = install_contract.run_install_setup_contract(
                extension_id=extension_id,
                stdin_text=self._payload(runtime_root),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
        self.assertIn("pip install failed for diffusers==0.35.1", result.diagnostics)
        failed_steps = {step.name: step for step in result.steps}
        self.assertEqual(failed_steps["install_family_dependencies"].status, "failed")

        with patch.dict(
            os.environ,
            {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
            clear=False,
        ):
            snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)
        record = bootstrap.get_extension_record(snapshot, extension_id)
        self.assertEqual(record["setup"]["status"], bootstrap.SETUP_STATUS_FAILED)
        self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_ERROR)
        self.assertEqual(record["error"], "pip install failed for diffusers==0.35.1")

    def test_torch_dependency_failure_persists_contextual_index_diagnostics(self) -> None:
        extension_id = "sd15"
        runtime_root = self._make_runtime_root(extension_id)
        plan = self._fake_torch_plan(extension_id, python_tag="cp312", cuda_version="12.8")
        torch_index_url = dependencies._TORCH_EXTRA_INDEX_URLS[plan.cuda_variant]
        raw_detail = "Could not fetch URL https://download.pytorch.org/whl/cu128/triton/: connection refused"
        expected_context = (
            f"install_shared_torch failed for cuda_variant '{plan.cuda_variant}' using PyTorch index "
            f"'{torch_index_url}'. Verify that the PyTorch index is reachable and compatible with "
            "the selected CUDA variant."
        )

        def fail_on_torch_step(*, venv_python, install_step, cwd):
            if install_step.name == "install_shared_torch":
                raise install_contract.SetupExecutionError(
                    step_name=install_step.name,
                    detail=raw_detail,
                )
            return None

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                )
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM)
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=plan)
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract._run_checked", side_effect=self._run_checked_side_effect)
            )
            stack.enter_context(
                patch("local_image_runtime.install_contract._install_dependency_step", side_effect=fail_on_torch_step)
            )
            stack.enter_context(
                patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports"))
            )
            result = install_contract.run_install_setup_contract(
                extension_id=extension_id,
                stdin_text=self._payload(runtime_root),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
        self.assertEqual(result.steps[-1].name, "install_shared_torch")
        self.assertEqual(result.steps[-1].status, "failed")
        self.assertIn(raw_detail, result.diagnostics)
        self.assertIn(expected_context, result.diagnostics)

        with patch.dict(
            os.environ,
            {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
            clear=False,
        ):
            snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)
        record = bootstrap.get_extension_record(snapshot, extension_id)
        self.assertEqual(record["setup"]["status"], bootstrap.SETUP_STATUS_FAILED)
        self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_ERROR)
        self.assertEqual(record["error"], raw_detail)
        self.assertIn(expected_context, record["setup"]["diagnostics"])

    def test_resolve_dependency_plan_builds_torch_step_for_cp311_cu124(self) -> None:
        plan = self._resolve_plan(python_tag="cp311", cuda_version="12.4")

        torch_step = plan.shared_steps[0]
        self.assertEqual(torch_step.name, "install_shared_torch")
        self.assertEqual(
            torch_step.extra_args,
            (
                "--index-url",
                dependencies._PYPI_INDEX_URL,
                "--extra-index-url",
                dependencies._TORCH_EXTRA_INDEX_URLS["cu124"],
                "--no-cache-dir",
            ),
        )
        self.assertEqual(
            torch_step.packages,
            (
                dependencies._TORCH_WHEELS["cu124"]["cp311"]["torch"],
                dependencies._TORCH_WHEELS["cu124"]["cp311"]["torchvision"],
            ),
        )

    def test_resolve_dependency_plan_builds_torch_step_for_cp312_cu128(self) -> None:
        plan = self._resolve_plan(python_tag="cp312", cuda_version="12.8")

        torch_step = plan.shared_steps[0]
        self.assertEqual(plan.cuda_variant, "cu128")
        self.assertEqual(torch_step.name, "install_shared_torch")
        self.assertEqual(
            torch_step.extra_args,
            (
                "--index-url",
                dependencies._PYPI_INDEX_URL,
                "--extra-index-url",
                dependencies._TORCH_EXTRA_INDEX_URLS["cu128"],
                "--no-cache-dir",
            ),
        )
        self.assertEqual(
            torch_step.packages,
            (
                dependencies._TORCH_WHEELS["cu128"]["cp312"]["torch"],
                dependencies._TORCH_WHEELS["cu128"]["cp312"]["torchvision"],
            ),
        )

    def test_pip_install_command_keeps_indexes_before_direct_wheels(self) -> None:
        command = dependencies.pip_install_command(
            venv_python="/tmp/runtime/bin/python",
            extra_args=(
                "--index-url",
                "https://pypi.org/simple",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu128",
                "--no-cache-dir",
            ),
            packages=(
                dependencies._TORCH_WHEELS["cu128"]["cp312"]["torch"],
                dependencies._TORCH_WHEELS["cu128"]["cp312"]["torchvision"],
            ),
        )

        self.assertEqual(
            command,
            [
                "/tmp/runtime/bin/python",
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://pypi.org/simple",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu128",
                "--no-cache-dir",
                dependencies._TORCH_WHEELS["cu128"]["cp312"]["torch"],
                dependencies._TORCH_WHEELS["cu128"]["cp312"]["torchvision"],
            ],
        )

    def test_select_cuda_variant_preserves_verified_matrix(self) -> None:
        self.assertEqual(dependencies._select_cuda_variant("12.8"), "cu128")
        self.assertEqual(dependencies._select_cuda_variant("12.4"), "cu124")

    def test_select_cuda_variant_rejects_unverified_cuda(self) -> None:
        with self.assertRaisesRegex(
            dependencies.DependencyPlanError,
            "Verified variants are cu124 and cu128 only",
        ):
            dependencies._select_cuda_variant("12.3")

    def test_bootstrap_reconciles_stale_ready_state_to_installed(self) -> None:
        extension_id = "sd15"
        runtime_root = self._make_runtime_root(extension_id)
        venv_python = runtime_root / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with patch.dict(
            os.environ,
            {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
            clear=False,
        ), patch(
            "local_image_runtime.bootstrap._smoke_test_runtime_imports",
            return_value=(True, "stubbed imports"),
        ):
            snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)
            snapshot = bootstrap.persist_extension_setup(
                snapshot,
                extension_id,
                status=bootstrap.SETUP_STATUS_READY,
                ext_dir=str(runtime_root),
                python_exe=sys.executable,
                venv_python=str(venv_python),
                steps=({"name": "verify_venv_python", "status": "ok", "detail": "present"},),
                diagnostics=(),
                platform_info=SUPPORTED_PLATFORM,
            )

        raw_models_state = json.loads(snapshot.paths.models_state_file.read_text(encoding="utf-8"))
        raw_models_state["extensions"][extension_id]["status"] = bootstrap.EXTENSION_STATUS_NOT_INSTALLED
        raw_models_state["extensions"][extension_id]["installed_at"] = None
        snapshot.paths.models_state_file.write_text(
            json.dumps(raw_models_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        with patch.dict(
            os.environ,
            {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
            clear=False,
        ), patch(
            "local_image_runtime.bootstrap._smoke_test_runtime_imports",
            return_value=(True, "stubbed imports"),
        ):
            healed_snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)

        record = bootstrap.get_extension_record(healed_snapshot, extension_id)
        self.assertEqual(record["setup"]["status"], bootstrap.SETUP_STATUS_READY)
        self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_INSTALLED)
        self.assertIsNotNone(record["installed_at"])

    def test_generator_reaches_real_runner_success_after_ready_setup(self) -> None:
        cases = (
            ("sd15", "stable-diffusion", "stable-text"),
            ("sdxl-base", "sdxl", "sdxl-text"),
            ("flux-schnell", "flux", "flux-text"),
        )

        for extension_id, expected_family, expected_marker in cases:
            with self.subTest(extension_id=extension_id):
                runtime_root, result = self._run_setup_success(extension_id)
                self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
                stdout = StringIO()
                outputs_dir = Path(tempfile.mkdtemp(prefix=f"generator-main-{extension_id}-"))
                invocations: list[dict[str, object]] = []
                payload = {
                    "nodeId": "text-to-image",
                    "workspaceDir": str(outputs_dir),
                    "input": {"text": f"legacy prompt {extension_id}"},
                    "params": {
                        "prompt": f"hero image {extension_id}",
                        "negative_prompt": f"avoid artifacts {extension_id}",
                        "steps": 4,
                        "width": 512,
                        "height": 512,
                        "guidance_scale": 7.5,
                        "seed": 42,
                    },
                }

                with patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                ), patch(
                    "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                    return_value=(True, "stubbed imports"),
                ), patch(
                    "local_image_runtime.pipeline.subprocess.Popen",
                    side_effect=self._run_real_runner_popen(
                        loader_map={(expected_family, "text-to-image"): self._make_real_runner_loader(
                            marker=expected_marker,
                            invocations=invocations,
                        )}
                    ),
                ):
                    exit_code = runtime_adapter.run_generator_main(
                        extension_id=extension_id,
                        runtime_root=str(runtime_root),
                        stdin=StringIO(json.dumps(payload) + "\n"),
                        stdout=stdout,
                    )

                output = stdout.getvalue()
                events = self._parse_ndjson_events(output)
                done_event = events[-1]
                self.assertEqual(exit_code, 0)
                self.assertEqual(len(invocations), 1)
                self.assertEqual(invocations[0]["marker"], expected_marker)
                self.assertEqual(invocations[0]["kwargs"]["negative_prompt"], payload["params"]["negative_prompt"])
                self.assertTrue(Path(done_event["result"]["output_path"]).exists())
                self.assertTrue(str(done_event["result"]["output_path"]).startswith(str(outputs_dir)))
                self.assertEqual(
                    done_event["result"]["metadata"],
                    {
                        "family": expected_family,
                        "node_id": "text-to-image",
                        "seed": 42,
                        "negative_prompt_used": True,
                        "source_image_used": False,
                    },
                )
                self.assertIn('"label": "runtime-ready"', output)
                self.assertIn('"label": "backend-dispatch"', output)

    def test_run_generator_main_surfaces_child_runner_errors_clearly(self) -> None:
        runtime_root, result = self._run_setup_success("sd15")
        self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
        stdout = StringIO()

        payload = {
            "nodeId": "text-to-image",
            "workspaceDir": str(Path(tempfile.mkdtemp(prefix="generator-main-error-"))),
            "input": {"text": "broken"},
            "params": {"prompt": "broken", "steps": 4},
        }

        with patch.dict(
            os.environ,
            {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
            clear=False,
        ), patch(
            "local_image_runtime.bootstrap._smoke_test_runtime_imports",
            return_value=(True, "stubbed imports"),
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            side_effect=self._run_real_runner_popen(loader_map={}, source_image_token=object()),
        ):
            exit_code = runtime_adapter.run_generator_main(
                extension_id="sd15",
                runtime_root=str(runtime_root),
                stdin=StringIO(json.dumps(payload) + "\n"),
                stdout=stdout,
            )

        events = self._parse_ndjson_events(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(
            events[-1]["message"],
            "Unsupported inference backend for family 'stable-diffusion' and node 'text-to-image'",
        )

    def test_run_payload_reaches_real_runner_success_after_ready_setup(self) -> None:
        cases = (
            ("sd15", "stable-diffusion", "stable-text"),
            ("sdxl-base", "sdxl", "sdxl-text"),
            ("flux-schnell", "flux", "flux-text"),
        )

        for extension_id, expected_family, expected_marker in cases:
            with self.subTest(extension_id=extension_id):
                runtime_root, result = self._run_setup_success(extension_id)
                self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
                outputs_dir = Path(tempfile.mkdtemp(prefix=f"run-payload-{extension_id}-"))
                invocations: list[dict[str, object]] = []
                payload = {
                    "nodeId": "text-to-image",
                    "workspaceDir": str(outputs_dir),
                    "input": {"text": f"legacy {extension_id}"},
                    "params": {
                        "prompt": f"payload prompt {extension_id}",
                        "negative_prompt": f"payload negative {extension_id}",
                        "steps": 4,
                        "width": 512,
                        "height": 512,
                        "guidance_scale": 7.5,
                        "seed": 42,
                    },
                }

                with patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                ), patch(
                    "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                    return_value=(True, "stubbed imports"),
                ), patch(
                    "local_image_runtime.pipeline.subprocess.Popen",
                    side_effect=self._run_real_runner_popen(
                        loader_map={(expected_family, "text-to-image"): self._make_real_runner_loader(
                            marker=expected_marker,
                            invocations=invocations,
                        )}
                    ),
                ):
                    result_payload = runtime_adapter.run_payload(
                        payload,
                        extension_id=extension_id,
                        runtime_root=str(runtime_root),
                    )

                self.assertEqual(result_payload["extension_id"], extension_id)
                self.assertEqual(len(invocations), 1)
                self.assertEqual(invocations[0]["kwargs"]["negative_prompt"], payload["params"]["negative_prompt"])
                self.assertTrue(Path(result_payload["result"]["output_path"]).exists())
                self.assertTrue(str(result_payload["result"]["output_path"]).startswith(str(outputs_dir)))
                self.assertEqual(
                    result_payload["result"]["metadata"],
                    {
                        "family": expected_family,
                        "node_id": "text-to-image",
                        "seed": 42,
                        "negative_prompt_used": True,
                        "source_image_used": False,
                    },
                )

    def test_run_payload_surfaces_child_runner_errors_clearly(self) -> None:
        runtime_root, result = self._run_setup_success("sd15")
        self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
        outputs_dir = Path(tempfile.mkdtemp(prefix="run-payload-error-"))
        payload = {
            "nodeId": "text-to-image",
            "workspaceDir": str(outputs_dir),
            "input": {"text": "broken"},
            "params": {"prompt": "broken", "steps": 4},
        }

        with patch.dict(
            os.environ,
            {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
            clear=False,
        ), patch(
            "local_image_runtime.bootstrap._smoke_test_runtime_imports",
            return_value=(True, "stubbed imports"),
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            side_effect=self._run_real_runner_popen(loader_map={}),
        ):
            with self.assertRaisesRegex(
                runtime_adapter.DomainError,
                "Unsupported inference backend for family 'stable-diffusion' and node 'text-to-image'",
            ):
                runtime_adapter.run_payload(
                    payload,
                    extension_id="sd15",
                    runtime_root=str(runtime_root),
                )

    def test_extension_generator_classes_expose_basegenerator_contract(self) -> None:
        expected_nodes = {
            "sd15": "text-to-image",
            "sdxl-base": "text-to-image",
            "flux-schnell": "text-to-image",
        }

        for extension_id, node_id in expected_nodes.items():
            with self.subTest(extension_id=extension_id, node_id=node_id):
                generator_class = self._load_generator_class(extension_id)
                model_dir = self._make_model_dir(extension_id, node_id)
                outputs_dir = Path(tempfile.mkdtemp(prefix=f"outputs-{extension_id}-"))

                generator = generator_class(model_dir, outputs_dir)

                self.assertEqual(generator.model_dir, model_dir)
                self.assertEqual(generator.outputs_dir, outputs_dir)
                self.assertTrue(callable(generator.load))
                self.assertTrue(callable(generator.generate))
                self.assertTrue(callable(generator.unload))
                self.assertTrue(callable(generator.params_schema))
                self.assertIsInstance(generator.params_schema(), list)
                self.assertGreater(len(generator.params_schema()), 0)

    def test_extension_generator_resolves_node_from_model_dir_name(self) -> None:
        cases = (
            ("sd15", "text-to-image", "text-to-image"),
            ("sd15", "image-to-image", "image-to-image"),
            ("sdxl-base", "text-to-image", "text-to-image"),
            ("sdxl-base", "image-to-image", "image-to-image"),
            ("flux-schnell", "weights-cache", "text-to-image"),
        )

        for extension_id, model_dir_name, expected_node_id in cases:
            with self.subTest(
                extension_id=extension_id,
                model_dir_name=model_dir_name,
                expected_node_id=expected_node_id,
            ):
                generator_class = self._load_generator_class(extension_id)
                model_dir = self._make_model_dir(extension_id, model_dir_name)
                outputs_dir = Path(tempfile.mkdtemp(prefix=f"outputs-{extension_id}-"))

                generator = generator_class(model_dir, outputs_dir)

                self.assertEqual(generator.node_id, expected_node_id)

    def test_extension_generator_rejects_unsupported_model_dir_name(self) -> None:
        generator_class = self._load_generator_class("sd15")
        model_dir = self._make_model_dir("sd15", "not-a-real-node")
        outputs_dir = Path(tempfile.mkdtemp(prefix="outputs-sd15-"))

        with self.assertRaisesRegex(
            runtime_adapter.DomainError,
            "Could not resolve node for extension 'sd15' from model_dir.name 'not-a-real-node'",
        ):
            generator_class(model_dir, outputs_dir)

    def test_extension_generator_load_and_unload_manage_runtime_snapshot(self) -> None:
        generator_class = self._load_generator_class("sd15")
        model_dir = self._make_model_dir("sd15", "text-to-image")
        outputs_dir = Path(tempfile.mkdtemp(prefix="outputs-sd15-"))
        runtime_snapshot = object()

        with patch("local_image_runtime.runtime_adapter.bootstrap_runtime", return_value=runtime_snapshot) as bootstrap_mock:
            generator = generator_class(model_dir, outputs_dir)

            generator.load()
            generator.load()
            self.assertIs(generator._runtime_snapshot, runtime_snapshot)
            self.assertIs(generator._model, runtime_snapshot)
            self.assertEqual(bootstrap_mock.call_count, 1)

            generator.unload()
            self.assertIsNone(generator._runtime_snapshot)
            self.assertIsNone(generator._model)

            generator.load()
            self.assertEqual(bootstrap_mock.call_count, 2)

    def test_extension_generator_generate_reaches_real_runner_for_text_to_image(self) -> None:
        cases = (
            ("sd15", "stable-diffusion", "stable-text"),
            ("sdxl-base", "sdxl", "sdxl-text"),
            ("flux-schnell", "flux", "flux-text"),
        )

        for extension_id, expected_family, expected_marker in cases:
            with self.subTest(extension_id=extension_id):
                runtime_root = self._make_runtime_root(extension_id)
                _, result = self._run_setup_success(extension_id, runtime_root=runtime_root)
                self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
                generator_class = self._load_generator_class(extension_id)
                original_runtime_root = generator_class.runtime_root
                generator_class.runtime_root = str(runtime_root)
                model_dir = self._make_model_dir(extension_id, "text-to-image")
                outputs_dir = Path(tempfile.mkdtemp(prefix=f"outputs-{extension_id}-"))
                progress_events: list[tuple[int, str]] = []
                invocations: list[dict[str, object]] = []
                params = {
                    "prompt": f"generator prompt {extension_id}",
                    "negative_prompt": f"generator negative {extension_id}",
                    "steps": 4,
                    "width": 512,
                    "height": 512,
                    "guidance_scale": 7.5,
                    "seed": 42,
                    "input": {"text": f"legacy generator text {extension_id}"},
                }

                try:
                    with patch(
                        "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                        return_value=(True, "stubbed imports"),
                    ), patch(
                        "local_image_runtime.pipeline.subprocess.Popen",
                        side_effect=self._run_real_runner_popen(
                            loader_map={(expected_family, "text-to-image"): self._make_real_runner_loader(
                                marker=expected_marker,
                                invocations=invocations,
                            )}
                        ),
                    ):
                        generator = generator_class(model_dir, outputs_dir)
                        actual_path = generator.generate(
                            b"ignored-image-bytes",
                            params,
                            progress_cb=lambda percent, label: progress_events.append((percent, label)),
                        )
                finally:
                    generator_class.runtime_root = original_runtime_root

                self.assertEqual(len(invocations), 1)
                self.assertEqual(invocations[0]["kwargs"]["prompt"], params["prompt"])
                self.assertEqual(invocations[0]["kwargs"]["negative_prompt"], params["negative_prompt"])
                self.assertFalse((outputs_dir / ".modly-inputs").exists())
                self.assertTrue(actual_path.exists())
                self.assertTrue(str(actual_path).startswith(str(outputs_dir)))
                self.assertIn((75, "backend-dispatch"), progress_events)

    def test_extension_generator_generate_serializes_effective_model_dir_to_child_payload(self) -> None:
        cases = (
            (
                "sd15",
                "text-to-image",
                "stable-diffusion",
                "stable-text",
                b"ignored-image-bytes",
                {
                    "prompt": "generator prompt sd15",
                    "steps": 4,
                    "input": {"text": "legacy generator text sd15"},
                },
                None,
            ),
            (
                "sdxl-base",
                "image-to-image",
                "sdxl",
                "sdxl-image",
                b"fake-image-bytes",
                {"prompt": "variation", "strength": 0.35, "steps": 4},
                object(),
            ),
        )

        for (
            extension_id,
            node_id,
            expected_family,
            expected_marker,
            image_bytes,
            params,
            source_image_token,
        ) in cases:
            with self.subTest(extension_id=extension_id, node_id=node_id):
                runtime_root = self._make_runtime_root(extension_id)
                _, result = self._run_setup_success(extension_id, runtime_root=runtime_root)
                self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
                generator_class = self._load_generator_class(extension_id)
                original_runtime_root = generator_class.runtime_root
                generator_class.runtime_root = str(runtime_root)
                model_dir = self._make_model_dir(extension_id, node_id)
                outputs_dir = Path(tempfile.mkdtemp(prefix=f"outputs-{extension_id}-{node_id}-"))
                serialized_payloads: list[dict[str, object]] = []
                invocations: list[dict[str, object]] = []

                real_runner_side_effect = self._run_real_runner_popen(
                    loader_map={
                        (expected_family, node_id): self._make_real_runner_loader(
                            marker=expected_marker,
                            invocations=invocations,
                        )
                    },
                    source_image_token=source_image_token,
                )

                def capture_serialized_payload(command, *, stdin, stdout, stderr, text, bufsize, cwd, env):
                    self.assertIs(stdin, subprocess.PIPE)
                    self.assertIs(stdout, subprocess.PIPE)
                    self.assertIs(stderr, subprocess.PIPE)
                    self.assertTrue(text)
                    self.assertEqual(bufsize, 1)

                    def on_stdin_close(payload_text: str) -> tuple[list[str], list[str], int]:
                        serialized_payloads.append(json.loads(payload_text))
                        fake_process = real_runner_side_effect(
                            command,
                            stdin=stdin,
                            stdout=stdout,
                            stderr=stderr,
                            text=text,
                            bufsize=bufsize,
                            cwd=cwd,
                            env=env,
                        )
                        fake_process.stdin.write(payload_text)
                        fake_process.stdin.close()
                        return (
                            fake_process.stdout._lines,
                            fake_process.stderr._lines,
                            fake_process._expected_returncode,
                        )

                    return self._FakePopen(stdout_lines=[], stderr_lines=[], on_stdin_close=on_stdin_close)

                try:
                    with patch(
                        "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                        return_value=(True, "stubbed imports"),
                    ), patch(
                        "local_image_runtime.pipeline.subprocess.Popen",
                        side_effect=capture_serialized_payload,
                    ):
                        generator = generator_class(model_dir, outputs_dir)
                        actual_path = generator.generate(image_bytes, params)
                finally:
                    generator_class.runtime_root = original_runtime_root

                self.assertEqual(len(serialized_payloads), 1)
                self.assertEqual(serialized_payloads[0]["family"], expected_family)
                self.assertEqual(serialized_payloads[0]["node_id"], node_id)
                self.assertEqual(
                    serialized_payloads[0]["model_dir"],
                    str(model_dir.expanduser().resolve()),
                )
                self.assertEqual(len(invocations), 1)
                self.assertEqual(invocations[0]["model_dir"], str(model_dir.expanduser().resolve()))
                self.assertTrue(actual_path.exists())
                self.assertTrue(str(actual_path).startswith(str(outputs_dir)))

    def test_extension_generator_generate_maps_image_to_image_request(self) -> None:
        cases = (
            ("sd15", {"prompt": "variation", "strength": 0.35, "steps": 4}),
            ("sdxl-base", {"strength": 0.8, "guidance_scale": 6.5}),
        )

        for extension_id, params in cases:
            with self.subTest(extension_id=extension_id, params=params):
                generator_class = self._load_generator_class(extension_id)
                model_dir = self._make_model_dir(extension_id, "image-to-image")
                outputs_dir = Path(tempfile.mkdtemp(prefix=f"outputs-{extension_id}-"))
                runtime_snapshot = object()
                result_path = outputs_dir / f"{extension_id}-variation.png"
                image_bytes = b"fake-image-bytes"

                def execute_side_effect(request, runtime, extension_id, emit_progress, emit_log):
                    self.assertEqual(request.node_id, "image-to-image")
                    self.assertEqual(request.workspace_dir, str(outputs_dir))
                    materialized_input = Path(request.input["filePath"])
                    self.assertTrue(materialized_input.is_absolute())
                    self.assertTrue(materialized_input.exists())
                    self.assertEqual(materialized_input.read_bytes(), image_bytes)
                    self.assertEqual(materialized_input.parent, outputs_dir / ".modly-inputs")
                    self.assertEqual(request.params.get("strength"), params["strength"])
                    return {"output_path": str(result_path)}

                with patch(
                    "local_image_runtime.runtime_adapter.bootstrap_runtime",
                    return_value=runtime_snapshot,
                ), patch(
                    "local_image_runtime.runtime_adapter.execute",
                    side_effect=execute_side_effect,
                ):
                    generator = generator_class(model_dir, outputs_dir)
                    actual_path = generator.generate(image_bytes, params)

                self.assertEqual(actual_path, result_path)

    def test_build_generate_request_for_text_to_image_forwards_effective_model_dir_override(self) -> None:
        generator_class = self._load_generator_class("sd15")
        model_dir = self._make_model_dir("sd15", "text-to-image")
        outputs_dir = Path(tempfile.mkdtemp(prefix="outputs-sd15-"))
        generator = generator_class(model_dir, outputs_dir)

        params = {
            "prompt": "generator prompt sd15",
            "steps": 4,
            "input": {"text": "legacy generator text sd15"},
        }

        request = generator._build_generate_request(b"ignored-image-bytes", params)

        self.assertEqual(request.node_id, "text-to-image")
        self.assertEqual(request.input, {"text": "legacy generator text sd15"})
        self.assertEqual(request.workspace_dir, str(outputs_dir))
        self.assertEqual(
            request.model_dir_override,
            str(model_dir.expanduser().resolve()),
        )

    def test_build_generate_request_for_image_to_image_forwards_effective_model_dir_override(self) -> None:
        generator_class = self._load_generator_class("sdxl-base")
        model_dir = self._make_model_dir("sdxl-base", "image-to-image")
        outputs_dir = Path(tempfile.mkdtemp(prefix="outputs-sdxl-base-"))
        generator = generator_class(model_dir, outputs_dir)

        request = generator._build_generate_request(
            b"fake-image-bytes",
            {"prompt": "variation", "strength": 0.35, "steps": 4},
        )

        self.assertEqual(request.node_id, "image-to-image")
        self.assertEqual(request.workspace_dir, str(outputs_dir))
        self.assertEqual(
            request.model_dir_override,
            str(model_dir.expanduser().resolve()),
        )
        materialized_input = Path(request.input["filePath"])
        self.assertTrue(materialized_input.is_absolute())
        self.assertTrue(materialized_input.exists())
        self.assertEqual(materialized_input.read_bytes(), b"fake-image-bytes")
        self.assertEqual(materialized_input.parent, outputs_dir / ".modly-inputs")

    def test_extension_generator_generate_rejects_invalid_image_to_image_strength(self) -> None:
        generator_class = self._load_generator_class("sd15")
        model_dir = self._make_model_dir("sd15", "image-to-image")
        outputs_dir = Path(tempfile.mkdtemp(prefix="outputs-sd15-"))
        runtime_snapshot = object()

        def validate_request_only(request, runtime, extension_id, emit_progress, emit_log):
            pipeline._validate_node_payload(request, legacy_model_id=None)
            return {"output_path": str(outputs_dir / "never-created.png")}

        with patch(
            "local_image_runtime.runtime_adapter.bootstrap_runtime",
            return_value=runtime_snapshot,
        ), patch(
            "local_image_runtime.runtime_adapter.execute",
            side_effect=validate_request_only,
        ):
            generator = generator_class(model_dir, outputs_dir)
            with self.assertRaisesRegex(
                pipeline.RequestValidationError,
                "image-to-image requires params.strength between 0.0 and 1.0",
            ):
                generator.generate(b"fake-image-bytes", {"prompt": "variation"})

    def test_pipeline_validate_node_payload_rejects_nonexistent_image_to_image_source_file(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-missing-source-"))
        request = pipeline.ExecutionRequest(
            node_id="image-to-image",
            input={"filePath": "missing-source.png"},
            params={"prompt": "variation", "strength": 0.55, "steps": 4},
            workspace_dir=str(workspace_dir),
        )

        with self.assertRaisesRegex(
            pipeline.RequestValidationError,
            "image-to-image input.filePath must point to an existing local file",
        ):
            pipeline._validate_node_payload(request, legacy_model_id=None)

    def test_extension_generator_generate_accepts_output_path_within_outputs_dir(self) -> None:
        generator_class = self._load_generator_class("sd15")
        model_dir = self._make_model_dir("sd15", "text-to-image")
        outputs_dir = Path(tempfile.mkdtemp(prefix="outputs-sd15-"))
        runtime_snapshot = object()
        nested_dir = outputs_dir / "nested"
        nested_dir.mkdir(parents=True, exist_ok=True)
        expected_path = nested_dir / "result.png"

        with patch(
            "local_image_runtime.runtime_adapter.bootstrap_runtime",
            return_value=runtime_snapshot,
        ), patch(
            "local_image_runtime.runtime_adapter.execute",
            return_value={"output_path": str(expected_path)},
        ):
            generator = generator_class(model_dir, outputs_dir)
            actual_path = generator.generate(b"", {"prompt": "contained output"})

        self.assertEqual(actual_path, expected_path)

    def test_extension_generator_generate_rejects_output_path_outside_outputs_dir(self) -> None:
        generator_class = self._load_generator_class("sd15")
        model_dir = self._make_model_dir("sd15", "text-to-image")
        outputs_dir = Path(tempfile.mkdtemp(prefix="outputs-sd15-"))
        runtime_snapshot = object()
        outside_path = outputs_dir.parent / "escaped.png"

        with patch(
            "local_image_runtime.runtime_adapter.bootstrap_runtime",
            return_value=runtime_snapshot,
        ), patch(
            "local_image_runtime.runtime_adapter.execute",
            return_value={"output_path": str(outside_path)},
        ):
            generator = generator_class(model_dir, outputs_dir)
            with self.assertRaisesRegex(
                runtime_adapter.DomainError,
                "outside configured outputs_dir",
            ):
                generator.generate(b"", {"prompt": "escaped output"})

    def test_pipeline_execute_serializes_subprocess_payload_by_family_and_node(self) -> None:
        cases = (
            (
                "sd15",
                pipeline.ExecutionRequest(
                    node_id="text-to-image",
                    input={"text": "legacy lighthouse prompt"},
                    params={
                        "prompt": "lighthouse at dusk",
                        "negative_prompt": "blurry",
                        "steps": 4,
                        "width": 512,
                        "height": 512,
                        "guidance_scale": 7.5,
                        "seed": 42,
                    },
                    model_dir_override="/models/modly/sd15",
                ),
                "stable-diffusion",
                None,
                "/models/modly/sd15",
                "stable-text",
            ),
            (
                "sdxl-base",
                pipeline.ExecutionRequest(
                    node_id="image-to-image",
                    input={},
                    params={
                        "prompt": "cinematic variation",
                        "negative_prompt": "low quality",
                        "strength": 0.55,
                        "steps": 5,
                    },
                ),
                "sdxl",
                "source.png",
                None,
                "sdxl-image",
            ),
        )

        for extension_id, request, expected_family, source_name, expected_model_dir, expected_marker in cases:
            with self.subTest(extension_id=extension_id, node_id=request.node_id):
                workspace_dir = Path(tempfile.mkdtemp(prefix=f"workspace-{extension_id}-"))
                runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
                runtime_root = Path(tempfile.mkdtemp(prefix=f"ext-root-{extension_id}-"))
                (runtime_root / "src").mkdir(parents=True, exist_ok=True)
                venv_python = self._make_executable_python(runtime_root)
                extension_record = {
                    "venv_python": str(venv_python),
                    "model_dir": str(runtime.paths.models_dir / extension_id),
                }
                progress_events: list[tuple[int, str]] = []
                logs: list[str] = []
                serialized_payloads: list[dict[str, object]] = []
                invocations: list[dict[str, object]] = []

                if source_name is not None:
                    source_path = workspace_dir / source_name
                    source_path.write_bytes(b"fake-image")
                    source_image_token = object()
                    request = pipeline.ExecutionRequest(
                        node_id=request.node_id,
                        input={"filePath": source_name},
                        params=request.params,
                        workspace_dir=str(workspace_dir),
                        model_dir_override=request.model_dir_override,
                    )
                else:
                    source_path = None
                    source_image_token = None
                    request = pipeline.ExecutionRequest(
                        node_id=request.node_id,
                        input=request.input,
                        params=request.params,
                        workspace_dir=str(workspace_dir),
                        model_dir_override=request.model_dir_override,
                    )

                real_runner_side_effect = self._run_real_runner_popen(
                    loader_map={
                        (expected_family, request.node_id): self._make_real_runner_loader(
                            marker=expected_marker,
                            invocations=invocations,
                        )
                    },
                    source_image_token=source_image_token,
                )

                def capture_serialized_payload(command, *, stdin, stdout, stderr, text, bufsize, cwd, env):
                    def on_stdin_close(payload_text: str) -> tuple[list[str], list[str], int]:
                        serialized_payloads.append(json.loads(payload_text))
                        fake_process = real_runner_side_effect(
                            command,
                            stdin=stdin,
                            stdout=stdout,
                            stderr=stderr,
                            text=text,
                            bufsize=bufsize,
                            cwd=cwd,
                            env=env,
                        )
                        fake_process.stdin.write(payload_text)
                        fake_process.stdin.close()
                        return (
                            list(fake_process.stdout._lines),
                            list(fake_process.stderr._lines),
                            fake_process._expected_returncode,
                        )

                    return self._FakePopen(
                        stdout_lines=[],
                        stderr_lines=[],
                        on_stdin_close=on_stdin_close,
                    )

                with patch(
                    "local_image_runtime.pipeline.extension_is_installed",
                    return_value=True,
                ), patch(
                    "local_image_runtime.pipeline.get_extension_record",
                    return_value=extension_record,
                ), patch(
                    "local_image_runtime.pipeline.subprocess.Popen",
                    side_effect=capture_serialized_payload,
                ):
                    result = pipeline.execute(
                        request,
                        runtime,
                        extension_id=extension_id,
                        emit_progress=lambda percent, label: progress_events.append((percent, label)),
                        emit_log=logs.append,
                    )

                self.assertEqual(len(serialized_payloads), 1)
                serialized_payload = serialized_payloads[0]
                self.assertEqual(serialized_payload["extension_id"], extension_id)
                self.assertEqual(serialized_payload["family"], expected_family)
                self.assertEqual(serialized_payload["node_id"], request.node_id)
                self.assertEqual(serialized_payload["workspace_dir"], str(workspace_dir))
                self.assertEqual(
                    serialized_payload["model_dir"],
                    expected_model_dir or extension_record["model_dir"],
                )
                self.assertEqual(serialized_payload["prompt"], request.params.get("prompt"))
                self.assertEqual(serialized_payload["negative_prompt"], request.params.get("negative_prompt"))
                self.assertEqual(
                    serialized_payload["params"],
                    {key: value for key, value in request.params.items() if key != "negative_prompt"},
                )
                if source_path is None:
                    self.assertIsNone(serialized_payload["source_image_path"])
                else:
                    self.assertEqual(serialized_payload["source_image_path"], str(source_path.resolve()))

                expected_output_path = Path(str(serialized_payload["output_path"])).resolve()
                self.assertEqual(
                    result,
                    {
                        "output_path": str(expected_output_path),
                        "metadata": {
                            "family": expected_family,
                            "node_id": request.node_id,
                            "seed": request.params.get("seed"),
                            "negative_prompt_used": bool(request.params.get("negative_prompt")),
                            "source_image_used": source_path is not None,
                        },
                    },
                )
                self.assertEqual(
                    progress_events,
                    [
                        (35, "validating-request"),
                        (55, "checking-extension"),
                        (75, "backend-dispatch"),
                        (80, "loading-pipeline"),
                        (90, "running-inference"),
                        (95, "saving-output"),
                    ],
                )
                self.assertEqual(len(invocations), 1)
                self.assertEqual(invocations[0]["marker"], expected_marker)
                if source_image_token is None:
                    self.assertNotIn("image", invocations[0]["kwargs"])
                else:
                    self.assertIs(invocations[0]["kwargs"]["image"], source_image_token)
                self.assertIn(f"Validated node '{request.node_id}' for extension '{extension_id}'", logs[0])
                self.assertIn(f"Workspace: {workspace_dir}.", logs[0])
                self.assertEqual(
                    logs[1],
                    f"Dispatching backend family '{expected_family}' for node '{request.node_id}' using venv '{venv_python}'.",
                )
                self.assertEqual(
                    logs[2:],
                    [
                        "Loading inference pipeline.",
                        "Running inference.",
                        "Saving output image.",
                    ],
                )

    def test_build_backend_job_prefers_model_dir_override_over_extension_record(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-job-override-"))
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-job-override-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(extension_root)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "lighthouse"},
            params={"prompt": "lighthouse", "steps": 4},
            workspace_dir=str(workspace_dir),
            model_dir_override="/models/modly/sdxl",
        )
        payload_details = pipeline.ValidatedPayload(
            prompt="lighthouse",
            source_image_path=None,
            numeric_params={"steps": 4},
            legacy_model_id=None,
        )

        job = pipeline._build_backend_job(
            request=request,
            extension_id="sd15",
            extension_record={
                "venv_python": str(venv_python),
                "model_dir": "/runtime/local/sdxl",
            },
            payload_details=payload_details,
            effective_workspace_dir=str(workspace_dir),
        )

        self.assertEqual(job.payload["model_dir"], "/models/modly/sdxl")

    def test_build_backend_job_falls_back_to_extension_record_when_override_missing(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-job-fallback-"))
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-job-fallback-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(extension_root)
        payload_details = pipeline.ValidatedPayload(
            prompt="forest",
            source_image_path=None,
            numeric_params={"steps": 8},
            legacy_model_id=None,
        )

        for raw_override in (None, "", "   "):
            with self.subTest(model_dir_override=raw_override):
                request = pipeline.ExecutionRequest(
                    node_id="text-to-image",
                    input={"text": "forest"},
                    params={"prompt": "forest", "steps": 8},
                    workspace_dir=str(workspace_dir),
                    model_dir_override=raw_override,
                )

                job = pipeline._build_backend_job(
                    request=request,
                    extension_id="sd15",
                    extension_record={
                        "venv_python": str(venv_python),
                        "model_dir": "/runtime/local/sdxl",
                    },
                    payload_details=payload_details,
                    effective_workspace_dir=str(workspace_dir),
                )

                self.assertEqual(job.payload["model_dir"], "/runtime/local/sdxl")

    def test_pipeline_execute_prepends_existing_host_pythonpath_without_losing_env(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-pythonpath-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        runtime_root = Path(tempfile.mkdtemp(prefix="ext-root-pythonpath-"))
        runtime_src = runtime_root / "src"
        runtime_src.mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(runtime_root)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "preserve pythonpath"},
            params={"prompt": "preserve pythonpath", "steps": 4},
            workspace_dir=str(workspace_dir),
        )
        expected_output_path = workspace_dir / "pythonpath-output.png"

        def popen_side_effect(command, *, stdin, stdout, stderr, text, bufsize, cwd, env):
            self.assertEqual(command, [str(venv_python), "-m", "local_image_runtime.inference_runner"])
            self.assertIs(stdin, subprocess.PIPE)
            self.assertIs(stdout, subprocess.PIPE)
            self.assertIs(stderr, subprocess.PIPE)
            self.assertTrue(text)
            self.assertEqual(bufsize, 1)
            self.assertEqual(cwd, str(runtime_src))
            self.assertEqual(
                env["PYTHONPATH"],
                str(runtime_src) + os.pathsep + "/host/a:/host/b",
            )
            self.assertEqual(env["KEEP_ME"], "1")

            def on_stdin_close(payload_text: str) -> tuple[list[str], list[str], int]:
                payload = json.loads(payload_text)
                self.assertEqual(payload["workspace_dir"], str(workspace_dir))
                return (
                    [json.dumps({"type": "done", "result": {"output_path": str(expected_output_path)}}) + "\n"],
                    [],
                    0,
                )

            return self._FakePopen(
                stdout_lines=[],
                stderr_lines=[],
                on_stdin_close=on_stdin_close,
            )

        with patch.dict(os.environ, {"PYTHONPATH": "/host/a:/host/b", "KEEP_ME": "1"}, clear=True), patch(
            "local_image_runtime.pipeline.extension_is_installed",
            return_value=True,
        ), patch(
            "local_image_runtime.pipeline.get_extension_record",
            return_value={
                "venv_python": str(venv_python),
                "model_dir": str(runtime.paths.models_dir / "sd15"),
            },
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            side_effect=popen_side_effect,
        ):
            result = pipeline.execute(
                request,
                runtime,
                extension_id="sd15",
                emit_progress=lambda percent, label: None,
                emit_log=lambda message: None,
            )

        self.assertEqual(result, {"output_path": str(expected_output_path.resolve())})

    def test_pipeline_execute_fails_before_spawn_when_runtime_src_is_missing(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-missing-src-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        runtime_root = Path(tempfile.mkdtemp(prefix="ext-root-missing-src-"))
        venv_python = self._make_executable_python(runtime_root)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "missing runtime src"},
            params={"prompt": "missing runtime src", "steps": 4},
            workspace_dir=str(workspace_dir),
        )

        with patch(
            "local_image_runtime.pipeline.extension_is_installed",
            return_value=True,
        ), patch(
            "local_image_runtime.pipeline.get_extension_record",
            return_value={
                "venv_python": str(venv_python),
                "model_dir": str(runtime.paths.models_dir / "sd15"),
            },
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
        ) as subprocess_popen:
            with self.assertRaisesRegex(
                pipeline.DomainError,
                "Missing vendored runtime src for extension 'sd15'",
            ):
                pipeline.execute(
                    request,
                    runtime,
                    extension_id="sd15",
                    emit_progress=lambda percent, label: None,
                    emit_log=lambda message: None,
                )

        subprocess_popen.assert_not_called()

    def test_build_backend_job_derives_runtime_src_from_posix_venv_without_ext_dir(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-job-posix-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-job-posix-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(extension_root)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "lighthouse"},
            params={"prompt": "lighthouse", "steps": 4},
            workspace_dir=str(workspace_dir),
        )
        payload_details = pipeline.ValidatedPayload(
            prompt="lighthouse",
            source_image_path=None,
            numeric_params={"steps": 4},
            legacy_model_id=None,
        )

        with patch.dict(os.environ, {}, clear=True):
            job = pipeline._build_backend_job(
                request=request,
                extension_id="sd15",
                extension_record={
                    "venv_python": str(venv_python),
                    "model_dir": str(runtime.paths.models_dir / "sd15"),
                },
                payload_details=payload_details,
                effective_workspace_dir=str(workspace_dir),
            )

        self.assertEqual(
            job.command,
            (str(venv_python), "-m", "local_image_runtime.inference_runner"),
        )
        self.assertEqual(job.cwd, extension_root / "src")
        self.assertEqual(job.workspace_dir, workspace_dir.resolve())
        self.assertEqual(job.env["PYTHONPATH"], str(extension_root / "src"))

    def test_build_backend_job_supports_windows_venv_layout_and_prepends_pythonpath(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-job-windows-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-job-windows-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_windows_executable_python(extension_root)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "forest"},
            params={"prompt": "forest", "steps": 8},
            workspace_dir=str(workspace_dir),
        )
        payload_details = pipeline.ValidatedPayload(
            prompt="forest",
            source_image_path=None,
            numeric_params={"steps": 8},
            legacy_model_id=None,
        )

        with patch.dict(os.environ, {"PYTHONPATH": "/host/a:/host/b", "KEEP_ME": "1"}, clear=True):
            job = pipeline._build_backend_job(
                request=request,
                extension_id="sd15",
                extension_record={
                    "venv_python": str(venv_python),
                    "model_dir": str(runtime.paths.models_dir / "sd15"),
                },
                payload_details=payload_details,
                effective_workspace_dir=str(workspace_dir),
            )

        self.assertEqual(
            job.command,
            (str(venv_python), "-m", "local_image_runtime.inference_runner"),
        )
        self.assertEqual(job.cwd, extension_root / "src")
        self.assertEqual(
            job.env["PYTHONPATH"],
            str(extension_root / "src") + os.pathsep + "/host/a:/host/b",
        )
        self.assertEqual(job.env["KEEP_ME"], "1")

    def test_pipeline_execute_requires_executable_venv_python_before_spawn(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-venv-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "venv check"},
            params={"prompt": "venv check", "steps": 4},
            workspace_dir=str(workspace_dir),
        )
        runtime_root = Path(tempfile.mkdtemp(prefix="ext-root-venv-"))
        missing_python = runtime_root / "venv" / "bin" / "python"
        non_executable_python = runtime_root / "venv" / "bin" / "python-not-executable"
        non_executable_python.parent.mkdir(parents=True, exist_ok=True)
        non_executable_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        cases = (
            ({"venv_python": None, "model_dir": str(runtime.paths.models_dir / "sd15")}, "Missing executable venv_python"),
            ({"venv_python": str(missing_python), "model_dir": str(runtime.paths.models_dir / "sd15")}, str(missing_python)),
            ({"venv_python": str(non_executable_python), "model_dir": str(runtime.paths.models_dir / "sd15")}, str(non_executable_python)),
        )

        for extension_record, expected_detail in cases:
            with self.subTest(venv_python=extension_record["venv_python"]):
                with patch(
                    "local_image_runtime.pipeline.extension_is_installed",
                    return_value=True,
                ), patch(
                    "local_image_runtime.pipeline.get_extension_record",
                    return_value=extension_record,
                ), patch(
                    "local_image_runtime.pipeline.subprocess.Popen",
                ) as subprocess_popen:
                    with self.assertRaisesRegex(pipeline.DomainError, expected_detail):
                        pipeline.execute(
                            request,
                            runtime,
                            extension_id="sd15",
                            emit_progress=lambda percent, label: None,
                            emit_log=lambda message: None,
                        )

                subprocess_popen.assert_not_called()

    def test_pipeline_execute_rejects_flux_image_to_image_before_spawn(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-flux-img2img-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        source_path = workspace_dir / "source.png"
        source_path.write_bytes(b"fake-image")
        request = pipeline.ExecutionRequest(
            node_id="image-to-image",
            input={"filePath": str(source_path)},
            params={"prompt": "variation", "strength": 0.55, "steps": 4},
            workspace_dir=str(workspace_dir),
        )

        with patch(
            "local_image_runtime.pipeline.extension_is_installed",
            return_value=True,
        ), patch(
            "local_image_runtime.pipeline.get_extension_record",
            return_value={
                "venv_python": str(self._make_executable_python(Path(tempfile.mkdtemp(prefix="ext-root-flux-")))),
                "model_dir": str(runtime.paths.models_dir / "flux-schnell"),
            },
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
        ) as subprocess_popen:
            with self.assertRaisesRegex(
                pipeline.RequestValidationError,
                "Extension 'flux-schnell' does not support node 'image-to-image'",
            ):
                pipeline.execute(
                    request,
                    runtime,
                    extension_id="flux-schnell",
                    emit_progress=lambda percent, label: None,
                    emit_log=lambda message: None,
                )

        subprocess_popen.assert_not_called()

    def test_pipeline_execute_rejects_child_output_path_outside_workspace_dir(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-output-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        runtime_root = Path(tempfile.mkdtemp(prefix="ext-root-output-"))
        (runtime_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(runtime_root)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "workspace guard"},
            params={"prompt": "workspace guard", "steps": 4},
            workspace_dir=str(workspace_dir),
        )
        outside_path = workspace_dir.parent / "escaped.png"

        with patch(
            "local_image_runtime.pipeline.extension_is_installed",
            return_value=True,
        ), patch(
            "local_image_runtime.pipeline.get_extension_record",
            return_value={
                "venv_python": str(venv_python),
                "model_dir": str(runtime.paths.models_dir / "sd15"),
            },
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            return_value=self._FakePopen(
                stdout_lines=[json.dumps({"type": "done", "result": {"output_path": str(outside_path)}}) + "\n"],
                stderr_lines=[],
            ),
        ):
            with self.assertRaisesRegex(pipeline.DomainError, "outside workspace_dir"):
                pipeline.execute(
                    request,
                    runtime,
                    extension_id="sd15",
                    emit_progress=lambda percent, label: None,
                    emit_log=lambda message: None,
                )

    def test_pipeline_execute_uses_request_workspace_dir_in_logs_and_errors(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-override-"))
        runtime_outputs_dir = Path(tempfile.mkdtemp(prefix="runtime-outputs-"))
        runtime_models_dir = Path(tempfile.mkdtemp(prefix="runtime-models-"))
        runtime_root = Path(tempfile.mkdtemp(prefix="workspace-override-ext-"))
        (runtime_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(runtime_root)
        runtime = SimpleNamespace(
            paths=SimpleNamespace(outputs_dir=runtime_outputs_dir, models_dir=runtime_models_dir)
        )
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "workspace prompt"},
            params={"prompt": "workspace prompt", "steps": 4},
            workspace_dir=str(workspace_dir),
        )
        progress_events: list[tuple[int, str]] = []
        logs: list[str] = []
        invocations: list[dict[str, object]] = []

        real_runner_side_effect = self._run_real_runner_popen(
            loader_map={
                ("stable-diffusion", "text-to-image"): self._make_real_runner_loader(
                    marker="workspace-override",
                    invocations=invocations,
                )
            }
        )

        with patch(
            "local_image_runtime.pipeline.extension_is_installed",
            return_value=True,
        ), patch(
            "local_image_runtime.pipeline.get_extension_record",
            return_value={
                "venv_python": str(venv_python),
                "model_dir": str(runtime_models_dir / "sd15"),
            },
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            side_effect=real_runner_side_effect,
        ):
            result = pipeline.execute(
                request,
                runtime,
                extension_id="sd15",
                emit_progress=lambda percent, label: progress_events.append((percent, label)),
                emit_log=logs.append,
            )

        self.assertIn(f"Workspace: {workspace_dir}.", logs[0])
        self.assertTrue(result["output_path"].startswith(str(workspace_dir)))
        self.assertEqual(
            progress_events,
            [
                (35, "validating-request"),
                (55, "checking-extension"),
                (75, "backend-dispatch"),
                (80, "loading-pipeline"),
                (90, "running-inference"),
                (95, "saving-output"),
            ],
        )
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0]["marker"], "workspace-override")
        self.assertEqual(
            logs[2:],
            [
                "Loading inference pipeline.",
                "Running inference.",
                "Saving output image.",
            ],
        )

    def test_run_backend_job_streams_progress_and_logs_before_child_exit(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-streaming-progress-"))
        job = self._make_backend_job(workspace_dir=workspace_dir)
        fake_process = self._FakePopen(
            stdout_lines=[
                json.dumps({"type": "progress", "percent": 80, "label": "loading-pipeline"}) + "\n",
                json.dumps({"type": "log", "message": "warming backend"}) + "\n",
                json.dumps({"type": "done", "result": {"output_path": job.payload["output_path"]}}) + "\n",
            ],
            stderr_lines=[],
        )
        progress_events: list[tuple[int, str]] = []
        logs: list[str] = []

        def emit_progress(percent: int, label: str) -> None:
            self.assertFalse(fake_process.wait_called)
            progress_events.append((percent, label))

        def emit_log(message: str) -> None:
            self.assertFalse(fake_process.wait_called)
            logs.append(message)

        with patch("local_image_runtime.pipeline.subprocess.run") as subprocess_run, patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            return_value=fake_process,
        ):
            result = pipeline._run_backend_job(
                job,
                emit_progress=emit_progress,
                emit_log=emit_log,
            )

        subprocess_run.assert_not_called()
        self.assertEqual(fake_process.stdin.value, json.dumps(job.payload) + "\n")
        self.assertEqual(progress_events, [(80, "loading-pipeline")])
        self.assertEqual(logs, ["warming backend"])
        self.assertEqual(result, {"output_path": str(Path(job.payload["output_path"]).resolve())})

    def test_run_backend_job_drains_stderr_separately_from_stdout_ndjson(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-streaming-stderr-"))
        job = self._make_backend_job(workspace_dir=workspace_dir)
        fake_process = self._FakePopen(
            stdout_lines=[
                json.dumps({"type": "progress", "percent": 90, "label": "running-inference"}) + "\n",
                json.dumps({"type": "done", "result": {"output_path": job.payload["output_path"]}}) + "\n",
            ],
            stderr_lines=["{not-json}\n", '{"type":"error","message":"stderr-only"}\n', "gpu warning\n"],
        )
        progress_events: list[tuple[int, str]] = []

        with patch("local_image_runtime.pipeline.subprocess.run") as subprocess_run, patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            return_value=fake_process,
        ):
            result = pipeline._run_backend_job(
                job,
                emit_progress=lambda percent, label: progress_events.append((percent, label)),
                emit_log=lambda message: self.fail(f"Unexpected log forwarded: {message}"),
            )

        subprocess_run.assert_not_called()
        self.assertGreater(fake_process.stderr.read_count, 0)
        self.assertEqual(progress_events, [(90, "running-inference")])
        self.assertEqual(result, {"output_path": str(Path(job.payload["output_path"]).resolve())})

    def test_run_backend_job_raises_protocol_error_for_invalid_stdout_line(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-streaming-invalid-"))
        job = self._make_backend_job(workspace_dir=workspace_dir)
        fake_process = self._FakePopen(
            stdout_lines=[
                json.dumps({"type": "progress", "percent": 95, "label": "saving-output"}) + "\n",
                "{not-json}\n",
            ],
            stderr_lines=["child warning\n"],
        )
        progress_events: list[tuple[int, str]] = []

        with patch("local_image_runtime.pipeline.subprocess.run") as subprocess_run, patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            return_value=fake_process,
        ):
            with self.assertRaisesRegex(pipeline.DomainError, "invalid NDJSON"):
                pipeline._run_backend_job(
                    job,
                    emit_progress=lambda percent, label: progress_events.append((percent, label)),
                    emit_log=lambda message: None,
                )

        subprocess_run.assert_not_called()
        self.assertEqual(progress_events, [(95, "saving-output")])
        self.assertGreater(fake_process.stderr.read_count, 0)

    def test_run_backend_job_aborts_hung_child_on_total_timeout(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-streaming-timeout-total-"))
        job = self._make_backend_job(workspace_dir=workspace_dir)
        fake_process = self._FakePopen(
            stdout_lines=[],
            stderr_lines=[],
            wait_timeout_after_terminate=True,
        )
        clock = self._ScriptedClock()
        scripted_queue = self._ScriptedQueue(
            clock=clock,
            items=[
                (1.0, self._ScriptedQueue.EMPTY),
                (6.2, self._ScriptedQueue.EMPTY),
            ],
        )

        with patch("local_image_runtime.pipeline._read_stream", side_effect=lambda *args, **kwargs: None), patch(
            "local_image_runtime.pipeline.queue.Queue",
            return_value=scripted_queue,
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            return_value=fake_process,
        ):
            with self.assertRaisesRegex(pipeline.DomainError, "total backend timeout"):
                pipeline._run_backend_job(
                    job,
                    emit_progress=lambda percent, label: None,
                    emit_log=lambda message: None,
                    timeout_config=pipeline.BackendTimeoutConfig(
                        total_seconds=5.0,
                        idle_seconds=30.0,
                        terminate_grace_seconds=0.25,
                        poll_seconds=0.1,
                    ),
                    monotonic=clock.monotonic,
                )

        self.assertTrue(fake_process.terminate_called)
        self.assertTrue(fake_process.kill_called)

    def test_run_backend_job_reports_idle_timeout_with_last_stage(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-streaming-timeout-idle-"))
        job = self._make_backend_job(workspace_dir=workspace_dir)
        fake_process = self._FakePopen(stdout_lines=[], stderr_lines=[])
        clock = self._ScriptedClock()
        scripted_queue = self._ScriptedQueue(
            clock=clock,
            items=[
                (
                    0.5,
                    (
                        "line",
                        "stdout",
                        json.dumps({"type": "progress", "percent": 90, "label": "running-inference"}) + "\n",
                    ),
                ),
                (1.9, self._ScriptedQueue.EMPTY),
                (3.0, self._ScriptedQueue.EMPTY),
            ],
        )
        progress_events: list[tuple[int, str]] = []

        with patch("local_image_runtime.pipeline._read_stream", side_effect=lambda *args, **kwargs: None), patch(
            "local_image_runtime.pipeline.queue.Queue",
            return_value=scripted_queue,
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            return_value=fake_process,
        ):
            with self.assertRaisesRegex(pipeline.DomainError, "running-inference"):
                pipeline._run_backend_job(
                    job,
                    emit_progress=lambda percent, label: progress_events.append((percent, label)),
                    emit_log=lambda message: None,
                    timeout_config=pipeline.BackendTimeoutConfig(
                        total_seconds=10.0,
                        idle_seconds=2.0,
                        terminate_grace_seconds=0.25,
                        poll_seconds=0.1,
                    ),
                    monotonic=clock.monotonic,
                )

        self.assertEqual(progress_events, [(90, "running-inference")])
        self.assertTrue(fake_process.terminate_called)

    def test_run_backend_job_resets_idle_watchdog_on_progress_or_log_activity(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-streaming-timeout-reset-"))
        job = self._make_backend_job(workspace_dir=workspace_dir)
        fake_process = self._FakePopen(stdout_lines=[], stderr_lines=[])
        clock = self._ScriptedClock()
        scripted_queue = self._ScriptedQueue(
            clock=clock,
            items=[
                (
                    0.5,
                    (
                        "line",
                        "stdout",
                        json.dumps({"type": "progress", "percent": 90, "label": "running-inference"}) + "\n",
                    ),
                ),
                (2.4, self._ScriptedQueue.EMPTY),
                (
                    2.49,
                    ("line", "stdout", json.dumps({"type": "log", "message": "still alive"}) + "\n"),
                ),
                (4.3, self._ScriptedQueue.EMPTY),
                (
                    4.35,
                    (
                        "line",
                        "stdout",
                        json.dumps({"type": "done", "result": {"output_path": job.payload["output_path"]}}) + "\n",
                    ),
                ),
                (4.36, ("eof", "stdout", None)),
                (4.36, ("eof", "stderr", None)),
            ],
        )
        progress_events: list[tuple[int, str]] = []
        logs: list[str] = []

        with patch("local_image_runtime.pipeline._read_stream", side_effect=lambda *args, **kwargs: None), patch(
            "local_image_runtime.pipeline.queue.Queue",
            return_value=scripted_queue,
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            return_value=fake_process,
        ):
            result = pipeline._run_backend_job(
                job,
                emit_progress=lambda percent, label: progress_events.append((percent, label)),
                emit_log=logs.append,
                timeout_config=pipeline.BackendTimeoutConfig(
                    total_seconds=10.0,
                    idle_seconds=2.0,
                    terminate_grace_seconds=0.25,
                    poll_seconds=0.1,
                ),
                monotonic=clock.monotonic,
            )

        self.assertEqual(progress_events, [(90, "running-inference")])
        self.assertEqual(logs, ["still alive"])
        self.assertEqual(result, {"output_path": str(Path(job.payload["output_path"]).resolve())})
        self.assertFalse(fake_process.terminate_called)

    def test_inference_runner_reads_json_job_selects_loader_and_emits_done(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FakeImage:
            def __init__(self) -> None:
                self.saved_paths: list[str] = []

            def save(self, output_path: str) -> None:
                self.saved_paths.append(output_path)

        class FakePipeline:
            def __init__(self, *, marker: str) -> None:
                self.marker = marker
                self.calls: list[dict[str, object]] = []
                self.output_image = FakeImage()

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(images=[self.output_image])

        cases = (
            {
                "family": "stable-diffusion",
                "node_id": "text-to-image",
                "expected_loader": "stable-text",
                "source_image_path": None,
            },
            {
                "family": "sdxl",
                "node_id": "image-to-image",
                "expected_loader": "sdxl-image",
                "source_image_path": str(Path(tempfile.mkdtemp(prefix="runner-source-")) / "source.png"),
            },
            {
                "family": "flux",
                "node_id": "text-to-image",
                "expected_loader": "flux-text",
                "source_image_path": None,
            },
        )

        for case in cases:
            with self.subTest(family=case["family"], node_id=case["node_id"]):
                workspace_dir = Path(tempfile.mkdtemp(prefix="runner-workspace-"))
                output_path = workspace_dir / f"{case['family']}-{case['node_id']}.png"
                source_image_token = object()
                fake_pipeline = FakePipeline(marker=str(case["expected_loader"]))
                fake_loader = SimpleNamespace(from_pretrained=lambda model_dir: fake_pipeline)
                job = {
                    "extension_id": "test-extension",
                    "family": case["family"],
                    "node_id": case["node_id"],
                    "model_dir": str(workspace_dir / "model"),
                    "workspace_dir": str(workspace_dir),
                    "output_path": str(output_path),
                    "prompt": "test prompt",
                    "negative_prompt": "avoid blur",
                    "source_image_path": case["source_image_path"],
                    "params": {
                        "steps": 4,
                        "width": 512,
                        "height": 512,
                        "guidance_scale": 7.5,
                        "strength": 0.55,
                        "seed": 42,
                    },
                }

                stdin = StringIO(json.dumps(job) + "\n")
                stdout = StringIO()

                with patch.dict(
                    inference_runner._PIPELINE_LOADERS,
                    {
                        ("stable-diffusion", "text-to-image"): fake_loader,
                        ("stable-diffusion", "image-to-image"): SimpleNamespace(from_pretrained=lambda model_dir: None),
                        ("sdxl", "text-to-image"): SimpleNamespace(from_pretrained=lambda model_dir: None),
                        ("sdxl", "image-to-image"): fake_loader,
                        ("flux", "text-to-image"): fake_loader,
                    },
                    clear=True,
                ), patch.object(
                    inference_runner, "_seeded_generator", return_value="generator-token"
                ), patch.object(
                    inference_runner, "_open_source_image", return_value=source_image_token
                ) as open_source_image:
                    exit_code = inference_runner.run_child_main(stdin=stdin, stdout=stdout)

                self.assertEqual(exit_code, 0)
                if case["source_image_path"] is None:
                    open_source_image.assert_not_called()
                else:
                    open_source_image.assert_called_once_with(case["source_image_path"])

                self.assertEqual(len(fake_pipeline.calls), 1)
                invocation = fake_pipeline.calls[0]
                self.assertEqual(invocation["prompt"], "test prompt")
                self.assertEqual(invocation["negative_prompt"], "avoid blur")
                self.assertEqual(invocation["num_inference_steps"], 4)
                self.assertEqual(invocation["width"], 512)
                self.assertEqual(invocation["height"], 512)
                self.assertEqual(invocation["guidance_scale"], 7.5)
                self.assertEqual(invocation["generator"], "generator-token")
                if case["source_image_path"] is None:
                    self.assertNotIn("image", invocation)
                    self.assertNotIn("strength", invocation)
                else:
                    self.assertIs(invocation["image"], source_image_token)
                    self.assertEqual(invocation["strength"], 0.55)

                self.assertEqual(fake_pipeline.output_image.saved_paths, [str(output_path)])
                events = self._parse_ndjson_events(stdout.getvalue())
                self.assertEqual(events[-1]["type"], "done")
                self.assertEqual(
                    events[-1]["result"],
                    {
                        "output_path": str(output_path),
                        "metadata": {
                            "family": case["family"],
                            "node_id": case["node_id"],
                            "seed": 42,
                            "negative_prompt_used": True,
                            "source_image_used": case["source_image_path"] is not None,
                        },
                    },
                )

    def test_inference_runner_emits_stage_progress_events_before_done(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-progress-"))
        output_path = workspace_dir / "result.png"

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        class FakePipeline:
            def __call__(self, **kwargs):
                return SimpleNamespace(images=[FakeImage()])

        job = {
            "extension_id": "test-extension",
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "model_dir": str(workspace_dir / "model"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(output_path),
            "prompt": "test prompt",
            "params": {"steps": 4, "seed": 42},
        }

        stdout = StringIO()
        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("stable-diffusion", "text-to-image"): SimpleNamespace(from_pretrained=lambda model_dir: FakePipeline())},
            clear=True,
        ), patch.object(inference_runner, "_seeded_generator", return_value="generator-token"):
            exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

        self.assertEqual(exit_code, 0)
        events = self._parse_ndjson_events(stdout.getvalue())
        self.assertEqual(
            [event["label"] for event in events[:-1] if event["type"] == "progress"],
            ["loading-pipeline", "running-inference", "saving-output"],
        )
        self.assertEqual(events[-1]["type"], "done")

    def test_inference_runner_keeps_done_terminal_contract_with_intermediate_events(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-terminal-compat-"))
        output_path = workspace_dir / "result.png"

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        class FakePipeline:
            def __call__(self, **kwargs):
                return SimpleNamespace(images=[FakeImage()])

        job = {
            "extension_id": "test-extension",
            "family": "flux",
            "node_id": "text-to-image",
            "model_dir": str(workspace_dir / "model"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(output_path),
            "prompt": "terminal compatibility",
            "negative_prompt": "avoid blur",
            "params": {"steps": 8, "seed": 7},
        }

        stdout = StringIO()
        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("flux", "text-to-image"): SimpleNamespace(from_pretrained=lambda model_dir: FakePipeline())},
            clear=True,
        ), patch.object(inference_runner, "_seeded_generator", return_value="generator-token"):
            exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

        self.assertEqual(exit_code, 0)
        events = self._parse_ndjson_events(stdout.getvalue())
        self.assertEqual(events[-1]["type"], "done")
        self.assertNotIn(events[-1]["type"], {"progress", "log", "error"})
        self.assertEqual(
            events[-1]["result"],
            {
                "output_path": str(output_path),
                "metadata": {
                    "family": "flux",
                    "node_id": "text-to-image",
                    "seed": 7,
                    "negative_prompt_used": True,
                    "source_image_used": False,
                },
            },
        )

    def test_inference_runner_emits_error_ndjson_for_invalid_job_or_unsupported_loader(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        cases = (
            (
                StringIO("{not-json}\n"),
                {},
                "Invalid JSON job received by inference runner",
            ),
            (
                StringIO(
                    json.dumps(
                        {
                            "family": "flux",
                            "node_id": "image-to-image",
                            "model_dir": "/tmp/model",
                            "workspace_dir": "/tmp/workspace",
                            "output_path": "/tmp/workspace/out.png",
                            "prompt": "oops",
                            "negative_prompt": None,
                            "source_image_path": None,
                            "params": {},
                        }
                    )
                    + "\n"
                ),
                {("stable-diffusion", "text-to-image"): object()},
                "Unsupported inference backend for family 'flux' and node 'image-to-image'",
            ),
        )

        for stdin, loader_map, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                stdout = StringIO()
                with patch.dict(inference_runner._PIPELINE_LOADERS, loader_map, clear=True):
                    exit_code = inference_runner.run_child_main(stdin=stdin, stdout=stdout)

                self.assertEqual(exit_code, 1)
                events = self._parse_ndjson_events(stdout.getvalue())
                self.assertEqual(events, [{"type": "error", "message": expected_message}])

    def test_weight_readiness_is_node_scoped_and_reports_exact_missing_path(self) -> None:
        extension_id = "sd15"
        with tempfile.TemporaryDirectory(prefix="models-dir-") as temp_dir:
            models_dir = Path(temp_dir)
            ready_check = models_dir / extension_id / "text-to-image" / "model_index.json"
            ready_check.parent.mkdir(parents=True, exist_ok=True)
            ready_check.write_text("{}\n", encoding="utf-8")

            readiness = weights.evaluate_extension_weights(extension_id, models_dir=models_dir)

        text_node = readiness["nodes"]["text-to-image"]
        image_node = readiness["nodes"]["image-to-image"]
        expected_missing = str(models_dir / extension_id / "image-to-image" / "model_index.json")

        self.assertEqual(readiness["status"], "missing")
        self.assertEqual(text_node["status"], "ready")
        self.assertEqual(image_node["status"], "missing")
        self.assertEqual(
            image_node["check_path"],
            expected_missing,
        )
        self.assertIn(expected_missing, "\n".join(image_node["diagnostics"]))

    def test_vendored_runtime_matches_canonical_runtime_sources(self) -> None:
        for relative_name in (
            "dependencies.py",
            "install_contract.py",
            "pipeline.py",
            "runtime_adapter.py",
            "inference_runner.py",
        ):
            canonical_text = self._canonical_runtime_file(relative_name).read_text(encoding="utf-8")
            for extension_id in EXTENSION_IDS:
                with self.subTest(extension_id=extension_id, relative_name=relative_name):
                    vendored_text = self._vendored_runtime_file(extension_id, relative_name).read_text(
                        encoding="utf-8"
                    )
                    self.assertEqual(vendored_text, canonical_text)


if __name__ == "__main__":
    unittest.main()

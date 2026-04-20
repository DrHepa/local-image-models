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
        return runtime_root

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

    def test_generator_reaches_backend_not_implemented_after_ready_setup(self) -> None:
        for extension_id in EXTENSION_IDS:
            with self.subTest(extension_id=extension_id):
                runtime_root, result = self._run_setup_success(extension_id)
                self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)

                stdout = StringIO()
                with patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                ), patch(
                    "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                    return_value=(True, "stubbed imports"),
                ):
                    exit_code = runtime_adapter.run_generator_main(
                        extension_id=extension_id,
                        runtime_root=str(runtime_root),
                        stdin=self._generator_payload(),
                        stdout=stdout,
                    )

                output = stdout.getvalue()
                self.assertEqual(exit_code, 1)
                self.assertIn('"label": "runtime-ready"', output)
                self.assertIn('"label": "backend-dispatch"', output)
                self.assertIn("Runtime ready at", output)
                self.assertIn("Generation backend is not implemented yet", output)
                self.assertNotIn("not installed", output.lower())

    def test_run_payload_reaches_backend_not_implemented_after_ready_setup(self) -> None:
        payload = json.loads(self._generator_payload().getvalue())

        for extension_id in EXTENSION_IDS:
            with self.subTest(extension_id=extension_id):
                runtime_root, result = self._run_setup_success(extension_id)
                self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)

                with patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                ), patch(
                    "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                    return_value=(True, "stubbed imports"),
                ):
                    with self.assertRaisesRegex(
                        pipeline.BackendNotImplementedError,
                        "Generation backend is not implemented yet",
                    ) as exc_info:
                        runtime_adapter.run_payload(
                            payload,
                            extension_id=extension_id,
                            runtime_root=str(runtime_root),
                        )

                self.assertIn("Validation passed and the runtime scaffold is ready", str(exc_info.exception))
                self.assertNotIn("not installed", str(exc_info.exception).lower())

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

    def test_extension_generator_generate_maps_text_to_image_request(self) -> None:
        cases = (
            ("sd15", {"prompt": "primary prompt", "input": {"text": "legacy prompt"}}, "primary prompt"),
            ("sdxl-base", {"input": {"text": "legacy only prompt"}}, None),
            ("flux-schnell", {"prompt": "flux prompt", "input": {"text": "ignored fallback"}}, "flux prompt"),
        )

        for extension_id, params, expected_prompt in cases:
            with self.subTest(extension_id=extension_id, params=params):
                generator_class = self._load_generator_class(extension_id)
                model_dir = self._make_model_dir(extension_id, "text-to-image")
                outputs_dir = Path(tempfile.mkdtemp(prefix=f"outputs-{extension_id}-"))
                runtime_snapshot = object()
                result_path = outputs_dir / f"{extension_id}.png"
                progress_events: list[tuple[int, str]] = []

                def execute_side_effect(request, runtime, extension_id, emit_progress, emit_log):
                    self.assertEqual(request.node_id, "text-to-image")
                    self.assertEqual(request.workspace_dir, str(outputs_dir))
                    self.assertEqual(request.input.get("text"), params.get("input", {}).get("text"))
                    self.assertEqual(request.params.get("prompt"), expected_prompt)
                    self.assertFalse((outputs_dir / ".modly-inputs").exists())
                    emit_progress(61, "mapped")
                    emit_log("mapping-ready")
                    return {"output_path": str(result_path)}

                with patch(
                    "local_image_runtime.runtime_adapter.bootstrap_runtime",
                    return_value=runtime_snapshot,
                ), patch(
                    "local_image_runtime.runtime_adapter.execute",
                    side_effect=execute_side_effect,
                ):
                    generator = generator_class(model_dir, outputs_dir)
                    actual_path = generator.generate(
                        b"ignored-image-bytes",
                        params,
                        progress_cb=lambda percent, label: progress_events.append((percent, label)),
                    )

                self.assertEqual(actual_path, result_path)
                self.assertEqual(progress_events, [(61, "mapped")])

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

    def test_pipeline_execute_uses_request_workspace_dir_in_logs_and_errors(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-override-"))
        runtime_outputs_dir = Path(tempfile.mkdtemp(prefix="runtime-outputs-"))
        runtime_models_dir = Path(tempfile.mkdtemp(prefix="runtime-models-"))
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

        with patch(
            "local_image_runtime.pipeline.extension_is_installed",
            return_value=True,
        ), patch(
            "local_image_runtime.pipeline.get_extension_record",
            return_value={},
        ):
            with self.assertRaisesRegex(pipeline.BackendNotImplementedError, str(workspace_dir)):
                pipeline.execute(
                    request,
                    runtime,
                    extension_id="sd15",
                    emit_progress=lambda percent, label: progress_events.append((percent, label)),
                    emit_log=logs.append,
                )

        self.assertIn(f"Workspace: {workspace_dir}.", logs[0])
        self.assertEqual(progress_events[-1], (75, "backend-dispatch"))

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
            "runtime_adapter.py",
            "pipeline.py",
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

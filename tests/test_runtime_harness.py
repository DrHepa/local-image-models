from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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
    bundle_metadata,
    bootstrap,
    dependencies,
    descriptors,
    install_contract,
    pipeline,
    release_gates,
    runtime_adapter,
    smoke_evidence,
    weights,
)
from local_image_runtime.dependencies import DependencyInstallStep, DependencyPlan  # noqa: E402


SUPPORTED_PLATFORM = {"system": "linux", "machine": "aarch64"}
UNSUPPORTED_PLATFORM = {"system": "linux", "machine": "x86_64"}
WINDOWS_PLATFORM = {"system": "windows", "machine": "AMD64"}
EXTENSION_IDS = ("sd15", "sdxl-base", "flux-schnell")
WINDOWS_PLAN_STATES = {
    "sd15": "candidate_install",
    "sdxl-base": "candidate_install",
    "flux-schnell": "candidate_install",
}


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

    def test_release_preflight_blocks_wrong_repo_and_excludes_gitignore(self) -> None:
        expected_root = Path("/home/drhepa/shiro-stack/repos/Tools/local-image-models")

        wrong_repo = release_gates.evaluate_workspace_preflight(
            repo_root=Path("/home/drhepa/shiro-stack/repos/Tools/modly-Codex-image-extension"),
            expected_repo_root=expected_root,
            branch="main",
            ahead_count=1,
            staged_files=(),
            unstaged_files=(".gitignore",),
            untracked_files=(),
        )
        self.assertFalse(wrong_repo.ready)
        self.assertIn("outside expected repository", " ".join(wrong_repo.diagnostics))

        correct_repo = release_gates.evaluate_workspace_preflight(
            repo_root=expected_root,
            expected_repo_root=expected_root,
            branch="main",
            ahead_count=1,
            staged_files=("README.md", "extensions/sd15/manifest.json"),
            unstaged_files=(".gitignore",),
            untracked_files=("docs/release-smoke.md",),
        )
        self.assertTrue(correct_repo.ready)
        self.assertEqual(
            correct_repo.release_evidence_files,
            ("README.md", "docs/release-smoke.md", "extensions/sd15/manifest.json"),
        )
        self.assertNotIn(".gitignore", correct_repo.release_evidence_files)
        self.assertIn(".gitignore preserved as unrelated", " ".join(correct_repo.diagnostics))

    def test_previous_release_ref_must_point_to_archived_commit_before_promotion(self) -> None:
        missing_ref = release_gates.evaluate_previous_release_ref(
            ref_name="release/previous-0.1.x",
            target_commit="8f1eb6b",
            refs={"refs/heads/main": "HEAD"},
        )
        self.assertFalse(missing_ref.ready)
        self.assertIn("release/previous-0.1.x", " ".join(missing_ref.diagnostics))
        self.assertIn("8f1eb6b", " ".join(missing_ref.diagnostics))

        matching_ref = release_gates.evaluate_previous_release_ref(
            ref_name="release/previous-0.1.x",
            target_commit="8f1eb6b",
            refs={"refs/heads/release/previous-0.1.x": "8f1eb6b"},
        )
        self.assertTrue(matching_ref.ready)
        self.assertEqual(matching_ref.ref_name, "release/previous-0.1.x")
        self.assertEqual(matching_ref.commit, "8f1eb6b")

    def test_bundle_version_policy_rejects_manifest_drift_or_missing_next_policy(self) -> None:
        drift = bundle_metadata.evaluate_bundle_version_policy(
            manifest_versions={"sd15": "0.1.0", "sdxl-base": "0.1.1", "flux-schnell": "0.1.0"},
            package_versions={"local-image-runtime": "0.1.1"},
            next_version_policy=None,
        )
        self.assertFalse(drift.ready)
        self.assertIn("version drift", " ".join(drift.diagnostics))
        self.assertIn("missing next-version policy", " ".join(drift.diagnostics))

        policy = bundle_metadata.evaluate_bundle_version_policy(
            manifest_versions={"sd15": "0.1.1", "sdxl-base": "0.1.1", "flux-schnell": "0.1.1"},
            package_versions={"local-image-runtime": "0.1.1"},
            next_version_policy="next patch version 0.1.2 after smoke gates pass",
        )
        self.assertTrue(policy.ready)
        self.assertEqual(policy.current_version, "0.1.1")
        self.assertEqual(policy.next_version_policy, "next patch version 0.1.2 after smoke gates pass")

    def test_bundle_metadata_prepares_next_ip_adapter_patch_version(self) -> None:
        manifest_paths = {
            extension_id: REPO_ROOT / "extensions" / extension_id / "manifest.json"
            for extension_id in EXTENSION_IDS
        }
        manifest_versions = bundle_metadata.load_manifest_versions(manifest_paths)

        policy = bundle_metadata.evaluate_bundle_version_policy(
            manifest_versions=manifest_versions,
            package_versions={"local-image-runtime": bundle_metadata.BUNDLE_VERSION},
            next_version_policy="prepared IP-Adapter patch version 0.1.1; Windows remains candidate/unverified",
        )

        self.assertTrue(policy.ready, policy.diagnostics)
        self.assertEqual(policy.current_version, "0.1.1")
        self.assertEqual(set(manifest_versions.values()), {"0.1.1"})

    def test_smoke_evidence_requires_fields_and_validation_only_is_not_promotion_ready(self) -> None:
        incomplete = smoke_evidence.validate_smoke_evidence(
            {
                "kind": "validation",
                "feature": "sd15_ip_adapter",
                "platform": "linux/arm64",
                "ref": "8f1eb6b",
                "version": "0.1.0",
                "result": "pass",
            }
        )
        self.assertFalse(incomplete.valid)
        self.assertIn("commands", incomplete.missing_fields)
        self.assertIn("assets", incomplete.missing_fields)
        self.assertIn("packages", incomplete.missing_fields)
        self.assertIn("logs", incomplete.missing_fields)
        self.assertIn("outputs", incomplete.missing_fields)

        validation_only = smoke_evidence.validate_smoke_evidence(
            {
                "kind": "validation",
                "feature": "sd15_ip_adapter",
                "platform": "linux/arm64",
                "ref": "8f1eb6b",
                "version": "0.1.0",
                "commands": ["python3 -m unittest discover tests -v"],
                "assets": [],
                "packages": {"python": "3.12"},
                "outputs": [],
                "logs": ["168 tests passed"],
                "result": "pass",
            }
        )
        self.assertTrue(validation_only.valid)

        gate = smoke_evidence.evaluate_feature_promotion_evidence([validation_only.payload], feature="sd15_ip_adapter")
        self.assertFalse(gate.ready)
        self.assertIn("install_repair", gate.missing_kinds)
        self.assertIn("gpu_generation", gate.missing_kinds)

    def test_sdxl_style_local_smoke_evidence_is_separate_from_validation_only_evidence(self) -> None:
        validation_only = {
            "kind": "validation",
            "feature": "sdxl_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "local-smoke-release-gates",
            "version": "0.1.1",
            "commands": ["python3 -m unittest discover tests -v"],
            "assets": [],
            "packages": {"python": "3.12", "cuda": "12.4", "torch": "2.7.0", "diffusers": "0.35.1"},
            "outputs": [],
            "logs": ["validation-only schema checks passed"],
            "result": "pass",
        }

        validation_gate = smoke_evidence.evaluate_sdxl_style_reference_local_smoke_evidence([validation_only])

        self.assertFalse(validation_gate.ready)
        validation_diagnostics = " ".join(validation_gate.diagnostics)
        self.assertIn("validation-only", validation_diagnostics)
        self.assertIn("install/repair", validation_diagnostics)
        self.assertIn("GPU generation", validation_diagnostics)
        self.assertIn("readiness", validation_diagnostics)
        self.assertIn("local asset", validation_diagnostics)

        install_repair = {
            "kind": "install_repair",
            "feature": "sdxl_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "local-smoke-release-gates",
            "version": "0.1.1",
            "commands": ["python3 extensions/sdxl-base/setup.py < install-payload.json"],
            "assets": [
                {
                    "id": "sdxl_ip_adapter_style",
                    "path": "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/ip-adapter_sdxl.bin",
                    "sha256": "adapter-sha",
                }
            ],
            "packages": {"python": "3.12", "cuda": "12.4", "torch": "2.7.0", "diffusers": "0.35.1"},
            "outputs": [],
            "logs": ["Install/Repair completed and runtime imports passed"],
            "result": "pass",
            "readiness": "ready",
            "local_assets": [
                "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/ip-adapter_sdxl.bin",
                "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/image_encoder/config.json",
            ],
        }
        incomplete_gpu_generation = {
            "kind": "gpu_generation",
            "feature": "sdxl_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "local-smoke-release-gates",
            "version": "0.1.1",
            "commands": ["modly generate sdxl-base image-to-image --style-reference local.png"],
            "assets": install_repair["assets"],
            "packages": install_repair["packages"],
            "outputs": [],
            "logs": [],
            "result": "pass",
            "local_files_only": True,
        }

        incomplete_generation_gate = smoke_evidence.evaluate_sdxl_style_reference_local_smoke_evidence(
            [install_repair, incomplete_gpu_generation]
        )

        self.assertFalse(incomplete_generation_gate.ready)
        incomplete_diagnostics = " ".join(incomplete_generation_gate.diagnostics)
        self.assertIn("GPU generation", incomplete_diagnostics)
        self.assertIn("output", incomplete_diagnostics)
        self.assertIn("log", incomplete_diagnostics)

        complete_gpu_generation = {
            **incomplete_gpu_generation,
            "outputs": ["/outputs/sdxl-style-local-smoke.png"],
            "logs": ["Generated SDXL style-reference output using local_files_only=True"],
        }
        installed_parity = {
            "kind": "installed_parity",
            "feature": "sdxl_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "local-smoke-release-gates",
            "version": "0.1.1",
            "commands": ["python3 tools/sync_extension_runtime.py --check"],
            "assets": install_repair["assets"],
            "packages": install_repair["packages"],
            "outputs": [],
            "logs": ["installed runtime parity passed"],
            "result": "pass",
            "parity": "fresh",
        }

        ready_gate = smoke_evidence.evaluate_sdxl_style_reference_local_smoke_evidence(
            [validation_only, install_repair, complete_gpu_generation, installed_parity]
        )

        self.assertTrue(ready_gate.ready)
        self.assertEqual(ready_gate.feature, "sdxl_ip_adapter_style")
        self.assertEqual(ready_gate.missing_kinds, ())

        generic_gate = smoke_evidence.evaluate_feature_promotion_evidence(
            [validation_only, install_repair, complete_gpu_generation], feature="sdxl_ip_adapter_style"
        )
        self.assertTrue(generic_gate.ready)

    def test_sdxl_style_evidence_persists_without_gpu_smoke_but_is_not_promotion_ready(self) -> None:
        evidence_path = Path(tempfile.mkdtemp(prefix="sdxl-style-evidence-missing-gpu-")) / "evidence.json"
        install_repair = {
            "kind": "install_repair",
            "feature": "sdxl_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "local-smoke-release-gates",
            "version": "0.1.1",
            "commands": ["python3 extensions/sdxl-base/setup.py < install-payload.json"],
            "assets": [
                {
                    "id": "sdxl_ip_adapter_style",
                    "path": "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/ip-adapter_sdxl.bin",
                    "sha256": "adapter-sha",
                }
            ],
            "packages": {"python": "3.12", "cuda": "12.4", "torch": "2.7.0", "diffusers": "0.35.1"},
            "outputs": [],
            "logs": ["Install/Repair completed and runtime imports passed"],
            "result": "pass",
            "readiness": "ready",
            "local_assets": [
                "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/ip-adapter_sdxl.bin",
                "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/image_encoder/config.json",
            ],
        }

        gate = smoke_evidence.write_sdxl_style_reference_smoke_evidence(
            evidence_path,
            install_repair=install_repair,
        )
        stored_records = smoke_evidence.read_smoke_evidence(evidence_path)

        self.assertFalse(gate.ready)
        self.assertEqual(gate.missing_kinds, ("model_load", "generate_no_download", "gpu_generation", "installed_parity"))
        diagnostics = " ".join(gate.diagnostics)
        self.assertIn("GPU generation", diagnostics)
        self.assertIn("output", diagnostics)
        self.assertIn("log", diagnostics)
        self.assertEqual(stored_records, [install_repair])
        self.assertEqual(stored_records[0]["readiness"], "ready")
        self.assertEqual(stored_records[0]["local_assets"], install_repair["local_assets"])

    def test_complete_sdxl_style_evidence_promotes_sdxl_only_not_sd15_or_controlnet(self) -> None:
        evidence_path = Path(tempfile.mkdtemp(prefix="sdxl-style-evidence-complete-")) / "evidence.json"
        install_repair = {
            "kind": "install_repair",
            "feature": "sdxl_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "local-smoke-release-gates",
            "version": "0.1.1",
            "commands": ["python3 extensions/sdxl-base/setup.py < install-payload.json"],
            "assets": [
                {
                    "id": "sdxl_ip_adapter_style",
                    "path": "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/ip-adapter_sdxl.bin",
                    "sha256": "adapter-sha",
                }
            ],
            "packages": {"python": "3.12", "cuda": "12.4", "torch": "2.7.0", "diffusers": "0.35.1"},
            "outputs": [],
            "logs": ["Install/Repair completed and runtime imports passed"],
            "result": "pass",
            "readiness": "ready",
            "local_assets": [
                "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/ip-adapter_sdxl.bin",
                "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/image_encoder/config.json",
            ],
        }
        gpu_generation = {
            "kind": "gpu_generation",
            "feature": "sdxl_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "local-smoke-release-gates",
            "version": "0.1.1",
            "commands": ["modly generate sdxl-base image-to-image --style-reference local.png"],
            "assets": install_repair["assets"],
            "packages": install_repair["packages"],
            "outputs": ["/outputs/sdxl-style-local-smoke.png"],
            "logs": ["Generated SDXL style-reference output using local_files_only=True"],
            "result": "pass",
            "local_files_only": True,
        }
        installed_parity = {
            "kind": "installed_parity",
            "feature": "sdxl_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "local-smoke-release-gates",
            "version": "0.1.1",
            "commands": ["python3 tools/sync_extension_runtime.py --check"],
            "assets": install_repair["assets"],
            "packages": install_repair["packages"],
            "outputs": [],
            "logs": ["installed runtime parity passed"],
            "result": "pass",
            "parity": "fresh",
        }

        sdxl_gate = smoke_evidence.write_sdxl_style_reference_smoke_evidence(
            evidence_path,
            install_repair=install_repair,
            gpu_generation=gpu_generation,
        )
        smoke_evidence.write_smoke_evidence(evidence_path, [install_repair, gpu_generation, installed_parity])
        sdxl_gate = smoke_evidence.evaluate_sdxl_style_reference_local_smoke_evidence(
            smoke_evidence.read_smoke_evidence(evidence_path)
        )
        stored_records = smoke_evidence.read_smoke_evidence(evidence_path)
        sd15_gate = smoke_evidence.evaluate_feature_promotion_evidence(stored_records, feature="sd15_ip_adapter")
        controlnet_gate = smoke_evidence.evaluate_feature_promotion_evidence(stored_records, feature="controlnet")

        self.assertTrue(sdxl_gate.ready)
        self.assertEqual(stored_records, [install_repair, gpu_generation, installed_parity])
        self.assertEqual(stored_records[1]["outputs"], ["/outputs/sdxl-style-local-smoke.png"])
        self.assertEqual(stored_records[1]["logs"], ["Generated SDXL style-reference output using local_files_only=True"])
        self.assertFalse(sd15_gate.ready)
        self.assertEqual(sd15_gate.missing_kinds, ("install_repair", "gpu_generation"))
        self.assertFalse(controlnet_gate.ready)
        self.assertEqual(controlnet_gate.missing_kinds, ("install_repair", "gpu_generation"))

    def test_sdxl_style_asset_identity_records_local_paths_hashes_repo_and_unknown_revision(self) -> None:
        models_dir = Path(tempfile.mkdtemp(prefix="sdxl-style-assets-"))
        feature_root = models_dir / "sdxl-base" / "optional" / "sdxl_ip_adapter_style"
        payloads = {
            "sdxl_models/ip-adapter_sdxl.bin": b"adapter-bytes",
            "sdxl_models/image_encoder/config.json": b'{"architectures":["CLIPVisionModelWithProjection"]}',
            "sdxl_models/image_encoder/model.safetensors": b"encoder-bytes",
        }
        for relative_path, content in payloads.items():
            target = feature_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        ledger = weights.collect_optional_feature_asset_identity(
            "sdxl-base",
            "sdxl_ip_adapter_style",
            models_dir=models_dir,
        )

        self.assertEqual(ledger["status"], "ready")
        self.assertEqual(ledger["extension_id"], "sdxl-base")
        self.assertEqual(ledger["feature_id"], "sdxl_ip_adapter_style")
        self.assertEqual(ledger["repo"], "h94/IP-Adapter")
        self.assertEqual(ledger["revision"], "unknown")
        self.assertEqual(ledger["model_dir"], str(feature_root))
        self.assertEqual(ledger["missing_files"], ())
        by_relative_path = {asset["relative_path"]: asset for asset in ledger["assets"]}
        self.assertEqual(tuple(by_relative_path), tuple(payloads))
        for relative_path, content in payloads.items():
            asset = by_relative_path[relative_path]
            self.assertEqual(asset["path"], str(feature_root / relative_path))
            self.assertEqual(asset["size"], len(content))
            self.assertEqual(asset["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(asset["repo"], "h94/IP-Adapter")
            self.assertEqual(asset["revision"], "unknown")

    def test_sdxl_style_asset_identity_missing_assets_is_actionable_and_local_only(self) -> None:
        models_dir = Path(tempfile.mkdtemp(prefix="sdxl-style-missing-assets-"))
        feature_root = models_dir / "sdxl-base" / "optional" / "sdxl_ip_adapter_style"
        adapter_path = feature_root / "sdxl_models" / "ip-adapter_sdxl.bin"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"adapter-only")

        ledger = weights.collect_optional_feature_asset_identity(
            "sdxl-base",
            "sdxl_ip_adapter_style",
            models_dir=models_dir,
            revision="main",
        )

        self.assertEqual(ledger["status"], "missing")
        self.assertEqual(ledger["revision"], "main")
        self.assertEqual(
            ledger["missing_files"],
            (
                "sdxl_models/image_encoder/config.json",
                "sdxl_models/image_encoder/model.safetensors",
            ),
        )
        diagnostics = " ".join(ledger["diagnostics"])
        self.assertIn("Run Install/Repair", diagnostics)
        self.assertIn("no download was attempted", diagnostics)
        self.assertIn(str(feature_root / "sdxl_models" / "image_encoder" / "config.json"), diagnostics)

    def test_sd15_style_asset_identity_reports_discovered_missing_bundle_without_download(self) -> None:
        models_dir = Path(tempfile.mkdtemp(prefix="sd15-style-discovered-assets-"))

        ledger = weights.collect_optional_feature_asset_identity(
            "sd15",
            "sd15_ip_adapter_style",
            models_dir=models_dir,
        )

        self.assertEqual(ledger["status"], "missing")
        self.assertEqual(ledger["extension_id"], "sd15")
        self.assertEqual(ledger["feature_id"], "sd15_ip_adapter_style")
        self.assertEqual(ledger["repo"], "h94/IP-Adapter")
        self.assertEqual(ledger["revision"], "018e402774aeeddd60609b4ecdb7e298259dc729")
        self.assertEqual(
            ledger["required_files"],
            (
                "models/ip-adapter_sd15.safetensors",
                "models/image_encoder/config.json",
                "models/image_encoder/model.safetensors",
            ),
        )
        self.assertEqual(ledger["allow_patterns"], ledger["required_files"])
        self.assertEqual(ledger["assets"], ())
        self.assertTrue(ledger["local_files_only"])
        diagnostics = " ".join(ledger["diagnostics"])
        self.assertIn("Install/Repair", diagnostics)
        self.assertIn("no download was attempted", diagnostics)
        self.assertIn("models/ip-adapter_sd15.safetensors", diagnostics)

    def test_sdxl_style_promotion_requires_all_installed_local_only_gates_not_repo_only(self) -> None:
        packages = {"python": "3.12", "cuda": "12.4", "torch": "2.7.0", "diffusers": "0.35.1"}
        asset = {
            "id": "sdxl_ip_adapter_style",
            "path": "/models/sdxl-base/optional/sdxl_ip_adapter_style/sdxl_models/ip-adapter_sdxl.bin",
            "sha256": "adapter-sha",
            "repo": "h94/IP-Adapter",
            "revision": "unknown",
        }
        repo_only_records = [
            {
                "kind": "asset_identity",
                "feature": "sdxl_ip_adapter_style",
                "platform": "linux/arm64",
                "ref": "repo-working-tree",
                "version": "0.1.1",
                "commands": ["python3 -m unittest tests.test_runtime_harness -v"],
                "assets": [asset],
                "packages": packages,
                "outputs": [],
                "logs": ["repo asset ledger passed"],
                "result": "pass",
                "scope": "repo-runtime",
            },
            {
                "kind": "model_load",
                "feature": "sdxl_ip_adapter_style",
                "platform": "linux/arm64",
                "ref": "repo-working-tree",
                "version": "0.1.1",
                "commands": ["python3 local-only-load.py"],
                "assets": [asset],
                "packages": packages,
                "outputs": [],
                "logs": ["load_ip_adapter used local_files_only=True"],
                "result": "pass",
                "local_files_only": True,
                "scope": "repo-runtime",
            },
            {
                "kind": "generate_no_download",
                "feature": "sdxl_ip_adapter_style",
                "platform": "linux/arm64",
                "ref": "repo-working-tree",
                "version": "0.1.1",
                "commands": ["python3 generate-no-download-guard.py"],
                "assets": [asset],
                "packages": packages,
                "outputs": [],
                "logs": ["no acquisition helpers called"],
                "result": "pass",
                "local_files_only": True,
                "forbidden_calls": [],
                "scope": "repo-runtime",
            },
        ]

        repo_gate = smoke_evidence.evaluate_sdxl_style_reference_local_smoke_evidence(repo_only_records)

        self.assertFalse(repo_gate.ready)
        self.assertEqual(repo_gate.status, "partial")
        repo_diagnostics = " ".join(repo_gate.diagnostics)
        self.assertIn("repo-runtime", repo_diagnostics)
        self.assertIn("installed parity", repo_diagnostics)
        self.assertIn("GPU generation", repo_diagnostics)

        installed_records = repo_only_records + [
            {
                "kind": "install_repair",
                "feature": "sdxl_ip_adapter_style",
                "platform": "linux/arm64",
                "ref": "installed-runtime-ref",
                "version": "0.1.1",
                "commands": ["python3 extensions/sdxl-base/setup.py < install-payload.json"],
                "assets": [asset],
                "packages": packages,
                "outputs": [],
                "logs": ["Install/Repair ready"],
                "result": "pass",
                "readiness": "ready",
                "local_assets": [asset["path"]],
                "scope": "installed-runtime",
            },
            {
                "kind": "installed_parity",
                "feature": "sdxl_ip_adapter_style",
                "platform": "linux/arm64",
                "ref": "installed-runtime-ref",
                "version": "0.1.1",
                "commands": ["python3 tools/sync_extension_runtime.py --check"],
                "assets": [asset],
                "packages": packages,
                "outputs": [],
                "logs": ["installed runtime parity passed"],
                "result": "pass",
                "parity": "fresh",
                "scope": "installed-runtime",
            },
            {
                "kind": "gpu_generation",
                "feature": "sdxl_ip_adapter_style",
                "platform": "linux/arm64",
                "ref": "installed-runtime-ref",
                "version": "0.1.1",
                "commands": ["modly generate sdxl-base image-to-image --style-reference local.png"],
                "assets": [asset],
                "packages": packages,
                "outputs": ["/outputs/sdxl-style-local-smoke.png"],
                "logs": ["Generated SDXL style-reference output using local_files_only=True"],
                "result": "pass",
                "local_files_only": True,
                "scope": "installed-runtime",
            },
        ]

        installed_gate = smoke_evidence.evaluate_sdxl_style_reference_local_smoke_evidence(installed_records)

        self.assertTrue(installed_gate.ready)
        self.assertEqual(installed_gate.status, "ready")
        self.assertEqual(installed_gate.missing_kinds, ())

    def test_sd15_style_promotion_blocks_unknown_revision_and_missing_assets_before_load(self) -> None:
        packages = {"python": "3.12", "cuda": "13.0", "torch": "2.11.0+cu130", "diffusers": "0.35.1"}
        incomplete_asset_identity = {
            "kind": "asset_identity",
            "feature": "sd15_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "repo-working-tree",
            "version": "0.1.1",
            "commands": ["python3 -m unittest tests.test_runtime_harness -v"],
            "assets": [
                {
                    "id": "sd15_ip_adapter_style",
                    "path": "/models/sd15/optional/sd15_ip_adapter_style/models/ip-adapter_sd15.bin",
                    "repo": "unknown",
                    "revision": "unknown",
                    "size": 1024,
                    "sha256": "adapter-sha",
                }
            ],
            "packages": packages,
            "outputs": [],
            "logs": ["candidate SD1.5 adapter without verified image encoder"],
            "result": "pass",
            "scope": "repo-runtime",
        }
        blocked_model_load = {
            **incomplete_asset_identity,
            "kind": "model_load",
            "logs": ["model load must not run until exact SD1.5 adapter and image encoder metadata are known"],
            "local_files_only": True,
        }

        gate = smoke_evidence.evaluate_sd15_style_reference_discovery_evidence(
            [incomplete_asset_identity, blocked_model_load]
        )

        self.assertFalse(gate.ready)
        self.assertEqual(gate.status, "blocked")
        self.assertIn("install_repair", gate.missing_kinds)
        self.assertIn("generate_no_download", gate.missing_kinds)
        diagnostics = " ".join(gate.diagnostics)
        self.assertIn("unknown revision", diagnostics)
        self.assertIn("image encoder", diagnostics)
        self.assertIn("before model load", diagnostics)

    def test_sd15_repair_readiness_alone_is_partial_not_manifest_or_promotion_ready(self) -> None:
        packages = {"python": "3.12", "cuda": "13.0", "torch": "2.11.0+cu130", "diffusers": "0.35.1"}
        revision = "018e402774aeeddd60609b4ecdb7e298259dc729"
        assets = [
            {
                "id": "sd15_ip_adapter_style",
                "relative_path": "models/ip-adapter_sd15.safetensors",
                "path": "/models/sd15/optional/sd15_ip_adapter_style/models/ip-adapter_sd15.safetensors",
                "repo": "h94/IP-Adapter",
                "revision": revision,
                "size": 44642768,
                "sha256": "7a8b1bbda0d1379df61b4dd8e8fad2e82578e0c52450a871f443da338f385cf1",
            },
            {
                "id": "sd15_ip_adapter_style",
                "relative_path": "models/image_encoder/config.json",
                "path": "/models/sd15/optional/sd15_ip_adapter_style/models/image_encoder/config.json",
                "repo": "h94/IP-Adapter",
                "revision": revision,
                "size": 560,
                "sha256": "config-local-sha256",
            },
            {
                "id": "sd15_ip_adapter_style",
                "relative_path": "models/image_encoder/model.safetensors",
                "path": "/models/sd15/optional/sd15_ip_adapter_style/models/image_encoder/model.safetensors",
                "repo": "h94/IP-Adapter",
                "revision": revision,
                "size": 2528373448,
                "sha256": "1686fef5ab13a4c8dcc32876ef7b557b296cb78ec2f1ec259360ae9135044209",
            },
        ]
        install_repair = {
            "kind": "install_repair",
            "feature": "sd15_ip_adapter_style",
            "platform": "linux/arm64",
            "ref": "authorized-repair",
            "version": "0.1.1",
            "commands": ["python3 extensions/sd15/setup.py < repair-payload.json"],
            "assets": assets,
            "packages": packages,
            "outputs": [],
            "logs": ["Repair acquired SD1.5 optional assets"],
            "result": "pass",
            "readiness": "ready",
            "local_assets": [asset["path"] for asset in assets],
            "scope": "installed-runtime",
        }

        gate = smoke_evidence.evaluate_sd15_style_reference_discovery_evidence([install_repair])

        self.assertFalse(gate.ready)
        self.assertEqual(gate.status, "partial")
        self.assertIn("asset_identity", gate.missing_kinds)
        self.assertIn("model_load", gate.missing_kinds)
        self.assertIn("generate_no_download", gate.missing_kinds)
        self.assertIn("gpu_generation", gate.missing_kinds)
        self.assertIn("installed_parity", gate.missing_kinds)

    def test_complete_sd15_style_evidence_promotes_sd15_only_not_windows_public_sdxl_or_controlnet(self) -> None:
        packages = {"python": "3.12", "cuda": "13.0", "torch": "2.11.0+cu130", "diffusers": "0.35.1"}
        revision = "018e402774aeeddd60609b4ecdb7e298259dc729"
        assets = [
            {
                "id": "sd15_ip_adapter_style",
                "relative_path": "models/ip-adapter_sd15.safetensors",
                "path": "/models/sd15/optional/sd15_ip_adapter_style/models/ip-adapter_sd15.safetensors",
                "repo": "h94/IP-Adapter",
                "revision": revision,
                "size": 44642768,
                "sha256": "adapter-local-sha256",
            },
            {
                "id": "sd15_ip_adapter_style",
                "relative_path": "models/image_encoder/config.json",
                "path": "/models/sd15/optional/sd15_ip_adapter_style/models/image_encoder/config.json",
                "repo": "h94/IP-Adapter",
                "revision": revision,
                "size": 560,
                "sha256": "config-local-sha256",
            },
            {
                "id": "sd15_ip_adapter_style",
                "relative_path": "models/image_encoder/model.safetensors",
                "path": "/models/sd15/optional/sd15_ip_adapter_style/models/image_encoder/model.safetensors",
                "repo": "h94/IP-Adapter",
                "revision": revision,
                "size": 2528373448,
                "sha256": "encoder-local-sha256",
            },
        ]

        def record(kind: str, **overrides: object) -> dict[str, object]:
            payload: dict[str, object] = {
                "kind": kind,
                "feature": "sd15_ip_adapter_style",
                "platform": "linux/arm64",
                "ref": "authorized-manifest-promotion",
                "version": "0.1.1",
                "commands": [f"collect {kind}"],
                "assets": assets,
                "packages": packages,
                "outputs": ["/outputs/sd15-style.png"] if kind == "gpu_generation" else [],
                "logs": [f"{kind} passed"],
                "result": "pass",
                "scope": "installed-runtime",
            }
            payload.update(overrides)
            return payload

        complete_sd15 = [
            record("asset_identity"),
            record("install_repair", readiness="ready", local_assets=[asset["path"] for asset in assets]),
            record("model_load", local_files_only=True),
            record("generate_no_download", local_files_only=True),
            record("gpu_generation", local_files_only=True),
            record("installed_parity", parity="fresh"),
        ]
        windows_records = [dict(item, platform="windows-amd64") for item in complete_sd15]
        public_release_records = [dict(item, scope="public-release") for item in complete_sd15]
        sdxl_records = [dict(item, feature="sdxl_ip_adapter_style") for item in complete_sd15]
        controlnet_records = [dict(item, feature="sd15_controlnet_canny") for item in complete_sd15]

        sd15_gate = smoke_evidence.evaluate_sd15_style_reference_discovery_evidence(complete_sd15)
        windows_gate = smoke_evidence.evaluate_sd15_style_reference_discovery_evidence(windows_records)
        public_gate = smoke_evidence.evaluate_sd15_style_reference_discovery_evidence(public_release_records)
        sdxl_gate = smoke_evidence.evaluate_sd15_style_reference_discovery_evidence(sdxl_records)
        controlnet_gate = smoke_evidence.evaluate_sd15_style_reference_discovery_evidence(controlnet_records)

        self.assertTrue(sd15_gate.ready)
        self.assertEqual(sd15_gate.status, "ready")
        self.assertEqual(sd15_gate.missing_kinds, ())
        self.assertFalse(windows_gate.ready)
        self.assertFalse(public_gate.ready)
        self.assertFalse(sdxl_gate.ready)
        self.assertFalse(controlnet_gate.ready)
        self.assertIn("Windows", " ".join(windows_gate.diagnostics))
        self.assertIn("public release", " ".join(public_gate.diagnostics))
        self.assertIn("SDXL", " ".join(sdxl_gate.diagnostics))
        self.assertIn("ControlNet", " ".join(controlnet_gate.diagnostics))

    def test_sdxl_style_promotion_diagnostics_reject_non_sdxl_windows_and_public_release_scope(self) -> None:
        records = []
        for feature, platform, scope in (
            ("sd15_ip_adapter_style", "linux/arm64", "installed-runtime"),
            ("sdxl_controlnet_canny", "linux/arm64", "installed-runtime"),
            ("ip_adapter", "linux/arm64", "installed-runtime"),
            ("sdxl_ip_adapter_style", "windows-amd64", "installed-runtime"),
            ("sdxl_ip_adapter_style", "linux/arm64", "public-release"),
        ):
            records.append(
                {
                    "kind": "gpu_generation",
                    "feature": feature,
                    "platform": platform,
                    "ref": "out-of-scope",
                    "version": "0.1.1",
                    "commands": ["not-promoted"],
                    "assets": [{"path": "/tmp/asset", "sha256": "sha"}],
                    "packages": {"python": "3.12", "cuda": "12.4", "torch": "2.7.0", "diffusers": "0.35.1"},
                    "outputs": ["/tmp/out.png"],
                    "logs": ["out of scope"],
                    "result": "pass",
                    "local_files_only": True,
                    "scope": scope,
                }
            )

        gate = smoke_evidence.evaluate_sdxl_style_reference_local_smoke_evidence(records)

        self.assertFalse(gate.ready)
        self.assertEqual(gate.status, "blocked")
        diagnostics = " ".join(gate.diagnostics)
        self.assertIn("sd15_ip_adapter_style", diagnostics)
        self.assertIn("ControlNet", diagnostics)
        self.assertIn("generic IP-Adapter", diagnostics)
        self.assertIn("Windows", diagnostics)
        self.assertIn("public release", diagnostics)

    def test_generate_missing_sdxl_style_assets_reports_local_readiness_without_download(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class LocalOnlyPipeline:
            def load_ip_adapter(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("missing local assets must be detected before IP-Adapter loading")

            def __call__(self, **kwargs: object) -> object:
                raise AssertionError("generation must not run when local style assets are missing")

        class LocalOnlyLoader:
            def from_pretrained(self, model_dir: str, **kwargs: object) -> LocalOnlyPipeline:
                return LocalOnlyPipeline()

        workspace_dir = Path(tempfile.mkdtemp(prefix="generate-missing-sdxl-style-"))
        extension_model_dir = workspace_dir / "models" / "sdxl-base"
        source_path = workspace_dir / "source.png"
        reference_path = workspace_dir / "style-reference.png"
        source_path.write_bytes(b"source")
        reference_path.write_bytes(b"style")
        job = {
            "extension_id": "sdxl-base",
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(extension_model_dir / "image-to-image"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(workspace_dir / "result.png"),
            "prompt": "variation",
            "source_image_path": str(source_path),
            "params": {"steps": 4, "strength": 0.55, "reference_strength": 0.4},
            "conditioning": {"references": [{"role": "style", "filePath": str(reference_path)}]},
        }

        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("sdxl", "image-to-image"): LocalOnlyLoader()},
            clear=True,
        ), patch(
            "local_image_runtime.weights.acquire_optional_feature_weights",
            side_effect=AssertionError("Generate must not acquire optional feature assets"),
        ) as acquire_optional, patch(
            "local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download",
            side_effect=AssertionError("Generate must not download optional feature assets"),
        ) as snapshot_download:
            stdout = StringIO()
            exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

        self.assertEqual(exit_code, 1)
        events = self._parse_ndjson_events(stdout.getvalue())
        self.assertEqual(events[-1]["type"], "error")
        message = str(events[-1]["message"])
        self.assertIn("local-readiness", message)
        self.assertIn("local", message)
        self.assertIn("assets", message)
        self.assertNotIn("downloaded", message.casefold())
        self.assertNotIn("attempted to download", message.casefold())
        acquire_optional.assert_not_called()
        snapshot_download.assert_not_called()

    def test_install_repair_may_acquire_supported_sdxl_style_assets_without_real_download(self) -> None:
        runtime_root = self._make_runtime_root("sdxl-base")
        acquired: list[dict[str, object]] = []

        def record_optional_feature_acquisition(extension_id, feature_id, models_dir, *, downloader=None):
            acquired.append({"extension_id": extension_id, "feature_id": feature_id, "models_dir": Path(models_dir)})
            return self._fake_optional_feature_acquisition(extension_id, feature_id, models_dir, downloader=downloader)

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False))
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM))
            stack.enter_context(patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=self._fake_plan("sdxl-base")))
            stack.enter_context(patch("local_image_runtime.install_contract._run_checked", side_effect=self._run_checked_side_effect))
            stack.enter_context(patch("local_image_runtime.install_contract._install_dependency_step", return_value=None))
            stack.enter_context(patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports")))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.acquire_optional_feature_weights",
                    side_effect=record_optional_feature_acquisition,
                )
            )
            stack.enter_context(
                patch(
                    "local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download",
                    side_effect=AssertionError("mocked Install/Repair acquisition must not perform a real download"),
                )
            )

            result = install_contract.run_install_setup_contract(
                extension_id="sdxl-base",
                stdin_text=self._payload(runtime_root),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
        self.assertEqual(len(acquired), 1)
        self.assertEqual(acquired[0]["extension_id"], "sdxl-base")
        self.assertEqual(acquired[0]["feature_id"], "sdxl_ip_adapter_style")

    def test_generate_accepts_sdxl_style_reference_when_local_assets_are_present(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FakeImage:
            def save(self, output_path: str, **kwargs: object) -> None:
                Path(output_path).write_bytes(b"generated")

        class LocalStylePipeline:
            def __init__(self) -> None:
                self.loaded_adapter: dict[str, object] | None = None
                self.reference_strength: object | None = None
                self.invocations: list[dict[str, object]] = []

            def load_ip_adapter(self, model_dir: str, **kwargs: object) -> None:
                self.loaded_adapter = {"model_dir": model_dir, **kwargs}

            def set_ip_adapter_scale(self, value: object) -> None:
                self.reference_strength = value

            def __call__(self, **kwargs: object) -> object:
                self.invocations.append(kwargs)
                return SimpleNamespace(images=[FakeImage()])

        class LocalStyleLoader:
            def __init__(self, pipeline_instance: LocalStylePipeline) -> None:
                self.pipeline_instance = pipeline_instance

            def from_pretrained(self, model_dir: str, **kwargs: object) -> LocalStylePipeline:
                return self.pipeline_instance

        workspace_dir = Path(tempfile.mkdtemp(prefix="generate-ready-sdxl-style-"))
        extension_model_dir = workspace_dir / "models" / "sdxl-base"
        asset_dir = extension_model_dir / "optional" / "sdxl_ip_adapter_style"
        for relative_path in (
            "sdxl_models/ip-adapter_sdxl.bin",
            "sdxl_models/image_encoder/config.json",
            "sdxl_models/image_encoder/model.safetensors",
        ):
            asset_path = asset_dir / relative_path
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(b"asset")
        source_path = workspace_dir / "source.png"
        reference_path = workspace_dir / "style-reference.png"
        source_path.write_bytes(b"source")
        reference_path.write_bytes(b"style")
        pipeline_instance = LocalStylePipeline()
        source_image_token = object()
        reference_image_token = object()
        opened_images: list[str] = []

        def open_image(path: str) -> object:
            opened_images.append(path)
            if path == str(source_path):
                return source_image_token
            if path == str(reference_path):
                return reference_image_token
            raise AssertionError(f"unexpected image path {path}")

        job = {
            "extension_id": "sdxl-base",
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(extension_model_dir / "image-to-image"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(workspace_dir / "result.png"),
            "prompt": "variation",
            "source_image_path": str(source_path),
            "params": {"steps": 4, "strength": 0.55, "reference_strength": 0.4},
            "conditioning": {"references": [{"role": "style", "filePath": str(reference_path)}]},
        }

        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("sdxl", "image-to-image"): LocalStyleLoader(pipeline_instance)},
            clear=True,
        ), patch("local_image_runtime.inference_runner._open_source_image", side_effect=open_image), patch(
            "local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download",
            side_effect=AssertionError("Generate must use present local assets without download"),
        ) as snapshot_download:
            result = inference_runner.run_child_job(job)

        self.assertEqual(result["output_path"], str(workspace_dir / "result.png"))
        self.assertEqual(pipeline_instance.loaded_adapter["model_dir"], str(asset_dir))
        self.assertTrue(pipeline_instance.loaded_adapter["local_files_only"])
        self.assertEqual(pipeline_instance.reference_strength, 0.4)
        self.assertEqual(pipeline_instance.invocations[0]["image"], source_image_token)
        self.assertEqual(pipeline_instance.invocations[0]["ip_adapter_image"], reference_image_token)
        self.assertEqual(opened_images, [str(source_path), str(reference_path)])
        snapshot_download.assert_not_called()

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

    def _make_installed_extension_record(self, *, extension_id: str, workspace_dir: Path) -> dict[str, str]:
        extension_root = Path(tempfile.mkdtemp(prefix=f"ext-root-{extension_id}-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        return {
            "venv_python": str(self._make_executable_python(extension_root)),
            "model_dir": str(workspace_dir / "models" / extension_id),
        }

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

    def _windows_payload(self, runtime_root: Path) -> str:
        return json.dumps(
            {
                "python_exe": str(runtime_root / "venv" / "Scripts" / "python.exe"),
                "ext_dir": str(runtime_root),
                "gpu_sm": "89",
                "cuda_version": "12.8",
            }
        )

    def _sd15_windows_evidence(self, **overrides: object) -> dict[str, object]:
        evidence: dict[str, object] = {
            "extension_id": "sd15",
            "status": "verified",
            "reviewed": True,
            "platform_key": "windows-amd64",
            "os_name": "Windows",
            "os_version": "11 Pro",
            "os_build": "22631",
            "machine": "AMD64",
            "python_version": "3.12.4",
            "python_abi": "cp312",
            "sysconfig_platform": "win-amd64",
            "pip_version": "24.2",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "nvidia_driver": "555.99",
            "torch_cuda_available": True,
            "torch_version": "2.7.0",
            "torchvision_version": "0.22.0",
            "torch_cuda_version": "12.8",
            "cuda_variant": "cu128",
            "pip_freeze": ["torch==2.7.0+cu128", "torchvision==0.22.0"],
            "torch_wheel": "torch-2.7.0+cu128-cp312-cp312-win_amd64.whl",
            "torchvision_wheel": "torchvision-0.22.0-cp312-cp312-win_amd64.whl",
            "import_results": {
                "torch": "ok",
                "torchvision": "ok",
                "diffusers": "ok",
                "transformers": "ok",
                "sentencepiece": "ok",
                "scipy": "ok",
            },
            "model_layout": {"model_index.json": "present", "unet": "present"},
            "model_repo": "runwayml/stable-diffusion-v1-5",
            "model_load": {"status": "ok", "pipeline": "StableDiffusionPipeline"},
            "smoke_inference": {"status": "ok", "node_id": "text-to-image", "output": "metadata-only"},
            "timestamp": "2026-04-25T20:00:00Z",
            "operator": "manual-windows-review",
            "tool_version": "local-image-runtime-test",
            "failure_diagnostics": [],
        }
        evidence.update(overrides)
        return evidence

    def _write_sd15_windows_evidence(self, payload: dict[str, object]) -> Path:
        evidence_path = Path(tempfile.mkdtemp(prefix="sd15-windows-evidence-")) / "evidence.json"
        evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return evidence_path

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

    def _fake_optional_feature_acquisition(self, extension_id, feature_id, models_dir, *, downloader=None):
        target_dir = Path(models_dir) / extension_id / "optional" / feature_id
        check_path = target_dir / "sdxl_models" / "ip-adapter_sdxl.bin"
        check_path.parent.mkdir(parents=True, exist_ok=True)
        check_path.write_bytes(b"adapter")
        return {
            "status": "ready",
            "extension_id": extension_id,
            "feature_id": feature_id,
            "model_dir": str(target_dir),
            "check_path": str(check_path),
            "downloaded": True,
        }

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
                patch(
                    "local_image_runtime.install_contract.acquire_optional_feature_weights",
                    side_effect=self._fake_optional_feature_acquisition,
                )
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

    def test_flux_setup_readiness_stays_ready_when_weights_are_missing(self) -> None:
        extension_id = "flux-schnell"
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
            snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)

        record = bootstrap.get_extension_record(snapshot, extension_id)
        expected_model_dir = runtime_root / ".local-image-runtime" / "models" / "flux-schnell" / "text-to-image"
        self.assertEqual(record["setup"]["status"], bootstrap.SETUP_STATUS_READY)
        self.assertEqual(record["readiness"], bootstrap.SETUP_STATUS_READY)
        self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_INSTALLED)
        self.assertEqual(record["weights_readiness"], "missing")
        self.assertEqual(record["weights"]["nodes"]["text-to-image"]["model_dir"], str(expected_model_dir))

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

    def test_linux_arm64_dependency_plan_preserves_verified_state_and_steps(self) -> None:
        plan = self._resolve_plan(python_tag="cp311", cuda_version="12.4")

        self.assertEqual(plan.plan_state, "verified")
        self.assertTrue(plan.platform_supported)
        self.assertEqual(plan.platform_key, "linux-aarch64")
        self.assertEqual(
            [step.name for step in (*plan.shared_steps, *plan.family_steps)],
            [
                "install_shared_torch",
                "install_shared_runtime",
                "install_family_dependencies",
            ],
        )
        self.assertEqual(plan.diagnostics, ())

    def test_gb10_sm121_does_not_inherit_verified_linux_arm64_cu124_plan(self) -> None:
        plan = dependencies.resolve_dependency_plan(
            extension_id="sdxl-base",
            dependency_family="sdxl-base",
            readiness_imports=("torch", "diffusers"),
            platform_info=SUPPORTED_PLATFORM,
            python_tag="cp312",
            cuda_version="12.4",
            gpu_sm="121",
            torch_arch_list=("sm_50", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a"),
            torch_version="2.5.1+cu124",
        )

        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_UNSUPPORTED)
        self.assertFalse(plan.platform_supported)
        self.assertEqual(plan.shared_steps, ())
        self.assertEqual(plan.family_steps, ())
        diagnostics_text = " ".join(plan.diagnostics)
        self.assertIn("sm_121", diagnostics_text)
        self.assertIn("2.5.1+cu124", diagnostics_text)
        self.assertIn("torch/CUDA", diagnostics_text)
        self.assertIn("dependency/runtime compatibility", diagnostics_text)
        self.assertIn("no kernel image", diagnostics_text)
        self.assertIn("not as an IP-Adapter", diagnostics_text)
        self.assertIn("ControlNet", diagnostics_text)
        self.assertIn("local asset", diagnostics_text)

    def test_gb10_sm121_does_not_inherit_verified_linux_arm64_cu128_plan_without_runtime_proof(self) -> None:
        plan = dependencies.resolve_dependency_plan(
            extension_id="sd15",
            dependency_family="sd15",
            readiness_imports=("torch",),
            platform_info=SUPPORTED_PLATFORM,
            python_tag="cp312",
            cuda_version="12.8",
            gpu_sm="121",
            torch_arch_list=("sm_50", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a"),
            torch_version="2.7.0+cu128",
        )

        self.assertEqual(plan.cuda_variant, "cu128")
        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_UNSUPPORTED)
        self.assertFalse(plan.platform_supported)
        diagnostics_text = " ".join(plan.diagnostics)
        self.assertIn("sm_121", diagnostics_text)
        self.assertIn("cu128", diagnostics_text)
        self.assertIn("runtime proof", diagnostics_text)

    def test_linux_arm64_non_gb10_plan_remains_verified_when_gpu_sm_is_supported_or_unknown(self) -> None:
        supported_sm_plan = dependencies.resolve_dependency_plan(
            extension_id="sd15",
            dependency_family="sd15",
            readiness_imports=("torch",),
            platform_info=SUPPORTED_PLATFORM,
            python_tag="cp312",
            cuda_version="12.8",
            gpu_sm="90",
            torch_arch_list=("sm_80", "sm_90", "sm_90a"),
            torch_version="2.7.0+cu128",
        )
        unknown_sm_plan = dependencies.resolve_dependency_plan(
            extension_id="sd15",
            dependency_family="sd15",
            readiness_imports=("torch",),
            platform_info=SUPPORTED_PLATFORM,
            python_tag="cp312",
            cuda_version="12.8",
            gpu_sm=None,
        )

        self.assertEqual(supported_sm_plan.plan_state, dependencies.PLAN_STATE_VERIFIED)
        self.assertTrue(supported_sm_plan.platform_supported)
        self.assertEqual(supported_sm_plan.diagnostics, ())
        self.assertEqual(unknown_sm_plan.plan_state, dependencies.PLAN_STATE_VERIFIED)
        self.assertTrue(unknown_sm_plan.platform_supported)
        self.assertEqual(unknown_sm_plan.shared_steps[0].packages, dependencies._shared_runtime_steps("cu128", "cp312")[0].packages)

    def test_gb10_sm121_runtime_proof_selects_runtime_proven_cu130_not_native_sm121(self) -> None:
        plan = dependencies.resolve_dependency_plan(
            extension_id="sdxl-base",
            dependency_family="sdxl-base",
            readiness_imports=("torch", "diffusers"),
            platform_info=SUPPORTED_PLATFORM,
            python_tag="cp312",
            cuda_version="13.0",
            gpu_sm="121",
            torch_arch_list=("sm_80", "sm_90", "sm_100", "sm_110", "sm_120"),
            torch_version="2.11.0+cu130",
            gb10_runtime_evidence={
                "source_index": "https://download.pytorch.org/whl/cu130",
                "torch_version": "2.11.0+cu130",
                "torchvision_version": "0.26.0+cu130",
                "python_tag": "cp312",
                "platform_tag": "manylinux_2_28_aarch64",
                "cuda_variant": "cu130",
                "torch_cuda": "13.0",
                "driver": "580.142",
                "gpu_name": "NVIDIA GB10",
                "capability": [12, 1],
                "cuda_available": True,
                "matmul_passed": True,
                "synchronize_passed": True,
                "dependency_imports": {
                    "diffusers": "0.35.1",
                    "transformers": "4.57.6",
                    "accelerate": "1.13.0",
                    "safetensors": "0.7.0",
                    "sentencepiece": "0.2.1",
                    "scipy": "1.17.1",
                    "local_image_runtime": "ok",
                },
            },
        )

        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_VERIFIED)
        self.assertTrue(plan.platform_supported)
        self.assertEqual(plan.cuda_variant, "cu130")
        torch_step = plan.shared_steps[0]
        self.assertEqual(torch_step.name, "install_shared_torch")
        self.assertEqual(torch_step.extra_args[3], "https://download.pytorch.org/whl/cu130")
        self.assertEqual(
            torch_step.packages,
            (
                dependencies._TORCH_WHEELS["cu130"]["cp312"]["torch"],
                dependencies._TORCH_WHEELS["cu130"]["cp312"]["torchvision"],
            ),
        )
        diagnostics_text = " ".join(plan.diagnostics)
        self.assertIn("runtime-proven GB10 (12,1)", diagnostics_text)
        self.assertIn("not native sm_121 arch-list support", diagnostics_text)
        self.assertNotIn("public release", diagnostics_text.lower())
        self.assertNotIn("Windows", diagnostics_text)

    def test_gb10_sm121_cu130_index_availability_without_runtime_evidence_is_rejected(self) -> None:
        plan = dependencies.resolve_dependency_plan(
            extension_id="sdxl-base",
            dependency_family="sdxl-base",
            readiness_imports=("torch", "diffusers"),
            platform_info=SUPPORTED_PLATFORM,
            python_tag="cp312",
            cuda_version="13.0",
            gpu_sm="121",
            torch_arch_list=("sm_80", "sm_90", "sm_100", "sm_110", "sm_120"),
            torch_version="2.11.0+cu130",
            gb10_runtime_evidence={
                "source_index": "https://download.pytorch.org/whl/cu130",
                "torch_version": "2.11.0+cu130",
                "torchvision_version": "0.26.0+cu130",
                "python_tag": "cp312",
                "platform_tag": "manylinux_2_28_aarch64",
                "cuda_variant": "cu130",
            },
        )

        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_UNSUPPORTED)
        self.assertFalse(plan.platform_supported)
        self.assertEqual(plan.shared_steps, ())
        diagnostics_text = " ".join(plan.diagnostics)
        self.assertIn("index availability", diagnostics_text)
        self.assertIn("runtime/dependency evidence", diagnostics_text)
        self.assertIn("matmul", diagnostics_text)
        self.assertIn("synchronize", diagnostics_text)

    def test_gb10_sm121_cu130_runtime_evidence_without_matmul_synchronize_is_rejected(self) -> None:
        plan = dependencies.resolve_dependency_plan(
            extension_id="sdxl-base",
            dependency_family="sdxl-base",
            readiness_imports=("torch", "diffusers"),
            platform_info=SUPPORTED_PLATFORM,
            python_tag="cp312",
            cuda_version="13.0",
            gpu_sm="121",
            torch_arch_list=("sm_80", "sm_90", "sm_100", "sm_110", "sm_120"),
            torch_version="2.11.0+cu130",
            gb10_runtime_evidence={
                "source_index": "https://download.pytorch.org/whl/cu130",
                "torch_version": "2.11.0+cu130",
                "torchvision_version": "0.26.0+cu130",
                "python_tag": "cp312",
                "platform_tag": "manylinux_2_28_aarch64",
                "cuda_variant": "cu130",
                "torch_cuda": "13.0",
                "driver": "580.142",
                "gpu_name": "NVIDIA GB10",
                "capability": [12, 1],
                "cuda_available": True,
                "matmul_passed": False,
                "synchronize_passed": False,
                "dependency_imports": {
                    "diffusers": "0.35.1",
                    "transformers": "4.57.6",
                    "accelerate": "1.13.0",
                    "safetensors": "0.7.0",
                    "sentencepiece": "0.2.1",
                    "scipy": "1.17.1",
                    "local_image_runtime": "ok",
                },
            },
        )

        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_UNSUPPORTED)
        self.assertFalse(plan.platform_supported)
        diagnostics_text = " ".join(plan.diagnostics)
        self.assertIn("matmul", diagnostics_text)
        self.assertIn("synchronize", diagnostics_text)
        self.assertIn("runtime proof", diagnostics_text)

    def test_install_contract_forwards_gpu_sm_to_dependency_planning(self) -> None:
        runtime_root = self._make_runtime_root("sd15")
        unsupported_plan = DependencyPlan(
            extension_id="sd15",
            dependency_family="sd15",
            platform_system="linux",
            platform_machine="aarch64",
            python_tag="cp312",
            cuda_variant="cu124",
            shared_steps=(),
            family_steps=(),
            plan_state=dependencies.PLAN_STATE_UNSUPPORTED,
            platform_key="linux-aarch64",
            platform_supported=False,
            diagnostics=("sm_121 requires torch/CUDA runtime proof before dependency install.",),
        )

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False))
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM))
            stack.enter_context(patch("local_image_runtime.install_contract.python_tag_from_interpreter", return_value="cp312"))
            resolve_plan = stack.enter_context(
                patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=unsupported_plan)
            )

            result = install_contract.run_install_setup_contract(
                extension_id="sd15",
                stdin_text=json.dumps(
                    {
                        "python_exe": sys.executable,
                        "ext_dir": str(runtime_root),
                        "gpu_sm": "121",
                        "cuda_version": "12.4",
                    }
                ),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
        self.assertEqual(resolve_plan.call_args.kwargs["gpu_sm"], "121")

    def test_install_contract_forwards_installed_gb10_runtime_evidence_to_dependency_planning(self) -> None:
        runtime_root = self._make_runtime_root("sdxl-base")
        gb10_runtime_evidence = {
            "source_index": "https://download.pytorch.org/whl/cu130",
            "torch_version": "2.11.0+cu130",
            "torchvision_version": "0.26.0+cu130",
            "python_tag": "cp312",
            "platform_tag": "manylinux_2_28_aarch64",
            "cuda_variant": "cu130",
            "torch_cuda": "13.0",
            "driver": "580.142",
            "gpu_name": "NVIDIA GB10",
            "capability": [12, 1],
            "cuda_available": True,
            "matmul_passed": True,
            "synchronize_passed": True,
            "dependency_imports": {
                "diffusers": "0.35.1",
                "transformers": "4.57.6",
                "accelerate": "1.13.0",
                "safetensors": "0.7.0",
                "sentencepiece": "0.2.1",
                "scipy": "1.17.1",
                "local_image_runtime": "ok",
            },
        }
        unsupported_plan = DependencyPlan(
            extension_id="sdxl-base",
            dependency_family="sdxl-base",
            platform_system="linux",
            platform_machine="aarch64",
            python_tag="cp312",
            cuda_variant="cu130",
            shared_steps=(),
            family_steps=(),
            plan_state=dependencies.PLAN_STATE_UNSUPPORTED,
            platform_key="linux-aarch64",
            platform_supported=False,
            diagnostics=("probe-only test plan",),
        )

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False))
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM))
            stack.enter_context(patch("local_image_runtime.install_contract.python_tag_from_interpreter", return_value="cp312"))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract._gb10_runtime_evidence_from_installed_venv",
                    return_value=gb10_runtime_evidence,
                )
            )
            resolve_plan = stack.enter_context(
                patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=unsupported_plan)
            )

            result = install_contract.run_install_setup_contract(
                extension_id="sdxl-base",
                stdin_text=json.dumps(
                    {
                        "python_exe": sys.executable,
                        "ext_dir": str(runtime_root),
                        "gpu_sm": "121",
                        "cuda_version": "13.0",
                    }
                ),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
        self.assertEqual(resolve_plan.call_args.kwargs["gpu_sm"], "121")
        self.assertEqual(resolve_plan.call_args.kwargs["gb10_runtime_evidence"], gb10_runtime_evidence)

    def test_sd15_windows_candidate_matrix_is_visible_without_verified_evidence(self) -> None:
        plan = dependencies.resolve_dependency_plan(
            extension_id="sd15",
            dependency_family="sd15",
            readiness_imports=("torch", "diffusers"),
            platform_info=WINDOWS_PLATFORM,
            python_tag="cp312",
            cuda_version="12.8",
        )

        self.assertEqual(plan.platform_key, "windows-amd64")
        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertFalse(plan.platform_supported)
        self.assertEqual(plan.cuda_variant, "cu128")
        self.assertEqual(plan.python_tag, "cp312")
        self.assertEqual(plan.shared_steps[0].name, "install_shared_torch")
        self.assertEqual(
            plan.shared_steps[0].packages,
            ("torch==2.7.0", "torchvision==0.22.0"),
        )
        self.assertEqual(
            plan.shared_steps[0].extra_args,
            (
                "--index-url",
                dependencies._PYPI_INDEX_URL,
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu128",
                "--no-cache-dir",
            ),
        )
        self.assertEqual(
            plan.family_steps[0].packages,
            ("diffusers==0.35.1", "transformers>=4.46,<5", "sentencepiece", "scipy"),
        )
        diagnostics_text = " ".join(plan.diagnostics)
        self.assertIn("first-pass", diagnostics_text)
        self.assertIn("candidate", diagnostics_text)
        self.assertIn("experimental", diagnostics_text)

    def test_sdxl_windows_candidate_matrix_is_visible_with_executable_steps(self) -> None:
        with patch("local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download") as download:
            plan = dependencies.resolve_dependency_plan(
                extension_id="sdxl-base",
                dependency_family="sdxl-base",
                readiness_imports=("torch", "diffusers"),
                platform_info=WINDOWS_PLATFORM,
                python_tag="cp312",
                cuda_version="12.8",
            )

        self.assertFalse(download.called)
        self.assertEqual(plan.extension_id, "sdxl-base")
        self.assertEqual(plan.dependency_family, "sdxl-base")
        self.assertEqual(plan.platform_key, "windows-amd64")
        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertFalse(plan.platform_supported)
        self.assertEqual(plan.cuda_variant, "cu128")
        self.assertEqual(plan.python_tag, "cp312")
        self.assertEqual(
            [step.name for step in (*plan.shared_steps, *plan.family_steps)],
            ["install_shared_torch", "install_shared_runtime", "install_family_dependencies"],
        )
        self.assertEqual(plan.shared_steps[0].packages, ("torch==2.7.0", "torchvision==0.22.0"))
        self.assertEqual(
            plan.shared_steps[0].extra_args,
            (
                "--index-url",
                dependencies._PYPI_INDEX_URL,
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu128",
                "--no-cache-dir",
            ),
        )
        self.assertEqual(
            plan.family_steps[0].packages,
            ("diffusers==0.35.1", "transformers>=4.46,<5", "sentencepiece", "scipy"),
        )
        diagnostics_text = " ".join(plan.diagnostics)
        self.assertIn("SDXL", diagnostics_text)
        self.assertIn("candidate", diagnostics_text)
        self.assertIn("first-pass", diagnostics_text)
        self.assertIn("unverified", diagnostics_text)
        self.assertIn("not verified compatibility", diagnostics_text)

    def test_flux_windows_candidate_matrix_is_visible_with_executable_steps(self) -> None:
        with patch("local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download") as download:
            plan = dependencies.resolve_dependency_plan(
                extension_id="flux-schnell",
                dependency_family="flux-schnell",
                readiness_imports=("torch", "diffusers", "transformers"),
                platform_info=WINDOWS_PLATFORM,
                python_tag="cp312",
                cuda_version="12.8",
            )

        self.assertFalse(download.called)
        self.assertEqual(plan.extension_id, "flux-schnell")
        self.assertEqual(plan.dependency_family, "flux-schnell")
        self.assertEqual(plan.platform_key, "windows-amd64")
        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertFalse(plan.platform_supported)
        self.assertEqual(plan.cuda_variant, "cu128")
        self.assertEqual(plan.python_tag, "cp312")
        self.assertEqual(
            [step.name for step in (*plan.shared_steps, *plan.family_steps)],
            ["install_shared_torch", "install_shared_runtime", "install_family_dependencies"],
        )
        self.assertEqual(plan.shared_steps[0].packages, ("torch==2.7.0", "torchvision==0.22.0"))
        self.assertEqual(
            plan.shared_steps[0].extra_args,
            (
                "--index-url",
                dependencies._PYPI_INDEX_URL,
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu128",
                "--no-cache-dir",
            ),
        )
        self.assertEqual(
            plan.shared_steps[1].packages,
            ("Pillow", "numpy", "huggingface_hub", "safetensors", "accelerate"),
        )
        self.assertEqual(
            plan.family_steps[0].packages,
            ("diffusers==0.35.1", "transformers>=4.46,<5", "sentencepiece", "protobuf<6"),
        )
        diagnostics_text = " ".join(plan.diagnostics)
        self.assertIn("Flux", diagnostics_text)
        self.assertIn("candidate", diagnostics_text)
        self.assertIn("first-pass", diagnostics_text)
        self.assertIn("unverified", diagnostics_text)
        self.assertIn("not verified compatibility", diagnostics_text)
        self.assertIn("model", diagnostics_text)
        self.assertIn("token", diagnostics_text)
        self.assertIn("terms", diagnostics_text)
        self.assertIn("VRAM", diagnostics_text)

    def test_flux_windows_candidate_matrix_reaches_executable_steps_for_runtime_python_tag(self) -> None:
        plan = dependencies.resolve_dependency_plan(
            extension_id="flux-schnell",
            dependency_family="flux-schnell",
            readiness_imports=("torch", "diffusers", "transformers"),
            platform_info=WINDOWS_PLATFORM,
            python_tag="cp311",
            cuda_version="12.8",
        )

        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertTrue(install_contract._candidate_install_allowed(plan))
        self.assertEqual(
            [step.name for step in (*plan.shared_steps, *plan.family_steps)],
            ["install_shared_torch", "install_shared_runtime", "install_family_dependencies"],
        )

    def test_sdxl_windows_candidate_matrix_reaches_executable_steps_for_runtime_python_tag(self) -> None:
        plan = dependencies.resolve_dependency_plan(
            extension_id="sdxl-base",
            dependency_family="sdxl-base",
            readiness_imports=("torch", "diffusers"),
            platform_info=WINDOWS_PLATFORM,
            python_tag="cp311",
            cuda_version="12.8",
        )

        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertTrue(install_contract._candidate_install_allowed(plan))
        self.assertEqual(
            [step.name for step in (*plan.shared_steps, *plan.family_steps)],
            ["install_shared_torch", "install_shared_runtime", "install_family_dependencies"],
        )

    def test_install_contract_allows_only_exact_sd15_windows_candidate_plan(self) -> None:
        valid_plan = DependencyPlan(
            extension_id="sd15",
            dependency_family="sd15",
            platform_system="windows",
            platform_machine="amd64",
            python_tag="cp312",
            cuda_variant="cu128",
            shared_steps=(DependencyInstallStep(name="install_shared_torch", packages=("torch==2.7.0",)),),
            family_steps=(DependencyInstallStep(name="install_family_dependencies", packages=("diffusers==0.35.1",)),),
            plan_state=dependencies.PLAN_STATE_CANDIDATE_INSTALL,
            platform_key="windows-amd64",
            platform_supported=False,
        )

        self.assertTrue(install_contract._candidate_install_allowed(valid_plan))

        rejected_cases = (
            ("wrong extension", {"extension_id": "sdxl-base"}),
            ("wrong platform", {"platform_key": "linux-aarch64", "platform_system": "linux", "platform_machine": "aarch64"}),
            ("wrong cuda", {"cuda_variant": "cu124"}),
            ("wrong state", {"plan_state": dependencies.PLAN_STATE_VERIFIED, "platform_supported": True}),
            ("empty steps", {"shared_steps": (), "family_steps": ()}),
        )
        for label, overrides in rejected_cases:
            with self.subTest(label=label):
                candidate = DependencyPlan(**{**valid_plan.__dict__, **overrides})
                self.assertFalse(install_contract._candidate_install_allowed(candidate))

    def test_install_contract_allows_only_exact_sdxl_windows_candidate_plan(self) -> None:
        valid_plan = DependencyPlan(
            extension_id="sdxl-base",
            dependency_family="sdxl-base",
            platform_system="windows",
            platform_machine="amd64",
            python_tag="cp312",
            cuda_variant="cu128",
            shared_steps=(DependencyInstallStep(name="install_shared_torch", packages=("torch==2.7.0",)),),
            family_steps=(DependencyInstallStep(name="install_family_dependencies", packages=("diffusers==0.35.1",)),),
            plan_state=dependencies.PLAN_STATE_CANDIDATE_INSTALL,
            platform_key="windows-amd64",
            platform_supported=False,
        )

        self.assertTrue(install_contract._candidate_install_allowed(valid_plan))

        rejected_cases = (
            ("wrong extension", {"extension_id": "not-sdxl"}),
            ("wrong family", {"dependency_family": "sd15"}),
            ("wrong platform", {"platform_key": "linux-aarch64", "platform_system": "linux", "platform_machine": "aarch64"}),
            ("wrong cuda", {"cuda_variant": "cu124"}),
            ("wrong state", {"plan_state": dependencies.PLAN_STATE_VERIFIED, "platform_supported": True}),
            ("empty steps", {"shared_steps": (), "family_steps": ()}),
        )
        for label, overrides in rejected_cases:
            with self.subTest(label=label):
                candidate = DependencyPlan(**{**valid_plan.__dict__, **overrides})
                self.assertFalse(install_contract._candidate_install_allowed(candidate))

    def test_install_contract_allows_only_exact_flux_windows_candidate_plan(self) -> None:
        valid_plan = DependencyPlan(
            extension_id="flux-schnell",
            dependency_family="flux-schnell",
            platform_system="windows",
            platform_machine="amd64",
            python_tag="cp312",
            cuda_variant="cu128",
            shared_steps=(DependencyInstallStep(name="install_shared_torch", packages=("torch==2.7.0",)),),
            family_steps=(DependencyInstallStep(name="install_family_dependencies", packages=("diffusers==0.35.1",)),),
            plan_state=dependencies.PLAN_STATE_CANDIDATE_INSTALL,
            platform_key="windows-amd64",
            platform_supported=False,
        )

        self.assertTrue(install_contract._candidate_install_allowed(valid_plan))

        rejected_cases = (
            ("wrong extension", {"extension_id": "not-flux"}),
            ("wrong family", {"dependency_family": "sd15"}),
            ("wrong platform", {"platform_key": "linux-aarch64", "platform_system": "linux", "platform_machine": "aarch64"}),
            ("wrong cuda", {"cuda_variant": "cu124"}),
            ("wrong state", {"plan_state": dependencies.PLAN_STATE_VERIFIED, "platform_supported": True}),
            ("empty steps", {"shared_steps": (), "family_steps": ()}),
        )
        for label, overrides in rejected_cases:
            with self.subTest(label=label):
                candidate = DependencyPlan(**{**valid_plan.__dict__, **overrides})
                self.assertFalse(install_contract._candidate_install_allowed(candidate))

    def test_sd15_windows_evidence_rejects_missing_malformed_and_unreviewed_artifacts(self) -> None:
        cases = (
            (self._write_sd15_windows_evidence({"extension_id": "sd15"}), "missing required evidence field"),
            (
                self._write_sd15_windows_evidence(self._sd15_windows_evidence(reviewed=False)),
                "reviewed",
            ),
            (
                self._write_sd15_windows_evidence(self._sd15_windows_evidence(torch_cuda_available=False)),
                "torch_cuda_available",
            ),
            (
                self._write_sd15_windows_evidence(self._sd15_windows_evidence(model_repo="stable-diffusion-v1-5/stable-diffusion-v1-5")),
                "model_repo",
            ),
        )

        for evidence_path, expected_diagnostic in cases:
            with self.subTest(expected_diagnostic=expected_diagnostic):
                plan = dependencies.resolve_dependency_plan(
                    extension_id="sd15",
                    dependency_family="sd15",
                    readiness_imports=("torch",),
                    platform_info=WINDOWS_PLATFORM,
                    python_tag="cp312",
                    cuda_version="12.8",
                    evidence_path=evidence_path,
                )

                self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_CANDIDATE_INSTALL)
                self.assertFalse(plan.platform_supported)
                self.assertIn(expected_diagnostic, " ".join(plan.diagnostics))

    def test_sd15_windows_complete_reviewed_evidence_promotes_planner_only_verified_state(self) -> None:
        evidence_path = self._write_sd15_windows_evidence(self._sd15_windows_evidence())

        with patch("local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download") as download:
            plan = dependencies.resolve_dependency_plan(
                extension_id="sd15",
                dependency_family="sd15",
                readiness_imports=("torch", "diffusers"),
                platform_info=WINDOWS_PLATFORM,
                python_tag="cp312",
                cuda_version="12.8",
                evidence_path=evidence_path,
            )

        self.assertFalse(download.called)
        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_VERIFIED)
        self.assertTrue(plan.platform_supported)
        self.assertEqual(plan.platform_key, "windows-amd64")
        self.assertEqual(plan.cuda_variant, "cu128")
        self.assertEqual(plan.shared_steps[0].packages, ("torch==2.7.0", "torchvision==0.22.0"))
        self.assertEqual(plan.family_steps[0].packages[0], "diffusers==0.35.1")

    def test_sd15_hf_repo_identity_and_windows_cpu_only_status_are_not_changed(self) -> None:
        descriptor = bootstrap.get_extension_descriptor("sd15")
        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.hf_repo, "runwayml/stable-diffusion-v1-5")

        plan = dependencies.resolve_dependency_plan(
            extension_id="sd15",
            dependency_family="sd15",
            readiness_imports=("torch",),
            platform_info=WINDOWS_PLATFORM,
            python_tag="cp312",
            cuda_version=None,
        )

        diagnostics_text = " ".join(plan.diagnostics).lower()
        self.assertEqual(plan.plan_state, dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertNotIn("cpu-only", diagnostics_text)
        self.assertNotIn("cpu verified", diagnostics_text)

    def test_sd15_windows_evidence_example_is_a_non_verified_manual_checklist(self) -> None:
        example_path = REPO_ROOT / "docs" / "integration" / "sd15-windows-evidence.example.json"
        payload = json.loads(example_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["extension_id"], "sd15")
        self.assertEqual(payload["platform_key"], "windows-amd64")
        self.assertEqual(payload["python_abi"], "cp312")
        self.assertEqual(payload["cuda_variant"], "cu128")
        self.assertEqual(payload["torch_version"], "2.7.0")
        self.assertEqual(payload["torchvision_version"], "0.22.0")
        self.assertEqual(payload["model_repo"], "runwayml/stable-diffusion-v1-5")
        self.assertNotEqual(payload["status"], dependencies.PLAN_STATE_VERIFIED)
        self.assertFalse(payload["reviewed"])
        self.assertIn("manual_checklist", payload)

    def test_windows_dependency_plans_are_explicit_per_extension(self) -> None:
        observed_states: dict[str, str] = {}
        for extension_id in EXTENSION_IDS:
            descriptor = bootstrap.get_extension_descriptor(extension_id)
            self.assertIsNotNone(descriptor)

            plan = dependencies.resolve_dependency_plan(
                extension_id=extension_id,
                dependency_family=descriptor.dependency_family,
                readiness_imports=descriptor.readiness_imports,
                platform_info=WINDOWS_PLATFORM,
                python_tag="cp312",
                cuda_version="12.8",
            )

            observed_states[extension_id] = plan.plan_state
            self.assertEqual(plan.platform_key, "windows-amd64")
            self.assertEqual(plan.plan_state, WINDOWS_PLAN_STATES[extension_id])
            self.assertFalse(plan.platform_supported)
            if extension_id in {"sd15", "sdxl-base", "flux-schnell"}:
                self.assertEqual(plan.shared_steps[0].packages, ("torch==2.7.0", "torchvision==0.22.0"))
                if extension_id == "flux-schnell":
                    self.assertEqual(
                        plan.family_steps[0].packages,
                        ("diffusers==0.35.1", "transformers>=4.46,<5", "sentencepiece", "protobuf<6"),
                    )
                else:
                    self.assertEqual(plan.family_steps[0].packages[0], "diffusers==0.35.1")
            else:
                self.assertEqual(plan.shared_steps, ())
                self.assertEqual(plan.family_steps, ())
            self.assertIn("Windows", " ".join(plan.diagnostics))

        self.assertEqual(observed_states, WINDOWS_PLAN_STATES)

    def test_sd15_windows_candidate_setup_attempts_fake_install_and_persists_failure(self) -> None:
        runtime_root = self._make_runtime_root("sd15")
        self._make_windows_executable_python(runtime_root)
        install_calls: list[str] = []

        def fake_run_checked(*, command, step_name, cwd=None):
            if step_name == "create_venv":
                return None
            if step_name == "upgrade_pip":
                install_calls.append(step_name)
                return None
            raise AssertionError(f"unexpected command execution: {step_name}")

        def fake_install_step(*, venv_python, install_step, cwd):
            install_calls.append(install_step.name)
            raise install_contract.SetupExecutionError(
                step_name=install_step.name,
                detail="fake pip failure for SD15 Windows candidate install",
            )

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                )
            )
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=WINDOWS_PLATFORM))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.expected_venv_python",
                    side_effect=lambda ext_dir: Path(ext_dir) / "venv" / "Scripts" / "python.exe",
                )
            )
            stack.enter_context(patch("local_image_runtime.install_contract.python_tag_from_interpreter", return_value="cp312"))
            stack.enter_context(patch("local_image_runtime.install_contract._run_checked", side_effect=fake_run_checked))
            stack.enter_context(patch("local_image_runtime.install_contract._install_dependency_step", side_effect=fake_install_step))
            hf_download = stack.enter_context(
                patch(
                    "local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download",
                    side_effect=AssertionError("unit tests must not download Hugging Face weights"),
                )
            )
            smoke_imports = stack.enter_context(
                patch(
                    "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                    side_effect=AssertionError("failed setup must not run runtime import smoke tests"),
                )
            )

            result = install_contract.run_install_setup_contract(
                extension_id="sd15",
                stdin_text=self._windows_payload(runtime_root),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
        self.assertIn("upgrade_pip", install_calls)
        self.assertIn("install_shared_torch", install_calls)
        self.assertIn("fake pip failure for SD15 Windows candidate install", result.diagnostics)
        self.assertFalse(hf_download.called)
        self.assertFalse(smoke_imports.called)

        with patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False):
            snapshot = bootstrap.bootstrap_runtime(extension_id="sd15")
        record = bootstrap.get_extension_record(snapshot, "sd15")
        self.assertEqual(record["setup"]["status"], bootstrap.SETUP_STATUS_FAILED)
        self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_ERROR)
        self.assertEqual(record["setup_state"], dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertEqual(record["dependency_plan_state"], dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertFalse(record["platform_supported"])

    def test_sd15_windows_candidate_setup_continues_when_optional_acquisition_fails(self) -> None:
        runtime_root = self._make_runtime_root("sd15")
        self._make_windows_executable_python(runtime_root)
        install_calls: list[str] = []
        acquisition_calls: list[tuple[str, str, Path]] = []

        def fake_run_checked(*, command, step_name, cwd=None):
            if step_name == "create_venv":
                return None
            if step_name == "upgrade_pip":
                install_calls.append(step_name)
                return None
            raise AssertionError(f"unexpected command execution: {step_name}")

        def fake_install_step(*, venv_python, install_step, cwd):
            install_calls.append(install_step.name)
            return None

        def fail_optional_acquisition(extension_id, feature_id, models_dir, *, downloader=None):
            acquisition_calls.append((extension_id, feature_id, Path(models_dir)))
            raise RuntimeError("simulated unauthenticated Hugging Face rate limit for SD15 image encoder")

        windows_venv_python = runtime_root / "venv" / "Scripts" / "python.exe"
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                )
            )
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=WINDOWS_PLATFORM))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.expected_venv_python",
                    side_effect=lambda ext_dir: Path(ext_dir) / "venv" / "Scripts" / "python.exe",
                )
            )
            stack.enter_context(
                patch(
                    "local_image_runtime.bootstrap.expected_venv_python",
                    side_effect=lambda ext_dir: Path(ext_dir) / "venv" / "Scripts" / "python.exe",
                )
            )
            stack.enter_context(patch("local_image_runtime.install_contract.python_tag_from_interpreter", return_value="cp312"))
            stack.enter_context(patch("local_image_runtime.install_contract._run_checked", side_effect=fake_run_checked))
            stack.enter_context(patch("local_image_runtime.install_contract._install_dependency_step", side_effect=fake_install_step))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.acquire_optional_feature_weights",
                    side_effect=fail_optional_acquisition,
                )
            )
            stack.enter_context(
                patch(
                    "local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download",
                    side_effect=AssertionError("unit tests must not download Hugging Face weights"),
                )
            )
            smoke_imports = stack.enter_context(
                patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports"))
            )

            result = install_contract.run_install_setup_contract(
                extension_id="sd15",
                stdin_text=self._windows_payload(runtime_root),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
        self.assertEqual(
            install_calls,
            ["upgrade_pip", "install_shared_torch", "install_shared_runtime", "install_family_dependencies"],
        )
        self.assertEqual(
            acquisition_calls,
            [("sd15", "sd15_ip_adapter_style", runtime_root / ".local-image-runtime" / "models")],
        )
        warning_steps = [step for step in result.steps if step.name == "acquire_optional_feature_sd15_ip_adapter_style"]
        self.assertEqual(len(warning_steps), 1)
        self.assertEqual(warning_steps[0].status, "warning")
        warning_detail = warning_steps[0].detail or ""
        self.assertIn("optional feature", warning_detail)
        self.assertIn("candidate", warning_detail)
        self.assertIn("rate limit", warning_detail)
        smoke_imports.assert_any_call(windows_venv_python, descriptors.get_extension_descriptor("sd15").readiness_imports)

        with patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False), patch(
            "local_image_runtime.bootstrap.expected_venv_python",
            side_effect=lambda ext_dir: Path(ext_dir) / "venv" / "Scripts" / "python.exe",
        ), patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports")):
            snapshot = bootstrap.bootstrap_runtime(extension_id="sd15")
        record = bootstrap.get_extension_record(snapshot, "sd15")
        self.assertEqual(record["setup"]["status"], bootstrap.SETUP_STATUS_READY)
        self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_INSTALLED)
        self.assertEqual(record["setup_state"], dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertEqual(record["dependency_plan_state"], dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertFalse(record["platform_supported"])
        optional = record["weights"]["optional_features"]["sd15_ip_adapter_style"]
        self.assertEqual(optional["status"], "missing")
        self.assertFalse(optional["ready"])

    def test_sd15_linux_verified_setup_fails_when_optional_acquisition_fails(self) -> None:
        runtime_root = self._make_runtime_root("sd15")

        def fail_optional_acquisition(extension_id, feature_id, models_dir, *, downloader=None):
            raise RuntimeError("simulated required Linux ARM64 optional acquisition failure")

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                )
            )
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM))
            stack.enter_context(patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=self._fake_plan("sd15")))
            stack.enter_context(patch("local_image_runtime.install_contract._run_checked", side_effect=self._run_checked_side_effect))
            stack.enter_context(patch("local_image_runtime.install_contract._install_dependency_step", return_value=None))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.acquire_optional_feature_weights",
                    side_effect=fail_optional_acquisition,
                )
            )
            stack.enter_context(patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports")))

            result = install_contract.run_install_setup_contract(
                extension_id="sd15",
                stdin_text=self._payload(runtime_root),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
        failed_steps = {step.name: step for step in result.steps}
        self.assertEqual(failed_steps["acquire_optional_feature_sd15_ip_adapter_style"].status, "failed")
        self.assertIn("simulated required Linux ARM64 optional acquisition failure", result.diagnostics)

    def test_sdxl_windows_candidate_setup_attempts_fake_install_and_persists_failure(self) -> None:
        runtime_root = self._make_runtime_root("sdxl-base")
        self._make_windows_executable_python(runtime_root)
        install_calls: list[str] = []

        def fake_run_checked(*, command, step_name, cwd=None):
            if step_name == "create_venv":
                return None
            if step_name == "upgrade_pip":
                install_calls.append(step_name)
                return None
            raise AssertionError(f"unexpected command execution: {step_name}")

        def fake_install_step(*, venv_python, install_step, cwd):
            install_calls.append(install_step.name)
            raise install_contract.SetupExecutionError(
                step_name=install_step.name,
                detail="fake pip failure for SDXL Windows candidate install",
            )

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                )
            )
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=WINDOWS_PLATFORM))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.expected_venv_python",
                    side_effect=lambda ext_dir: Path(ext_dir) / "venv" / "Scripts" / "python.exe",
                )
            )
            stack.enter_context(patch("local_image_runtime.install_contract.python_tag_from_interpreter", return_value="cp312"))
            stack.enter_context(patch("local_image_runtime.install_contract._run_checked", side_effect=fake_run_checked))
            stack.enter_context(patch("local_image_runtime.install_contract._install_dependency_step", side_effect=fake_install_step))
            hf_download = stack.enter_context(
                patch(
                    "local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download",
                    side_effect=AssertionError("unit tests must not download Hugging Face weights"),
                )
            )
            smoke_imports = stack.enter_context(
                patch(
                    "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                    side_effect=AssertionError("failed setup must not run runtime import smoke tests"),
                )
            )

            result = install_contract.run_install_setup_contract(
                extension_id="sdxl-base",
                stdin_text=self._windows_payload(runtime_root),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
        self.assertIn("upgrade_pip", install_calls)
        self.assertIn("install_shared_torch", install_calls)
        self.assertIn("fake pip failure for SDXL Windows candidate install", result.diagnostics)
        diagnostics_text = " ".join(result.diagnostics + tuple(step.detail or "" for step in result.steps))
        self.assertIn("candidate", diagnostics_text)
        self.assertIn("not verified compatibility", diagnostics_text)
        self.assertFalse(hf_download.called)
        self.assertFalse(smoke_imports.called)

        with patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False):
            snapshot = bootstrap.bootstrap_runtime(extension_id="sdxl-base")
        record = bootstrap.get_extension_record(snapshot, "sdxl-base")
        self.assertEqual(record["setup"]["status"], bootstrap.SETUP_STATUS_FAILED)
        self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_ERROR)
        self.assertEqual(record["setup_state"], dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertEqual(record["dependency_plan_state"], dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertFalse(record["platform_supported"])

    def test_flux_windows_candidate_setup_attempts_fake_install_and_persists_failure(self) -> None:
        runtime_root = self._make_runtime_root("flux-schnell")
        self._make_windows_executable_python(runtime_root)
        install_calls: list[str] = []

        def fake_run_checked(*, command, step_name, cwd=None):
            if step_name == "create_venv":
                return None
            if step_name == "upgrade_pip":
                install_calls.append(step_name)
                return None
            raise AssertionError(f"unexpected command execution: {step_name}")

        def fake_install_step(*, venv_python, install_step, cwd):
            install_calls.append(install_step.name)
            raise install_contract.SetupExecutionError(
                step_name=install_step.name,
                detail="fake pip failure for Flux Windows candidate install",
            )

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                )
            )
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=WINDOWS_PLATFORM))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.expected_venv_python",
                    side_effect=lambda ext_dir: Path(ext_dir) / "venv" / "Scripts" / "python.exe",
                )
            )
            stack.enter_context(patch("local_image_runtime.install_contract.python_tag_from_interpreter", return_value="cp312"))
            stack.enter_context(patch("local_image_runtime.install_contract._run_checked", side_effect=fake_run_checked))
            stack.enter_context(patch("local_image_runtime.install_contract._install_dependency_step", side_effect=fake_install_step))
            hf_download = stack.enter_context(
                patch(
                    "local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download",
                    side_effect=AssertionError("unit tests must not download Hugging Face weights"),
                )
            )
            smoke_imports = stack.enter_context(
                patch(
                    "local_image_runtime.bootstrap._smoke_test_runtime_imports",
                    side_effect=AssertionError("failed setup must not run runtime import smoke tests"),
                )
            )

            result = install_contract.run_install_setup_contract(
                extension_id="flux-schnell",
                stdin_text=self._windows_payload(runtime_root),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
        self.assertIn("upgrade_pip", install_calls)
        self.assertIn("install_shared_torch", install_calls)
        self.assertIn("fake pip failure for Flux Windows candidate install", result.diagnostics)
        diagnostics_text = " ".join(result.diagnostics + tuple(step.detail or "" for step in result.steps))
        self.assertIn("candidate", diagnostics_text)
        self.assertIn("not verified compatibility", diagnostics_text)
        self.assertFalse(hf_download.called)
        self.assertFalse(smoke_imports.called)

        with patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False):
            snapshot = bootstrap.bootstrap_runtime(extension_id="flux-schnell")
        record = bootstrap.get_extension_record(snapshot, "flux-schnell")
        self.assertEqual(record["setup"]["status"], bootstrap.SETUP_STATUS_FAILED)
        self.assertEqual(record["status"], bootstrap.EXTENSION_STATUS_ERROR)
        self.assertEqual(record["setup_state"], dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertEqual(record["dependency_plan_state"], dependencies.PLAN_STATE_CANDIDATE_INSTALL)
        self.assertFalse(record["platform_supported"])

    def test_windows_setup_fails_safely_without_dependency_install_attempt_for_blocked_targets(self) -> None:
        for extension_id in EXTENSION_IDS:
            if extension_id in {"sd15", "sdxl-base", "flux-schnell"}:
                continue
            with self.subTest(extension_id=extension_id):
                runtime_root = self._make_runtime_root(extension_id)
                self._make_windows_executable_python(runtime_root)

                with ExitStack() as stack:
                    stack.enter_context(
                        patch.dict(
                            os.environ,
                            {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                            clear=False,
                        )
                    )
                    stack.enter_context(
                        patch("local_image_runtime.install_contract.detect_platform", return_value=WINDOWS_PLATFORM)
                    )
                    stack.enter_context(
                        patch("local_image_runtime.install_contract.python_tag_from_interpreter", return_value="cp311")
                    )
                    run_checked = stack.enter_context(
                        patch("local_image_runtime.install_contract._run_checked")
                    )
                    install_step = stack.enter_context(
                        patch("local_image_runtime.install_contract._install_dependency_step")
                    )

                    result = install_contract.run_install_setup_contract(
                        extension_id=extension_id,
                        stdin_text=self._payload(runtime_root),
                    )

                self.assertEqual(result.status, bootstrap.SETUP_STATUS_FAILED)
                self.assertFalse(run_checked.called)
                self.assertFalse(install_step.called)
                diagnostics_text = " ".join(result.diagnostics)
                self.assertIn(WINDOWS_PLAN_STATES[extension_id], diagnostics_text)
                self.assertIn("Windows", diagnostics_text)

                with patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                ):
                    snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)
                record = bootstrap.get_extension_record(snapshot, extension_id)
                self.assertEqual(record["setup_state"], WINDOWS_PLAN_STATES[extension_id])
                self.assertEqual(record["dependency_plan_state"], WINDOWS_PLAN_STATES[extension_id])
                self.assertEqual(record["platform_key"], "windows-amd64")
                self.assertFalse(record["platform_supported"])

    def test_windows_readiness_fields_are_additive_and_weights_remain_offline(self) -> None:
        for extension_id in EXTENSION_IDS:
            with self.subTest(extension_id=extension_id):
                runtime_root = self._make_runtime_root(extension_id)
                with patch.dict(
                    os.environ,
                    {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)},
                    clear=False,
                ), patch("local_image_runtime.bootstrap.platform.system", return_value="Windows"), patch(
                    "local_image_runtime.bootstrap.platform.machine", return_value="AMD64"
                ), patch(
                    "local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download",
                    side_effect=AssertionError("readiness must not download Hugging Face weights"),
                ):
                    snapshot = bootstrap.bootstrap_runtime(extension_id=extension_id)

                record = bootstrap.get_extension_record(snapshot, extension_id)
                self.assertEqual(record["platform_key"], "windows-amd64")
                self.assertEqual(record["dependency_plan_state"], WINDOWS_PLAN_STATES[extension_id])
                self.assertEqual(record["setup_state"], WINDOWS_PLAN_STATES[extension_id])
                self.assertIn(record["model_weight_state"], {"missing", "unknown"})
                self.assertIsInstance(record["diagnostics"], list)
                self.assertIn("Windows", " ".join(record["diagnostics"]))

    def test_extension_setup_entrypoints_remain_python_only_without_shell_commands(self) -> None:
        for extension_id in EXTENSION_IDS:
            setup_text = (REPO_ROOT / "extensions" / extension_id / "setup.py").read_text(encoding="utf-8")
            with self.subTest(extension_id=extension_id):
                self.assertNotIn("shell=True", setup_text)
                self.assertNotIn("powershell", setup_text.lower())
                self.assertNotIn("subprocess.run", setup_text)
                self.assertIn("run_extension_setup_cli", setup_text)

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
                        "guidance_scale": 0.0 if expected_family == "flux" else 7.5,
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
                if expected_family == "flux":
                    self.assertNotIn("negative_prompt", invocations[0]["kwargs"])
                else:
                    self.assertEqual(invocations[0]["kwargs"]["negative_prompt"], payload["params"]["negative_prompt"])
                self.assertTrue(Path(done_event["result"]["output_path"]).exists())
                self.assertTrue(str(done_event["result"]["output_path"]).startswith(str(outputs_dir)))
                self.assertEqual(
                    done_event["result"]["metadata"],
                    {
                        "family": expected_family,
                        "node_id": "text-to-image",
                        "seed": 42,
                        "negative_prompt_used": expected_family != "flux",
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

    def test_run_generator_main_emits_bootstrap_progress_from_shared_lifecycle_module(self) -> None:
        custom_bootstrap_steps = (
            (7, "bootstrap-read"),
            (17, "bootstrap-ready"),
        )
        payload = {
            "nodeId": "text-to-image",
            "input": {"text": "hello"},
            "params": {"prompt": "hello", "steps": 4},
        }
        stdout = StringIO()

        with patch(
            "local_image_runtime.runtime_adapter.lifecycle.bootstrap_steps",
            return_value=custom_bootstrap_steps,
        ), patch(
            "local_image_runtime.runtime_adapter.prepare_execution",
            return_value=SimpleNamespace(
                runtime=SimpleNamespace(paths=SimpleNamespace(runtime_dir="/tmp/runtime")),
                request=object(),
            ),
        ), patch(
            "local_image_runtime.runtime_adapter.execute",
            side_effect=lambda *args, **kwargs: (
                kwargs["emit_progress"](35, "validating-request"),
                {"output_path": "/tmp/generated.png", "metadata": {}},
            )[1],
        ):
            exit_code = runtime_adapter.run_generator_main(
                extension_id="sd15",
                stdin=StringIO(json.dumps(payload) + "\n"),
                stdout=stdout,
            )

        self.assertEqual(exit_code, 0)
        labels = [event["label"] for event in self._parse_ndjson_events(stdout.getvalue()) if event["type"] == "progress"]
        self.assertEqual(labels[:2], [label for _, label in custom_bootstrap_steps])
        self.assertEqual(labels[2:], ["validating-request"])

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
                        "guidance_scale": 0.0 if expected_family == "flux" else 7.5,
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
                if expected_family == "flux":
                    self.assertNotIn("negative_prompt", invocations[0]["kwargs"])
                else:
                    self.assertEqual(invocations[0]["kwargs"]["negative_prompt"], payload["params"]["negative_prompt"])
                self.assertTrue(Path(result_payload["result"]["output_path"]).exists())
                self.assertTrue(str(result_payload["result"]["output_path"]).startswith(str(outputs_dir)))
                self.assertEqual(
                    result_payload["result"]["metadata"],
                    {
                        "family": expected_family,
                        "node_id": "text-to-image",
                        "seed": 42,
                        "negative_prompt_used": expected_family != "flux",
                        "source_image_used": False,
                    },
                )

    def test_run_payload_keeps_bootstrap_labels_outside_canonical_generation_lifecycle(self) -> None:
        canonical_progress = [
            {"percent": 35, "label": "validating-request"},
            {"percent": 55, "label": "checking-extension"},
            {"percent": 75, "label": "backend-dispatch"},
            {"percent": 80, "label": "loading-pipeline"},
            {"percent": 90, "label": "running-inference"},
            {"percent": 95, "label": "saving-output"},
        ]
        cases = (
            {
                "nodeId": "text-to-image",
                "input": {"text": "text prompt"},
                "params": {"prompt": "text prompt", "steps": 4},
            },
            {
                "nodeId": "image-to-image",
                "input": {"filePath": "/tmp/source.png"},
                "params": {"prompt": "variation", "strength": 0.45, "steps": 4},
            },
        )

        for payload in cases:
            with self.subTest(node_id=payload["nodeId"]):
                with patch(
                    "local_image_runtime.runtime_adapter.prepare_execution",
                    return_value=SimpleNamespace(request=object(), runtime=object()),
                ), patch(
                    "local_image_runtime.runtime_adapter.execute",
                    side_effect=lambda *args, **kwargs: (
                        [kwargs["emit_progress"](event["percent"], event["label"]) for event in canonical_progress],
                        {"output_path": "/tmp/generated.png"},
                    )[1],
                ):
                    result = runtime_adapter.run_payload(payload, extension_id="sd15")

                self.assertEqual(result["progress"], canonical_progress)
                self.assertNotIn("payload-received", [event["label"] for event in result["progress"]])
                self.assertNotIn("runtime-ready", [event["label"] for event in result["progress"]])

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
                    "guidance_scale": 0.0 if expected_family == "flux" else 7.5,
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
                if expected_family == "flux":
                    self.assertNotIn("negative_prompt", invocations[0]["kwargs"])
                else:
                    self.assertEqual(invocations[0]["kwargs"]["negative_prompt"], params["negative_prompt"])
                self.assertFalse((outputs_dir / ".modly-inputs").exists())
                self.assertTrue(actual_path.exists())
                self.assertTrue(str(actual_path).startswith(str(outputs_dir)))
                self.assertIn((75, "backend-dispatch"), progress_events)

    def test_lifecycle_module_defines_canonical_generation_and_bootstrap_sequences(self) -> None:
        from local_image_runtime import lifecycle

        self.assertEqual(
            lifecycle.canonical_generation_steps(),
            (
                (35, "validating-request"),
                (55, "checking-extension"),
                (75, "backend-dispatch"),
                (80, "loading-pipeline"),
                (90, "running-inference"),
                (95, "saving-output"),
            ),
        )
        self.assertEqual(
            lifecycle.bootstrap_steps(),
            (
                (5, "payload-received"),
                (20, "runtime-ready"),
            ),
        )

    def test_lifecycle_module_splits_host_and_child_steps_without_bootstrap_overlap(self) -> None:
        from local_image_runtime import lifecycle

        canonical_labels = {label for _, label in lifecycle.canonical_generation_steps()}
        bootstrap_labels = {label for _, label in lifecycle.bootstrap_steps()}

        self.assertEqual(
            lifecycle.host_generation_steps(),
            (
                (35, "validating-request"),
                (55, "checking-extension"),
                (75, "backend-dispatch"),
            ),
        )
        self.assertEqual(
            lifecycle.child_generation_steps(),
            (
                (80, "loading-pipeline"),
                (90, "running-inference"),
                (95, "saving-output"),
            ),
        )
        self.assertFalse(canonical_labels & bootstrap_labels)

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

    def test_build_generate_request_for_sdxl_image_to_image_propagates_style_reference_to_conditioning(self) -> None:
        for input_key in ("Style reference", "style_reference"):
            with self.subTest(input_key=input_key):
                generator_class = self._load_generator_class("sdxl-base")
                model_dir = self._make_model_dir("sdxl-base", "image-to-image")
                outputs_dir = Path(tempfile.mkdtemp(prefix="outputs-sdxl-style-generator-"))
                style_reference = outputs_dir / "style-reference.png"
                style_reference.write_bytes(f"style-reference-{input_key}".encode("utf-8"))
                generator = generator_class(model_dir, outputs_dir)

                request = generator._build_generate_request(
                    b"primary-image-bytes",
                    {
                        "prompt": "variation",
                        "strength": 0.35,
                        "reference_strength": 0.4,
                        "input": {input_key: str(style_reference)},
                    },
                )
                payload_details = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sdxl-base")
                self.assertIsNotNone(payload_details.conditioning)
                self.assertEqual(
                    payload_details.conditioning.to_backend_payload(),
                    {"references": [{"role": "style", "filePath": str(style_reference.resolve())}]},
                )
                self.assertEqual(payload_details.numeric_params["reference_strength"], 0.4)
                materialized_input = Path(request.input["filePath"])
                self.assertEqual(materialized_input.read_bytes(), b"primary-image-bytes")

    def test_build_generate_request_for_sdxl_image_to_image_maps_left_image_path_to_style_reference(self) -> None:
        generator_class = self._load_generator_class("sdxl-base")
        model_dir = self._make_model_dir("sdxl-base", "image-to-image")
        outputs_dir = Path(tempfile.mkdtemp(prefix="outputs-sdxl-left-generator-"))
        style_reference = outputs_dir / "left-style-reference.png"
        style_reference.write_bytes(b"left-style-reference")
        generator = generator_class(model_dir, outputs_dir)

        request = generator._build_generate_request(
            b"primary-front-image-bytes",
            {
                "prompt": "variation",
                "strength": 0.35,
                "reference_strength": 0.45,
                "left_image_path": str(style_reference),
            },
        )
        payload_details = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sdxl-base")

        self.assertIsNotNone(payload_details.conditioning)
        self.assertEqual(
            payload_details.conditioning.to_backend_payload(),
            {"references": [{"role": "style", "filePath": str(style_reference.resolve())}]},
        )
        self.assertEqual(payload_details.numeric_params["reference_strength"], 0.45)
        self.assertNotIn("left_image_path", request.params)
        materialized_input = Path(request.input["filePath"])
        self.assertEqual(materialized_input.read_bytes(), b"primary-front-image-bytes")

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

    def test_image_to_image_validation_rejects_zero_effective_denoising_steps(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="img2img-zero-steps-"))
        source_path = workspace_dir / "primary-source.png"
        source_path.write_bytes(b"primary")
        request = pipeline.ExecutionRequest(
            node_id="image-to-image",
            input={"filePath": str(source_path)},
            params={"prompt": "variation", "strength": 0.35, "steps": 1},
            workspace_dir=str(workspace_dir),
        )

        with self.assertRaisesRegex(
            pipeline.RequestValidationError,
            "steps.*strength.*at least one.*denoising.*step",
        ):
            pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sdxl-base")

    def test_image_to_image_validation_accepts_low_combo_with_effective_denoising_step(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="img2img-one-step-"))
        source_path = workspace_dir / "primary-source.png"
        source_path.write_bytes(b"primary")
        request = pipeline.ExecutionRequest(
            node_id="image-to-image",
            input={"filePath": str(source_path)},
            params={"prompt": "variation", "strength": 0.8, "steps": 2},
            workspace_dir=str(workspace_dir),
        )

        validated = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sdxl-base")

        self.assertEqual(validated.numeric_params["steps"], 2)
        self.assertEqual(validated.numeric_params["strength"], 0.8)
        self.assertEqual(validated.source_image_path, str(source_path.resolve()))

    def test_inference_runner_rejects_zero_effective_denoising_steps_before_diffusers(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FailingIfDiffusersInvoked:
            def from_pretrained(self, model_dir: str, **kwargs: object) -> object:
                raise AssertionError("Diffusers must not be invoked for invalid zero-step img2img params")

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-img2img-zero-steps-"))
        job = {
            "extension_id": "sdxl-base",
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(workspace_dir / "model"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(workspace_dir / "result.png"),
            "prompt": "variation",
            "source_image_path": str(workspace_dir / "source.png"),
            "params": {"steps": 1, "strength": 0.35},
        }

        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("sdxl", "image-to-image"): FailingIfDiffusersInvoked()},
            clear=True,
        ):
            with self.assertRaisesRegex(
                inference_runner.InferenceRunnerError,
                "steps.*strength.*at least one.*denoising.*step",
            ):
                inference_runner.run_child_job(job)

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

    def test_baseline_text_to_image_payload_and_runner_kwargs_omit_conditioning(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="baseline-text-conditioning-"))
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-baseline-text-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(extension_root)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "baseline text"},
            params={"prompt": "baseline text", "steps": 4, "guidance_scale": 7.5},
            workspace_dir=str(workspace_dir),
        )

        payload_details = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sd15")
        self.assertIsNone(payload_details.conditioning)
        job = pipeline._build_backend_job(
            request=request,
            extension_id="sd15",
            extension_record={"venv_python": str(venv_python), "model_dir": "/runtime/local/sd15"},
            payload_details=payload_details,
            effective_workspace_dir=str(workspace_dir),
        )

        self.assertNotIn("conditioning", job.payload)
        runner_kwargs = inference_runner._build_pipeline_kwargs(job.payload, execution_device="cpu")
        self.assertEqual(runner_kwargs["prompt"], "baseline text")
        self.assertEqual(runner_kwargs["num_inference_steps"], 4)
        self.assertEqual(runner_kwargs["guidance_scale"], 7.5)
        self.assertNotIn("image", runner_kwargs)
        self.assertNotIn("strength", runner_kwargs)
        self.assertNotIn("ip_adapter_image", runner_kwargs)
        self.assertNotIn("control_image", runner_kwargs)

    def test_baseline_image_to_image_keeps_primary_source_and_omits_conditioning(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="baseline-img2img-conditioning-"))
        source_path = workspace_dir / "primary-source.png"
        source_path.write_bytes(b"primary")
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-baseline-img2img-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(extension_root)
        source_image_token = object()
        request = pipeline.ExecutionRequest(
            node_id="image-to-image",
            input={"filePath": str(source_path)},
            params={"prompt": "variation", "strength": 0.55, "steps": 4},
            workspace_dir=str(workspace_dir),
        )

        payload_details = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sdxl-base")
        self.assertEqual(payload_details.source_image_path, str(source_path.resolve()))
        self.assertIsNone(payload_details.conditioning)
        job = pipeline._build_backend_job(
            request=request,
            extension_id="sdxl-base",
            extension_record={"venv_python": str(venv_python), "model_dir": "/runtime/local/sdxl"},
            payload_details=payload_details,
            effective_workspace_dir=str(workspace_dir),
        )

        self.assertEqual(job.payload["source_image_path"], str(source_path.resolve()))
        self.assertNotIn("conditioning", job.payload)
        with patch("local_image_runtime.inference_runner._open_source_image", return_value=source_image_token) as open_source_image:
            runner_kwargs = inference_runner._build_pipeline_kwargs(job.payload, execution_device="cpu")

        open_source_image.assert_called_once_with(str(source_path.resolve()))
        self.assertIs(runner_kwargs["image"], source_image_token)
        self.assertEqual(runner_kwargs["strength"], 0.55)
        self.assertNotIn("ip_adapter_image", runner_kwargs)
        self.assertNotIn("control_image", runner_kwargs)

    def test_empty_conditioning_contract_is_inert_for_image_to_image(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="empty-conditioning-"))
        source_path = workspace_dir / "primary-source.png"
        source_path.write_bytes(b"primary")
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-empty-conditioning-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(extension_root)
        request = pipeline.ExecutionRequest(
            node_id="image-to-image",
            input={"filePath": str(source_path), "conditioning": {"references": [], "controls": []}},
            params={"prompt": "variation", "strength": 0.55, "steps": 4},
            workspace_dir=str(workspace_dir),
        )

        payload_details = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sdxl-base")
        self.assertIsNone(payload_details.conditioning)
        job = pipeline._build_backend_job(
            request=request,
            extension_id="sdxl-base",
            extension_record={"venv_python": str(venv_python), "model_dir": "/runtime/local/sdxl"},
            payload_details=payload_details,
            effective_workspace_dir=str(workspace_dir),
        )

        self.assertEqual(job.payload["source_image_path"], str(source_path.resolve()))
        self.assertNotIn("conditioning", job.payload)

    def test_base_image_to_image_rejects_handcrafted_control_conditioning_payloads(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="base-img2img-control-guard-"))
        source_path = workspace_dir / "primary-source.png"
        control_path = workspace_dir / "control-source.png"
        source_path.write_bytes(b"primary")
        control_path.write_bytes(b"control")

        cases = (
            (
                "sdxl-base",
                {"filePath": str(source_path), "conditioning": {"controls": [{"role": "structure", "control_type": "canny", "filePath": str(control_path)}]}},
                {"prompt": "variation", "strength": 0.55, "steps": 4},
            ),
            (
                "sd15",
                {"filePath": str(source_path)},
                {"prompt": "variation", "strength": 0.55, "steps": 4, "conditioning": {"controls": [{"role": "structure", "control_type": "depth", "filePath": str(control_path)}]}},
            ),
        )
        for extension_id, input_payload, params in cases:
            with self.subTest(extension_id=extension_id):
                request = pipeline.ExecutionRequest(
                    node_id="image-to-image",
                    input=input_payload,
                    params=params,
                    workspace_dir=str(workspace_dir),
                )

                with self.assertRaisesRegex(
                    pipeline.RequestValidationError,
                    "ControlNet.*control conditioning.*not supported.*base image-to-image.*explicit ControlNet nodes",
                ):
                    pipeline._validate_node_payload(request, legacy_model_id=None, extension_id=extension_id)

    def test_conditioning_contract_rejects_anonymous_ordered_image_inputs(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="anonymous-conditioning-"))
        source_path = workspace_dir / "primary-source.png"
        source_path.write_bytes(b"primary")

        for anonymous_key in ("Image 2", "Image 3", "image_4"):
            with self.subTest(anonymous_key=anonymous_key):
                request = pipeline.ExecutionRequest(
                    node_id="image-to-image",
                    input={"filePath": str(source_path), anonymous_key: str(source_path)},
                    params={"prompt": "variation", "strength": 0.55, "steps": 4},
                    workspace_dir=str(workspace_dir),
                )

                with self.assertRaisesRegex(pipeline.RequestValidationError, "Anonymous image input"):
                    pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sdxl-base")

    def test_sdxl_style_reference_payload_defaults_strength_and_serializes_conditioning(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="sdxl-style-reference-"))
        source_path = workspace_dir / "primary-source.png"
        reference_path = workspace_dir / "style-reference.png"
        source_path.write_bytes(b"primary")
        reference_path.write_bytes(b"style")
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-sdxl-style-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(extension_root)
        request = pipeline.ExecutionRequest(
            node_id="image-to-image",
            input={"filePath": str(source_path), "Style reference": str(reference_path)},
            params={"prompt": "variation", "strength": 0.55, "steps": 4},
            workspace_dir=str(workspace_dir),
        )

        payload_details = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sdxl-base")
        self.assertIsNotNone(payload_details.conditioning)
        self.assertEqual(payload_details.numeric_params["reference_strength"], 0.6)
        self.assertEqual(
            payload_details.conditioning.to_backend_payload(),
            {"references": [{"role": "style", "filePath": str(reference_path.resolve())}]},
        )
        job = pipeline._build_backend_job(
            request=request,
            extension_id="sdxl-base",
            extension_record={"venv_python": str(venv_python), "model_dir": "/runtime/local/sdxl"},
            payload_details=payload_details,
            effective_workspace_dir=str(workspace_dir),
        )

        self.assertEqual(job.payload["source_image_path"], str(source_path.resolve()))
        self.assertEqual(job.payload["params"]["reference_strength"], 0.6)
        self.assertEqual(job.payload["conditioning"], {"references": [{"role": "style", "filePath": str(reference_path.resolve())}]})

    def test_sdxl_style_reference_validates_strength_bounds_family_and_count(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="sdxl-style-validation-"))
        source_path = workspace_dir / "primary-source.png"
        reference_path = workspace_dir / "style-reference.png"
        second_reference_path = workspace_dir / "second-reference.png"
        source_path.write_bytes(b"primary")
        reference_path.write_bytes(b"style")
        second_reference_path.write_bytes(b"second")

        invalid_cases = (
            (
                "sdxl-base",
                {"filePath": str(source_path), "Style reference": str(reference_path)},
                {"prompt": "variation", "strength": 0.55, "reference_strength": 1.2},
                "reference_strength.*<= 1.0",
            ),
            (
                "sdxl-base",
                {"filePath": str(source_path), "Style reference": str(reference_path), "style_reference": str(second_reference_path)},
                {"prompt": "variation", "strength": 0.55},
                "one style reference",
            ),
            (
                "flux-schnell",
                {"filePath": str(source_path), "Style reference": str(reference_path)},
                {"prompt": "variation", "strength": 0.55},
                "Style reference.*only for SDXL Base or SD1.5",
            ),
        )
        for extension_id, input_payload, params, expected_message in invalid_cases:
            with self.subTest(extension_id=extension_id, expected_message=expected_message):
                request = pipeline.ExecutionRequest(
                    node_id="image-to-image",
                    input=input_payload,
                    params=params,
                    workspace_dir=str(workspace_dir),
                )
                with self.assertRaisesRegex(pipeline.RequestValidationError, expected_message):
                    pipeline._validate_node_payload(request, legacy_model_id=None, extension_id=extension_id)

    def test_sdxl_manifest_exposes_optional_style_reference_without_multi_init_language(self) -> None:
        manifest = self._extension_manifest_data("sdxl-base")
        nodes = {node["id"]: node for node in manifest["nodes"]}
        image_node = nodes["image-to-image"]
        params_schema = {schema["id"]: schema for schema in image_node["params_schema"]}

        self.assertEqual(image_node["style_reference"], {"label": "Style reference", "optional": True, "role": "style"})
        self.assertEqual(
            image_node["inputs"],
            [
                {"name": "front", "label": "Primary image", "type": "image", "required": True},
                {"name": "left", "label": "Style reference", "type": "image", "required": False},
            ],
        )
        self.assertEqual(
            params_schema["reference_strength"],
            {
                "id": "reference_strength",
                "label": "Reference Strength",
                "type": "float",
                "default": 0.6,
                "min": 0,
                "max": 1,
                "tooltip": "Controls optional SDXL IP-Adapter style reference guidance; the primary image remains the init image.",
            },
        )
        manifest_text = self._extension_manifest("sdxl-base").casefold()
        self.assertNotIn("image 2", manifest_text)
        self.assertNotIn("multi-init", manifest_text)
        self.assertNotIn("batch", manifest_text)
        self.assertNotIn("controlnet", manifest_text)

    def test_sd15_manifest_exposes_style_reference_after_authorized_promotion(self) -> None:
        manifest = self._extension_manifest_data("sd15")
        nodes = {node["id"]: node for node in manifest["nodes"]}
        text_node = nodes["text-to-image"]
        image_node = nodes["image-to-image"]
        params_schema = {schema["id"]: schema for schema in image_node["params_schema"]}
        manifest_text = self._extension_manifest("sd15").casefold()

        self.assertNotIn("style_reference", text_node)
        self.assertNotIn("inputs", text_node)
        self.assertEqual(image_node["style_reference"], {"label": "Style reference", "optional": True, "role": "style"})
        self.assertEqual(
            image_node["inputs"],
            [
                {"name": "front", "label": "Primary image", "type": "image", "required": True},
                {"name": "left", "label": "Style reference", "type": "image", "required": False},
            ],
        )
        self.assertEqual(
            params_schema["reference_strength"],
            {
                "id": "reference_strength",
                "label": "Reference Strength",
                "type": "float",
                "default": 0.6,
                "min": 0,
                "max": 1,
                "tooltip": "Controls optional SD1.5 IP-Adapter style reference guidance; the primary image remains the init image.",
            },
        )
        self.assertIn("style reference", manifest_text)
        self.assertNotIn("image 2", manifest_text)
        self.assertNotIn("image 3", manifest_text)
        self.assertNotIn("controlnet", manifest_text)

    def test_sd15_style_reference_request_is_accepted_after_authorized_promotion(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="sd15-style-gate-"))
        source_path = workspace_dir / "primary-source.png"
        reference_path = workspace_dir / "style-reference.png"
        source_path.write_bytes(b"primary")
        reference_path.write_bytes(b"style")
        sd15_request = pipeline.ExecutionRequest(
            node_id="image-to-image",
            input={"filePath": str(source_path), "Style reference": str(reference_path)},
            params={"prompt": "variation", "strength": 0.55, "steps": 4, "reference_strength": 0.4},
            workspace_dir=str(workspace_dir),
        )

        validated = pipeline._validate_node_payload(sd15_request, legacy_model_id=None, extension_id="sd15")
        self.assertIsNotNone(validated.conditioning)
        self.assertEqual(validated.numeric_params["reference_strength"], 0.4)
        self.assertEqual(
            validated.conditioning.to_backend_payload(),
            {"references": [{"role": "style", "filePath": str(reference_path.resolve())}]},
        )

    def test_sd15_optional_ip_adapter_readiness_reports_missing_or_ready_after_promotion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sd15-models-") as temp_dir:
            models_dir = Path(temp_dir)
            for node_id in ("text-to-image", "image-to-image"):
                base_check = models_dir / "sd15" / node_id / "model_index.json"
                base_check.parent.mkdir(parents=True, exist_ok=True)
                base_check.write_text("{}\n", encoding="utf-8")
            readiness = weights.evaluate_extension_weights("sd15", models_dir=models_dir)

            feature_root = models_dir / "sd15" / "optional" / "sd15_ip_adapter_style"
            for relative_path in (
                "models/ip-adapter_sd15.safetensors",
                "models/image_encoder/config.json",
                "models/image_encoder/model.safetensors",
            ):
                asset_file = feature_root / relative_path
                asset_file.parent.mkdir(parents=True, exist_ok=True)
                asset_file.write_bytes(b"asset")
            ready_readiness = weights.evaluate_extension_weights("sd15", models_dir=models_dir)

        self.assertEqual(readiness["status"], "ready")
        optional = readiness["optional_features"]["sd15_ip_adapter_style"]
        self.assertEqual(optional["status"], "missing")
        self.assertFalse(optional["ready"])
        self.assertEqual(
            optional["missing_files"],
            (
                "models/ip-adapter_sd15.safetensors",
                "models/image_encoder/config.json",
                "models/image_encoder/model.safetensors",
            ),
        )
        diagnostics_text = " ".join(optional["diagnostics"])
        self.assertIn("SD1.5", diagnostics_text)
        self.assertIn("Install/Repair", diagnostics_text)
        self.assertEqual(ready_readiness["optional_features"]["sd15_ip_adapter_style"]["status"], "ready")
        self.assertIn("Style reference", self._extension_manifest("sd15"))

    def test_sd15_setup_contract_acquires_only_scoped_ip_adapter_assets_for_install_repair(self) -> None:
        runtime_root = self._make_runtime_root("sd15")
        explicit_models_dir = Path(tempfile.mkdtemp(prefix="sd15-repair-models-"))
        acquired: list[dict[str, object]] = []

        def fake_acquire_optional_feature_weights(extension_id, feature_id, models_dir, *, downloader=None):
            acquired.append({"extension_id": extension_id, "feature_id": feature_id, "models_dir": Path(models_dir)})
            self.assertEqual(extension_id, "sd15")
            self.assertEqual(feature_id, "sd15_ip_adapter_style")
            target_dir = Path(models_dir) / extension_id / "optional" / feature_id
            required_files = (
                "models/ip-adapter_sd15.safetensors",
                "models/image_encoder/config.json",
                "models/image_encoder/model.safetensors",
            )
            for relative_path in required_files:
                target = target_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"asset")
            return {
                "status": "ready",
                "extension_id": extension_id,
                "feature_id": feature_id,
                "model_dir": str(target_dir),
                "check_path": str(target_dir / "models" / "ip-adapter_sd15.safetensors"),
                "required_files": required_files,
                "missing_files": (),
                "downloaded": True,
            }

        payload = json.loads(self._payload(runtime_root))
        payload["modelsDir"] = str(explicit_models_dir)

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False))
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM))
            stack.enter_context(patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=self._fake_plan("sd15")))
            stack.enter_context(patch("local_image_runtime.install_contract._run_checked", side_effect=self._run_checked_side_effect))
            stack.enter_context(patch("local_image_runtime.install_contract._install_dependency_step", return_value=None))
            stack.enter_context(patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports")))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.acquire_optional_feature_weights",
                    side_effect=fake_acquire_optional_feature_weights,
                )
            )

            result = install_contract.run_install_setup_contract(
                extension_id="sd15",
                stdin_text=json.dumps(payload),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
        self.assertEqual(
            acquired,
            [{"extension_id": "sd15", "feature_id": "sd15_ip_adapter_style", "models_dir": explicit_models_dir}],
        )
        persisted_state = json.loads(
            (runtime_root / ".local-image-runtime" / "state" / "models-state.json").read_text(encoding="utf-8")
        )
        sd15_optional = persisted_state["extensions"]["sd15"]["weights"]["optional_features"]["sd15_ip_adapter_style"]
        self.assertEqual(sd15_optional["status"], "ready")
        self.assertNotIn("sdxl_models", json.dumps(sd15_optional, sort_keys=True))

    def test_sd15_optional_feature_weight_acquisition_uses_discovered_pinned_bundle_not_sdxl_paths(self) -> None:
        class FakeDownloader:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def snapshot_download(
                self,
                *,
                repo_id: str,
                local_dir: Path,
                allow_patterns: tuple[str, ...] | None = None,
                revision: str | None = None,
            ) -> Path:
                self.calls.append(
                    {"repo_id": repo_id, "local_dir": local_dir, "allow_patterns": allow_patterns, "revision": revision}
                )
                for relative_path in allow_patterns or ():
                    target = local_dir / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"asset")
                return local_dir

        with tempfile.TemporaryDirectory(prefix="sd15-models-") as temp_dir:
            models_dir = Path(temp_dir)
            downloader = FakeDownloader()

            result = weights.acquire_optional_feature_weights(
                "sd15",
                "sd15_ip_adapter_style",
                models_dir,
                downloader=downloader,
            )

        expected_target = models_dir / "sd15" / "optional" / "sd15_ip_adapter_style"
        expected_files = (
            "models/ip-adapter_sd15.safetensors",
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
        )
        self.assertEqual(
            downloader.calls,
            [
                {
                    "repo_id": "h94/IP-Adapter",
                    "local_dir": expected_target,
                    "allow_patterns": expected_files,
                    "revision": "018e402774aeeddd60609b4ecdb7e298259dc729",
                }
            ],
        )
        self.assertEqual(result["required_files"], expected_files)
        self.assertEqual(result["revision"], "018e402774aeeddd60609b4ecdb7e298259dc729")
        self.assertNotIn("sdxl_models", json.dumps(result, sort_keys=True))

    def test_base_image_to_image_manifests_do_not_expose_controlnet_params_or_anonymous_images(self) -> None:
        forbidden_param_fragments = (
            "controlnet",
            "control_image",
            "control image",
            "control_type",
            "control strength",
            "control_strength",
            "canny",
            "depth",
            "normal",
            "pose",
        )

        for extension_id in ("sdxl-base", "sd15"):
            with self.subTest(extension_id=extension_id):
                manifest = self._extension_manifest_data(extension_id)
                nodes = {node["id"]: node for node in manifest["nodes"]}
                image_node = nodes["image-to-image"]
                params_schema = image_node["params_schema"]
                self.assertGreater(len(params_schema), 0)

                searchable_param_text = json.dumps(params_schema, sort_keys=True).casefold()
                for fragment in forbidden_param_fragments:
                    self.assertNotIn(fragment, searchable_param_text)

                input_slots = image_node.get("inputs", [])
                self.assertNotIn("Image 2", json.dumps(input_slots, sort_keys=True))
                self.assertNotIn("Image 3", json.dumps(input_slots, sort_keys=True))
                self.assertNotIn("Image 4", json.dumps(input_slots, sort_keys=True))

    def test_controlnet_manifest_exposure_is_explicit_separate_and_non_baseline_if_added(self) -> None:
        for extension_id in ("sdxl-base", "sd15"):
            with self.subTest(extension_id=extension_id):
                manifest = self._extension_manifest_data(extension_id)
                nodes = manifest["nodes"]
                base_image_node = next(node for node in nodes if node["id"] == "image-to-image")
                base_text = json.dumps(base_image_node, sort_keys=True).casefold()
                self.assertNotIn("controlnet", base_text)
                self.assertNotIn("control image", base_text)
                self.assertNotIn("control_image", base_text)

                control_nodes = [
                    node
                    for node in nodes
                    if "controlnet" in json.dumps(node, sort_keys=True).casefold()
                    or "structural control" in json.dumps(node, sort_keys=True).casefold()
                ]
                for control_node in control_nodes:
                    label_text = " ".join(
                        str(control_node.get(key, "")) for key in ("id", "name", "description")
                    ).casefold()
                    self.assertNotEqual(control_node["id"], "image-to-image")
                    self.assertRegex(label_text, r"controlnet|structural control|canny|depth")
                    self.assertNotIn("Image 2", json.dumps(control_node, sort_keys=True))
                    self.assertNotEqual(control_node.get("readiness"), "ready")

    def test_baseline_dependency_plans_exclude_controlnet_preprocessors_and_keep_them_optional(self) -> None:
        forbidden_packages = ("controlnet_aux", "opencv-python", "opencv-python-headless", "cv2")

        for extension_id in ("sdxl-base", "sd15"):
            with self.subTest(extension_id=extension_id):
                descriptor = descriptors.get_extension_descriptor(extension_id)
                self.assertIsNotNone(descriptor)
                assert descriptor is not None
                plan = dependencies.resolve_dependency_plan(
                    extension_id=extension_id,
                    dependency_family=descriptor.dependency_family,
                    readiness_imports=descriptor.readiness_imports,
                    platform_info=SUPPORTED_PLATFORM,
                    python_tag="cp311",
                    cuda_version="12.4",
                )
                baseline_packages = tuple(
                    package.casefold()
                    for step in (*plan.shared_steps, *plan.family_steps)
                    for package in step.packages
                )
                baseline_imports = tuple(module.casefold() for module in plan.readiness_imports)
                optional_groups = dependencies.get_optional_dependency_groups(extension_id)

                for forbidden in forbidden_packages:
                    self.assertFalse(any(forbidden in package for package in baseline_packages))
                    self.assertNotIn(forbidden, baseline_imports)
                self.assertIn(f"{extension_id}_controlnet_preprocessors", optional_groups)
                controlnet_group = optional_groups[f"{extension_id}_controlnet_preprocessors"]
                self.assertFalse(controlnet_group["baseline"])
                self.assertEqual(controlnet_group["state"], dependencies.PLAN_STATE_UNSUPPORTED)
                self.assertIn("controlnet_aux", controlnet_group["packages"])
                self.assertTrue(any(package.startswith("opencv-python") for package in controlnet_group["packages"]))

    def test_controlnet_readiness_metadata_is_planned_unsupported_and_separate_from_base_nodes(self) -> None:
        expected_feature_ids = {
            "sdxl-base": "sdxl_controlnet_canny",
            "sd15": "sd15_controlnet_canny",
        }

        with tempfile.TemporaryDirectory(prefix="controlnet-readiness-") as temp_dir:
            models_dir = Path(temp_dir)
            for extension_id in ("sdxl-base", "sd15"):
                for node_id in ("text-to-image", "image-to-image"):
                    base_check = models_dir / extension_id / node_id / "model_index.json"
                    base_check.parent.mkdir(parents=True, exist_ok=True)
                    base_check.write_text("{}\n", encoding="utf-8")

            for extension_id, feature_id in expected_feature_ids.items():
                with self.subTest(extension_id=extension_id):
                    specs = descriptors.get_optional_feature_specs(extension_id)
                    self.assertIn(feature_id, specs)
                    feature = specs[feature_id]
                    self.assertFalse(feature["supported"])
                    self.assertTrue(feature["explicit_node_required"])
                    self.assertNotEqual(feature["node_id"], "image-to-image")
                    self.assertIn("ControlNet", feature["label"])
                    self.assertIn("per-control", feature["node_strategy"])

                    readiness = weights.evaluate_extension_weights(extension_id, models_dir=models_dir)
                    self.assertEqual(readiness["status"], "ready")
                    controlnet_readiness = readiness["optional_features"][feature_id]
                    self.assertEqual(controlnet_readiness["status"], "unsupported")
                    self.assertFalse(controlnet_readiness["ready"])
                    diagnostics_text = " ".join(controlnet_readiness["diagnostics"])
                    self.assertIn("ControlNet", diagnostics_text)
                    self.assertIn("separate", diagnostics_text.casefold())
                    self.assertIn("smoke", diagnostics_text.casefold())

    def test_sdxl_ip_adapter_readiness_is_optional_and_separate_from_base_weights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sdxl-models-") as temp_dir:
            models_dir = Path(temp_dir)
            for node_id in ("text-to-image", "image-to-image"):
                base_check = models_dir / "sdxl-base" / node_id / "model_index.json"
                base_check.parent.mkdir(parents=True, exist_ok=True)
                base_check.write_text("{}\n", encoding="utf-8")
            readiness = weights.evaluate_extension_weights("sdxl-base", models_dir=models_dir)

        image_node = readiness["nodes"]["image-to-image"]
        optional = readiness["optional_features"]["sdxl_ip_adapter_style"]
        self.assertEqual(readiness["status"], "ready")
        self.assertEqual(image_node["status"], "ready")
        self.assertEqual(optional["status"], "missing")
        self.assertFalse(optional["ready"])
        self.assertIn("optional", " ".join(optional["diagnostics"]).casefold())
        self.assertIn("IP-Adapter", " ".join(optional["diagnostics"]))

    def test_sdxl_ip_adapter_descriptor_requires_adapter_and_image_encoder_assets(self) -> None:
        specs = descriptors.get_optional_feature_specs("sdxl-base")

        feature = specs["sdxl_ip_adapter_style"]

        self.assertEqual(feature["download_check"], "sdxl_models/ip-adapter_sdxl.bin")
        self.assertEqual(
            feature["required_files"],
            (
                "sdxl_models/ip-adapter_sdxl.bin",
                "sdxl_models/image_encoder/config.json",
                "sdxl_models/image_encoder/model.safetensors",
            ),
        )
        self.assertEqual(feature["allow_patterns"], feature["required_files"])
        self.assertNotIn("sdxl_models/image_encoder/pytorch_model.bin", feature["allow_patterns"])

    def test_sdxl_optional_feature_weight_acquisition_uses_descriptor_target(self) -> None:
        class FakeDownloader:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def snapshot_download(
                self,
                *,
                repo_id: str,
                local_dir: Path,
                allow_patterns: tuple[str, ...] | None = None,
            ) -> Path:
                self.calls.append({"repo_id": repo_id, "local_dir": local_dir, "allow_patterns": allow_patterns})
                for relative_path in allow_patterns or ():
                    target = local_dir / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"asset")
                return local_dir

        with tempfile.TemporaryDirectory(prefix="sdxl-models-") as temp_dir:
            models_dir = Path(temp_dir)
            downloader = FakeDownloader()

            result = weights.acquire_optional_feature_weights(
                "sdxl-base",
                "sdxl_ip_adapter_style",
                models_dir,
                downloader=downloader,
            )

        expected_target = models_dir / "sdxl-base" / "optional" / "sdxl_ip_adapter_style"
        self.assertEqual(
            downloader.calls,
            [
                {
                    "repo_id": "h94/IP-Adapter",
                    "local_dir": expected_target,
                    "allow_patterns": (
                        "sdxl_models/ip-adapter_sdxl.bin",
                        "sdxl_models/image_encoder/config.json",
                        "sdxl_models/image_encoder/model.safetensors",
                    ),
                }
            ],
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["feature_id"], "sdxl_ip_adapter_style")
        self.assertEqual(result["model_dir"], str(expected_target))
        self.assertEqual(result["check_path"], str(expected_target / "sdxl_models" / "ip-adapter_sdxl.bin"))
        self.assertEqual(
            result["required_files"],
            (
                "sdxl_models/ip-adapter_sdxl.bin",
                "sdxl_models/image_encoder/config.json",
                "sdxl_models/image_encoder/model.safetensors",
            ),
        )

    def test_sdxl_optional_feature_weight_acquisition_limits_snapshot_to_download_check(self) -> None:
        class FakeDownloader:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def snapshot_download(
                self,
                *,
                repo_id: str,
                local_dir: Path,
                allow_patterns: tuple[str, ...] | None = None,
            ) -> Path:
                self.calls.append(
                    {"repo_id": repo_id, "local_dir": local_dir, "allow_patterns": allow_patterns}
                )
                for relative_path in allow_patterns or ():
                    target = local_dir / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"asset")
                return local_dir

        with tempfile.TemporaryDirectory(prefix="sdxl-models-") as temp_dir:
            models_dir = Path(temp_dir)
            downloader = FakeDownloader()

            weights.acquire_optional_feature_weights(
                "sdxl-base",
                "sdxl_ip_adapter_style",
                models_dir,
                downloader=downloader,
            )

        expected_target = models_dir / "sdxl-base" / "optional" / "sdxl_ip_adapter_style"
        self.assertEqual(
            downloader.calls,
            [
                {
                    "repo_id": "h94/IP-Adapter",
                    "local_dir": expected_target,
                    "allow_patterns": (
                        "sdxl_models/ip-adapter_sdxl.bin",
                        "sdxl_models/image_encoder/config.json",
                        "sdxl_models/image_encoder/model.safetensors",
                    ),
                }
            ],
        )

    def test_sdxl_optional_feature_weight_acquisition_requests_minimal_required_asset_bundle(self) -> None:
        class FakeDownloader:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def snapshot_download(
                self,
                *,
                repo_id: str,
                local_dir: Path,
                allow_patterns: tuple[str, ...] | None = None,
            ) -> Path:
                self.calls.append({"repo_id": repo_id, "local_dir": local_dir, "allow_patterns": allow_patterns})
                for relative_path in allow_patterns or ():
                    target = local_dir / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"asset")
                return local_dir

        with tempfile.TemporaryDirectory(prefix="sdxl-models-") as temp_dir:
            models_dir = Path(temp_dir)
            downloader = FakeDownloader()

            result = weights.acquire_optional_feature_weights(
                "sdxl-base",
                "sdxl_ip_adapter_style",
                models_dir,
                downloader=downloader,
            )

        expected_required_files = (
            "sdxl_models/ip-adapter_sdxl.bin",
            "sdxl_models/image_encoder/config.json",
            "sdxl_models/image_encoder/model.safetensors",
        )
        self.assertEqual(downloader.calls[0]["allow_patterns"], expected_required_files)
        self.assertEqual(result["required_files"], expected_required_files)
        self.assertEqual(result["missing_files"], ())

    def test_sdxl_optional_feature_readiness_requires_adapter_and_image_encoder_assets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sdxl-models-") as temp_dir:
            models_dir = Path(temp_dir)
            feature_root = models_dir / "sdxl-base" / "optional" / "sdxl_ip_adapter_style"
            adapter_file = feature_root / "sdxl_models" / "ip-adapter_sdxl.bin"
            adapter_file.parent.mkdir(parents=True, exist_ok=True)
            adapter_file.write_bytes(b"adapter")

            readiness = weights.evaluate_extension_weights("sdxl-base", models_dir=models_dir)

            (feature_root / "sdxl_models" / "image_encoder" / "config.json").parent.mkdir(parents=True, exist_ok=True)
            (feature_root / "sdxl_models" / "image_encoder" / "config.json").write_text("{}\n", encoding="utf-8")
            (feature_root / "sdxl_models" / "image_encoder" / "model.safetensors").write_bytes(b"encoder")
            ready_readiness = weights.evaluate_extension_weights("sdxl-base", models_dir=models_dir)

        optional = readiness["optional_features"]["sdxl_ip_adapter_style"]
        self.assertEqual(optional["status"], "missing")
        self.assertFalse(optional["ready"])
        self.assertEqual(
            optional["missing_files"],
            (
                "sdxl_models/image_encoder/config.json",
                "sdxl_models/image_encoder/model.safetensors",
            ),
        )
        self.assertIn("image_encoder/config.json", " ".join(optional["diagnostics"]))
        self.assertEqual(ready_readiness["optional_features"]["sdxl_ip_adapter_style"]["status"], "ready")

    def test_parse_setup_payload_accepts_models_dir_aliases(self) -> None:
        snake_payload = install_contract.parse_setup_payload(
            stdin_text=json.dumps(
                {
                    "python_exe": sys.executable,
                    "ext_dir": "/tmp/ext",
                    "models_dir": "/tmp/global-models",
                }
            )
        )
        camel_payload = install_contract.parse_setup_payload(
            stdin_text=json.dumps(
                {
                    "python_exe": sys.executable,
                    "ext_dir": "/tmp/ext",
                    "modelsDir": "/tmp/global-models-camel",
                }
            )
        )

        self.assertEqual(snake_payload.models_dir, "/tmp/global-models")
        self.assertEqual(camel_payload.models_dir, "/tmp/global-models-camel")

    def test_sdxl_setup_contract_acquires_optional_ip_adapter_assets(self) -> None:
        runtime_root = self._make_runtime_root("sdxl-base")
        acquired: list[dict[str, object]] = []

        def fake_acquire_optional_feature_weights(extension_id, feature_id, models_dir, *, downloader=None):
            acquired.append({"extension_id": extension_id, "feature_id": feature_id, "models_dir": Path(models_dir)})
            target_dir = Path(models_dir) / extension_id / "optional" / feature_id
            required_files = (
                "sdxl_models/ip-adapter_sdxl.bin",
                "sdxl_models/image_encoder/config.json",
                "sdxl_models/image_encoder/model.safetensors",
            )
            for relative_path in required_files:
                target = target_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"asset")
            check_path = target_dir / "sdxl_models" / "ip-adapter_sdxl.bin"
            return {
                "status": "ready",
                "extension_id": extension_id,
                "feature_id": feature_id,
                "model_dir": str(target_dir),
                "check_path": str(check_path),
                "required_files": required_files,
                "missing_files": (),
                "downloaded": True,
            }

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False))
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM))
            stack.enter_context(patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=self._fake_plan("sdxl-base")))
            stack.enter_context(patch("local_image_runtime.install_contract._run_checked", side_effect=self._run_checked_side_effect))
            stack.enter_context(patch("local_image_runtime.install_contract._install_dependency_step", return_value=None))
            stack.enter_context(patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports")))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.acquire_optional_feature_weights",
                    side_effect=fake_acquire_optional_feature_weights,
                )
            )

            result = install_contract.run_install_setup_contract(
                extension_id="sdxl-base",
                stdin_text=self._payload(runtime_root),
            )

        runtime_models_dir = runtime_root / ".local-image-runtime" / "models"
        self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
        self.assertEqual(
            acquired,
            [
                {
                    "extension_id": "sdxl-base",
                    "feature_id": "sdxl_ip_adapter_style",
                    "models_dir": runtime_models_dir,
                }
            ],
        )
        readiness = weights.evaluate_extension_weights("sdxl-base", models_dir=runtime_models_dir)
        self.assertEqual(readiness["optional_features"]["sdxl_ip_adapter_style"]["status"], "ready")
        persisted_state = json.loads(
            (runtime_root / ".local-image-runtime" / "state" / "models-state.json").read_text(
                encoding="utf-8"
            )
        )
        sdxl_weights = persisted_state["extensions"]["sdxl-base"]["weights"]
        self.assertEqual(sdxl_weights["models_dir"], str(runtime_models_dir.resolve()))
        self.assertEqual(sdxl_weights["source"], "argument")
        self.assertEqual(
            sdxl_weights["optional_features"]["sdxl_ip_adapter_style"]["status"],
            "ready",
        )

    def test_sdxl_setup_contract_uses_explicit_models_dir_for_optional_assets(self) -> None:
        runtime_root = self._make_runtime_root("sdxl-base")
        explicit_models_dir = Path(tempfile.mkdtemp(prefix="modly-models-"))
        acquired: list[dict[str, object]] = []

        def fake_acquire_optional_feature_weights(extension_id, feature_id, models_dir, *, downloader=None):
            acquired.append({"extension_id": extension_id, "feature_id": feature_id, "models_dir": Path(models_dir)})
            target_dir = Path(models_dir) / extension_id / "optional" / feature_id
            required_files = (
                "sdxl_models/ip-adapter_sdxl.bin",
                "sdxl_models/image_encoder/config.json",
                "sdxl_models/image_encoder/model.safetensors",
            )
            for relative_path in required_files:
                target = target_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"asset")
            check_path = target_dir / "sdxl_models" / "ip-adapter_sdxl.bin"
            return {
                "status": "ready",
                "extension_id": extension_id,
                "feature_id": feature_id,
                "model_dir": str(target_dir),
                "check_path": str(check_path),
                "required_files": required_files,
                "missing_files": (),
                "downloaded": True,
            }

        payload = json.loads(self._payload(runtime_root))
        payload["models_dir"] = str(explicit_models_dir)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {bootstrap.EXTENSION_ROOT_OVERRIDE_ENV: str(runtime_root)}, clear=False))
            stack.enter_context(patch("local_image_runtime.install_contract.detect_platform", return_value=SUPPORTED_PLATFORM))
            stack.enter_context(patch("local_image_runtime.install_contract.resolve_dependency_plan", return_value=self._fake_plan("sdxl-base")))
            stack.enter_context(patch("local_image_runtime.install_contract._run_checked", side_effect=self._run_checked_side_effect))
            stack.enter_context(patch("local_image_runtime.install_contract._install_dependency_step", return_value=None))
            stack.enter_context(patch("local_image_runtime.bootstrap._smoke_test_runtime_imports", return_value=(True, "stubbed imports")))
            stack.enter_context(
                patch(
                    "local_image_runtime.install_contract.acquire_optional_feature_weights",
                    side_effect=fake_acquire_optional_feature_weights,
                )
            )

            result = install_contract.run_install_setup_contract(
                extension_id="sdxl-base",
                stdin_text=json.dumps(payload),
            )

        self.assertEqual(result.status, bootstrap.SETUP_STATUS_READY)
        self.assertEqual(
            acquired,
            [
                {
                    "extension_id": "sdxl-base",
                    "feature_id": "sdxl_ip_adapter_style",
                    "models_dir": explicit_models_dir,
                }
            ],
        )
        persisted_state = json.loads(
            (runtime_root / ".local-image-runtime" / "state" / "models-state.json").read_text(
                encoding="utf-8"
            )
        )
        sdxl_weights = persisted_state["extensions"]["sdxl-base"]["weights"]
        self.assertEqual(sdxl_weights["models_dir"], str(explicit_models_dir.resolve()))
        self.assertEqual(sdxl_weights["source"], "setup_payload")
        self.assertEqual(
            sdxl_weights["optional_features"]["sdxl_ip_adapter_style"]["status"],
            "ready",
        )
        self.assertEqual(
            sdxl_weights["optional_features"]["sdxl_ip_adapter_style"]["model_dir"],
            str(
                explicit_models_dir.resolve()
                / "sdxl-base"
                / "optional"
                / "sdxl_ip_adapter_style"
            ),
        )

    def test_inference_runner_adds_ip_adapter_kwargs_and_scale_only_for_sdxl_style_reference(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-sdxl-style-"))
        source_image_token = object()
        reference_image_token = object()
        job = {
            "extension_id": "sdxl-base",
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(workspace_dir / "model"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(workspace_dir / "result.png"),
            "prompt": "test prompt",
            "source_image_path": "/tmp/source.png",
            "params": {"steps": 4, "strength": 0.55, "reference_strength": 0.35},
            "conditioning": {"references": [{"role": "style", "filePath": "/tmp/style.png"}]},
        }

        with patch.object(inference_runner, "_open_source_image", side_effect=[source_image_token, reference_image_token]) as open_image:
            runner_kwargs = inference_runner._build_pipeline_kwargs(job, execution_device="cpu")

        self.assertEqual([call.args[0] for call in open_image.call_args_list], ["/tmp/source.png", "/tmp/style.png"])
        self.assertIs(runner_kwargs["image"], source_image_token)
        self.assertIs(runner_kwargs["ip_adapter_image"], reference_image_token)
        self.assertNotIn("cross_attention_kwargs", runner_kwargs)

    def test_inference_runner_lazily_configures_ip_adapter_only_when_style_reference_exists(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FakePipeline:
            def __init__(self) -> None:
                self.loaded_adapters: list[dict[str, object]] = []
                self.scales: list[float] = []

            def load_ip_adapter(self, asset_dir: str, **kwargs: object) -> None:
                self.loaded_adapters.append({"asset_dir": asset_dir, "kwargs": kwargs})

            def set_ip_adapter_scale(self, scale: float) -> None:
                self.scales.append(scale)

        model_dir = Path(tempfile.mkdtemp(prefix="sdxl-model-dir-"))
        for relative_path in (
            "sdxl_models/ip-adapter_sdxl.bin",
            "sdxl_models/image_encoder/config.json",
            "sdxl_models/image_encoder/model.safetensors",
        ):
            asset_file = model_dir / "optional" / "sdxl_ip_adapter_style" / relative_path
            asset_file.parent.mkdir(parents=True, exist_ok=True)
            asset_file.write_bytes(b"asset")
        style_job = {
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(model_dir),
            "params": {"reference_strength": 0.45},
            "conditioning": {"references": [{"role": "style", "filePath": "/tmp/style.png"}]},
        }
        baseline_job = {"family": "sdxl", "node_id": "image-to-image", "model_dir": str(model_dir), "params": {}}
        style_pipeline = FakePipeline()
        baseline_pipeline = FakePipeline()

        inference_runner._configure_ip_adapter_if_present(style_pipeline, style_job)
        inference_runner._configure_ip_adapter_if_present(baseline_pipeline, baseline_job)

        self.assertEqual(
            style_pipeline.loaded_adapters,
            [
                {
                    "asset_dir": str(model_dir / "optional" / "sdxl_ip_adapter_style"),
                    "kwargs": {
                        "subfolder": "sdxl_models",
                        "weight_name": "ip-adapter_sdxl.bin",
                        "image_encoder_folder": "sdxl_models/image_encoder",
                        "local_files_only": True,
                    },
                }
            ],
        )
        self.assertEqual(style_pipeline.scales, [0.45])
        self.assertEqual(baseline_pipeline.loaded_adapters, [])
        self.assertEqual(baseline_pipeline.scales, [])

    def test_inference_runner_resolves_ip_adapter_assets_from_node_scoped_model_dir(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FakePipeline:
            def __init__(self) -> None:
                self.loaded_adapters: list[dict[str, object]] = []

            def load_ip_adapter(self, asset_dir: str, **kwargs: object) -> None:
                self.loaded_adapters.append({"asset_dir": asset_dir, "kwargs": kwargs})

        extension_model_dir = Path(tempfile.mkdtemp(prefix="sdxl-extension-model-dir-")) / "sdxl-base"
        node_model_dir = extension_model_dir / "image-to-image"
        node_model_dir.mkdir(parents=True, exist_ok=True)
        for relative_path in (
            "sdxl_models/ip-adapter_sdxl.bin",
            "sdxl_models/image_encoder/config.json",
            "sdxl_models/image_encoder/model.safetensors",
        ):
            asset_file = extension_model_dir / "optional" / "sdxl_ip_adapter_style" / relative_path
            asset_file.parent.mkdir(parents=True, exist_ok=True)
            asset_file.write_bytes(b"asset")
        job = {
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(node_model_dir),
            "params": {"reference_strength": 0.45},
            "conditioning": {"references": [{"role": "style", "filePath": "/tmp/style.png"}]},
        }
        pipeline = FakePipeline()

        inference_runner._configure_ip_adapter_if_present(pipeline, job)

        self.assertEqual(
            pipeline.loaded_adapters,
            [
                {
                    "asset_dir": str(extension_model_dir / "optional" / "sdxl_ip_adapter_style"),
                    "kwargs": {
                        "subfolder": "sdxl_models",
                        "weight_name": "ip-adapter_sdxl.bin",
                        "image_encoder_folder": "sdxl_models/image_encoder",
                        "local_files_only": True,
                    },
                }
            ],
        )

    def test_inference_runner_resolves_ip_adapter_assets_from_extension_scoped_model_dir(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FakePipeline:
            def __init__(self) -> None:
                self.loaded_adapters: list[dict[str, object]] = []

            def load_ip_adapter(self, asset_dir: str, **kwargs: object) -> None:
                self.loaded_adapters.append({"asset_dir": asset_dir, "kwargs": kwargs})

        extension_model_dir = Path(tempfile.mkdtemp(prefix="sdxl-extension-model-dir-")) / "sdxl-base"
        for relative_path in (
            "sdxl_models/ip-adapter_sdxl.bin",
            "sdxl_models/image_encoder/config.json",
            "sdxl_models/image_encoder/model.safetensors",
        ):
            asset_file = extension_model_dir / "optional" / "sdxl_ip_adapter_style" / relative_path
            asset_file.parent.mkdir(parents=True, exist_ok=True)
            asset_file.write_bytes(b"asset")
        job = {
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(extension_model_dir),
            "params": {"reference_strength": 0.45},
            "conditioning": {"references": [{"role": "style", "filePath": "/tmp/style.png"}]},
        }
        pipeline = FakePipeline()

        inference_runner._configure_ip_adapter_if_present(pipeline, job)

        self.assertEqual(
            pipeline.loaded_adapters,
            [
                {
                    "asset_dir": str(extension_model_dir / "optional" / "sdxl_ip_adapter_style"),
                    "kwargs": {
                        "subfolder": "sdxl_models",
                        "weight_name": "ip-adapter_sdxl.bin",
                        "image_encoder_folder": "sdxl_models/image_encoder",
                        "local_files_only": True,
                    },
                }
            ],
        )

    def test_inference_runner_sd15_ip_adapter_loader_config_is_feature_specific_or_blocked(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        extension_model_dir = Path(tempfile.mkdtemp(prefix="sd15-loader-config-")) / "sd15"

        sd15_config = inference_runner.resolve_ip_adapter_loader_config(
            extension_id="sd15",
            family="stable-diffusion",
            node_id="image-to-image",
            feature_id="sd15_ip_adapter_style",
            extension_model_dir=extension_model_dir,
        )

        self.assertTrue(sd15_config.ready)
        self.assertEqual(sd15_config.status, "ready")
        self.assertTrue(sd15_config.local_files_only)
        self.assertEqual(sd15_config.subfolder, "models")
        self.assertEqual(sd15_config.weight_name, "ip-adapter_sd15.safetensors")
        self.assertEqual(sd15_config.image_encoder_folder, "models/image_encoder")
        self.assertNotEqual(sd15_config.subfolder, "sdxl_models")
        self.assertNotEqual(sd15_config.weight_name, "ip-adapter_sdxl.bin")
        self.assertNotEqual(sd15_config.image_encoder_folder, "sdxl_models/image_encoder")

        sdxl_config = inference_runner.resolve_ip_adapter_loader_config(
            extension_id="sdxl-base",
            family="sdxl",
            node_id="image-to-image",
            feature_id="sdxl_ip_adapter_style",
            extension_model_dir=extension_model_dir.parent / "sdxl-base",
        )
        self.assertTrue(sdxl_config.ready)
        self.assertEqual(sdxl_config.subfolder, "sdxl_models")
        self.assertEqual(sdxl_config.weight_name, "ip-adapter_sdxl.bin")
        self.assertEqual(sdxl_config.image_encoder_folder, "sdxl_models/image_encoder")
        self.assertTrue(sdxl_config.local_files_only)

    def test_generate_missing_sd15_style_assets_fails_locally_without_acquisition_or_sdxl_paths(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FailingIfLoadedPipeline:
            def load_ip_adapter(self, asset_dir: str, **kwargs: object) -> None:
                raise AssertionError("missing SD1.5 style assets must be detected before IP-Adapter loading")

        workspace_dir = Path(tempfile.mkdtemp(prefix="generate-missing-sd15-style-"))
        extension_model_dir = workspace_dir / "models" / "sd15"
        source_path = workspace_dir / "source.png"
        reference_path = workspace_dir / "style-reference.png"
        source_path.write_bytes(b"source")
        reference_path.write_bytes(b"style")
        job = {
            "extension_id": "sd15",
            "family": "stable-diffusion",
            "node_id": "image-to-image",
            "model_dir": str(extension_model_dir / "image-to-image"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(workspace_dir / "result.png"),
            "prompt": "variation",
            "source_image_path": str(source_path),
            "params": {"steps": 4, "strength": 0.55, "reference_strength": 0.4},
            "conditioning": {"references": [{"role": "style", "filePath": str(reference_path)}]},
        }

        with patch(
            "local_image_runtime.weights.acquire_optional_feature_weights",
            side_effect=AssertionError("Generate must not acquire SD1.5 optional feature assets"),
        ) as acquire_optional, patch(
            "local_image_runtime.weights.HuggingFaceSnapshotDownloader.snapshot_download",
            side_effect=AssertionError("Generate must not download SD1.5 optional feature assets"),
        ) as snapshot_download:
            with self.assertRaisesRegex(inference_runner.InferenceRunnerError, "SD1.5.*local-readiness") as cm:
                inference_runner._configure_ip_adapter_if_present(FailingIfLoadedPipeline(), job)

        message = str(cm.exception)
        self.assertIn("models/ip-adapter_sd15.safetensors", message)
        self.assertNotIn("sdxl_models", message)
        acquire_optional.assert_not_called()
        snapshot_download.assert_not_called()

    def test_inference_runner_fails_locally_before_loading_missing_ip_adapter_image_encoder_assets(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FailingIfLoadedPipeline:
            def load_ip_adapter(self, asset_dir: str, **kwargs: object) -> None:
                raise AssertionError("generation must precheck the full local IP-Adapter bundle")

        model_dir = Path(tempfile.mkdtemp(prefix="sdxl-missing-encoder-"))
        node_model_dir = model_dir / "image-to-image"
        node_model_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = model_dir / "optional" / "sdxl_ip_adapter_style" / "sdxl_models" / "ip-adapter_sdxl.bin"
        adapter_file.parent.mkdir(parents=True, exist_ok=True)
        adapter_file.write_bytes(b"adapter")
        expected_encoder_file = model_dir / "optional" / "sdxl_ip_adapter_style" / "sdxl_models" / "image_encoder" / "config.json"
        job = {
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(node_model_dir),
            "params": {"reference_strength": 0.6},
            "conditioning": {"references": [{"role": "style", "filePath": "/tmp/style.png"}]},
        }

        with self.assertRaisesRegex(inference_runner.InferenceRunnerError, str(expected_encoder_file)):
            inference_runner._configure_ip_adapter_if_present(FailingIfLoadedPipeline(), job)

    def test_inference_runner_fails_locally_before_loading_missing_ip_adapter_asset(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FailingIfLoadedPipeline:
            def load_ip_adapter(self, asset_dir: str, **kwargs: object) -> None:
                raise AssertionError("generation must not ask Diffusers to download missing IP-Adapter assets")

        model_dir = Path(tempfile.mkdtemp(prefix="sdxl-missing-adapter-"))
        node_model_dir = model_dir / "image-to-image"
        node_model_dir.mkdir(parents=True, exist_ok=True)
        expected_adapter_file = model_dir / "optional" / "sdxl_ip_adapter_style" / "sdxl_models" / "ip-adapter_sdxl.bin"
        job = {
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(node_model_dir),
            "params": {"reference_strength": 0.6},
            "conditioning": {"references": [{"role": "style", "filePath": "/tmp/style.png"}]},
        }

        with self.assertRaisesRegex(inference_runner.InferenceRunnerError, str(expected_adapter_file)):
            inference_runner._configure_ip_adapter_if_present(FailingIfLoadedPipeline(), job)

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

    def test_quality_policy_resolve_effective_params_applies_family_defaults_only_when_missing(self) -> None:
        import local_image_runtime.quality_policy as quality_policy

        cases = (
            (
                "sd15",
                "text-to-image",
                {},
                {
                    "width": 512,
                    "height": 512,
                    "steps": 30,
                    "guidance_scale": 7.5,
                    "negative_prompt": "blurry, low quality, bad anatomy, deformed, extra digits",
                },
            ),
            (
                "sdxl-base",
                "image-to-image",
                {},
                {
                    "width": 1024,
                    "height": 1024,
                    "steps": 30,
                    "guidance_scale": 5.0,
                    "strength": 0.7,
                    "negative_prompt": "blurry, low quality, distorted, artifacts",
                },
            ),
        )

        for extension_id, node_id, params, expected_defaults in cases:
            with self.subTest(extension_id=extension_id, node_id=node_id):
                resolved = quality_policy.resolve_effective_params(
                    extension_id=extension_id,
                    node_id=node_id,
                    params=params,
                )
                for key, expected_value in expected_defaults.items():
                    self.assertEqual(resolved[key], expected_value)

        overridden = quality_policy.resolve_effective_params(
            extension_id="sd15",
            node_id="text-to-image",
            params={"steps": 12, "negative_prompt": "custom override"},
        )
        self.assertEqual(overridden["steps"], 12)
        self.assertEqual(overridden["negative_prompt"], "custom override")

    def test_quality_policy_resolve_effective_params_preserves_explicit_empty_negative_prompt(self) -> None:
        import local_image_runtime.quality_policy as quality_policy

        resolved = quality_policy.resolve_effective_params(
            extension_id="sd15",
            node_id="text-to-image",
            params={"negative_prompt": "", "steps": 12},
        )

        self.assertEqual(resolved["negative_prompt"], "")
        self.assertEqual(resolved["steps"], 12)

        flux_passthrough = quality_policy.resolve_effective_params(
            extension_id="flux-schnell",
            node_id="text-to-image",
            params={"steps": 4},
        )
        self.assertEqual(
            flux_passthrough,
            {
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance_scale": 0.0,
                "max_sequence_length": 256,
            },
        )

    def test_flux_text_to_image_missing_params_resolve_runtime_defaults(self) -> None:
        import local_image_runtime.quality_policy as quality_policy

        resolved = quality_policy.resolve_effective_params(
            extension_id="flux-schnell",
            node_id="text-to-image",
            params={},
        )

        self.assertEqual(
            resolved,
            {
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance_scale": 0.0,
                "max_sequence_length": 256,
            },
        )

    def test_flux_max_sequence_length_validates_and_reaches_runner_kwargs(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "flux prompt"},
            params={
                "prompt": "flux prompt",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance_scale": 0.0,
                "max_sequence_length": 128,
            },
        )

        validated = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="flux-schnell")
        self.assertEqual(validated.numeric_params["max_sequence_length"], 128)

        kwargs = inference_runner._build_pipeline_kwargs(
            {
                "family": "flux",
                "node_id": "text-to-image",
                "prompt": "flux prompt",
                "source_image_path": None,
                "params": validated.numeric_params,
            },
            execution_device="cpu",
        )
        self.assertEqual(kwargs["max_sequence_length"], 128)

    def test_flux_max_sequence_length_invalid_values_are_rejected_before_inference(self) -> None:
        cases = (0, 257, 128.5, "128")

        for value in cases:
            with self.subTest(max_sequence_length=value):
                request = pipeline.ExecutionRequest(
                    node_id="text-to-image",
                    input={"text": "flux prompt"},
                    params={
                        "prompt": "flux prompt",
                        "width": 1024,
                        "height": 1024,
                        "steps": 4,
                        "guidance_scale": 0.0,
                        "max_sequence_length": value,
                    },
                )

                with self.assertRaises(pipeline.RequestValidationError):
                    pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="flux-schnell")

    def test_flux_high_steps_are_preserved_before_backend_and_runner(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="flux-stale-steps-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        runtime_root = Path(tempfile.mkdtemp(prefix="ext-root-flux-steps-"))
        (runtime_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(runtime_root)
        extension_record = {
            "venv_python": str(venv_python),
            "model_dir": str(runtime.paths.models_dir / "flux-schnell"),
        }
        serialized_payloads: list[dict[str, object]] = []

        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "flux prompt"},
            params={
                "prompt": "flux prompt",
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 0.0,
                "max_sequence_length": 128,
            },
            workspace_dir=str(workspace_dir),
        )

        def capture_serialized_payload(command, *, stdin, stdout, stderr, text, bufsize, cwd, env):
            def on_stdin_close(payload_text: str) -> tuple[list[str], list[str], int]:
                serialized_payloads.append(json.loads(payload_text))
                return (
                    [json.dumps({"type": "done", "result": {"output_path": str(workspace_dir / "result.png")}}) + "\n"],
                    [],
                    0,
                )

            return self._FakePopen(
                stdout_lines=[],
                stderr_lines=[],
                on_stdin_close=on_stdin_close,
            )

        with patch("local_image_runtime.pipeline.extension_is_installed", return_value=True), patch(
            "local_image_runtime.pipeline.get_extension_record",
            return_value=extension_record,
        ), patch(
            "local_image_runtime.pipeline.subprocess.Popen",
            side_effect=capture_serialized_payload,
        ):
            pipeline.execute(
                request,
                runtime,
                extension_id="flux-schnell",
                emit_progress=lambda percent, label: None,
                emit_log=lambda message: None,
            )

        self.assertEqual(len(serialized_payloads), 1)
        self.assertEqual(serialized_payloads[0]["params"]["steps"], 30)
        runner_kwargs = inference_runner._build_pipeline_kwargs(serialized_payloads[0], execution_device="cpu")
        self.assertEqual(runner_kwargs["num_inference_steps"], 30)

    def test_flux_invalid_low_steps_are_still_rejected_before_inference(self) -> None:
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "flux prompt"},
            params={
                "prompt": "flux prompt",
                "width": 1024,
                "height": 1024,
                "steps": 0,
                "guidance_scale": 0.0,
                "max_sequence_length": 128,
            },
        )

        with self.assertRaises(pipeline.RequestValidationError):
            pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="flux-schnell")

    def test_flux_non_integer_steps_are_still_rejected_before_inference(self) -> None:
        for value in (1.5, "4"):
            with self.subTest(steps=value):
                request = pipeline.ExecutionRequest(
                    node_id="text-to-image",
                    input={"text": "flux prompt"},
                    params={
                        "prompt": "flux prompt",
                        "width": 1024,
                        "height": 1024,
                        "steps": value,
                        "guidance_scale": 0.0,
                        "max_sequence_length": 128,
                    },
                )

                with self.assertRaises(pipeline.RequestValidationError):
                    pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="flux-schnell")

    def test_flux_guidance_scale_five_reaches_backend_and_runner(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="flux-guidance-pass-through-"))
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-flux-guidance-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(extension_root)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "flux prompt"},
            params={
                "prompt": "flux prompt",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance_scale": 5.0,
                "max_sequence_length": 128,
            },
            workspace_dir=str(workspace_dir),
        )

        payload_details = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="flux-schnell")
        self.assertEqual(payload_details.numeric_params["guidance_scale"], 5.0)

        job = pipeline._build_backend_job(
            request=request,
            extension_id="flux-schnell",
            extension_record={"venv_python": str(venv_python), "model_dir": "/runtime/local/flux"},
            payload_details=payload_details,
            effective_workspace_dir=str(workspace_dir),
        )

        self.assertEqual(job.payload["params"]["guidance_scale"], 5.0)
        kwargs = inference_runner._build_pipeline_kwargs(job.payload, execution_device="cpu")
        self.assertEqual(kwargs["guidance_scale"], 5.0)

    def test_flux_invalid_guidance_scale_values_are_rejected_before_inference(self) -> None:
        for value in (-0.1, -1.0, "5"):
            with self.subTest(guidance_scale=value):
                request = pipeline.ExecutionRequest(
                    node_id="text-to-image",
                    input={"text": "flux prompt"},
                    params={
                        "prompt": "flux prompt",
                        "width": 1024,
                        "height": 1024,
                        "steps": 4,
                        "guidance_scale": value,
                        "max_sequence_length": 128,
                    },
                )

                with self.assertRaises(pipeline.RequestValidationError):
                    pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="flux-schnell")

    def test_flux_negative_prompt_is_omitted_from_backend_payload_and_runner_kwargs(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="flux-negative-suppression-"))
        extension_root = Path(tempfile.mkdtemp(prefix="ext-root-flux-negative-"))
        (extension_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(extension_root)
        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "flux prompt"},
            params={
                "prompt": "flux prompt",
                "negative_prompt": "avoid this",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance_scale": 0.0,
                "max_sequence_length": 128,
            },
            workspace_dir=str(workspace_dir),
        )
        payload_details = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="flux-schnell")

        job = pipeline._build_backend_job(
            request=request,
            extension_id="flux-schnell",
            extension_record={"venv_python": str(venv_python), "model_dir": "/runtime/local/flux"},
            payload_details=payload_details,
            effective_workspace_dir=str(workspace_dir),
        )

        self.assertNotIn("negative_prompt", job.payload)
        self.assertNotIn("negative_prompt", job.payload["params"])
        kwargs = inference_runner._build_pipeline_kwargs(job.payload, execution_device="cpu")
        self.assertNotIn("negative_prompt", kwargs)

    def test_num_images_per_prompt_validates_and_reaches_runner_kwargs(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        for extension_id, family, node_id in (
            ("sd15", "stable-diffusion", "text-to-image"),
            ("sdxl-base", "sdxl", "image-to-image"),
            ("flux-schnell", "flux", "text-to-image"),
        ):
            with self.subTest(extension_id=extension_id, node_id=node_id):
                workspace_dir = Path(tempfile.mkdtemp(prefix=f"num-images-{extension_id}-"))
                input_payload = {"text": "prompt"}
                params: dict[str, object] = {
                    "prompt": "prompt",
                    "steps": 4,
                    "num_images_per_prompt": 4,
                }
                if node_id == "image-to-image":
                    source_path = workspace_dir / "source.png"
                    source_path.write_bytes(b"source")
                    input_payload = {"filePath": str(source_path)}
                    params["strength"] = 0.55
                if family == "flux":
                    params["guidance_scale"] = 0.0

                request = pipeline.ExecutionRequest(
                    node_id=node_id,
                    input=input_payload,
                    params=params,
                    workspace_dir=str(workspace_dir),
                )

                validated = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id=extension_id)
                self.assertEqual(validated.numeric_params["num_images_per_prompt"], 4)

                runner_payload = {
                    "family": family,
                    "node_id": node_id,
                    "prompt": "prompt",
                    "source_image_path": validated.source_image_path,
                    "params": validated.numeric_params,
                }
                with patch("local_image_runtime.inference_runner._open_source_image", return_value=object()):
                    kwargs = inference_runner._build_pipeline_kwargs(runner_payload, execution_device="cpu")
                self.assertEqual(kwargs["num_images_per_prompt"], 4)

    def test_num_images_per_prompt_invalid_values_are_rejected_before_inference(self) -> None:
        for value in (0, 5, 2.5, "2", True):
            with self.subTest(num_images_per_prompt=value):
                request = pipeline.ExecutionRequest(
                    node_id="text-to-image",
                    input={"text": "prompt"},
                    params={"prompt": "prompt", "steps": 4, "num_images_per_prompt": value},
                )

                with self.assertRaises(pipeline.RequestValidationError):
                    pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sd15")

    def test_output_format_and_quality_invalid_values_are_rejected_before_inference(self) -> None:
        cases = (
            {"output_format": "gif"},
            {"output_format": ""},
            {"output_quality": 0, "output_format": "jpeg"},
            {"output_quality": 101, "output_format": "jpeg"},
            {"output_format": "jpeg", "output_quality": 90.5},
        )

        for params_delta in cases:
            with self.subTest(params_delta=params_delta):
                workspace_dir = Path(tempfile.mkdtemp(prefix="invalid-output-params-"))
                runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
                request = pipeline.ExecutionRequest(
                    node_id="text-to-image",
                    input={"text": "prompt"},
                    params={"prompt": "prompt", "steps": 4, **params_delta},
                    workspace_dir=str(workspace_dir),
                )

                with patch("local_image_runtime.pipeline.extension_is_installed", return_value=True), patch(
                    "local_image_runtime.pipeline.get_extension_record",
                    return_value=self._make_installed_extension_record(extension_id="sd15", workspace_dir=workspace_dir),
                ), patch("local_image_runtime.pipeline.subprocess.Popen") as subprocess_popen:
                    with self.assertRaises(pipeline.RequestValidationError):
                        pipeline.execute(
                            request,
                            runtime,
                            extension_id="sd15",
                            emit_progress=lambda percent, label: None,
                            emit_log=lambda message: None,
                        )
                subprocess_popen.assert_not_called()

    def test_build_backend_job_ignores_png_quality_and_preserves_jpeg_quality(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="output-format-job-"))
        extension_record = self._make_installed_extension_record(extension_id="sd15", workspace_dir=workspace_dir)

        for params, expected_suffix, expected_format, expected_quality in (
            ({"prompt": "png prompt", "steps": 4}, ".png", "png", None),
            ({"prompt": "png prompt", "steps": 4, "output_format": "png", "output_quality": 85}, ".png", "png", None),
            ({"prompt": "jpeg prompt", "steps": 4, "output_format": "jpeg", "output_quality": 85}, ".jpg", "jpeg", 85),
        ):
            with self.subTest(params=params):
                request = pipeline.ExecutionRequest(
                    node_id="text-to-image",
                    input={"text": "prompt"},
                    params=params,
                    workspace_dir=str(workspace_dir),
                )
                payload_details = pipeline._validate_node_payload(request, legacy_model_id=None, extension_id="sd15")

                job = pipeline._build_backend_job(
                    request=request,
                    extension_id="sd15",
                    extension_record=extension_record,
                    payload_details=payload_details,
                    effective_workspace_dir=str(workspace_dir),
                )

                output_path = Path(job.payload["output_path"])
                self.assertEqual(output_path.suffix, expected_suffix)
                self.assertEqual(job.payload["output_format"], expected_format)
                self.assertEqual(job.payload["output_quality"], expected_quality)
                if expected_format == "jpeg":
                    self.assertEqual(job.payload["params"].get("output_format"), "jpeg")
                    self.assertEqual(job.payload["params"].get("output_quality"), expected_quality)
                else:
                    if "output_format" not in params:
                        self.assertNotIn("output_format", job.payload["params"])
                    self.assertNotIn("output_quality", job.payload["params"])
                self.assertTrue(output_path.resolve().is_relative_to(workspace_dir.resolve()))

    def test_sd_families_keep_negative_prompt_guidance_steps_and_exclude_flux_sequence_length(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        cases = (
            ("stable-diffusion", "sd negative", 7.5, 30),
            ("sdxl", "sdxl negative", 5.0, 28),
        )

        for family, negative_prompt, guidance_scale, steps in cases:
            with self.subTest(family=family):
                kwargs = inference_runner._build_pipeline_kwargs(
                    {
                        "family": family,
                        "node_id": "text-to-image",
                        "prompt": f"{family} prompt",
                        "negative_prompt": negative_prompt,
                        "source_image_path": None,
                        "params": {
                            "steps": steps,
                            "guidance_scale": guidance_scale,
                            "width": 512,
                            "height": 512,
                            "max_sequence_length": 128,
                        },
                    },
                    execution_device="cpu",
                )

                self.assertEqual(kwargs["negative_prompt"], negative_prompt)
                self.assertEqual(kwargs["guidance_scale"], guidance_scale)
                self.assertEqual(kwargs["num_inference_steps"], steps)
                self.assertNotIn("max_sequence_length", kwargs)

    def test_inference_runner_default_single_png_contract_is_unchanged(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="single-png-runner-"))
        output_path = workspace_dir / "generated.png"
        saved_paths: list[tuple[str, dict[str, object]]] = []

        class FakeImage:
            def save(self, output_path: str, **kwargs: object) -> None:
                saved_paths.append((output_path, kwargs))
                Path(output_path).write_bytes(b"png")

        class FakePipeline:
            def __call__(self, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(images=[FakeImage()])

        job = {
            "model_dir": str(workspace_dir / "model"),
            "output_path": str(output_path),
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "prompt": "prompt",
            "source_image_path": None,
            "params": {"steps": 4},
        }

        loader = SimpleNamespace(from_pretrained=lambda model_dir: FakePipeline())
        with patch.dict(inference_runner._PIPELINE_LOADERS, {("stable-diffusion", "text-to-image"): loader}, clear=True), patch(
            "local_image_runtime.inference_runner._load_torch", return_value=None
        ):
            result = inference_runner.run_child_job(job, stdout=StringIO())

        self.assertEqual(saved_paths, [(str(output_path), {})])
        self.assertEqual(result["output_path"], str(output_path))
        self.assertEqual(result["output_paths"], [str(output_path)])
        self.assertEqual(result["output_count"], 1)
        self.assertEqual(result["output_format"], "png")
        self.assertTrue(output_path.exists())
        self.assertEqual(list(workspace_dir.glob("*.jpg")), [])

    def test_inference_runner_saves_all_returned_images_with_numbered_paths(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="multi-image-runner-"))
        output_path = workspace_dir / "generated-sd15-text.png"
        saved_paths: list[str] = []

        class FakeImage:
            def __init__(self, marker: str) -> None:
                self.marker = marker

            def save(self, output_path: str, **kwargs: object) -> None:
                saved_paths.append(output_path)
                Path(output_path).write_bytes(self.marker.encode("utf-8"))

        class FakePipeline:
            def __call__(self, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(images=[FakeImage("zero"), FakeImage("one"), FakeImage("two")])

        job = {
            "model_dir": str(workspace_dir / "model"),
            "output_path": str(output_path),
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "prompt": "prompt",
            "source_image_path": None,
            "output_format": "png",
            "params": {"steps": 4, "num_images_per_prompt": 3},
        }

        loader = SimpleNamespace(from_pretrained=lambda model_dir: FakePipeline())
        with patch.dict(inference_runner._PIPELINE_LOADERS, {("stable-diffusion", "text-to-image"): loader}, clear=True), patch(
            "local_image_runtime.inference_runner._load_torch", return_value=None
        ):
            result = inference_runner.run_child_job(job, stdout=StringIO())

        expected_paths = [
            output_path,
            workspace_dir / "generated-sd15-text-1.png",
            workspace_dir / "generated-sd15-text-2.png",
        ]
        self.assertEqual(saved_paths, [str(path) for path in expected_paths])
        self.assertEqual(result["output_path"], str(output_path))
        self.assertEqual(result["output_paths"], [str(path) for path in expected_paths])
        self.assertEqual(result["output_count"], 3)
        self.assertEqual(result["output_format"], "png")
        sidecar = output_path.with_suffix(output_path.suffix + ".json")
        self.assertEqual(
            json.loads(sidecar.read_text(encoding="utf-8")),
            {
                "output_path": str(output_path),
                "output_paths": [str(path) for path in expected_paths],
                "output_count": 3,
                "output_format": "png",
            },
        )
        self.assertEqual([path.read_text(encoding="utf-8") for path in expected_paths], ["zero", "one", "two"])

    def test_inference_runner_reports_actual_count_when_request_asks_for_two_but_one_returns(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="multi-image-under-return-"))
        output_path = workspace_dir / "generated-sd15-text.png"
        pipeline_kwargs: list[dict[str, object]] = []
        saved_paths: list[str] = []

        class FakeImage:
            def save(self, output_path: str, **kwargs: object) -> None:
                saved_paths.append(output_path)
                Path(output_path).write_bytes(b"only-returned-image")

        class FakePipeline:
            def __call__(self, **kwargs: object) -> SimpleNamespace:
                pipeline_kwargs.append(kwargs)
                return SimpleNamespace(images=[FakeImage()])

        job = {
            "model_dir": str(workspace_dir / "model"),
            "output_path": str(output_path),
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "prompt": "prompt",
            "source_image_path": None,
            "output_format": "png",
            "params": {"steps": 4, "num_images_per_prompt": 2},
        }

        loader = SimpleNamespace(from_pretrained=lambda model_dir: FakePipeline())
        with patch.dict(inference_runner._PIPELINE_LOADERS, {("stable-diffusion", "text-to-image"): loader}, clear=True), patch(
            "local_image_runtime.inference_runner._load_torch", return_value=None
        ):
            result = inference_runner.run_child_job(job, stdout=StringIO())

        self.assertEqual(pipeline_kwargs[0]["num_images_per_prompt"], 2)
        self.assertEqual(saved_paths, [str(output_path)])
        self.assertEqual(result["output_path"], str(output_path))
        self.assertEqual(result["output_paths"], [str(output_path)])
        self.assertEqual(result["output_count"], 1)
        self.assertEqual(output_path.read_bytes(), b"only-returned-image")
        self.assertFalse(output_path.with_name("generated-sd15-text-1.png").exists())

    def test_inference_runner_reports_actual_count_when_request_asks_for_two_but_three_return(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="multi-image-over-return-"))
        output_path = workspace_dir / "generated-sd15-text.png"
        pipeline_kwargs: list[dict[str, object]] = []
        saved_paths: list[str] = []

        class FakeImage:
            def __init__(self, marker: str) -> None:
                self.marker = marker

            def save(self, output_path: str, **kwargs: object) -> None:
                saved_paths.append(output_path)
                Path(output_path).write_bytes(self.marker.encode("utf-8"))

        class FakePipeline:
            def __call__(self, **kwargs: object) -> SimpleNamespace:
                pipeline_kwargs.append(kwargs)
                return SimpleNamespace(images=[FakeImage("zero"), FakeImage("one"), FakeImage("two")])

        job = {
            "model_dir": str(workspace_dir / "model"),
            "output_path": str(output_path),
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "prompt": "prompt",
            "source_image_path": None,
            "output_format": "png",
            "params": {"steps": 4, "num_images_per_prompt": 2},
        }

        loader = SimpleNamespace(from_pretrained=lambda model_dir: FakePipeline())
        with patch.dict(inference_runner._PIPELINE_LOADERS, {("stable-diffusion", "text-to-image"): loader}, clear=True), patch(
            "local_image_runtime.inference_runner._load_torch", return_value=None
        ):
            result = inference_runner.run_child_job(job, stdout=StringIO())

        expected_paths = [
            output_path,
            workspace_dir / "generated-sd15-text-1.png",
            workspace_dir / "generated-sd15-text-2.png",
        ]
        self.assertEqual(pipeline_kwargs[0]["num_images_per_prompt"], 2)
        self.assertEqual(saved_paths, [str(path) for path in expected_paths])
        self.assertEqual(result["output_path"], str(output_path))
        self.assertEqual(result["output_paths"], [str(path) for path in expected_paths])
        self.assertEqual(result["output_count"], 3)
        self.assertEqual([path.read_text(encoding="utf-8") for path in expected_paths], ["zero", "one", "two"])

    def test_inference_runner_saves_jpeg_with_rgb_conversion_and_quality(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="jpeg-runner-"))
        output_path = workspace_dir / "generated.jpg"
        save_calls: list[tuple[str, dict[str, object], str]] = []

        class FakeImage:
            def __init__(self, mode: str = "RGBA") -> None:
                self.mode = mode

            def convert(self, mode: str) -> "FakeImage":
                self.converted_to = mode
                return FakeImage(mode=mode)

            def save(self, output_path: str, **kwargs: object) -> None:
                save_calls.append((output_path, kwargs, self.mode))
                Path(output_path).write_bytes(b"jpeg")

        class FakePipeline:
            def __call__(self, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(images=[FakeImage()])

        job = {
            "model_dir": str(workspace_dir / "model"),
            "output_path": str(output_path),
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "prompt": "prompt",
            "source_image_path": None,
            "output_format": "jpeg",
            "params": {"steps": 4, "output_format": "jpeg", "output_quality": 85},
        }

        loader = SimpleNamespace(from_pretrained=lambda model_dir: FakePipeline())
        with patch.dict(inference_runner._PIPELINE_LOADERS, {("stable-diffusion", "text-to-image"): loader}, clear=True), patch(
            "local_image_runtime.inference_runner._load_torch", return_value=None
        ):
            result = inference_runner.run_child_job(job, stdout=StringIO())

        self.assertEqual(save_calls, [(str(output_path), {"quality": 85}, "RGB")])
        self.assertEqual(Path(result["output_path"]).suffix, ".jpg")
        self.assertEqual(result["output_paths"], [str(output_path)])
        self.assertEqual(result["output_format"], "jpeg")

    def test_pipeline_normalizes_and_contains_all_reported_output_paths(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="contained-output-paths-"))
        inside_primary = workspace_dir / "primary.png"
        inside_secondary = workspace_dir / "primary-1.png"
        outside = workspace_dir.parent / "escaped.png"

        normalized = pipeline._normalize_backend_result_paths(
            {
                "output_path": str(inside_primary),
                "output_paths": [str(inside_primary), str(inside_secondary)],
                "output_count": 2,
                "output_format": "png",
            },
            workspace_dir=workspace_dir.resolve(),
        )
        self.assertEqual(normalized["output_paths"], [str(inside_primary.resolve()), str(inside_secondary.resolve())])
        self.assertEqual(normalized["output_count"], 2)

        with self.assertRaisesRegex(pipeline.DomainError, "output_paths.*outside workspace_dir"):
            pipeline._normalize_backend_result_paths(
                {"output_path": str(inside_primary), "output_paths": [str(inside_primary), str(outside)]},
                workspace_dir=workspace_dir.resolve(),
            )

    def test_flux_manifest_exposes_sequence_length_and_hides_negative_prompt(self) -> None:
        schema = self._extension_manifest_data("flux-schnell")["nodes"][0]["params_schema"]
        params_by_id = {param["id"]: param for param in schema}

        self.assertNotIn("negative_prompt", params_by_id)
        self.assertEqual(
            params_by_id["steps"],
            {
                "id": "steps",
                "label": "Steps",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 30,
                "tooltip": "Recommended range is 1-4 for FLUX Schnell; higher values are experimental and may not improve quality.",
            },
        )
        self.assertEqual(
            params_by_id["max_sequence_length"],
            {
                "id": "max_sequence_length",
                "label": "Max Sequence Length",
                "type": "int",
                "default": 256,
                "min": 1,
                "max": 256,
                "tooltip": "Maximum FLUX text token sequence length passed to Diffusers.",
            },
        )
        self.assertEqual(
            params_by_id["guidance_scale"],
            {
                "id": "guidance_scale",
                "label": "Guidance Scale",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 50,
                "tooltip": "Recommended 0.0 for FLUX Schnell; higher values are experimental and may not improve quality unless the active pipeline supports them.",
            },
        )

    def test_pipeline_execute_applies_quality_policy_defaults_to_backend_payload(self) -> None:
        cases = (
            (
                "sd15",
                pipeline.ExecutionRequest(
                    node_id="text-to-image",
                    input={"text": "legacy lighthouse prompt"},
                    params={"prompt": "lighthouse at dusk"},
                ),
                {
                    "width": 512,
                    "height": 512,
                    "steps": 30,
                    "guidance_scale": 7.5,
                    "negative_prompt": "blurry, low quality, bad anatomy, deformed, extra digits",
                },
            ),
            (
                "sdxl-base",
                pipeline.ExecutionRequest(
                    node_id="image-to-image",
                    input={},
                    params={"prompt": "cinematic variation"},
                ),
                {
                    "width": 1024,
                    "height": 1024,
                    "steps": 30,
                    "guidance_scale": 5.0,
                    "strength": 0.7,
                    "negative_prompt": "blurry, low quality, distorted, artifacts",
                },
            ),
        )

        for extension_id, request, expected_defaults in cases:
            with self.subTest(extension_id=extension_id, node_id=request.node_id):
                workspace_dir = Path(tempfile.mkdtemp(prefix=f"quality-policy-{extension_id}-"))
                runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
                runtime_root = Path(tempfile.mkdtemp(prefix=f"ext-root-{extension_id}-"))
                (runtime_root / "src").mkdir(parents=True, exist_ok=True)
                venv_python = self._make_executable_python(runtime_root)
                extension_record = {
                    "venv_python": str(venv_python),
                    "model_dir": str(runtime.paths.models_dir / extension_id),
                }
                serialized_payloads: list[dict[str, object]] = []

                effective_request = pipeline.ExecutionRequest(
                    node_id=request.node_id,
                    input=request.input,
                    params=request.params,
                    workspace_dir=str(workspace_dir),
                )
                if request.node_id == "image-to-image":
                    source_path = workspace_dir / "source.png"
                    source_path.write_bytes(b"fake-image")
                    effective_request = pipeline.ExecutionRequest(
                        node_id=request.node_id,
                        input={"filePath": str(source_path)},
                        params=request.params,
                        workspace_dir=str(workspace_dir),
                    )

                def capture_serialized_payload(command, *, stdin, stdout, stderr, text, bufsize, cwd, env):
                    def on_stdin_close(payload_text: str) -> tuple[list[str], list[str], int]:
                        serialized_payloads.append(json.loads(payload_text))
                        return (
                            [json.dumps({"type": "done", "result": {"output_path": str(workspace_dir / "result.png")}}) + "\n"],
                            [],
                            0,
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
                    pipeline.execute(
                        effective_request,
                        runtime,
                        extension_id=extension_id,
                        emit_progress=lambda percent, label: None,
                        emit_log=lambda message: None,
                    )

                self.assertEqual(len(serialized_payloads), 1)
                serialized_payload = serialized_payloads[0]
                self.assertEqual(serialized_payload["negative_prompt"], expected_defaults["negative_prompt"])
                for key, expected_value in expected_defaults.items():
                    if key == "negative_prompt":
                        continue
                    self.assertEqual(serialized_payload["params"][key], expected_value)

    def test_pipeline_execute_preserves_explicit_empty_negative_prompt_override(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="quality-policy-empty-negative-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        runtime_root = Path(tempfile.mkdtemp(prefix="ext-root-sd15-"))
        (runtime_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(runtime_root)
        extension_record = {
            "venv_python": str(venv_python),
            "model_dir": str(runtime.paths.models_dir / "sd15"),
        }
        serialized_payloads: list[dict[str, object]] = []

        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "legacy lighthouse prompt"},
            params={"prompt": "lighthouse at dusk", "negative_prompt": ""},
            workspace_dir=str(workspace_dir),
        )

        def capture_serialized_payload(command, *, stdin, stdout, stderr, text, bufsize, cwd, env):
            def on_stdin_close(payload_text: str) -> tuple[list[str], list[str], int]:
                serialized_payloads.append(json.loads(payload_text))
                return (
                    [json.dumps({"type": "done", "result": {"output_path": str(workspace_dir / "result.png")}}) + "\n"],
                    [],
                    0,
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
            pipeline.execute(
                request,
                runtime,
                extension_id="sd15",
                emit_progress=lambda percent, label: None,
                emit_log=lambda message: None,
            )

        self.assertEqual(len(serialized_payloads), 1)
        self.assertEqual(serialized_payloads[0]["negative_prompt"], "")

    def test_pipeline_execute_applies_quality_policy_defaults_for_sd15_image_to_image(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="quality-policy-sd15-image-to-image-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        runtime_root = Path(tempfile.mkdtemp(prefix="ext-root-sd15-"))
        (runtime_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(runtime_root)
        extension_record = {
            "venv_python": str(venv_python),
            "model_dir": str(runtime.paths.models_dir / "sd15"),
        }
        serialized_payloads: list[dict[str, object]] = []
        source_path = workspace_dir / "source.png"
        source_path.write_bytes(b"fake-image")

        request = pipeline.ExecutionRequest(
            node_id="image-to-image",
            input={"filePath": str(source_path)},
            params={"prompt": "portrait remix"},
            workspace_dir=str(workspace_dir),
        )

        def capture_serialized_payload(command, *, stdin, stdout, stderr, text, bufsize, cwd, env):
            def on_stdin_close(payload_text: str) -> tuple[list[str], list[str], int]:
                serialized_payloads.append(json.loads(payload_text))
                return (
                    [json.dumps({"type": "done", "result": {"output_path": str(workspace_dir / "result.png")}}) + "\n"],
                    [],
                    0,
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
            pipeline.execute(
                request,
                runtime,
                extension_id="sd15",
                emit_progress=lambda percent, label: None,
                emit_log=lambda message: None,
            )

        self.assertEqual(len(serialized_payloads), 1)
        self.assertEqual(
            serialized_payloads[0]["negative_prompt"],
            "blurry, low quality, bad anatomy, deformed, extra digits",
        )
        self.assertEqual(serialized_payloads[0]["params"]["width"], 512)
        self.assertEqual(serialized_payloads[0]["params"]["height"], 512)
        self.assertEqual(serialized_payloads[0]["params"]["steps"], 30)
        self.assertEqual(serialized_payloads[0]["params"]["guidance_scale"], 7.5)
        self.assertEqual(serialized_payloads[0]["params"]["strength"], 0.75)

    def test_pipeline_execute_applies_quality_policy_defaults_for_sdxl_text_to_image(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="quality-policy-sdxl-text-to-image-"))
        runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
        runtime_root = Path(tempfile.mkdtemp(prefix="ext-root-sdxl-base-"))
        (runtime_root / "src").mkdir(parents=True, exist_ok=True)
        venv_python = self._make_executable_python(runtime_root)
        extension_record = {
            "venv_python": str(venv_python),
            "model_dir": str(runtime.paths.models_dir / "sdxl-base"),
        }
        serialized_payloads: list[dict[str, object]] = []

        request = pipeline.ExecutionRequest(
            node_id="text-to-image",
            input={"text": "legacy mountain prompt"},
            params={"prompt": "mountain vista at sunrise"},
            workspace_dir=str(workspace_dir),
        )

        def capture_serialized_payload(command, *, stdin, stdout, stderr, text, bufsize, cwd, env):
            def on_stdin_close(payload_text: str) -> tuple[list[str], list[str], int]:
                serialized_payloads.append(json.loads(payload_text))
                return (
                    [json.dumps({"type": "done", "result": {"output_path": str(workspace_dir / "result.png")}}) + "\n"],
                    [],
                    0,
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
            pipeline.execute(
                request,
                runtime,
                extension_id="sdxl-base",
                emit_progress=lambda percent, label: None,
                emit_log=lambda message: None,
            )

        self.assertEqual(len(serialized_payloads), 1)
        self.assertEqual(
            serialized_payloads[0]["negative_prompt"],
            "blurry, low quality, distorted, artifacts",
        )
        self.assertEqual(serialized_payloads[0]["params"]["width"], 1024)
        self.assertEqual(serialized_payloads[0]["params"]["height"], 1024)
        self.assertEqual(serialized_payloads[0]["params"]["steps"], 30)
        self.assertEqual(serialized_payloads[0]["params"]["guidance_scale"], 5.0)

    def test_sd_family_manifests_align_quality_defaults_and_help_with_shared_policy(self) -> None:
        import local_image_runtime.quality_policy as quality_policy

        for extension_id in ("sd15", "sdxl-base"):
            manifest = self._extension_manifest_data(extension_id)
            nodes = {node["id"]: node for node in manifest["nodes"]}
            for node_id in ("text-to-image", "image-to-image"):
                with self.subTest(extension_id=extension_id, node_id=node_id):
                    params_schema = {
                        schema["id"]: schema
                        for schema in nodes[node_id]["params_schema"]
                    }
                    expected_defaults = quality_policy.get_node_defaults(extension_id, node_id)
                    expected_help = quality_policy.get_node_help(extension_id, node_id)

                    for param_id, expected_value in expected_defaults.items():
                        self.assertEqual(params_schema[param_id]["default"], expected_value)
                    for param_id, expected_tooltip in expected_help.items():
                        self.assertEqual(params_schema[param_id]["tooltip"], expected_tooltip)

    def test_manifests_hide_multi_image_and_expose_output_format_parameters(self) -> None:
        expected_nodes = {
            "sd15": ("text-to-image", "image-to-image"),
            "sdxl-base": ("text-to-image", "image-to-image"),
            "flux-schnell": ("text-to-image",),
        }

        for extension_id, node_ids in expected_nodes.items():
            manifest = self._extension_manifest_data(extension_id)
            nodes = {node["id"]: node for node in manifest["nodes"]}
            for node_id in node_ids:
                with self.subTest(extension_id=extension_id, node_id=node_id):
                    params_schema = {schema["id"]: schema for schema in nodes[node_id]["params_schema"]}
                    self.assertNotIn("num_images_per_prompt", params_schema)
                    self.assertEqual(
                        params_schema["output_format"],
                        {
                            "id": "output_format",
                            "label": "Output Format",
                            "type": "select",
                            "default": "png",
                            "options": [
                                {"value": "png", "label": "PNG"},
                                {"value": "jpeg", "label": "JPEG"},
                            ],
                            "tooltip": "File format for generated images. PNG preserves existing behavior; JPEG supports output_quality.",
                        },
                    )
                    self.assertEqual(
                        params_schema["output_quality"],
                        {
                            "id": "output_quality",
                            "label": "JPEG Quality",
                            "type": "int",
                            "default": 85,
                            "min": 1,
                            "max": 100,
                            "tooltip": "JPEG quality from 1 to 100. Only valid when output_format is jpeg.",
                        },
                    )

    def test_pipeline_execute_serializes_subprocess_payload_by_family_and_node(self) -> None:
        import local_image_runtime.quality_policy as quality_policy

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
                    {
                        key: value
                        for key, value in quality_policy.resolve_effective_params(
                            extension_id=extension_id,
                            node_id=request.node_id,
                            params=request.params,
                        ).items()
                        if key != "negative_prompt"
                    },
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
                        "output_paths": [str(expected_output_path)],
                        "output_count": 1,
                        "output_format": "png",
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

    def test_pipeline_execute_emits_host_progress_from_shared_lifecycle_module(self) -> None:
        custom_host_steps = (
            (11, "host-validate"),
            (22, "host-check"),
            (33, "host-dispatch"),
        )
        cases = (
            pipeline.ExecutionRequest(
                node_id="text-to-image",
                input={"text": "sunrise"},
                params={"prompt": "sunrise", "steps": 4},
            ),
            pipeline.ExecutionRequest(
                node_id="image-to-image",
                input={},
                params={"prompt": "variation", "strength": 0.45, "steps": 4},
            ),
        )

        for request in cases:
            with self.subTest(node_id=request.node_id):
                workspace_dir = Path(tempfile.mkdtemp(prefix=f"host-lifecycle-{request.node_id}-"))
                runtime = self._make_runtime_snapshot(outputs_dir=workspace_dir)
                effective_request = request
                if request.node_id == "image-to-image":
                    source_path = workspace_dir / "source.png"
                    source_path.write_bytes(b"source-image")
                    effective_request = pipeline.ExecutionRequest(
                        node_id=request.node_id,
                        input={"filePath": str(source_path)},
                        params=request.params,
                        workspace_dir=str(workspace_dir),
                    )
                else:
                    effective_request = pipeline.ExecutionRequest(
                        node_id=request.node_id,
                        input=request.input,
                        params=request.params,
                        workspace_dir=str(workspace_dir),
                    )

                progress_events: list[tuple[int, str]] = []
                backend_job = self._make_backend_job(workspace_dir=workspace_dir)

                with patch(
                    "local_image_runtime.pipeline.lifecycle.host_generation_steps",
                    return_value=custom_host_steps,
                ), patch(
                    "local_image_runtime.pipeline.get_extension_record",
                    return_value={"model_dir": str(runtime.paths.models_dir / "sd15")},
                ), patch(
                    "local_image_runtime.pipeline.extension_is_installed",
                    return_value=True,
                ), patch(
                    "local_image_runtime.pipeline._build_backend_job",
                    return_value=backend_job,
                ), patch(
                    "local_image_runtime.pipeline._run_backend_job",
                    return_value={"output_path": str(workspace_dir / "result.png")},
                ):
                    pipeline.execute(
                        effective_request,
                        runtime,
                        extension_id="sd15",
                        emit_progress=lambda percent, label: progress_events.append((percent, label)),
                        emit_log=lambda message: None,
                    )

                self.assertEqual(progress_events, list(custom_host_steps))

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

    def test_inference_runner_resolves_execution_device_by_available_accelerator(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        class FakeCuda:
            def __init__(self, *, available: bool) -> None:
                self._available = available

            def is_available(self) -> bool:
                return self._available

        class FakeMps:
            def __init__(self, *, available: bool) -> None:
                self._available = available

            def is_available(self) -> bool:
                return self._available

        cases = (
            (SimpleNamespace(cuda=FakeCuda(available=True), backends=SimpleNamespace(mps=FakeMps(available=True))), "cuda"),
            (SimpleNamespace(cuda=FakeCuda(available=False), backends=SimpleNamespace(mps=FakeMps(available=True))), "mps"),
            (SimpleNamespace(cuda=FakeCuda(available=False), backends=SimpleNamespace(mps=FakeMps(available=False))), "cpu"),
            (None, "cpu"),
        )

        for torch_module, expected_device in cases:
            with self.subTest(expected_device=expected_device):
                self.assertEqual(
                    inference_runner._resolve_execution_device(torch_module=torch_module),
                    expected_device,
                )

    def test_diffusers_load_attempts_target_sd_families_with_ordered_fallbacks(self) -> None:
        import local_image_runtime.diffusers_memory as diffusers_memory

        fake_torch = SimpleNamespace(float16="float16")
        cases = (
            ("sd15", "stable-diffusion"),
            ("sdxl-base", "sdxl"),
            ("flux-schnell", "flux"),
        )

        for extension_id, family in cases:
            with self.subTest(extension_id=extension_id):
                attempts = diffusers_memory.build_diffusers_load_attempts(
                    extension_id=extension_id,
                    family=family,
                    node_id="text-to-image",
                    torch_module=fake_torch,
                )

                if extension_id == "flux-schnell":
                    self.assertEqual(attempts, (("baseline", {}),))
                    continue

                self.assertEqual(
                    [attempt_name for attempt_name, _ in attempts],
                    [
                        "optimized-fp16",
                        "optimized-no-variant",
                        "optimized-no-safetensors",
                        "optimized-no-low-cpu-mem",
                        "baseline",
                    ],
                )
                self.assertEqual(
                    attempts[0][1],
                    {
                        "torch_dtype": "float16",
                        "variant": "fp16",
                        "use_safetensors": True,
                        "low_cpu_mem_usage": True,
                    },
                )
                self.assertEqual(attempts[-1], ("baseline", {}))

    def test_diffusers_load_retry_classifier_limits_fallback_to_known_loader_failures(self) -> None:
        import local_image_runtime.diffusers_memory as diffusers_memory

        retryable_errors = (
            TypeError("unexpected keyword argument 'variant'"),
            OSError("no file named diffusion_pytorch_model.fp16.safetensors found"),
            ValueError("variant fp16 is not available for this checkpoint"),
        )
        terminal_errors = (
            RuntimeError("weights checksum mismatch"),
            OSError("permission denied while reading model directory"),
        )

        for error in retryable_errors:
            with self.subTest(error=str(error)):
                self.assertTrue(diffusers_memory.is_retryable_diffusers_load_error(error))

        for error in terminal_errors:
            with self.subTest(error=str(error)):
                self.assertFalse(diffusers_memory.is_retryable_diffusers_load_error(error))

    def test_diffusers_memory_snapshot_tolerates_missing_posix_resource_module(self) -> None:
        import local_image_runtime.diffusers_memory as diffusers_memory

        with patch.object(diffusers_memory, "_resource", None):
            snapshot = diffusers_memory.collect_stage_memory_snapshot(stage="windows-runtime")

        self.assertEqual(snapshot, {"stage": "windows-runtime"})

    def test_inference_runner_moves_pipeline_to_resolved_device_before_execution(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-device-placement-"))
        output_path = workspace_dir / "result.png"

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        class FakePipeline:
            def __init__(self) -> None:
                self.to_calls: list[str] = []
                self.invocations: list[dict[str, object]] = []

            def to(self, device: str) -> "FakePipeline":
                self.to_calls.append(device)
                return self

            def __call__(self, **kwargs):
                self.invocations.append(kwargs)
                return SimpleNamespace(images=[FakeImage()])

        fake_pipeline = FakePipeline()
        job = {
            "extension_id": "sd15",
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "model_dir": str(workspace_dir / "model"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(output_path),
            "prompt": "device placement",
            "params": {"steps": 4, "seed": 42},
        }

        stdout = StringIO()
        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("stable-diffusion", "text-to-image"): SimpleNamespace(from_pretrained=lambda model_dir: fake_pipeline)},
            clear=True,
        ), patch.object(inference_runner, "_resolve_execution_device", return_value="cuda"):
            exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

        self.assertEqual(exit_code, 0)
        self.assertEqual(fake_pipeline.to_calls, ["cuda"])
        self.assertEqual(len(fake_pipeline.invocations), 1)

    def test_inference_runner_uses_optimized_loader_args_for_sd15_and_sdxl_base(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-loader-optimized-"))

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        for extension_id, family in (("sd15", "stable-diffusion"), ("sdxl-base", "sdxl")):
            loader_calls: list[dict[str, object]] = []

            class FakePipeline:
                def to(self, device: str) -> "FakePipeline":
                    return self

                def __call__(self, **kwargs):
                    return SimpleNamespace(images=[FakeImage()])

            class FakeLoader:
                @staticmethod
                def from_pretrained(model_dir: str, **kwargs):
                    loader_calls.append({"model_dir": model_dir, "kwargs": kwargs})
                    return FakePipeline()

            job = {
                "extension_id": extension_id,
                "family": family,
                "node_id": "text-to-image",
                "model_dir": str(workspace_dir / extension_id / "model"),
                "workspace_dir": str(workspace_dir),
                "output_path": str(workspace_dir / f"{extension_id}.png"),
                "prompt": f"optimized {extension_id}",
                "params": {"steps": 4},
            }

            stdout = StringIO()
            with self.subTest(extension_id=extension_id), patch.dict(
                inference_runner._PIPELINE_LOADERS,
                {(family, "text-to-image"): FakeLoader()},
                clear=True,
            ), patch.object(inference_runner, "_resolve_execution_device", return_value="cpu"), patch.object(
                inference_runner,
                "_load_torch",
                return_value=SimpleNamespace(float16="float16"),
            ):
                exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(loader_calls), 1)
            self.assertEqual(
                loader_calls[0]["kwargs"],
                {
                    "torch_dtype": "float16",
                    "variant": "fp16",
                    "use_safetensors": True,
                    "low_cpu_mem_usage": True,
                },
            )

    def test_inference_runner_falls_back_to_baseline_loader_when_optimized_kwargs_fail(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-loader-fallback-"))
        loader_calls: list[dict[str, object]] = []

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        class FakePipeline:
            def to(self, device: str) -> "FakePipeline":
                return self

            def __call__(self, **kwargs):
                return SimpleNamespace(images=[FakeImage()])

        class FakeLoader:
            @staticmethod
            def from_pretrained(model_dir: str, **kwargs):
                loader_calls.append({"model_dir": model_dir, "kwargs": kwargs})
                if kwargs:
                    raise TypeError("unexpected keyword argument 'variant'")
                return FakePipeline()

        job = {
            "extension_id": "sd15",
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "model_dir": str(workspace_dir / "model"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(workspace_dir / "sd15.png"),
            "prompt": "fallback sd15",
            "params": {"steps": 4},
        }

        stdout = StringIO()
        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("stable-diffusion", "text-to-image"): FakeLoader()},
            clear=True,
        ), patch.object(inference_runner, "_resolve_execution_device", return_value="cpu"), patch.object(
            inference_runner,
            "_load_torch",
            return_value=SimpleNamespace(float16="float16"),
        ):
            exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

        self.assertEqual(exit_code, 0)
        self.assertEqual([call["kwargs"] for call in loader_calls][-1], {})
        self.assertGreaterEqual(len(loader_calls), 2)

    def test_seeded_generator_uses_resolved_execution_device(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        generator_calls: list[tuple[str, int]] = []

        class FakeGenerator:
            def __init__(self, *, device: str) -> None:
                self.device = device

            def manual_seed(self, seed: int) -> str:
                generator_calls.append((self.device, seed))
                return f"generator:{self.device}:{seed}"

        fake_torch = SimpleNamespace(Generator=lambda *, device: FakeGenerator(device=device))

        generator = inference_runner._seeded_generator(
            {"seed": 1234},
            execution_device="cuda",
            torch_module=fake_torch,
        )

        self.assertEqual(generator, "generator:cuda:1234")
        self.assertEqual(generator_calls, [("cuda", 1234)])

    def test_inference_runner_applies_guarded_post_load_memory_optimizations_for_sd_families(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-post-load-optimizations-"))

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        for extension_id, family, expect_calls in (
            ("sd15", "stable-diffusion", True),
            ("flux-schnell", "flux", False),
        ):
            optimization_calls: list[str] = []

            class FakePipeline:
                def to(self, device: str) -> "FakePipeline":
                    return self

                def enable_attention_slicing(self, mode: str) -> None:
                    optimization_calls.append(f"attention:{mode}")

                def enable_vae_slicing(self) -> None:
                    optimization_calls.append("vae")

                def __call__(self, **kwargs):
                    return SimpleNamespace(images=[FakeImage()])

            job = {
                "extension_id": extension_id,
                "family": family,
                "node_id": "text-to-image",
                "model_dir": str(workspace_dir / extension_id / "model"),
                "workspace_dir": str(workspace_dir),
                "output_path": str(workspace_dir / f"{extension_id}.png"),
                "prompt": f"optimize {extension_id}",
                "params": {"steps": 4},
            }

            stdout = StringIO()
            with self.subTest(extension_id=extension_id), patch.dict(
                inference_runner._PIPELINE_LOADERS,
                {(family, "text-to-image"): SimpleNamespace(from_pretrained=lambda model_dir, **kwargs: FakePipeline())},
                clear=True,
            ), patch.object(inference_runner, "_resolve_execution_device", return_value="cpu"):
                exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

            self.assertEqual(exit_code, 0)
            if expect_calls:
                self.assertEqual(optimization_calls, ["attention:auto", "vae"])
            else:
                self.assertEqual(optimization_calls, [])

    def test_inference_runner_skips_missing_post_load_memory_optimization_hooks(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-post-load-guards-"))

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        class FakePipeline:
            def to(self, device: str) -> "FakePipeline":
                return self

            def __call__(self, **kwargs):
                return SimpleNamespace(images=[FakeImage()])

        job = {
            "extension_id": "sdxl-base",
            "family": "sdxl",
            "node_id": "text-to-image",
            "model_dir": str(workspace_dir / "model"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(workspace_dir / "sdxl.png"),
            "prompt": "guard missing hooks",
            "params": {"steps": 4},
        }

        stdout = StringIO()
        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("sdxl", "text-to-image"): SimpleNamespace(from_pretrained=lambda model_dir, **kwargs: FakePipeline())},
            clear=True,
        ), patch.object(inference_runner, "_resolve_execution_device", return_value="cpu"):
            exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

        self.assertEqual(exit_code, 0)
        self.assertEqual(self._parse_ndjson_events(stdout.getvalue())[-1]["type"], "done")

    def test_sdxl_style_reference_preserves_vae_slicing_but_skips_attention_slicing(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-sdxl-style-memory-"))
        model_dir = workspace_dir / "models" / "sdxl-base" / "image-to-image"
        model_dir.mkdir(parents=True, exist_ok=True)
        for relative_path in (
            "sdxl_models/ip-adapter_sdxl.bin",
            "sdxl_models/image_encoder/config.json",
            "sdxl_models/image_encoder/model.safetensors",
        ):
            asset_file = model_dir.parent / "optional" / "sdxl_ip_adapter_style" / relative_path
            asset_file.parent.mkdir(parents=True, exist_ok=True)
            asset_file.write_bytes(b"asset")
        source_image_token = object()
        reference_image_token = object()
        pipeline_calls: list[dict[str, object]] = []
        optimization_calls: list[str] = []
        scale_calls: list[float] = []

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        class FakePipeline:
            def to(self, device: str) -> "FakePipeline":
                return self

            def load_ip_adapter(self, asset_dir: str, **kwargs: object) -> None:
                return None

            def set_ip_adapter_scale(self, scale: float) -> None:
                scale_calls.append(scale)

            def enable_attention_slicing(self, mode: str) -> None:
                optimization_calls.append(f"attention:{mode}")

            def enable_vae_slicing(self) -> None:
                optimization_calls.append("vae")

            def __call__(self, **kwargs):
                pipeline_calls.append(kwargs)
                return SimpleNamespace(images=[FakeImage()])

        job = {
            "extension_id": "sdxl-base",
            "family": "sdxl",
            "node_id": "image-to-image",
            "model_dir": str(model_dir),
            "workspace_dir": str(workspace_dir),
            "output_path": str(workspace_dir / "style.png"),
            "prompt": "style reference without sliced attention",
            "source_image_path": "/tmp/source.png",
            "params": {"steps": 4, "strength": 0.55, "reference_strength": 0.72},
            "conditioning": {"references": [{"role": "style", "filePath": "/tmp/style.png"}]},
        }

        stdout = StringIO()
        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("sdxl", "image-to-image"): SimpleNamespace(from_pretrained=lambda model_dir, **kwargs: FakePipeline())},
            clear=True,
        ), patch.object(inference_runner, "_resolve_execution_device", return_value="cpu"), patch.object(
            inference_runner,
            "_open_source_image",
            side_effect=[source_image_token, reference_image_token],
        ):
            exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

        self.assertEqual(exit_code, 0)
        self.assertEqual(scale_calls, [0.72])
        self.assertEqual(optimization_calls, ["vae"])
        self.assertEqual(len(pipeline_calls), 1)
        self.assertIs(pipeline_calls[0]["ip_adapter_image"], reference_image_token)
        self.assertNotIn("cross_attention_kwargs", pipeline_calls[0])

    def test_inference_runner_emits_running_inference_heartbeat_logs_while_pipeline_is_busy(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-heartbeat-"))
        output_path = workspace_dir / "result.png"
        started = threading.Event()
        release = threading.Event()

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        class BlockingPipeline:
            def to(self, device: str) -> "BlockingPipeline":
                return self

            def __call__(self, **kwargs):
                started.set()
                if not release.wait(timeout=1.0):
                    raise AssertionError("Timed out waiting to release blocking pipeline.")
                return SimpleNamespace(images=[FakeImage()])

        job = {
            "extension_id": "sd15",
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "model_dir": str(workspace_dir / "model"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(output_path),
            "prompt": "heartbeat",
            "params": {"steps": 4},
        }

        stdout = StringIO()
        result: dict[str, int] = {}

        def run_child() -> None:
            result["exit_code"] = inference_runner.run_child_main(
                stdin=StringIO(json.dumps(job) + "\n"),
                stdout=stdout,
            )

        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("stable-diffusion", "text-to-image"): SimpleNamespace(from_pretrained=lambda model_dir: BlockingPipeline())},
            clear=True,
        ), patch.object(inference_runner, "_resolve_execution_device", return_value="cpu"), patch.object(
            inference_runner,
            "_RUNNING_INFERENCE_HEARTBEAT_SECONDS",
            0.01,
        ):
            worker = threading.Thread(target=run_child)
            worker.start()
            self.assertTrue(started.wait(timeout=1.0))
            time.sleep(0.05)
            release.set()
            worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, {"exit_code": 0})
        events = self._parse_ndjson_events(stdout.getvalue())
        heartbeat_logs = [event for event in events if event["type"] == "log" and "heartbeat" in event["message"]]
        self.assertGreaterEqual(len(heartbeat_logs), 1)
        self.assertEqual(events[-1]["type"], "done")

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
                if case["family"] == "flux":
                    self.assertNotIn("negative_prompt", invocation)
                else:
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
                        "output_paths": [str(output_path)],
                        "output_count": 1,
                        "output_format": "png",
                        "metadata": {
                            "family": case["family"],
                            "node_id": case["node_id"],
                            "seed": 42,
                            "negative_prompt_used": case["family"] != "flux",
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

    def test_inference_runner_emits_child_progress_from_shared_lifecycle_module(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        custom_child_steps = (
            (81, "child-load"),
            (91, "child-run"),
            (96, "child-save"),
        )
        cases = (
            {
                "family": "stable-diffusion",
                "node_id": "text-to-image",
                "prompt": "text prompt",
                "params": {"steps": 4},
            },
            {
                "family": "stable-diffusion",
                "node_id": "image-to-image",
                "prompt": "variation prompt",
                "source_image_path": "/tmp/source.png",
                "params": {"steps": 4, "strength": 0.55},
            },
        )

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        class FakePipeline:
            def __call__(self, **kwargs):
                return SimpleNamespace(images=[FakeImage()])

        for case in cases:
            with self.subTest(node_id=case["node_id"]):
                workspace_dir = Path(tempfile.mkdtemp(prefix=f"runner-child-{case['node_id']}-"))
                output_path = workspace_dir / "result.png"
                job = {
                    "extension_id": "sd15",
                    "model_dir": str(workspace_dir / "model"),
                    "workspace_dir": str(workspace_dir),
                    "output_path": str(output_path),
                    **case,
                }
                stdout = StringIO()

                with patch(
                    "local_image_runtime.inference_runner.lifecycle.child_generation_steps",
                    return_value=custom_child_steps,
                ), patch.dict(
                    inference_runner._PIPELINE_LOADERS,
                    {
                        (case["family"], case["node_id"]): SimpleNamespace(
                            from_pretrained=lambda model_dir, **kwargs: FakePipeline()
                        )
                    },
                    clear=True,
                ), patch.object(
                    inference_runner,
                    "_resolve_execution_device",
                    return_value="cpu",
                ), patch.object(
                    inference_runner,
                    "_open_source_image",
                    return_value=object(),
                ):
                    exit_code = inference_runner.run_child_main(
                        stdin=StringIO(json.dumps(job) + "\n"),
                        stdout=stdout,
                    )

                self.assertEqual(exit_code, 0)
                events = self._parse_ndjson_events(stdout.getvalue())
                self.assertEqual(
                    [event["label"] for event in events if event["type"] == "progress"],
                    [label for _, label in custom_child_steps],
                )

    def test_inference_runner_emits_stage_memory_events_for_sd_families_without_breaking_done(self) -> None:
        import local_image_runtime.inference_runner as inference_runner

        workspace_dir = Path(tempfile.mkdtemp(prefix="runner-memory-events-"))
        output_path = workspace_dir / "result.png"

        class FakeImage:
            def save(self, target_path: str) -> None:
                Path(target_path).write_text("generated", encoding="utf-8")

        class FakePipeline:
            def to(self, device: str) -> "FakePipeline":
                return self

            def __call__(self, **kwargs):
                return SimpleNamespace(images=[FakeImage()])

        job = {
            "extension_id": "sd15",
            "family": "stable-diffusion",
            "node_id": "text-to-image",
            "model_dir": str(workspace_dir / "model"),
            "workspace_dir": str(workspace_dir),
            "output_path": str(output_path),
            "prompt": "memory stages",
            "params": {"steps": 4},
        }

        stdout = StringIO()
        with patch.dict(
            inference_runner._PIPELINE_LOADERS,
            {("stable-diffusion", "text-to-image"): SimpleNamespace(from_pretrained=lambda model_dir, **kwargs: FakePipeline())},
            clear=True,
        ), patch.object(inference_runner, "_resolve_execution_device", return_value="cpu"), patch.object(
            inference_runner,
            "collect_stage_memory_snapshot",
            side_effect=lambda **kwargs: {"stage": kwargs["stage"], "rss_mib": 12.5},
        ):
            exit_code = inference_runner.run_child_main(stdin=StringIO(json.dumps(job) + "\n"), stdout=stdout)

        self.assertEqual(exit_code, 0)
        events = self._parse_ndjson_events(stdout.getvalue())
        self.assertEqual(
            [event["stage"] for event in events if event["type"] == "memory"],
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
                "output_paths": [str(output_path)],
                "output_count": 1,
                "output_format": "png",
                "metadata": {
                    "family": "flux",
                    "node_id": "text-to-image",
                    "seed": 7,
                    "negative_prompt_used": False,
                    "source_image_used": False,
                },
            },
        )

    def test_run_backend_job_accepts_memory_events_and_resets_idle_watchdog(self) -> None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="workspace-streaming-memory-reset-"))
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
                (
                    2.49,
                    (
                        "line",
                        "stdout",
                        json.dumps({"type": "memory", "stage": "running-inference", "rss_mib": 12.5}) + "\n",
                    ),
                ),
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
        self.assertEqual(logs, [])
        self.assertEqual(result, {"output_path": str(Path(job.payload["output_path"]).resolve())})
        self.assertFalse(fake_process.terminate_called)

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

    def test_flux_weight_readiness_policy_is_text_to_image_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="flux-models-") as temp_dir:
            models_dir = Path(temp_dir)
            readiness = weights.evaluate_extension_weights("flux-schnell", models_dir=models_dir)

        expected_target = models_dir / "flux-schnell" / "text-to-image"
        self.assertEqual(readiness["status"], "missing")
        self.assertEqual(tuple(readiness["nodes"].keys()), ("text-to-image",))
        self.assertEqual(readiness["nodes"]["text-to-image"]["model_dir"], str(expected_target))
        self.assertEqual(
            readiness["nodes"]["text-to-image"]["check_path"],
            str(expected_target / "model_index.json"),
        )

    def test_flux_weight_acquisition_skips_downloader_when_check_file_already_exists(self) -> None:
        class UnexpectedDownloader:
            def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
                raise AssertionError("downloader should not run for ready Flux weights")

        with tempfile.TemporaryDirectory(prefix="flux-models-") as temp_dir:
            models_dir = Path(temp_dir)
            target_dir = models_dir / "flux-schnell" / "text-to-image"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "model_index.json").write_text("{}\n", encoding="utf-8")

            result = weights.acquire_flux_schnell_weights(
                models_dir=models_dir,
                downloader=UnexpectedDownloader(),
            )

        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["downloaded"])
        self.assertEqual(result["model_dir"], str(target_dir))

    def test_flux_weight_acquisition_api_is_exported_from_runtime_package(self) -> None:
        import local_image_runtime

        self.assertIs(
            local_image_runtime.acquire_flux_schnell_weights,
            weights.acquire_flux_schnell_weights,
        )

    def test_flux_weight_acquisition_uses_injected_downloader_and_text_to_image_target(self) -> None:
        class FakeDownloader:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
                self.calls.append({"repo_id": repo_id, "local_dir": local_dir})
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "model_index.json").write_text("{}\n", encoding="utf-8")
                return local_dir

        with tempfile.TemporaryDirectory(prefix="flux-models-") as temp_dir:
            models_dir = Path(temp_dir)
            downloader = FakeDownloader()

            result = weights.acquire_flux_schnell_weights(
                models_dir=models_dir,
                downloader=downloader,
            )
            expected_target = models_dir / "flux-schnell" / "text-to-image"
            self.assertEqual(
                downloader.calls,
                [
                    {
                        "repo_id": "black-forest-labs/FLUX.1-schnell",
                        "local_dir": expected_target,
                    }
                ],
            )
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["node_id"], "text-to-image")
            self.assertEqual(result["model_dir"], str(expected_target))
            self.assertTrue((expected_target / "model_index.json").exists())

    def test_flux_weight_acquisition_maps_downloader_failures_to_domain_errors(self) -> None:
        class FailingDownloader:
            def __init__(self, exc: Exception) -> None:
                self.exc = exc

            def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
                raise self.exc

        cases = (
            (PermissionError("token required"), weights.FluxWeightAuthError, "authentication"),
            (TimeoutError("request timed out"), weights.FluxWeightNetworkError, "network"),
            (OSError(28, "No space left on device"), weights.FluxWeightDiskError, "disk"),
        )
        for exc, expected_error, expected_message in cases:
            with self.subTest(exc=type(exc).__name__), tempfile.TemporaryDirectory(
                prefix="flux-models-"
            ) as temp_dir:
                with self.assertRaises(expected_error) as raised:
                    weights.acquire_flux_schnell_weights(
                        models_dir=Path(temp_dir),
                        downloader=FailingDownloader(exc),
                    )

                self.assertIn(expected_message, str(raised.exception).lower())

    def test_flux_weight_acquisition_http_auth_failures_include_hf_gated_guidance(self) -> None:
        class StatusError(RuntimeError):
            def __init__(
                self,
                message: str,
                *,
                status_code: int | None = None,
                response_status: int | None = None,
            ) -> None:
                super().__init__(message)
                if status_code is not None:
                    self.status_code = status_code
                if response_status is not None:
                    self.response = SimpleNamespace(status_code=response_status)

        class FailingDownloader:
            def __init__(self, exc: Exception) -> None:
                self.exc = exc

            def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
                raise self.exc

        cases = (
            StatusError("Unauthorized", response_status=401),
            StatusError("Forbidden", status_code=403),
        )

        for exc in cases:
            with self.subTest(exc=str(exc)), tempfile.TemporaryDirectory(prefix="flux-models-") as temp_dir:
                with self.assertRaises(weights.FluxWeightAuthError) as raised:
                    weights.acquire_flux_schnell_weights(
                        models_dir=Path(temp_dir),
                        downloader=FailingDownloader(exc),
                    )

                message = str(raised.exception)
                self.assertIn("https://huggingface.co/black-forest-labs/FLUX.1-schnell", message)
                self.assertIn("same Hugging Face account/token used by Modly", message)
                self.assertIn("accept the model conditions", message)
                self.assertIn("share contact information if requested", message)
                self.assertIn("retry the download", message)

    def test_flux_weight_acquisition_auth_text_failures_include_hf_gated_guidance(self) -> None:
        class FailingDownloader:
            def __init__(self, exc: Exception) -> None:
                self.exc = exc

            def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
                raise self.exc

        cases = (
            RuntimeError("gated repo requires approval"),
            RuntimeError("access denied for this model"),
            RuntimeError("unauthorized request"),
            RuntimeError("accept terms before download"),
            RuntimeError("token does not have permission"),
            RuntimeError("contact information must be shared"),
            RuntimeError("repository not found for this authenticated user"),
            PermissionError("repo not found for token"),
        )

        for exc in cases:
            with self.subTest(exc=str(exc)), tempfile.TemporaryDirectory(prefix="flux-models-") as temp_dir:
                with self.assertRaises(weights.FluxWeightAuthError) as raised:
                    weights.acquire_flux_schnell_weights(
                        models_dir=Path(temp_dir),
                        downloader=FailingDownloader(exc),
                    )

                message = str(raised.exception)
                self.assertIn("https://huggingface.co/black-forest-labs/FLUX.1-schnell", message)
                self.assertIn("same Hugging Face account/token used by Modly", message)
                self.assertIn("accept the model conditions", message)
                self.assertIn("share contact information if requested", message)
                self.assertIn("retry the download", message)

    def test_flux_weight_acquisition_generic_failures_do_not_include_hf_gated_guidance(self) -> None:
        class FailingDownloader:
            def __init__(self, exc: Exception) -> None:
                self.exc = exc

            def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
                raise self.exc

        cases = (
            RuntimeError("checksum mismatch for downloaded artifact"),
            RuntimeError("disk full while expanding archive"),
        )

        for exc in cases:
            with self.subTest(exc=str(exc)), tempfile.TemporaryDirectory(prefix="flux-models-") as temp_dir:
                with self.assertRaises(weights.FluxWeightDownloadError) as raised:
                    weights.acquire_flux_schnell_weights(
                        models_dir=Path(temp_dir),
                        downloader=FailingDownloader(exc),
                    )

                message = str(raised.exception)
                self.assertIn(f"Flux Schnell weight download failed: {exc}", message)
                self.assertNotIn("https://huggingface.co/black-forest-labs/FLUX.1-schnell", message)
                self.assertNotIn("same Hugging Face account/token used by Modly", message)

    def test_sd_weight_readiness_diagnostics_do_not_include_flux_hf_gated_guidance(self) -> None:
        for extension_id in ("sd15", "sdxl-base"):
            with self.subTest(extension_id=extension_id), tempfile.TemporaryDirectory(
                prefix=f"{extension_id}-models-"
            ) as temp_dir:
                readiness = weights.evaluate_extension_weights(extension_id, models_dir=Path(temp_dir))

                diagnostics = "\n".join(readiness["diagnostics"])
                self.assertIn(extension_id, diagnostics)
                self.assertNotIn("https://huggingface.co/black-forest-labs/FLUX.1-schnell", diagnostics)
                self.assertNotIn("same Hugging Face account/token used by Modly", diagnostics)

    def test_flux_weight_acquisition_success_does_not_emit_hf_gated_guidance(self) -> None:
        class SuccessfulDownloader:
            def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "model_index.json").write_text("{}\n", encoding="utf-8")
                return local_dir

        with tempfile.TemporaryDirectory(prefix="flux-models-") as temp_dir:
            result = weights.acquire_flux_schnell_weights(
                models_dir=Path(temp_dir),
                downloader=SuccessfulDownloader(),
            )

        result_text = json.dumps(result, sort_keys=True)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["downloaded"])
        self.assertNotIn("https://huggingface.co/black-forest-labs/FLUX.1-schnell", result_text)
        self.assertNotIn("same Hugging Face account/token used by Modly", result_text)

    def test_flux_weight_acquisition_rejects_partial_download_without_check_file(self) -> None:
        class PartialDownloader:
            def snapshot_download(self, *, repo_id: str, local_dir: Path) -> Path:
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "README.md").write_text("partial\n", encoding="utf-8")
                return local_dir

        with tempfile.TemporaryDirectory(prefix="flux-models-") as temp_dir:
            expected_target = Path(temp_dir) / "flux-schnell" / "text-to-image"
            with self.assertRaises(weights.FluxWeightPartialDownloadError) as raised:
                weights.acquire_flux_schnell_weights(
                    models_dir=Path(temp_dir),
                    downloader=PartialDownloader(),
                )

        message = str(raised.exception)
        self.assertIn("partial", message.lower())
        self.assertIn(str(expected_target / "model_index.json"), message)

    def test_vendored_runtime_matches_canonical_runtime_sources(self) -> None:
        for relative_name in (
            "__init__.py",
            "descriptors.py",
            "dependencies.py",
            "diffusers_memory.py",
            "quality_policy.py",
            "install_contract.py",
            "lifecycle.py",
            "pipeline.py",
            "runtime_adapter.py",
            "inference_runner.py",
            "weights.py",
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

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVIDENCE_KINDS = frozenset(
    {
        "validation",
        "asset_identity",
        "install_repair",
        "model_load",
        "generate_no_download",
        "gpu_generation",
        "installed_parity",
    }
)
EVIDENCE_RESULTS = frozenset({"pass", "fail", "candidate"})
REQUIRED_FIELDS = (
    "kind",
    "feature",
    "platform",
    "ref",
    "version",
    "commands",
    "assets",
    "packages",
    "outputs",
    "logs",
    "result",
)
PROMOTION_REQUIRED_KINDS = ("install_repair", "gpu_generation")
SDXL_STYLE_PROMOTION_REQUIRED_KINDS = (
    "asset_identity",
    "install_repair",
    "model_load",
    "generate_no_download",
    "gpu_generation",
    "installed_parity",
)
SDXL_STYLE_REFERENCE_FEATURE = "sdxl_ip_adapter_style"
SD15_STYLE_DISCOVERY_REQUIRED_KINDS = (
    "asset_identity",
    "install_repair",
    "model_load",
    "generate_no_download",
    "gpu_generation",
    "installed_parity",
)
SD15_STYLE_REFERENCE_FEATURE = "sd15_ip_adapter_style"
LOCAL_SMOKE_PACKAGE_FIELDS = ("python", "cuda", "torch", "diffusers")


@dataclass(frozen=True)
class SmokeEvidenceValidation:
    valid: bool
    payload: dict[str, Any]
    missing_fields: tuple[str, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class FeaturePromotionEvidenceResult:
    ready: bool
    feature: str
    observed_kinds: tuple[str, ...]
    missing_kinds: tuple[str, ...]
    diagnostics: tuple[str, ...]
    status: str = "ready"


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_smoke_evidence(payload: dict[str, Any]) -> SmokeEvidenceValidation:
    diagnostics: list[str] = []
    missing_fields = tuple(field for field in REQUIRED_FIELDS if field not in payload)

    if missing_fields:
        diagnostics.append(f"missing required evidence field(s): {', '.join(missing_fields)}")

    for field in ("kind", "feature", "platform", "ref", "version", "result"):
        if field in payload and not _is_non_empty_string(payload.get(field)):
            diagnostics.append(f"evidence field '{field}' must be a non-empty string.")

    kind = payload.get("kind")
    if _is_non_empty_string(kind) and str(kind).strip() not in EVIDENCE_KINDS:
        diagnostics.append(f"unsupported evidence kind '{kind}'.")

    result = payload.get("result")
    if _is_non_empty_string(result) and str(result).strip() not in EVIDENCE_RESULTS:
        diagnostics.append(f"unsupported evidence result '{result}'.")

    for field in ("commands", "assets", "outputs", "logs"):
        if field in payload and not isinstance(payload.get(field), list):
            diagnostics.append(f"evidence field '{field}' must be a JSON list.")
    if "packages" in payload and not isinstance(payload.get("packages"), dict):
        diagnostics.append("evidence field 'packages' must be a JSON object.")

    return SmokeEvidenceValidation(
        valid=not diagnostics,
        payload=dict(payload),
        missing_fields=missing_fields,
        diagnostics=tuple(diagnostics),
    )


def evaluate_feature_promotion_evidence(
    evidence_records: list[dict[str, Any]] | tuple[dict[str, Any], ...], *, feature: str
) -> FeaturePromotionEvidenceResult:
    valid_records = [
        record
        for record in evidence_records
        if validate_smoke_evidence(record).valid
        and record.get("feature") == feature
        and record.get("result") == "pass"
    ]
    observed_kinds = tuple(sorted({str(record["kind"]) for record in valid_records}))
    missing_kinds = tuple(kind for kind in PROMOTION_REQUIRED_KINDS if kind not in observed_kinds)
    diagnostics = (
        (f"Feature '{feature}' is missing promotion evidence kind(s): {', '.join(missing_kinds)}.",)
        if missing_kinds
        else ()
    )
    return FeaturePromotionEvidenceResult(
        ready=not missing_kinds,
        feature=feature,
        observed_kinds=observed_kinds,
        missing_kinds=missing_kinds,
        diagnostics=diagnostics,
        status="ready" if not missing_kinds else "partial",
    )


def evaluate_sdxl_style_reference_local_smoke_evidence(
    evidence_records: list[dict[str, Any]] | tuple[dict[str, Any], ...]
) -> FeaturePromotionEvidenceResult:
    feature = SDXL_STYLE_REFERENCE_FEATURE
    validations = [(record, validate_smoke_evidence(record)) for record in evidence_records]
    invalid_diagnostics = tuple(
        " ".join(validation.diagnostics) for _, validation in validations if not validation.valid
    )
    scoped_records = [record for record, validation in validations if validation.valid and record.get("result") == "pass"]
    excluded_scope_diagnostics = _excluded_sdxl_style_scope_diagnostics(scoped_records)
    valid_records = [
        record
        for record in scoped_records
        if record.get("feature") == feature
        and not _is_windows_platform(record.get("platform"))
        and str(record.get("scope", "")).strip() != "public-release"
    ]
    observed_kinds = tuple(sorted({str(record["kind"]) for record in valid_records}))
    satisfied_kinds = _satisfied_sdxl_style_promotion_kinds(valid_records)
    missing_kinds = tuple(kind for kind in SDXL_STYLE_PROMOTION_REQUIRED_KINDS if kind not in satisfied_kinds)
    diagnostics: list[str] = []
    diagnostics.extend(invalid_diagnostics)
    diagnostics.extend(excluded_scope_diagnostics)

    if "validation" in observed_kinds and missing_kinds:
        diagnostics.append(
            "validation-only evidence is not local-only SDXL style-reference smoke evidence; "
            "separate install/repair readiness/local asset evidence and GPU generation output/log evidence are required."
        )

    install_repair_records = [record for record in valid_records if record.get("kind") == "install_repair"]
    gpu_generation_records = [record for record in valid_records if record.get("kind") == "gpu_generation"]
    parity_records = [record for record in valid_records if record.get("kind") == "installed_parity"]

    install_ready = any(_has_install_repair_local_smoke_fields(record) for record in install_repair_records)
    gpu_ready = any(_has_gpu_generation_local_smoke_fields(record) for record in gpu_generation_records)
    parity_ready = any(_has_installed_parity_fields(record) for record in parity_records)

    if not install_ready:
        diagnostics.append(
            "missing install/repair local smoke evidence with readiness='ready', non-empty commands/logs, "
            "local asset paths/checks, and Python/CUDA/torch/diffusers package details."
        )
    if not gpu_ready:
        diagnostics.append(
            "missing GPU generation local smoke evidence with local_files_only=True, non-empty output path(s), "
            "log(s), command(s), local asset paths/checks, and Python/CUDA/torch/diffusers package details."
        )
    if not parity_ready:
        diagnostics.append(
            "missing installed parity evidence; repo-runtime validation/no-download evidence is partial only and cannot promote installed readiness."
        )
    if any(str(record.get("scope", "")).strip() == "repo-runtime" for record in valid_records) and missing_kinds:
        diagnostics.append("repo-runtime evidence is diagnostic/partial only until installed parity and GPU generation gates pass.")

    status = _sdxl_style_status(valid_records=valid_records, missing_kinds=missing_kinds, diagnostics=diagnostics)

    return FeaturePromotionEvidenceResult(
        ready=status == "ready",
        feature=feature,
        observed_kinds=observed_kinds,
        missing_kinds=missing_kinds,
        diagnostics=tuple(diagnostics),
        status=status,
    )


def evaluate_sd15_style_reference_discovery_evidence(
    evidence_records: list[dict[str, Any]] | tuple[dict[str, Any], ...]
) -> FeaturePromotionEvidenceResult:
    feature = SD15_STYLE_REFERENCE_FEATURE
    validations = [(record, validate_smoke_evidence(record)) for record in evidence_records]
    scoped_records = [record for record, validation in validations if validation.valid and record.get("result") == "pass"]
    excluded_scope_diagnostics = _excluded_sd15_style_scope_diagnostics(scoped_records)
    valid_records = [
        record
        for record in scoped_records
        if record.get("feature") == feature
        and not _is_windows_platform(record.get("platform"))
        and str(record.get("scope", "")).strip() != "public-release"
    ]
    observed_kinds = tuple(sorted({str(record["kind"]) for record in valid_records}))
    missing_kinds = tuple(kind for kind in SD15_STYLE_DISCOVERY_REQUIRED_KINDS if kind not in observed_kinds)
    diagnostics: list[str] = [
        " ".join(validation.diagnostics) for _, validation in validations if not validation.valid
    ]
    diagnostics.extend(excluded_scope_diagnostics)
    asset_records = [record for record in valid_records if record.get("kind") == "asset_identity"]
    assets = [asset for record in valid_records for asset in _record_assets(record)]

    if not asset_records:
        diagnostics.append("missing SD1.5 asset identity evidence; promotion is blocked before model load.")
    if not assets:
        diagnostics.append("missing SD1.5 adapter and image encoder asset metadata; promotion is blocked before model load.")
    if any(_asset_has_unknown_revision(asset) for asset in assets) or not assets:
        diagnostics.append("unknown revision blocks SD1.5 IP-Adapter promotion before model load.")
    if not any(_asset_path_mentions(asset, "ip-adapter") for asset in assets):
        diagnostics.append("missing SD1.5 IP-Adapter adapter asset metadata before model load.")
    if not any(_asset_path_mentions(asset, "image_encoder") for asset in assets):
        diagnostics.append("missing SD1.5 image encoder asset metadata before model load.")
    incomplete_assets = [asset for asset in assets if not _asset_has_identity_shape(asset)]
    if incomplete_assets:
        diagnostics.append(
            "SD1.5 asset identity evidence must include local path, repo/source, exact revision, size, and SHA256 for every asset."
        )
    if any(
        record.get("kind") in {"model_load", "generate_no_download"} and record.get("local_files_only") is not True
        for record in valid_records
    ):
        diagnostics.append("SD1.5 loader and Generate evidence must use local_files_only=True.")

    if missing_kinds:
        diagnostics.append(
            f"Feature '{feature}' is missing promotion evidence kind(s): {', '.join(missing_kinds)}."
        )

    blocking_diagnostics = tuple(
        diagnostic
        for diagnostic in diagnostics
        if "unknown revision" in diagnostic
        or "missing SD1.5 adapter" in diagnostic
        or "missing SD1.5 image encoder" in diagnostic
        or "must include local path" in diagnostic
        or "local_files_only=True" in diagnostic
    )
    if valid_records and not missing_kinds and not diagnostics:
        status = "ready"
    elif valid_records and assets and not blocking_diagnostics:
        status = "partial"
    else:
        status = "blocked"
    return FeaturePromotionEvidenceResult(
        ready=status == "ready",
        feature=feature,
        observed_kinds=observed_kinds,
        missing_kinds=missing_kinds,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
        status=status,
    )


def _record_assets(record: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_assets = record.get("assets")
    if not isinstance(raw_assets, list):
        return ()
    return tuple(asset for asset in raw_assets if isinstance(asset, dict))


def _asset_has_unknown_revision(asset: dict[str, Any]) -> bool:
    revision = str(asset.get("revision", "")).strip().casefold()
    return revision in {"", "unknown"}


def _asset_path_mentions(asset: dict[str, Any], needle: str) -> bool:
    haystack = " ".join(str(asset.get(key, "")) for key in ("path", "relative_path", "id")).casefold()
    return needle.casefold() in haystack


def _asset_has_identity_shape(asset: dict[str, Any]) -> bool:
    path = str(asset.get("path", "")).strip()
    repo = str(asset.get("repo", asset.get("source", ""))).strip()
    revision = str(asset.get("revision", "")).strip()
    sha256 = str(asset.get("sha256", "")).strip()
    size = asset.get("size")
    return bool(path and repo and revision and revision != "unknown" and sha256 and isinstance(size, int) and size > 0)


def _excluded_sd15_style_scope_diagnostics(records: list[dict[str, Any]]) -> tuple[str, ...]:
    diagnostics: list[str] = []
    for record in records:
        record_feature = str(record.get("feature", "")).strip()
        platform = record.get("platform")
        scope = str(record.get("scope", "")).strip()
        if record_feature == SD15_STYLE_REFERENCE_FEATURE and _is_windows_platform(platform):
            diagnostics.append("excluded scope: Windows evidence cannot promote Linux SD1.5 style-reference readiness.")
        if record_feature == SD15_STYLE_REFERENCE_FEATURE and scope == "public-release":
            diagnostics.append("excluded scope: public release readiness is not claimed by SD1.5 style-reference smoke evidence.")
        if record_feature == SD15_STYLE_REFERENCE_FEATURE:
            continue
        if "controlnet" in record_feature.casefold():
            diagnostics.append(f"excluded scope: ControlNet evidence '{record_feature}' is not SD1.5 IP-Adapter style promotion evidence.")
        elif "sdxl" in record_feature.casefold():
            diagnostics.append(f"excluded scope: SDXL IP-Adapter evidence '{record_feature}' is not promoted by this SD1.5 lane.")
        elif record_feature in {"ip_adapter", "generic_ip_adapter"}:
            diagnostics.append("excluded scope: generic IP-Adapter evidence is not scoped to sd15_ip_adapter_style.")
        elif record_feature:
            diagnostics.append(f"excluded scope: feature '{record_feature}' is not sd15_ip_adapter_style.")
    return tuple(dict.fromkeys(diagnostics))


def _satisfied_sdxl_style_promotion_kinds(records: list[dict[str, Any]]) -> set[str]:
    satisfied = {str(record["kind"]) for record in records}
    if any(_has_asset_checks(record) for record in records):
        satisfied.add("asset_identity")
    if any(record.get("local_files_only") is True and _has_asset_checks(record) for record in records):
        satisfied.add("model_load")
        satisfied.add("generate_no_download")
    if any(_has_install_repair_local_smoke_fields(record) for record in records):
        satisfied.add("install_repair")
    if any(_has_gpu_generation_local_smoke_fields(record) for record in records):
        satisfied.add("gpu_generation")
    if any(_has_installed_parity_fields(record) for record in records):
        satisfied.add("installed_parity")
    return satisfied


def _excluded_sdxl_style_scope_diagnostics(records: list[dict[str, Any]]) -> tuple[str, ...]:
    diagnostics: list[str] = []
    for record in records:
        record_feature = str(record.get("feature", "")).strip()
        platform = record.get("platform")
        scope = str(record.get("scope", "")).strip()
        if record_feature == SDXL_STYLE_REFERENCE_FEATURE and _is_windows_platform(platform):
            diagnostics.append("excluded scope: Windows evidence cannot promote Linux SDXL style-reference readiness.")
        if record_feature == SDXL_STYLE_REFERENCE_FEATURE and scope == "public-release":
            diagnostics.append("excluded scope: public release readiness is not claimed by SDXL style-reference smoke evidence.")
        if record_feature == SDXL_STYLE_REFERENCE_FEATURE:
            continue
        if "controlnet" in record_feature.casefold():
            diagnostics.append(f"excluded scope: ControlNet evidence '{record_feature}' is not SDXL IP-Adapter style promotion evidence.")
        elif record_feature in {"ip_adapter", "generic_ip_adapter"}:
            diagnostics.append("excluded scope: generic IP-Adapter evidence is not scoped to sdxl_ip_adapter_style.")
        elif "sd15" in record_feature.casefold():
            diagnostics.append(f"excluded scope: SD1.5 IP-Adapter evidence '{record_feature}' is not promoted by this lane.")
        elif record_feature:
            diagnostics.append(f"excluded scope: feature '{record_feature}' is not sdxl_ip_adapter_style.")
    return tuple(dict.fromkeys(diagnostics))


def _is_windows_platform(value: Any) -> bool:
    return isinstance(value, str) and "windows" in value.casefold()


def _sdxl_style_status(
    *, valid_records: list[dict[str, Any]], missing_kinds: tuple[str, ...], diagnostics: list[str]
) -> str:
    if not missing_kinds and not diagnostics:
        return "ready"
    if not valid_records:
        return "blocked"
    if diagnostics and all("excluded scope" in diagnostic for diagnostic in diagnostics):
        return "blocked"
    return "partial"


def write_sdxl_style_reference_smoke_evidence(
    path: str | Path,
    *,
    install_repair: dict[str, Any],
    gpu_generation: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
) -> FeaturePromotionEvidenceResult:
    evidence_records = [record for record in (validation, install_repair, gpu_generation) if record is not None]
    write_smoke_evidence(path, evidence_records)
    stored_records = read_smoke_evidence(path)
    return evaluate_sdxl_style_reference_local_smoke_evidence(stored_records)


def _has_install_repair_local_smoke_fields(record: dict[str, Any]) -> bool:
    return (
        _is_non_empty_sequence(record.get("commands"))
        and _is_non_empty_sequence(record.get("logs"))
        and _has_local_asset_evidence(record)
        and _has_required_package_evidence(record)
        and str(record.get("readiness", "")).strip() == "ready"
    )


def _has_gpu_generation_local_smoke_fields(record: dict[str, Any]) -> bool:
    return (
        _is_non_empty_sequence(record.get("commands"))
        and _is_non_empty_sequence(record.get("logs"))
        and _is_non_empty_sequence(record.get("outputs"))
        and _has_asset_checks(record)
        and _has_required_package_evidence(record)
        and record.get("local_files_only") is True
    )


def _has_installed_parity_fields(record: dict[str, Any]) -> bool:
    return (
        record.get("kind") == "installed_parity"
        and str(record.get("parity", "")).strip() in {"fresh", "pass", "ready"}
        and _is_non_empty_sequence(record.get("commands"))
        and _is_non_empty_sequence(record.get("logs"))
    )


def _has_local_asset_evidence(record: dict[str, Any]) -> bool:
    return _has_asset_checks(record) and _is_non_empty_sequence(record.get("local_assets"))


def _has_asset_checks(record: dict[str, Any]) -> bool:
    return _is_non_empty_sequence(record.get("assets"))


def _has_required_package_evidence(record: dict[str, Any]) -> bool:
    packages = record.get("packages")
    return isinstance(packages, dict) and all(_is_non_empty_string(packages.get(field)) for field in LOCAL_SMOKE_PACKAGE_FIELDS)


def _is_non_empty_sequence(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def write_smoke_evidence(path: str | Path, evidence_records: list[dict[str, Any]]) -> None:
    validations = [validate_smoke_evidence(record) for record in evidence_records]
    invalid = [validation for validation in validations if not validation.valid]
    if invalid:
        messages = "; ".join(" ".join(validation.diagnostics) for validation in invalid)
        raise ValueError(f"Cannot persist invalid smoke evidence: {messages}")

    evidence_path = Path(path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence_records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_smoke_evidence(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Smoke evidence file must contain a JSON list.")
    records = [record for record in payload if isinstance(record, dict)]
    if len(records) != len(payload):
        raise ValueError("Smoke evidence records must be JSON objects.")
    return records

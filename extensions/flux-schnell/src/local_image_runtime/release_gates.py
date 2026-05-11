from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


UNRELATED_RELEASE_FILES = frozenset({".gitignore"})


@dataclass(frozen=True)
class WorkspacePreflightResult:
    ready: bool
    repo_root: str
    expected_repo_root: str
    branch: str
    ahead_count: int
    release_evidence_files: tuple[str, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class PreviousReleaseRefResult:
    ready: bool
    ref_name: str
    commit: str
    diagnostics: tuple[str, ...]


def _normalize_repo_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _clean_file_list(paths: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    return tuple(sorted({path.strip() for path in paths if isinstance(path, str) and path.strip()}))


def _release_evidence_files(
    *, staged_files: tuple[str, ...], untracked_files: tuple[str, ...]
) -> tuple[str, ...]:
    candidates = (*staged_files, *untracked_files)
    return tuple(path for path in _clean_file_list(candidates) if path not in UNRELATED_RELEASE_FILES)


def evaluate_workspace_preflight(
    *,
    repo_root: str | Path,
    expected_repo_root: str | Path,
    branch: str,
    ahead_count: int,
    staged_files: tuple[str, ...] = (),
    unstaged_files: tuple[str, ...] = (),
    untracked_files: tuple[str, ...] = (),
) -> WorkspacePreflightResult:
    """Evaluate release-preflight safety without mutating git state."""

    resolved_repo = _normalize_repo_path(repo_root)
    resolved_expected = _normalize_repo_path(expected_repo_root)
    normalized_staged = _clean_file_list(staged_files)
    normalized_unstaged = _clean_file_list(unstaged_files)
    normalized_untracked = _clean_file_list(untracked_files)
    diagnostics: list[str] = []

    if resolved_repo != resolved_expected:
        diagnostics.append(
            f"Workspace is outside expected repository: {resolved_repo} != {resolved_expected}."
        )
    if branch != "main":
        diagnostics.append(f"Release preflight expected branch 'main' but found '{branch}'.")
    if ahead_count < 0:
        diagnostics.append("Ahead count cannot be negative.")
    if ".gitignore" in normalized_staged:
        diagnostics.append(".gitignore must not be staged or included in release evidence.")
    if ".gitignore" in normalized_unstaged:
        diagnostics.append(".gitignore preserved as unrelated local working-tree change.")

    release_files = _release_evidence_files(
        staged_files=normalized_staged,
        untracked_files=normalized_untracked,
    )
    blocking = [message for message in diagnostics if not message.startswith(".gitignore preserved")]
    return WorkspacePreflightResult(
        ready=not blocking,
        repo_root=str(resolved_repo),
        expected_repo_root=str(resolved_expected),
        branch=branch,
        ahead_count=ahead_count,
        release_evidence_files=release_files,
        diagnostics=tuple(diagnostics),
    )


def _candidate_ref_keys(ref_name: str) -> tuple[str, ...]:
    stripped = ref_name.strip()
    return (
        stripped,
        f"refs/heads/{stripped}",
        f"refs/tags/{stripped}",
    )


def evaluate_previous_release_ref(
    *, ref_name: str, target_commit: str, refs: dict[str, str]
) -> PreviousReleaseRefResult:
    """Verify a local branch/tag-like ref preserves the archived release commit."""

    normalized_target = target_commit.strip()
    observed_commit = ""
    for key in _candidate_ref_keys(ref_name):
        value = refs.get(key)
        if isinstance(value, str) and value.strip():
            observed_commit = value.strip()
            break

    diagnostics: list[str] = []
    if not observed_commit:
        diagnostics.append(
            f"Previous release ref '{ref_name}' is missing; create it locally at {normalized_target} before promotion."
        )
    elif observed_commit != normalized_target:
        diagnostics.append(
            f"Previous release ref '{ref_name}' points to {observed_commit}, expected {normalized_target}."
        )

    return PreviousReleaseRefResult(
        ready=not diagnostics,
        ref_name=ref_name,
        commit=observed_commit or normalized_target,
        diagnostics=tuple(diagnostics),
    )

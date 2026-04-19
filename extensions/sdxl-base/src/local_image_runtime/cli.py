from __future__ import annotations

import argparse
import json
import sys

from .bootstrap import (
    InvalidExtensionSourceError,
    RuntimeOperationError,
    UnknownExtensionError,
    bootstrap_runtime,
    extension_runtime_status,
    install_extension_from_local_dir,
)
from .install_contract import run_install_setup_contract


def build_extension_setup_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("bootstrap", help="Create runtime folders and state files.")
    subparsers.add_parser("status", help="Show this extension runtime ownership status.")

    install_parser = subparsers.add_parser(
        "install",
        help="Install this extension from a local directory.",
    )
    install_parser.add_argument(
        "--source-dir",
        required=True,
        help="Local directory containing the model assets for this extension.",
    )
    return parser


def emit_json(payload: dict[str, object], *, exit_code: int = 0) -> int:
    print(json.dumps(payload))
    return exit_code


def _read_stdin_payload() -> str | None:
    stdin = sys.stdin
    if stdin is None or stdin.isatty():
        return None
    payload = stdin.read()
    stripped = payload.strip()
    return stripped or None


def run_extension_setup_cli(*, extension_id: str, description: str) -> int:
    argv = sys.argv[1:]
    stdin_payload = _read_stdin_payload()
    if stdin_payload is not None or (argv and argv[0] not in {"bootstrap", "status", "install"}):
        result = run_install_setup_contract(
            extension_id=extension_id,
            argv=argv,
            stdin_text=stdin_payload,
        )
        return emit_json(result.to_dict(), exit_code=result.exit_code)

    parser = build_extension_setup_parser(description=description)
    args = parser.parse_args()

    try:
        if args.command in (None, "bootstrap"):
            snapshot = bootstrap_runtime(extension_id=extension_id)
            return emit_json(
                {
                    "status": "ok",
                    "command": "bootstrap",
                    "extension_id": extension_id,
                    "runtime": extension_runtime_status(snapshot, extension_id),
                    "message": f"Runtime bootstrap complete for '{extension_id}'.",
                }
            )

        if args.command == "status":
            snapshot = bootstrap_runtime(extension_id=extension_id)
            return emit_json(
                {
                    "status": "ok",
                    "command": "status",
                    "extension_id": extension_id,
                    "runtime": extension_runtime_status(snapshot, extension_id),
                    "message": f"Runtime status loaded for '{extension_id}'.",
                }
            )

        if args.command == "install":
            snapshot = bootstrap_runtime(extension_id=extension_id)
            result = install_extension_from_local_dir(snapshot, extension_id, args.source_dir)
            return emit_json(
                {
                    "status": "ok",
                    "command": "install",
                    "extension_id": result.extension_id,
                    "source_dir": str(result.source_dir),
                    "destination_dir": str(result.destination_dir),
                    "required_paths": list(result.required_paths),
                    "runtime": extension_runtime_status(result.snapshot, extension_id),
                    "message": f"Installed extension '{result.extension_id}' from local source.",
                }
            )

        parser.error(f"Unknown command '{args.command}'.")
    except (UnknownExtensionError, InvalidExtensionSourceError, RuntimeOperationError) as exc:
        return emit_json(
            {
                "status": "error",
                "command": args.command or "bootstrap",
                "extension_id": extension_id,
                "message": str(exc),
            },
            exit_code=1,
        )

    return 1

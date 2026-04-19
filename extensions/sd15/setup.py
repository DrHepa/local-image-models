from __future__ import annotations

import os
import sys
from pathlib import Path


EXTENSION_ID = "sd15"
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CANONICAL_RUNTIME = ROOT.parent.parent / "shared" / "runtime"

for import_root in (SRC, CANONICAL_RUNTIME):
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ.setdefault("LOCAL_IMAGE_RUNTIME_EXTENSION_ROOT", str(ROOT))

from local_image_runtime.cli import run_extension_setup_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        run_extension_setup_cli(
            extension_id=EXTENSION_ID,
            description="Bootstrap and manage the SD15 local image runtime.",
        )
    )

from __future__ import annotations

import os
import sys
from pathlib import Path


EXTENSION_ID = "sdxl-base"
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CANONICAL_RUNTIME = ROOT.parent.parent / "shared" / "runtime"

for import_root in (SRC, CANONICAL_RUNTIME):
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ.setdefault("LOCAL_IMAGE_RUNTIME_EXTENSION_ROOT", str(ROOT))

from local_image_runtime.runtime_adapter import (  # noqa: E402
    create_generator_class,
    run_generator_main,
)


SDXLBaseGenerator = create_generator_class(
    "SDXLBaseGenerator",
    extension_id=EXTENSION_ID,
    runtime_root=str(ROOT),
)


def main() -> int:
    return run_generator_main(extension_id=EXTENSION_ID, runtime_root=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

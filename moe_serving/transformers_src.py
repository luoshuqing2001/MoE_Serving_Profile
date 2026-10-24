"""Import the vendored Transformers 5.20.0 tree before the environment copy."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = (
    Path(__file__).resolve().parents[1] / "transformers-5.20.0" / "src"
)
REQUIRED_VERSION = "5.20.0"


def ensure_transformers_5200() -> Path:
    """Put Transformers 5.20.0 at the front of ``sys.path`` and check the version."""
    if not (SOURCE_ROOT / "transformers" / "models" / "qwen4_exp" / "modeling_qwen4_exp.py").is_file():
        raise FileNotFoundError(f"Transformers {REQUIRED_VERSION} source is missing at {SOURCE_ROOT}")
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    loaded = sys.modules.get("transformers")
    if loaded is not None:
        origin = Path(loaded.__file__).resolve()
        if SOURCE_ROOT.resolve() not in origin.parents:
            raise RuntimeError(
                f"transformers was already imported from {origin}. "
                "Import moe_serving.transformers_src before any other transformers import."
            )
    import transformers

    if transformers.__version__ != REQUIRED_VERSION:
        raise RuntimeError(f"expected transformers {REQUIRED_VERSION}, imported {transformers.__version__}")
    return SOURCE_ROOT

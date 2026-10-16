"""Import the vendored Transformers 5.19.0 tree before the environment copy."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

SOURCE_ROOT = (
    Path(__file__).resolve().parents[1] / "third_party" / "transformers-5.19.0" / "src"
)
REQUIRED_VERSION = "5.19.0"


class _TorchvisionStub(types.ModuleType):
    """Stand-in for a torchvision build whose C++ ops do not load.

    This torch nightly has no ``torchvision::nms`` registration, so importing
    the real package raises during its fake-op setup. Text routing never calls
    into torchvision; the stub only lets Transformers finish importing.
    """

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        submodule = _TorchvisionStub(f"{self.__name__}.{name}")
        setattr(self, name, submodule)
        sys.modules[submodule.__name__] = submodule
        return submodule


def _install_torchvision_stub_if_needed() -> None:
    if "torchvision" in sys.modules and not isinstance(sys.modules["torchvision"], _TorchvisionStub):
        return
    try:
        import torchvision  # noqa: F401
    except Exception:
        for key in list(sys.modules):
            if key == "torchvision" or key.startswith("torchvision."):
                del sys.modules[key]
        stub = _TorchvisionStub("torchvision")
        stub.__path__ = []
        stub.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None, is_package=True)
        sys.modules["torchvision"] = stub


def ensure_transformers_5190() -> Path:
    """Put Transformers 5.19.0 at the front of ``sys.path`` and check the version."""
    if not (SOURCE_ROOT / "transformers" / "models" / "qwen4_exp" / "modeling_qwen4_exp.py").is_file():
        raise FileNotFoundError(f"Transformers {REQUIRED_VERSION} source is missing at {SOURCE_ROOT}")
    _install_torchvision_stub_if_needed()
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

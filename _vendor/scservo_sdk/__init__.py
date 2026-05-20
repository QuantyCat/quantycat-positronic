"""Shim package that reuses the existing pure-Python scservo_sdk install.

This avoids polluting OpenPI's Python 3.11 environment with the entire
Python 3.13 site-packages tree from the RynnVLA environment.
"""

from __future__ import annotations

from pathlib import Path


_EXTERNAL_PACKAGE = Path(
    "/home/caroline/miniconda3/envs/rynnvla002/lib/python3.13/site-packages/scservo_sdk"
)

if not _EXTERNAL_PACKAGE.is_dir():
    raise ImportError(f"External scservo_sdk package not found: {_EXTERNAL_PACKAGE}")

__file__ = str(_EXTERNAL_PACKAGE / "__init__.py")
__path__ = [str(_EXTERNAL_PACKAGE)]
__package__ = __name__

with open(__file__, "rb") as _f:
    exec(compile(_f.read(), __file__, "exec"))

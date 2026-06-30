#!/usr/bin/env python3
"""Launch the shared OpenPI high-motion eval with the Dacha task prompt."""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path


def main() -> int:
    positronic_repo = Path(os.environ.get("QUANTYCAT_POSITRONIC_REPO", "/home/caroline/quantycat-positronic"))
    config_path = positronic_repo / "models/openpi/vendor_patches/src/quantycat_training_config.py"
    eval_dir = positronic_repo / "models/openpi/eval/core_evals"

    spec = importlib.util.spec_from_file_location("quantycat_training_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["quantycat_training_config"] = module

    sys.path.insert(0, str(eval_dir))

    import relative_eval

    relative_eval.PROMPT = os.environ.get(
        "DACHA_EVAL_PROMPT",
        "pick up the white square and put it in the cup",
    )
    return int(relative_eval.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Diagnose whether the local OpenPI runtime can use an NVIDIA GPU."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_openpi_repo() -> Path:
    return _repo_root() / "vendor/openpi"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-repo", type=Path, default=_default_openpi_repo())
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {"cmd": cmd, "error": repr(exc)}
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _installed_versions(names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _python_runtime() -> dict[str, Any]:
    payload: dict[str, Any] = {"python": sys.version}
    try:
        import jax
        import jaxlib

        payload["jax"] = {
            "version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        }
    except Exception as exc:
        payload["jax_error"] = repr(exc)

    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:
        payload["torch_error"] = repr(exc)

    return payload


def _host_runtime() -> dict[str, Any]:
    return {
        "nvidia_smi": _run(["nvidia-smi", "-L"]),
        "lspci": _run(["lspci"]),
        "driver_module": _run(["modinfo", "nvidia"]),
        "dev_nodes": sorted(str(path) for path in Path("/dev").glob("nvidia*")),
    }


def _assessment(payload: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    host = payload["host"]
    py = payload["python_runtime"]
    packages = payload["packages"]

    if packages.get("jax-cuda12-plugin") and packages.get("jax-cuda12-pjrt"):
        notes.append("OpenPI venv includes JAX CUDA plugin packages.")
    else:
        notes.append("OpenPI venv is missing one or more JAX CUDA plugin packages.")

    if packages.get("torch") and "+cu" in (py.get("torch", {}) or {}).get("version", ""):
        notes.append("OpenPI venv includes a CUDA-enabled PyTorch build.")
    else:
        notes.append("OpenPI venv does not appear to include a CUDA-enabled PyTorch build.")

    if host["dev_nodes"]:
        notes.append(f"Host exposes NVIDIA device nodes: {', '.join(host['dev_nodes'])}.")
    else:
        notes.append("Host does not expose /dev/nvidia* device nodes.")

    if host["driver_module"].get("returncode") == 0:
        notes.append("Kernel NVIDIA driver module metadata is present.")
    else:
        notes.append("Kernel NVIDIA driver module is not installed or not discoverable via modinfo.")

    jax_backend = (py.get("jax") or {}).get("default_backend")
    torch_cuda = (py.get("torch") or {}).get("cuda_available")
    if jax_backend == "gpu":
        notes.append("JAX is currently using GPU.")
    else:
        notes.append(f"JAX is currently using {jax_backend or 'an unknown backend'}.")
    if torch_cuda:
        notes.append("PyTorch currently sees at least one CUDA device.")
    else:
        notes.append("PyTorch currently sees zero CUDA devices.")

    if (
        packages.get("jax-cuda12-plugin")
        and packages.get("jax-cuda12-pjrt")
        and packages.get("torch")
        and not host["dev_nodes"]
        and host["driver_module"].get("returncode") != 0
        and jax_backend != "gpu"
        and not torch_cuda
    ):
        notes.append(
            "Assessment: the OpenPI Python environment is GPU-capable, but the host NVIDIA driver stack is absent or non-functional, so inference is forced onto CPU."
        )
    return notes


def main() -> int:
    args = _parse_args()
    os.chdir(args.openpi_repo)

    payload = {
        "openpi_repo": str(args.openpi_repo),
        "packages": _installed_versions(
            [
                "jax",
                "jaxlib",
                "jax-cuda12-plugin",
                "jax-cuda12-pjrt",
                "torch",
                "torchvision",
                "triton",
                "nvidia-cuda-runtime-cu12",
                "nvidia-cudnn-cu12",
            ]
        ),
        "python_runtime": _python_runtime(),
        "host": _host_runtime(),
    }
    payload["assessment"] = _assessment(payload)

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

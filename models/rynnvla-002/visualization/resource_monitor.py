#!/usr/bin/env python3
"""
Monitor and log CPU, RAM, and GPU usage to CSV while training runs.

Usage:
    # Start monitor in background before launching training:
    python visualization/resource_monitor.py \\
        --output training_output/my_run/resources.csv &
    MONITOR_PID=$!

    # ... run training ...

    # Stop monitor when training is done:
    kill $MONITOR_PID

Options:
    --output    Path to output CSV (required)
    --interval  Sampling interval in seconds (default: 5)
    --gpu       GPU index to monitor (default: 0)

Dependencies:
    pip install psutil pynvml
    pynvml is optional — falls back to nvidia-smi subprocess if missing.

Output CSV columns:
    timestamp, cpu_percent, ram_used_gb, ram_total_gb,
    gpu_util_percent, gpu_mem_used_mb, gpu_mem_total_mb
"""

import argparse
import csv
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Dependency checks ─────────────────────────────────────────────────────────

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print("Warning: psutil not found — CPU/RAM monitoring disabled.", file=sys.stderr)
    print("  pip install psutil", file=sys.stderr)

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


# ── Samplers ──────────────────────────────────────────────────────────────────

def _sample_cpu_ram() -> dict:
    if not _HAS_PSUTIL:
        return {"cpu_percent": None, "ram_used_gb": None, "ram_total_gb": None}
    mem = psutil.virtual_memory()
    return {
        "cpu_percent":  round(psutil.cpu_percent(interval=None), 1),
        "ram_used_gb":  round(mem.used  / (1024 ** 3), 2),
        "ram_total_gb": round(mem.total / (1024 ** 3), 2),
    }


def _sample_gpu_nvml(handle) -> dict:
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu_util_percent": util.gpu,
        "gpu_mem_used_mb":  mem.used  // (1024 * 1024),
        "gpu_mem_total_mb": mem.total // (1024 * 1024),
    }


def _sample_gpu_smi(gpu_index: int) -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_percent": float(parts[0]),
            "gpu_mem_used_mb":  float(parts[1]),
            "gpu_mem_total_mb": float(parts[2]),
        }
    except Exception:
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None}


# ── Main loop ─────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "timestamp",
    "cpu_percent", "ram_used_gb", "ram_total_gb",
    "gpu_util_percent", "gpu_mem_used_mb", "gpu_mem_total_mb",
]

_running = True


def _handle_signal(sig, frame):
    global _running
    _running = False


def main():
    parser = argparse.ArgumentParser(description="Log system resource usage to CSV")
    parser.add_argument("--output",   required=True, help="Output CSV path")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Sampling interval in seconds (default: 5)")
    parser.add_argument("--gpu",      type=int,   default=0,
                        help="GPU index to monitor (default: 0)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up GPU handle
    gpu_handle = None
    if _HAS_NVML:
        try:
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
            name = pynvml.nvmlDeviceGetName(gpu_handle)
            print(f"GPU {args.gpu}: {name} (via pynvml)")
        except Exception as e:
            print(f"Warning: pynvml GPU handle failed ({e}), falling back to nvidia-smi")
            gpu_handle = None

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    print(f"Resource monitor started  →  {output_path}  (every {args.interval}s)")
    print("Stop with Ctrl+C or SIGTERM.\n")

    n_samples = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        # psutil's first cpu_percent call returns 0 — warm it up
        if _HAS_PSUTIL:
            psutil.cpu_percent(interval=None)

        while _running:
            row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            row.update(_sample_cpu_ram())
            if gpu_handle is not None:
                row.update(_sample_gpu_nvml(gpu_handle))
            else:
                row.update(_sample_gpu_smi(args.gpu))

            writer.writerow(row)
            f.flush()
            n_samples += 1

            # Progress to stdout every 60 samples (~5 min at default interval)
            if n_samples % 60 == 0:
                gpu_str = (
                    f"GPU {row['gpu_util_percent']}% / {row['gpu_mem_used_mb']} MB"
                    if row["gpu_util_percent"] is not None else "GPU n/a"
                )
                print(
                    f"[{row['timestamp']}]  "
                    f"CPU {row['cpu_percent']}%  "
                    f"RAM {row['ram_used_gb']} GB  "
                    f"{gpu_str}  "
                    f"({n_samples} samples)"
                )

            time.sleep(args.interval)

    print(f"\nStopped after {n_samples} samples. Saved → {output_path}")


if __name__ == "__main__":
    main()

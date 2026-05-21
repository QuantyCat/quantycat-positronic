#!/usr/bin/env python3
"""Show episode timing and frame-rate info."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO / "my_data/input_data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=7)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()

    parquet = args.dataset_root / "data/chunk-000" / f"episode_{args.episode:06d}.parquet"
    df = pd.read_parquet(parquet)
    print(f"Columns: {list(df.columns)}")
    print(f"Frames: {len(df)}")
    print()

    # Show timestamp columns if any
    for col in df.columns:
        if "time" in col.lower() or "stamp" in col.lower() or "index" in col.lower():
            print(f"{col}: {df[col].iloc[0]} ... {df[col].iloc[-1]}")

    # Show first few and last few rows for timing
    if "timestamp" in df.columns:
        ts = df["timestamp"].values
        diffs = np.diff(ts)
        print(f"\nTimestamp range: {ts[0]:.4f} → {ts[-1]:.4f}  ({ts[-1]-ts[0]:.2f}s total)")
        print(f"Frame dt: mean={diffs.mean()*1000:.1f}ms  std={diffs.std()*1000:.1f}ms  => {1/diffs.mean():.1f} fps")
        print(f"First motion (frame 165): t={ts[165]:.2f}s")
        print(f"Full episode: {ts[-1]-ts[0]:.1f}s ({len(df)} frames @ {1/diffs.mean():.1f}fps)")

if __name__ == "__main__":
    main()

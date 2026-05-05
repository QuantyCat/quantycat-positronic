#!/usr/bin/env python3
"""Audit action-token extraction from pretokenized training records.

This checks the exact token pattern consumed by
`ChameleonXLLMXForConditionalGeneration_ck_action_head.get_action_hs_label`:
an action start token, `action_dim` action-value tokens, and the action end
token. It also optionally compares decoded action tokens to the source
conversation's saved `abs_action` files, which are already target deltas for
joints 0-4 and absolute gripper targets.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

ACTION_START = 10004
ACTION_END = 15004


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        default=Path("my_data/training_pipeline/tokens/vla_data/train/record.json"),
        help="Pretokenized record.json file.",
    )
    parser.add_argument(
        "--conversation",
        type=Path,
        default=Path("my_data/training_pipeline/conversations/libero_screwdriver_his_1_train_img_state_abs_ck_1_256_weighted.json"),
        help="Optional source conversation JSON aligned by record id.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("my_data/training_pipeline/min_max_action.txt"),
        help="Action min/max stats used for token normalization.",
    )
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _parse_stats_file(path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mins: list[float] = []
    maxs: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if "|" not in line:
                continue
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if len(nums) < 5:
                continue
            mins.append(float(nums[1]))
            maxs.append(float(nums[2]))
            if len(mins) == action_dim:
                break
    if len(mins) != action_dim:
        raise SystemExit(f"expected {action_dim} action stat rows in {path}, got {len(mins)}")
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _decode_tokens(tokens: list[int]) -> np.ndarray:
    bins = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    centers = (bins[:-1] + bins[1:]) / 2.0
    idx = np.asarray(tokens, dtype=np.int64) - 1 - ACTION_START
    idx = np.clip(idx - 1, 0, len(centers) - 1)
    return centers[idx]


def _norm_action(action: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (action - mins) / (maxs - mins + 1e-8) - 1.0, -1.0, 1.0)


def _find_action_sequences(labels: list[int], action_dim: int) -> list[tuple[int, list[int]]]:
    sequences: list[tuple[int, list[int]]] = []
    last_start = len(labels) - action_dim
    for start in range(max(0, last_start)):
        if labels[start] == ACTION_START and labels[start + action_dim + 1] == ACTION_END:
            sequences.append((start + 1, labels[start + 1 : start + 1 + action_dim]))
    return sequences


def _load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    args = _parse_args()
    record_path = args.record.resolve()
    record = _load_json(record_path)
    conversations = _load_json(args.conversation.resolve()) if args.conversation and args.conversation.exists() else None
    mins, maxs = _parse_stats_file(args.stats_file.resolve(), args.action_dim)

    rows = []
    mismatch_count = 0
    source_compare_count = 0
    source_max_abs_errors = []
    count_hist: dict[int, int] = {}

    for record_row in record[: args.limit]:
        pkl_path = Path(record_row["file"])
        item = _load_pkl(pkl_path)
        labels = list(item["label"])
        sequences = _find_action_sequences(labels, args.action_dim)
        count_hist[len(sequences)] = count_hist.get(len(sequences), 0) + 1

        expected_count = 5
        mismatch = len(sequences) != expected_count
        decoded = np.asarray([_decode_tokens(tokens) for _start, tokens in sequences], dtype=np.float32)
        source_error = None

        if conversations is not None and len(sequences) == expected_count:
            conv = conversations[int(record_row.get("id", item.get("id", 0)))]
            source = np.asarray([np.load(path).astype(np.float32) for path in conv["action"]], dtype=np.float32)
            source_norm = _norm_action(source, mins, maxs)
            source_error = float(np.max(np.abs(decoded - source_norm)))
            source_max_abs_errors.append(source_error)
            source_compare_count += 1

        if mismatch or (source_error is not None and source_error > 0.01):
            mismatch_count += 1
            rows.append(
                {
                    "record_id": int(record_row.get("id", item.get("id", -1))),
                    "file": str(pkl_path),
                    "sequence_count": len(sequences),
                    "source_max_abs_error": source_error,
                    "decoded_first": np.round(decoded[0], 6).tolist() if len(decoded) else None,
                }
            )

    summary = {
        "record": str(record_path),
        "checked": min(args.limit, len(record)),
        "action_dim": args.action_dim,
        "sequence_count_histogram": {str(k): v for k, v in sorted(count_hist.items())},
        "mismatch_count": mismatch_count,
        "source_compare_count": source_compare_count,
        "source_max_abs_error_max": float(max(source_max_abs_errors)) if source_max_abs_errors else None,
        "source_max_abs_error_mean": float(np.mean(source_max_abs_errors)) if source_max_abs_errors else None,
        "examples": rows[:20],
    }

    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

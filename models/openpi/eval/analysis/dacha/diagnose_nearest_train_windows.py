#!/usr/bin/env python3
"""Find nearest training windows for weak holdout windows in Dacha v5."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowRecord:
    episode: int
    start: int
    end: int
    state: np.ndarray
    delta: np.ndarray
    features: dict[str, np.ndarray]


def _parse_episodes(value: str) -> list[int]:
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return list(dict.fromkeys(episodes))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", required=True)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--joint", type=int, default=4)
    parser.add_argument("--rank-by", choices=("full", "state", "delta", "joint"), default="full")
    parser.add_argument("--slope-threshold", type=float, default=0.6)
    parser.add_argument("--sample-count", type=int, default=10)
    return parser.parse_args()


def _load_episode(dataset_root: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    df = pd.read_parquet(path)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)[:, :5]
    action = np.stack(df["action"].to_numpy()).astype(np.float32)[:, :5]
    return state, action - state


def _resample(arr: np.ndarray, sample_count: int) -> np.ndarray:
    if len(arr) == sample_count:
        return arr
    xs = np.linspace(0, len(arr) - 1, sample_count)
    lo = np.floor(xs).astype(int)
    hi = np.ceil(xs).astype(int)
    frac = (xs - lo).reshape(-1, 1)
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def _features(state: np.ndarray, delta: np.ndarray, sample_count: int, joint: int) -> dict[str, np.ndarray]:
    state_sample = _resample(state, sample_count).reshape(-1)
    delta_sample = _resample(delta, sample_count).reshape(-1)
    joint_sample = _resample(delta[:, joint : joint + 1], sample_count).reshape(-1)
    stats = np.concatenate(
        [
            state[0],
            state[-1] - state[0],
            delta.mean(axis=0),
            delta.std(axis=0),
            delta.min(axis=0),
            delta.max(axis=0),
        ]
    )
    return {
        "full": np.concatenate([stats, state_sample, delta_sample]).astype(np.float32),
        "state": np.concatenate([state[0], state[-1] - state[0], state_sample]).astype(np.float32),
        "delta": np.concatenate([delta.mean(axis=0), delta.std(axis=0), delta_sample]).astype(np.float32),
        "joint": np.concatenate(
            [
                np.array(
                    [
                        delta[:, joint].mean(),
                        delta[:, joint].std(),
                        delta[:, joint].min(),
                        delta[:, joint].max(),
                    ],
                    dtype=np.float32,
                ),
                joint_sample,
            ]
        ).astype(np.float32),
    }


def _make_window(
    dataset_root: Path,
    episode: int,
    start: int,
    window_size: int,
    sample_count: int,
    joint: int,
) -> WindowRecord:
    state, delta = _load_episode(dataset_root, episode)
    end = start + window_size - 1
    if start < 0 or end >= len(state):
        raise ValueError(f"episode {episode} cannot provide window {start}-{end}; length={len(state)}")
    state_window = state[start : end + 1]
    delta_window = delta[start : end + 1]
    return WindowRecord(
        episode=episode,
        start=start,
        end=end,
        state=state_window,
        delta=delta_window,
        features=_features(state_window, delta_window, sample_count, joint),
    )


def _build_train_bank(
    dataset_root: Path,
    episodes: list[int],
    window_size: int,
    stride: int,
    sample_count: int,
    joint: int,
) -> list[WindowRecord]:
    bank: list[WindowRecord] = []
    for episode in episodes:
        state, delta = _load_episode(dataset_root, episode)
        max_start = len(state) - window_size
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            end = start + window_size - 1
            state_window = state[start : end + 1]
            delta_window = delta[start : end + 1]
            bank.append(
                WindowRecord(
                    episode=episode,
                    start=start,
                    end=end,
                    state=state_window,
                    delta=delta_window,
                    features=_features(state_window, delta_window, sample_count, joint),
                )
            )
    return bank


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_flat = x.reshape(-1).astype(np.float64)
    y_flat = y.reshape(-1).astype(np.float64)
    var = float(np.var(x_flat))
    if var <= 1e-12:
        return float("nan")
    return float(np.cov(x_flat, y_flat, bias=True)[0, 1] / var)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x_flat = x.reshape(-1).astype(np.float64)
    y_flat = y.reshape(-1).astype(np.float64)
    if float(np.std(x_flat)) <= 1e-12 or float(np.std(y_flat)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_flat, y_flat)[0, 1])


def _window_metrics(query: WindowRecord, candidate: WindowRecord, joint: int) -> dict[str, float]:
    qj = query.delta[:, joint]
    cj = candidate.delta[:, joint]
    return {
        "j4_delta_corr": _corr(qj, cj),
        "j4_delta_slope_candidate_vs_query": _fit_slope(qj, cj),
        "j4_delta_mae": float(np.mean(np.abs(qj - cj))),
        "j4_query_span": float(np.max(qj) - np.min(qj)),
        "j4_candidate_span": float(np.max(cj) - np.min(cj)),
        "all_delta_mae": float(np.mean(np.abs(query.delta - candidate.delta))),
        "state_mae": float(np.mean(np.abs(query.state - candidate.state))),
    }


def _weak_cases(summary: dict[str, Any], joint: int, threshold: float) -> list[dict[str, Any]]:
    joint_key = f"joint_{joint}"
    cases: list[dict[str, Any]] = []
    for case in summary["cases"]:
        slope = case["focus_joints"][joint_key]["normalized_fit_slope"]
        if slope < threshold:
            cases.append(case)
    return cases


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_episodes = _parse_episodes(args.train_episodes)
    train_bank = _build_train_bank(
        dataset_root,
        train_episodes,
        args.window_size,
        args.stride,
        args.sample_count,
        args.joint,
    )
    if not train_bank:
        raise RuntimeError("No train windows found.")

    candidate_features = np.stack([record.features[args.rank_by] for record in train_bank]).astype(np.float32)
    candidate_mean = candidate_features.mean(axis=0)
    candidate_std = candidate_features.std(axis=0)
    candidate_std[candidate_std < 1e-6] = 1.0
    candidate_norm = (candidate_features - candidate_mean) / candidate_std

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "summary": str(summary_path),
        "train_episodes": train_episodes,
        "train_window_count": len(train_bank),
        "window_size": args.window_size,
        "stride": args.stride,
        "joint": args.joint,
        "slope_threshold": args.slope_threshold,
        "queries": [],
    }

    for case in _weak_cases(summary, args.joint, args.slope_threshold):
        query = _make_window(
            dataset_root,
            int(case["episode_index"]),
            int(case["start_step"]),
            args.window_size,
            args.sample_count,
            args.joint,
        )
        query_norm = (query.features[args.rank_by] - candidate_mean) / candidate_std
        distances = np.linalg.norm(candidate_norm - query_norm.reshape(1, -1), axis=1)
        nearest_indices = np.argsort(distances)[: args.top_k]

        query_slope = case["focus_joints"][f"joint_{args.joint}"]["normalized_fit_slope"]
        query_record: dict[str, Any] = {
            "episode_index": query.episode,
            "start_step": query.start,
            "end_step": query.end,
            "eval_j4_slope": query_slope,
            "eval_j4_corr": case["focus_joints"][f"joint_{args.joint}"]["normalized_same_corr"],
            "rank_by": args.rank_by,
            "neighbors": [],
        }
        for rank, candidate_index in enumerate(nearest_indices, start=1):
            candidate = train_bank[int(candidate_index)]
            metrics = _window_metrics(query, candidate, args.joint)
            row = {
                "query_episode": query.episode,
                "query_start": query.start,
                "query_end": query.end,
                "query_eval_j4_slope": query_slope,
                "rank": rank,
                "neighbor_episode": candidate.episode,
                "neighbor_start": candidate.start,
                "neighbor_end": candidate.end,
                "feature_distance": float(distances[int(candidate_index)]),
                **metrics,
            }
            rows.append(row)
            query_record["neighbors"].append(row)
        payload["queries"].append(query_record)

    json_out = output_dir / f"{summary_path.parent.name}_{args.rank_by}_nearest_train_windows.json"
    csv_out = output_dir / f"{summary_path.parent.name}_{args.rank_by}_nearest_train_windows.csv"
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"train_windows={len(train_bank)}")
    print(f"weak_queries={len(payload['queries'])}")
    print(f"json={json_out}")
    print(f"csv={csv_out}")
    for query in payload["queries"]:
        first = query["neighbors"][0]
        print(
            "query "
            f"ep{query['episode_index']:06d}:{query['start_step']}-{query['end_step']} "
            f"slope={query['eval_j4_slope']:.3f} nearest="
            f"ep{first['neighbor_episode']:06d}:{first['neighbor_start']}-{first['neighbor_end']} "
            f"dist={first['feature_distance']:.2f} "
            f"j4_corr={first['j4_delta_corr']:.3f} "
            f"j4_slope={first['j4_delta_slope_candidate_vs_query']:.3f} "
            f"j4_mae={first['j4_delta_mae']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

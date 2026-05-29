#!/usr/bin/env python3
"""Run high-motion evals against a LeRobot pi05 LoRA checkpoint.

Mirrors openpi_lerobot_high_motion_eval.py but loads the lerobot LoRA adapter
instead of an openpi checkpoint. Action convention is identical: joints 0-4 are
target-current deltas, gripper remains absolute.

Run with the lerobot venv:
    vendor/lerobot/.venv/bin/python models/lerobot/eval/lerobot_lora_high_motion_eval.py \
        --checkpoint ~/quantycat-data/checkpoints/lerobot/pi05/<run>/checkpoints/<step>/pretrained_model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
OPENPI_EVAL_DIR = REPO / "models/openpi/eval/model_eval"
DATA_HOME = Path(os.environ.get("QUANTYCAT_DATA_HOME", str(Path.home() / "quantycat-data")))
DEFAULT_DATASET = DATA_HOME / "datasets/screwdriver_so101"
DEFAULT_CHECKPOINT = (
    DATA_HOME
    / "checkpoints/lerobot/pi05/05282026_1657_pi05_lerobot/checkpoints/010000/pretrained_model"
)
PROMPT = "Put the screwdriver into the cup"
FOCUS_JOINTS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4")


@dataclass(frozen=True)
class EpisodeData:
    index: int
    parquet: Path
    front_video: Path
    wrist_video: Path
    states: np.ndarray
    actions: np.ndarray
    global_indices: np.ndarray  # maps per-episode step → global video frame index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to pretrained_model/ directory (contains adapter_config.json)")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--label", default=None,
                        help="Output label (default: derived from checkpoint path)")
    parser.add_argument("--joints", default="0,1,2,3,4")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=20)
    parser.add_argument("--sign-eps", type=float, default=0.01)
    parser.add_argument("--output-root", type=Path,
                        default=DATA_HOME / "eval_output/lerobot/pi05")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-traces", action="store_true")
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def _episode_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _load_episodes(dataset_root: Path) -> list[EpisodeData]:
    # v3.0 format: all episodes consolidated into chunk-*/file-*.parquet
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset_root / 'data'}")

    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"])

    # v3.0: one consolidated video per camera
    front_video = dataset_root / "videos/observation.images.front/chunk-000/file-000.mp4"
    wrist_video = dataset_root / "videos/observation.images.wrist/chunk-000/file-000.mp4"
    if not front_video.is_file():
        raise FileNotFoundError(front_video)
    if not wrist_video.is_file():
        raise FileNotFoundError(wrist_video)

    rows: list[EpisodeData] = []
    for episode_idx, group in df.groupby("episode_index"):
        episode = int(episode_idx)
        group = group.sort_values("frame_index")
        states = np.stack(group["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(group["action"].to_numpy()).astype(np.float32)
        global_indices = group["index"].to_numpy().astype(np.int64)
        rows.append(EpisodeData(episode, parquet_files[0], front_video, wrist_video,
                                states, actions, global_indices))
    return rows


def _gt_chunk(ep: EpisodeData, step: int, horizon: int) -> np.ndarray:
    """GT achieved-delta: joints 0-4 as future_state - current_state, gripper absolute."""
    if step < 0 or step + horizon > len(ep.actions):
        raise ValueError(f"episode {ep.index} step {step} cannot provide horizon {horizon}")
    state = ep.states[step]
    chunk = ep.actions[step : step + horizon, :6].copy()
    chunk[:, :5] -= state[:5].reshape(1, 5)
    return chunk.astype(np.float32)


def _step_scores(ep: EpisodeData, horizon: int, joint: int | None) -> np.ndarray:
    max_step = len(ep.actions) - horizon + 1
    scores = np.zeros(max_step, dtype=np.float32)
    for step in range(max_step):
        chunk = _gt_chunk(ep, step, horizon)
        if joint is None:
            scores[step] = float(np.mean(np.abs(chunk[:, :5])))
        else:
            scores[step] = float(np.mean(np.abs(chunk[:, joint])))
    return scores


def _find_windows(
    episodes: list[EpisodeData], joint: int, args: argparse.Namespace, output_dir: Path
) -> list[dict[str, Any]]:
    windows_json = output_dir / "selected_windows.json"
    if windows_json.exists() and not args.force:
        return json.loads(windows_json.read_text(encoding="utf-8"))["windows"]

    rows: list[dict[str, Any]] = []
    for ep in episodes:
        scores = _step_scores(ep, args.action_horizon, joint)
        if len(scores) < args.window_size:
            continue
        for start in range(0, len(scores) - args.window_size + 1):
            end = start + args.window_size - 1
            rows.append({
                "episode_index": ep.index,
                "episode": str(ep.parquet),
                "start_step": start,
                "end_step": end,
                "max_steps": args.window_size,
                f"joint_{joint}_window_score": float(np.mean(scores[start : end + 1])),
            })
    rows.sort(key=lambda row: row[f"joint_{joint}_window_score"], reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        if row["episode_index"] in seen:
            continue
        selected.append(row)
        seen.add(row["episode_index"])
        if len(selected) >= args.top_k:
            break
    if len(selected) < args.top_k:
        raise ValueError(f"Only found {len(selected)} unique episode windows for joint {joint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    windows_json.write_text(
        json.dumps({
            "dataset_root": str(args.dataset_root),
            "window_size": args.window_size,
            "action_horizon": args.action_horizon,
            "rank_metric": f"mean_abs_achieved_delta_joint_{joint}_over_chunk_and_window",
            "windows": selected,
        }, indent=2),
        encoding="utf-8",
    )
    return selected


def _load_frame(video_path: Path, frame_index: int) -> np.ndarray:
    import av
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        time_base = float(stream.time_base)
        seek_secs = max(0.0, frame_index / fps - 10.0)
        container.seek(int(seek_secs / time_base), stream=stream)
        last_frame = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_num = round(frame.pts * time_base * fps)
            last_frame = frame
            if frame_num >= frame_index:
                return frame.to_ndarray(format="rgb24")
        if last_frame is not None:
            return last_frame.to_ndarray(format="rgb24")
    raise ValueError(f"Could not read frame {frame_index} from {video_path}")


def _load_policy(checkpoint_dir: Path, device: str = "cuda"):
    """Load lerobot pi05 LoRA policy and preprocessor from a pretrained_model/ directory."""
    import torch
    from peft import PeftConfig, PeftModel
    from safetensors import safe_open
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401 - registers PI0.5 processor steps
    from lerobot.processor.pipeline import DataProcessorPipeline

    from lerobot.configs.policies import PreTrainedConfig

    _log(f"Loading LoRA adapter config from {checkpoint_dir}")
    peft_config = PeftConfig.from_pretrained(str(checkpoint_dir))
    _log(f"Base model: {peft_config.base_model_name_or_path}")

    # Load policy config from our checkpoint (has our camera names, not pi05_base defaults)
    policy_config = PreTrainedConfig.from_pretrained(str(checkpoint_dir))
    policy = PI05Policy.from_pretrained(peft_config.base_model_name_or_path, config=policy_config)
    policy = PeftModel.from_pretrained(policy, str(checkpoint_dir), config=peft_config)
    # Cast after PEFT wrapping so LoRA adapters + base are both bfloat16
    policy = policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()
    _log(f"Policy dtype: {next(policy.parameters()).dtype}")
    _log("Policy loaded")

    preprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device}},
    )
    _log("Preprocessor loaded")

    stats_file = checkpoint_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    with safe_open(str(stats_file), framework="pt") as f:
        action_mean = f.get_tensor("action.mean").float().numpy()
        action_std = f.get_tensor("action.std").float().numpy()
    _log(f"Action stats loaded: mean={np.round(action_mean, 4)}, std={np.round(action_std, 4)}")

    return policy, preprocessor, action_mean, action_std


def _policy_delta(
    policy,
    preprocessor,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    ep: EpisodeData,
    step: int,
    horizon: int,
    device: str,
) -> np.ndarray:
    """Run policy and return achieved-delta actions: relative for joints 0-4, absolute for gripper."""
    import torch

    state = ep.states[step, :6].astype(np.float32)
    global_frame = int(ep.global_indices[step])
    front_img = _load_frame(ep.front_video, global_frame)   # [H, W, C] uint8
    wrist_img = _load_frame(ep.wrist_video, global_frame)   # [H, W, C] uint8

    # Build single-sample batch (preprocessor's to_batch_processor will add batch dim)
    batch = {
        "observation.state": torch.from_numpy(state),                              # [6]
        "observation.images.front": torch.from_numpy(front_img).permute(2, 0, 1).float() / 255.0,  # [C,H,W]
        "observation.images.wrist": torch.from_numpy(wrist_img).permute(2, 0, 1).float() / 255.0,  # [C,H,W]
        "task": PROMPT,
    }

    processed = preprocessor(batch)

    # Cast float tensors to bfloat16 to match model dtype
    processed = {
        k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in processed.items()
    }

    device_type = torch.device(device).type
    autocast_enabled = device_type == "cuda"
    with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
        # Returns [1, chunk_size, action_dim] in MEAN_STD normalized space
        actions_norm = policy.predict_action_chunk(processed)

    actions_norm = actions_norm[0, :horizon].cpu().float().numpy()  # [horizon, 6]

    # Unnormalize from MEAN_STD → absolute target positions
    actions = actions_norm * action_std[:6] + action_mean[:6]

    # Convert to delta space to match GT (action - current_state for joints 0-4, gripper stays absolute)
    actions[:, :5] -= state[:5].reshape(1, 5)
    return actions


def _action_bounds(episodes: list[EpisodeData], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    mins = np.full(6, np.inf, dtype=np.float32)
    maxs = np.full(6, -np.inf, dtype=np.float32)
    for ep in episodes:
        for step in range(len(ep.actions) - horizon + 1):
            chunk = _gt_chunk(ep, step, horizon)
            mins = np.minimum(mins, np.min(chunk, axis=0))
            maxs = np.maximum(maxs, np.max(chunk, axis=0))
    return mins, maxs


def _joint_focus(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for joint in FOCUS_JOINTS:
        convention = summary["convention_checks"][joint]
        norm_convention = summary["normalized_convention_checks"][joint]
        result[joint] = {
            "raw_sign_agreement": summary["per_joint"]["sign_agreement"][joint],
            "raw_sign_count": summary["per_joint"]["sign_count"][joint],
            "raw_same_corr": convention["same_joint_corr"],
            "raw_negated_corr": convention["negated_same_joint_corr"],
            "raw_fit_slope": convention["fit_same_joint"]["slope"],
            "normalized_mae": summary["normalized_per_joint"]["mean_abs_error"][joint],
            "normalized_sign_agreement": summary["normalized_per_joint"]["sign_agreement"][joint],
            "normalized_centered_sign_agreement": summary["normalized_centered_per_joint"]["sign_agreement"][joint],
            "normalized_centered_sign_center": summary["normalized_centered_per_joint"]["sign_center"][joint],
            "normalized_same_corr": norm_convention["same_joint_corr"],
            "normalized_negated_corr": norm_convention["negated_same_joint_corr"],
            "normalized_fit_slope": norm_convention["fit_same_joint"]["slope"],
        }
    return result


def _fmt(value: Any) -> str:
    return "nan" if value is None else f"{value:.3f}"


def _print_focus(title: str, focus: dict[str, dict[str, Any]]) -> None:
    print(f"\n{title}", flush=True)
    for joint in FOCUS_JOINTS:
        row = focus[joint]
        print(
            f"  {joint}: sign={_fmt(row['raw_sign_agreement'])} "
            f"corr={_fmt(row['raw_same_corr'])} neg_corr={_fmt(row['raw_negated_corr'])} "
            f"slope={_fmt(row['raw_fit_slope'])} norm_mae={_fmt(row['normalized_mae'])}",
            flush=True,
        )


def _eval_joint(
    joint: int,
    output_dir: Path,
    windows: list[dict[str, Any]],
    episodes_by_index: dict[int, EpisodeData],
    policy,
    preprocessor,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    report = output_dir / "focused_high_motion_joint_sign.json"
    if report.exists() and not args.force:
        _log(f"report already exists for j{joint}: {report}")
        return report
    if report.exists():
        report.unlink()

    sys.path.insert(0, str(OPENPI_EVAL_DIR))
    import episode_batch_eval as batch_eval
    import episode_step_eval as step_eval

    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    cases_out: list[dict[str, Any]] = []
    trace_case_ids: list[str] = []
    trace_case_starts: list[int] = []

    for case_index, window in enumerate(windows, start=1):
        ep = episodes_by_index[int(window["episode_index"])]
        steps = list(range(window["start_step"], window["end_step"] + 1))
        _log(f"j{joint} case {case_index}/{len(windows)} episode_{ep.index:06d}: steps {steps[0]}-{steps[-1]}")
        pred_steps: list[np.ndarray] = []
        gt_steps: list[np.ndarray] = []

        for idx, step in enumerate(steps, start=1):
            gt = _gt_chunk(ep, step, args.action_horizon)
            pred = _policy_delta(policy, preprocessor, action_mean, action_std,
                                  ep, step, args.action_horizon, args.device)
            pred_steps.append(pred)
            gt_steps.append(gt)
            if idx == 1 or idx == len(steps) or idx % 10 == 0:
                metrics = step_eval._metrics(pred, gt)
                _log(f"  [{idx}/{len(steps)}] step={step} mean_abs={metrics['mean_abs']:.6f}")

        pred_arr = np.asarray(pred_steps, dtype=np.float32)
        gt_arr = np.asarray(gt_steps, dtype=np.float32)
        summary = batch_eval._summarize(pred_arr, gt_arr, args.sign_eps, min_values, max_values)
        summary["action_convention"] = (
            "lerobot LoRA achieved-delta eval: joints 0-4 compare policy target-current "
            "against stored future achieved state-current; gripper remains absolute."
        )
        focus = _joint_focus(summary)
        _print_focus(f"episode_{ep.index:06d} focus", focus)
        all_pred.append(pred_arr)
        all_gt.append(gt_arr)
        trace_case_ids.append(f"episode_{ep.index:06d}")
        trace_case_starts.append(steps[0])
        cases_out.append({
            "episode_index": ep.index,
            "episode": str(ep.parquet),
            "start_step": steps[0],
            "end_step": steps[-1],
            "step_count": len(steps),
            "summary": summary,
            "focus_joints": focus,
        })

    aggregate_pred = np.concatenate(all_pred, axis=0)
    aggregate_gt = np.concatenate(all_gt, axis=0)
    aggregate_summary = batch_eval._summarize(
        aggregate_pred, aggregate_gt, args.sign_eps, min_values, max_values
    )
    aggregate_summary["action_convention"] = (
        "lerobot LoRA achieved-delta eval: joints 0-4 compare policy target-current "
        "against stored future achieved state-current; gripper remains absolute."
    )
    aggregate_focus = _joint_focus(aggregate_summary)
    _print_focus(f"j{joint} aggregate focus", aggregate_focus)

    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "action_convention": aggregate_summary["action_convention"],
        "sign_eps": args.sign_eps,
        "case_count": len(cases_out),
        "total_step_count": int(aggregate_pred.shape[0]),
        "chunk_size": int(aggregate_pred.shape[1]),
        "action_dim": int(aggregate_pred.shape[2]),
        "aggregate_summary": aggregate_summary,
        "aggregate_focus_joints": aggregate_focus,
        "cases": cases_out,
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"saved focused report: {report}")

    if args.save_traces:
        trace_path = output_dir / "focused_high_motion_traces.npz"
        np.savez_compressed(
            trace_path,
            pred=np.asarray(all_pred, dtype=np.float32),
            gt=np.asarray(all_gt, dtype=np.float32),
            case_id=np.asarray(trace_case_ids),
            case_start_step=np.asarray(trace_case_starts, dtype=np.int32),
        )
        _log(f"saved trace arrays: {trace_path}")
    return report


def _summarize(
    label_dir: Path,
    reports: dict[int, Path],
    checkpoint: Path,
    dataset_root: Path,
) -> Path:
    rows: dict[str, Any] = {}
    for ranked_joint, report in sorted(reports.items()):
        data = json.loads(report.read_text(encoding="utf-8"))
        focus = data["aggregate_focus_joints"]
        rows[f"broad_j{ranked_joint}_high_motion_top{data['case_count']}"] = {
            "report": str(report),
            "case_count": data["case_count"],
            "total_step_count": data["total_step_count"],
            "diagonal_joint": f"joint_{ranked_joint}",
            "diagonal_metrics": focus.get(f"joint_{ranked_joint}", {}),
            "all_focus_joints": focus,
        }
    summary = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "created_at": _stamp(),
        "coverage": rows,
    }
    out = label_dir / "all_joint_focused_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def main() -> int:
    args = _parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()

    if not (args.checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter_config.json at {args.checkpoint} — "
            "pass --checkpoint pointing to the pretrained_model/ directory"
        )
    if not (args.dataset_root / "data").is_dir():
        raise FileNotFoundError(args.dataset_root / "data")

    if args.label is None:
        run_name = args.checkpoint.parents[1].name
        step = args.checkpoint.parent.name
        args.label = f"lerobot_lora_{run_name}_{step}"

    joints = [int(item.strip()) for item in args.joints.split(",") if item.strip()]
    episodes = _load_episodes(args.dataset_root)
    episodes_by_index = {ep.index: ep for ep in episodes}
    label_dir = args.output_root.expanduser().resolve() / args.label
    label_dir.mkdir(parents=True, exist_ok=True)

    _log(f"loaded {len(episodes)} episodes from {args.dataset_root}")
    _log("computing achieved-delta action bounds")
    min_values, max_values = _action_bounds(episodes, args.action_horizon)
    (label_dir / "achieved_delta_action_bounds.json").write_text(
        json.dumps({"min": min_values.tolist(), "max": max_values.tolist()}, indent=2),
        encoding="utf-8",
    )

    _log(f"loading lerobot LoRA policy from {args.checkpoint}")
    policy, preprocessor, action_mean, action_std = _load_policy(args.checkpoint, args.device)

    reports: dict[int, Path] = {}
    for joint in joints:
        output_dir = label_dir / f"broad_j{joint}_high_motion_top{args.top_k}"
        output_dir.mkdir(parents=True, exist_ok=True)
        _log(f"j{joint}: finding high-motion windows from achieved deltas")
        windows = _find_windows(episodes, joint, args, output_dir)
        _log(f"j{joint}: running focused eval on {len(windows)} windows")
        reports[joint] = _eval_joint(
            joint, output_dir, windows, episodes_by_index,
            policy, preprocessor, action_mean, action_std,
            min_values, max_values, args,
        )

    summary = _summarize(label_dir, reports, args.checkpoint, args.dataset_root)
    _log(f"wrote all-joint eval summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

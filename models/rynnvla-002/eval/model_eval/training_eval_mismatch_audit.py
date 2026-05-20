#!/usr/bin/env python3
"""Audit train-record teacher-forced predictions against episode inference.

This script intentionally separates the two paths:
  1. pretokenized train records through the training model.forward path
  2. raw episode samples through the deployed inference/eval solver path

It also validates that checkpoint tensors are actually applied after stripping
the `module.` prefix used by training checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

import episode_batch_eval as batch_eval
import episode_step_eval as step_eval


ACTION_START = 10004
ACTION_END = 15004
JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("my_data/training_pipeline/fine_tuning/screwdriver_so101_tiny_high_motion_j123_overfit10/epoch9"),
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=Path("my_data/training_pipeline/tokens/vla_data/tiny_high_motion_j123/record.json"),
    )
    parser.add_argument(
        "--conversation",
        type=Path,
        default=Path(
            "my_data/training_pipeline/conversations/"
            "libero_screwdriver_his_1_train_img_state_abs_ck_1_256_weighted.json"
        ),
    )
    parser.add_argument("--positronic-config", default="models/rynnvla-002/config.yaml")
    parser.add_argument("--rynnvla-repo", default=os.environ.get("RYNNVLA_REPO", step_eval._default_rynnvla_repo()))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--teacher-limit", type=int, default=64)
    parser.add_argument("--paired-limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sign-eps", type=float, default=0.02)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-paired-inference", action="store_true")
    parser.add_argument("--deterministic-crop", action="store_true")
    parser.add_argument(
        "--require-prefix-match",
        action="store_true",
        help="Exit nonzero if paired episode inference tokens do not match the teacher prefix through 10004.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


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
        raise SystemExit(f"expected {action_dim} action stats rows in {path}, got {len(mins)}")
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _norm_action(action: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (np.asarray(action, dtype=np.float32) - mins) / (maxs - mins + 1e-8) - 1.0, -1.0, 1.0)


def _unnorm_action(action_norm: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    action_norm = np.clip(np.asarray(action_norm, dtype=np.float32), -1.0, 1.0)
    return (action_norm + 1.0) / 2.0 * (maxs - mins + 1e-8) + mins


def _zero_norm(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return _norm_action(np.zeros_like(mins, dtype=np.float32), mins, maxs)


def _decode_tokens(tokens: list[int]) -> np.ndarray:
    bins = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    centers = (bins[:-1] + bins[1:]) / 2.0
    idx = np.asarray(tokens, dtype=np.int64) - 1 - ACTION_START
    idx = np.clip(idx - 1, 0, len(centers) - 1)
    return centers[idx]


def _find_action_sequences(labels: list[int], action_dim: int) -> list[tuple[int, list[int]]]:
    sequences: list[tuple[int, list[int]]] = []
    for start in range(max(0, len(labels) - action_dim - 1)):
        if labels[start] == ACTION_START and labels[start + action_dim + 1] == ACTION_END:
            sequences.append((start + 1, labels[start + 1 : start + 1 + action_dim]))
    return sequences


def _joint_table(values: np.ndarray) -> dict[str, float | None]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    result: dict[str, float | None] = {}
    for idx, name in enumerate(JOINT_LABELS[: len(flat)]):
        value = float(flat[idx])
        result[name] = None if np.isnan(value) else value
    return result


def _corr_1d(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None
    centered = x - np.mean(x)
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        return None
    return float(np.dot(centered, y - np.mean(y)) / denom)


def _summarize_norm(pred: np.ndarray, gt: np.ndarray, zero: np.ndarray, sign_eps: float) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, pred.shape[-1])
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, gt.shape[-1])
    diff = pred - gt
    pred_centered = pred - zero.reshape(1, -1)
    gt_centered = gt - zero.reshape(1, -1)
    active = np.abs(gt_centered) > sign_eps
    sign_match = (np.sign(pred_centered) == np.sign(gt_centered)) & active
    counts = active.sum(axis=0).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sign_agreement = np.divide(
            sign_match.sum(axis=0),
            counts,
            out=np.full_like(counts, np.nan, dtype=np.float32),
            where=counts > 0,
        )
    corr = []
    slope = []
    for idx in range(gt.shape[1]):
        corr.append(_corr_1d(gt_centered[:, idx], pred_centered[:, idx]))
        slope.append(_fit_slope(gt_centered[:, idx], pred_centered[:, idx]))
    return {
        "count": int(pred.shape[0]),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "per_joint_mae": _joint_table(np.mean(np.abs(diff), axis=0)),
        "per_joint_bias": _joint_table(np.mean(diff, axis=0)),
        "per_joint_sign_agreement_centered": _joint_table(sign_agreement),
        "per_joint_sign_count": _joint_table(counts),
        "per_joint_corr_centered": {JOINT_LABELS[i]: corr[i] for i in range(gt.shape[1])},
        "per_joint_fit_slope_centered": {JOINT_LABELS[i]: slope[i] for i in range(gt.shape[1])},
        "pred_distribution": {
            "mean": _joint_table(np.mean(pred, axis=0)),
            "std": _joint_table(np.std(pred, axis=0)),
            "min": _joint_table(np.min(pred, axis=0)),
            "max": _joint_table(np.max(pred, axis=0)),
        },
        "gt_distribution": {
            "mean": _joint_table(np.mean(gt, axis=0)),
            "std": _joint_table(np.std(gt, axis=0)),
            "min": _joint_table(np.min(gt, axis=0)),
            "max": _joint_table(np.max(gt, axis=0)),
        },
    }


def _strip_module(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd):
        return {k.removeprefix("module."): v for k, v in sd.items()}
    return sd


def _tensor_norm(t: torch.Tensor) -> float:
    return float(t.detach().float().norm().cpu())


def _load_training_model(checkpoint: Path, args: argparse.Namespace, action_dim: int, time_horizon: int) -> tuple[Any, dict[str, Any]]:
    rynnvla_dir = Path(args.rynnvla_repo).expanduser().resolve()
    if str(rynnvla_dir) not in sys.path:
        sys.path.insert(0, str(rynnvla_dir))

    from model import ChameleonXLLMXForConditionalGeneration_ck_action_head
    from pretrain_solver_awm_w_ck_action_head import add_lora_to_model

    saved_args_path = checkpoint / "args.json"
    saved_args = _load_json(saved_args_path) if saved_args_path.is_file() else {}
    base_init_from = Path(saved_args.get("init_from", checkpoint)).expanduser()
    if not base_init_from.is_absolute():
        base_init_from = (checkpoint / base_init_from).resolve()
    tokenizer_path = saved_args.get("tokenizer_path")

    model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
        str(base_init_from),
        action_dim=action_dim,
        time_horizon=time_horizon,
        max_position_embeddings=int(saved_args.get("max_seq_len", 4096)),
        mask_image_logits=bool(saved_args.get("mask_image_logits", False)),
        dropout=float(saved_args.get("dropout", 0.0)),
        z_loss_weight=float(saved_args.get("z_loss_weight", 0.0)),
        action_sign_loss_weight=float(saved_args.get("action_sign_loss_weight", 0.0)),
        action_sign_eps=float(saved_args.get("action_sign_eps", 0.03)),
        action_sign_margin=float(saved_args.get("action_sign_margin", 0.02)),
        action_sign_joint_weights=saved_args.get("action_sign_joint_weights"),
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}",
    )

    if hasattr(model.model, "vqmodel"):
        del model.model.vqmodel
        torch.cuda.empty_cache()

    ckpt_file = checkpoint / "model.safetensors"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"model.safetensors not found: {ckpt_file}")
    raw_sd = load_file(str(ckpt_file), device="cpu")
    sd = _strip_module(raw_sd)

    lora_keys = [k for k in sd if k.endswith(".lora_weight_A")]
    if lora_keys:
        lora_r = int(sd[lora_keys[0]].shape[0])
        lora_alpha = int(saved_args.get("lora_alpha", lora_r * 2))
        add_lora_to_model(
            model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            lora_dropout=0.0,
            dtype=torch.bfloat16,
        )

    before = {name: _tensor_norm(param) for name, param in model.named_parameters()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    after_params = dict(model.named_parameters())

    groups = {
        "action_head": [k for k in sd if k.startswith("action_head.")],
        "lm_head": [k for k in sd if k.startswith("lm_head.")],
        "lora": [k for k in sd if "lora_weight_" in k],
    }
    group_reports: dict[str, Any] = {}
    for group, keys in groups.items():
        diffs = []
        applied = 0
        changed = 0
        examples = []
        for key in keys:
            param = after_params.get(key)
            if param is None:
                continue
            ckpt_tensor = sd[key].to(device=param.device, dtype=param.dtype)
            diff = float((param.detach() - ckpt_tensor).float().abs().max().cpu())
            diffs.append(diff)
            applied += 1
            before_norm = before.get(key)
            after_norm = _tensor_norm(param)
            if before_norm is None or abs(after_norm - before_norm) > 1e-7:
                changed += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "key": key,
                        "checkpoint_norm": _tensor_norm(ckpt_tensor),
                        "before_norm": before_norm,
                        "after_norm": after_norm,
                        "max_abs_diff_after_load": diff,
                    }
                )
        group_reports[group] = {
            "checkpoint_key_count": len(keys),
            "applied_key_count": applied,
            "changed_norm_count": changed,
            "max_abs_diff_after_load": max(diffs) if diffs else None,
            "mean_abs_diff_after_load": float(np.mean(diffs)) if diffs else None,
            "examples": examples,
        }

    model.eval()
    audit = {
        "checkpoint": str(checkpoint),
        "base_init_from": str(base_init_from),
        "tokenizer_path": tokenizer_path,
        "raw_checkpoint_key_count": len(raw_sd),
        "stripped_checkpoint_key_count": len(sd),
        "had_module_prefix": any(k.startswith("module.") for k in raw_sd),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_first20": list(missing[:20]),
        "unexpected_first20": list(unexpected[:20]),
        "groups": group_reports,
    }
    del raw_sd, sd
    return model, audit


def _record_source_info(record_row: dict[str, Any], conversations: list[dict[str, Any]] | None) -> dict[str, Any]:
    record_id = int(record_row.get("id", -1))
    info: dict[str, Any] = {"record_id": record_id, "file": record_row["file"]}
    if conversations is None or record_id < 0 or record_id >= len(conversations):
        return info
    conv = conversations[record_id]
    info["conversation_index"] = record_id
    info["prompt"] = conv["conversations"][0]["value"]
    info["image"] = conv.get("image")
    info["state"] = conv.get("state")
    info["action"] = conv.get("action")
    if conv.get("action"):
        action_path = Path(conv["action"][0])
        info["step"] = int(action_path.parent.name.removeprefix("action_"))
        info["episode"] = str(action_path.parents[2])
    return info


def _load_source_action_norm(info: dict[str, Any], mins: np.ndarray, maxs: np.ndarray) -> np.ndarray | None:
    paths = info.get("action")
    if not paths:
        return None
    raw = np.asarray([np.load(p).astype(np.float32) for p in paths], dtype=np.float32)
    return _norm_action(raw, mins, maxs)


def _teacher_forced_eval(
    model: Any,
    records: list[dict[str, Any]],
    conversations: list[dict[str, Any]] | None,
    args: argparse.Namespace,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    zero = _zero_norm(mins, maxs)
    selected = records[: args.teacher_limit]
    rows_by_record: dict[int, dict[str, Any]] = {}
    all_pred = []
    all_label = []
    source_errors = []
    debug_last: dict[str, float] = {}

    for batch_start in range(0, len(selected), args.batch_size):
        batch_rows = selected[batch_start : batch_start + args.batch_size]
        examples = []
        labels = []
        items = []
        for row in batch_rows:
            item = _load_pkl(Path(row["file"]))
            examples.append(list(item["token"]))
            labels.append(list(item["label"]))
            items.append(item)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _c_loss, _additional, _logits, _hidden, labels_c, predicted_actions, _loss_ct = model(
                input_ids=examples,
                labels=labels,
                output_hidden_states=True,
                training=True,
                att_mask=True,
            )

        label_tokens, sequences = model.get_action_hs_label(None, labels_c)
        label_actions = model.decode_token_ids_to_actions(label_tokens)
        pred_np = predicted_actions.detach().float().cpu().numpy()
        label_np = label_actions.detach().float().cpu().numpy()
        all_pred.append(pred_np)
        all_label.append(label_np)
        debug = getattr(model, "action_loss_debug", {})
        debug_last = {
            key: float(value.detach().float().cpu()) if torch.is_tensor(value) and value.numel() == 1 else None
            for key, value in debug.items()
        }

        per_batch: dict[int, list[int]] = {}
        for seq_idx, (batch_idx, _start) in enumerate(sequences):
            per_batch.setdefault(int(batch_idx), []).append(seq_idx)

        for batch_idx, row in enumerate(batch_rows):
            record_id = int(row.get("id", items[batch_idx].get("id", -1)))
            seq_indices = per_batch.get(batch_idx, [])
            pred_record = pred_np[seq_indices]
            label_record = label_np[seq_indices]
            info = _record_source_info(row, conversations)
            source_norm = _load_source_action_norm(info, mins, maxs)
            source_max_abs_error = None
            if source_norm is not None and source_norm.shape == label_record.shape:
                source_max_abs_error = float(np.max(np.abs(source_norm - label_record)))
                source_errors.append(source_max_abs_error)
            label_sequences = _find_action_sequences(labels[batch_idx], model.action_dim)
            rows_by_record[record_id] = {
                **info,
                "token_length": len(examples[batch_idx]),
                "label_length": len(labels[batch_idx]),
                "teacher_input_ids": list(examples[batch_idx]),
                "teacher_labels": list(labels[batch_idx]),
                "action_sequence_count": len(label_sequences),
                "action_sequence_starts": [start for start, _tokens in label_sequences],
                "action_tokens_first": label_sequences[0][1] if label_sequences else None,
                "decoded_first": _decode_tokens(label_sequences[0][1]).round(6).tolist() if label_sequences else None,
                "source_max_abs_error_vs_label": source_max_abs_error,
                "teacher_pred_norm": pred_record.tolist(),
                "teacher_label_norm": label_record.tolist(),
                "teacher_pred_raw": _unnorm_action(pred_record, mins, maxs).tolist(),
                "teacher_label_raw": _unnorm_action(label_record, mins, maxs).tolist(),
                "teacher_summary": _summarize_norm(pred_record, label_record, zero, args.sign_eps),
            }

    pred_all = np.concatenate(all_pred, axis=0)
    label_all = np.concatenate(all_label, axis=0)
    summary = {
        "record": str(args.record.resolve()),
        "checked_records": len(selected),
        "checked_action_rows": int(pred_all.shape[0]),
        "batch_size": args.batch_size,
        "action_dim": int(pred_all.shape[1]),
        "zero_norm": _joint_table(zero),
        "summary": _summarize_norm(pred_all, label_all, zero, args.sign_eps),
        "source_max_abs_error_max": max(source_errors) if source_errors else None,
        "source_max_abs_error_mean": float(np.mean(source_errors)) if source_errors else None,
        "last_action_loss_debug": debug_last,
    }
    return summary, rows_by_record


def _run_paired_inference(
    teacher_rows: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    cfg: dict[str, Any],
    root: Path,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> dict[str, Any]:
    selected_rows = [
        row for row in teacher_rows.values()
        if "episode" in row and "step" in row
    ][: args.paired_limit]
    if not selected_rows:
        return {"checked_records": 0, "reason": "no records with episode/step metadata"}

    ckpt_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir / "paired_episode_inference_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    solver_args = batch_eval._make_solver_args(args, cfg, ckpt_path, output_dir)
    solver_args.deterministic_crop = bool(args.deterministic_crop or cfg.get("deterministic_crop", False))
    solver = batch_eval._load_solver(args.rynnvla_repo, solver_args)

    zero = _zero_norm(mins, maxs)
    teacher_pred_norm_all = []
    teacher_label_norm_all = []
    infer_norm_all = []
    teacher_prefix_infer_norm_all = []
    gt_norm_all = []
    teacher_vs_infer_norm_all = []
    teacher_vs_teacher_prefix_infer_norm_all = []
    rows_out = []
    prefix_match_flags = []
    prefix_mismatch_counts = []
    his = int(cfg["his"])

    for row in selected_rows:
        episode_dir = Path(row["episode"])
        step = int(row["step"])
        sample = step_eval._build_sample(episode_dir, step=step, his=his)
        prompt = step_eval._extract_task_from_episode(episode_dir)
        step_eval._reset_solver_history(solver, sample)
        captured_input_ids: dict[str, Any] = {}
        original_generate_action_head = solver.model.generate_action_head

        def _capture_generate_action_head(input_ids, generation_config):
            ids = input_ids.detach().cpu().tolist()
            captured_input_ids["input_ids"] = ids[0] if ids else []
            captured_input_ids["shape"] = list(input_ids.shape)
            return original_generate_action_head(input_ids, generation_config)

        solver.model.generate_action_head = _capture_generate_action_head
        pred_raw = solver.get_action_wrist_action_head_state(
            front_image=sample["front_current"],
            wrist_image=sample["wrist_current"],
            state=sample["state"],
            prompt=prompt,
        )
        solver.model.generate_action_head = original_generate_action_head
        pred_raw = np.asarray(pred_raw, dtype=np.float32)
        gt_raw = np.asarray(sample["gt_action"], dtype=np.float32)
        infer_norm = _norm_action(pred_raw, mins, maxs)
        gt_norm = _norm_action(gt_raw, mins, maxs)
        teacher_pred_norm = np.asarray(row["teacher_pred_norm"], dtype=np.float32)
        teacher_label_norm = np.asarray(row["teacher_label_norm"], dtype=np.float32)
        inference_input_ids = list(captured_input_ids.get("input_ids", []))
        teacher_input_ids = list(row.get("teacher_input_ids", []))
        compare_len = min(len(inference_input_ids), len(teacher_input_ids))
        mismatch_indices = [
            idx for idx in range(compare_len)
            if inference_input_ids[idx] != teacher_input_ids[idx]
        ]
        first_action_start = row["action_sequence_starts"][0] - 1 if row.get("action_sequence_starts") else None
        expected_prefix_len = first_action_start + 1 if first_action_start is not None else None
        prefix_matches = (
            expected_prefix_len == len(inference_input_ids)
            and inference_input_ids == teacher_input_ids[:expected_prefix_len]
        )
        prefix_match_flags.append(bool(prefix_matches))
        prefix_mismatch_counts.append(len(mismatch_indices))
        teacher_prefix_pred_norm = None
        if expected_prefix_len is not None:
            teacher_prefix_ids = torch.tensor(
                teacher_input_ids[:expected_prefix_len],
                dtype=torch.int64,
                device=solver.model.device,
            ).unsqueeze(0)
            with torch.no_grad():
                teacher_prefix_pred = original_generate_action_head(teacher_prefix_ids, None)
            teacher_prefix_pred_norm = teacher_prefix_pred.detach().float().cpu().numpy()

        teacher_pred_norm_all.append(teacher_pred_norm)
        teacher_label_norm_all.append(teacher_label_norm)
        infer_norm_all.append(infer_norm)
        gt_norm_all.append(gt_norm)
        teacher_vs_infer_norm_all.append(infer_norm - teacher_pred_norm)
        if teacher_prefix_pred_norm is not None:
            teacher_prefix_infer_norm_all.append(teacher_prefix_pred_norm)
            teacher_vs_teacher_prefix_infer_norm_all.append(teacher_prefix_pred_norm - teacher_pred_norm)
        rows_out.append(
            {
                "record_id": row["record_id"],
                "episode": str(episode_dir),
                "step": step,
                "prompt": prompt,
                "token_length": row["token_length"],
                "action_sequence_starts": row["action_sequence_starts"],
                "teacher_input_ids": teacher_input_ids,
                "inference_input_ids": inference_input_ids,
                "inference_input_shape": captured_input_ids.get("shape"),
                "token_alignment": {
                    "teacher_length": len(teacher_input_ids),
                    "inference_length": len(inference_input_ids),
                    "expected_teacher_prefix_len_through_action_start": expected_prefix_len,
                    "inference_equals_teacher_prefix": prefix_matches,
                    "compared_prefix_len": compare_len,
                    "prefix_mismatch_count": len(mismatch_indices),
                    "first_prefix_mismatch": (
                        {
                            "index": mismatch_indices[0],
                            "teacher_token": teacher_input_ids[mismatch_indices[0]],
                            "inference_token": inference_input_ids[mismatch_indices[0]],
                        }
                        if mismatch_indices
                        else None
                    ),
                    "teacher_prefix_tail": teacher_input_ids[max(0, compare_len - 20) : compare_len],
                    "inference_tail": inference_input_ids[-20:],
                },
                "front_current_path": str(sample["front_current_path"]),
                "wrist_current_path": str(sample["wrist_current_path"]),
                "state_path": str(sample["state_path"]),
                "action_dir": str(sample["action_dir"]),
                "teacher_vs_label": row["teacher_summary"],
                "inference_vs_gt": _summarize_norm(infer_norm, gt_norm, zero, args.sign_eps),
                "inference_vs_teacher_pred": _summarize_norm(infer_norm, teacher_pred_norm, zero, args.sign_eps),
                "teacher_prefix_inference_vs_gt": (
                    _summarize_norm(teacher_prefix_pred_norm, gt_norm, zero, args.sign_eps)
                    if teacher_prefix_pred_norm is not None
                    else None
                ),
                "teacher_prefix_inference_vs_teacher_pred": (
                    _summarize_norm(teacher_prefix_pred_norm, teacher_pred_norm, zero, args.sign_eps)
                    if teacher_prefix_pred_norm is not None
                    else None
                ),
                "teacher_label_vs_episode_gt": _summarize_norm(teacher_label_norm, gt_norm, zero, args.sign_eps),
                "teacher_pred_norm_first": teacher_pred_norm[0].round(6).tolist(),
                "inference_pred_norm_first": infer_norm[0].round(6).tolist(),
                "teacher_prefix_inference_pred_norm_first": (
                    teacher_prefix_pred_norm[0].round(6).tolist()
                    if teacher_prefix_pred_norm is not None
                    else None
                ),
                "gt_norm_first": gt_norm[0].round(6).tolist(),
            }
        )

    teacher_pred_norm_all_np = np.asarray(teacher_pred_norm_all, dtype=np.float32)
    teacher_label_norm_all_np = np.asarray(teacher_label_norm_all, dtype=np.float32)
    infer_norm_all_np = np.asarray(infer_norm_all, dtype=np.float32)
    teacher_prefix_infer_norm_all_np = (
        np.asarray(teacher_prefix_infer_norm_all, dtype=np.float32)
        if teacher_prefix_infer_norm_all
        else None
    )
    gt_norm_all_np = np.asarray(gt_norm_all, dtype=np.float32)
    prefix_alignment = {
        "all_match": bool(prefix_match_flags and all(prefix_match_flags)),
        "match_count": int(sum(prefix_match_flags)),
        "checked_count": int(len(prefix_match_flags)),
        "mismatch_count_total": int(sum(prefix_mismatch_counts)),
        "mismatch_count_max": int(max(prefix_mismatch_counts)) if prefix_mismatch_counts else None,
    }
    if args.require_prefix_match and not prefix_alignment["all_match"]:
        first_bad = next((row for row in rows_out if not row["token_alignment"]["inference_equals_teacher_prefix"]), None)
        detail = first_bad["token_alignment"] if first_bad else prefix_alignment
        raise SystemExit(f"prefix alignment check failed: {json.dumps(detail, indent=2)}")
    return {
        "checked_records": len(rows_out),
        "deterministic_crop": bool(solver_args.deterministic_crop),
        "prefix_alignment": prefix_alignment,
        "summary": {
            "teacher_vs_label": _summarize_norm(teacher_pred_norm_all_np, teacher_label_norm_all_np, zero, args.sign_eps),
            "inference_vs_gt": _summarize_norm(infer_norm_all_np, gt_norm_all_np, zero, args.sign_eps),
            "inference_vs_teacher_pred": _summarize_norm(infer_norm_all_np, teacher_pred_norm_all_np, zero, args.sign_eps),
            "teacher_prefix_inference_vs_gt": (
                _summarize_norm(teacher_prefix_infer_norm_all_np, gt_norm_all_np, zero, args.sign_eps)
                if teacher_prefix_infer_norm_all_np is not None
                else None
            ),
            "teacher_prefix_inference_vs_teacher_pred": (
                _summarize_norm(teacher_prefix_infer_norm_all_np, teacher_pred_norm_all_np, zero, args.sign_eps)
                if teacher_prefix_infer_norm_all_np is not None
                else None
            ),
            "teacher_label_vs_episode_gt": _summarize_norm(teacher_label_norm_all_np, gt_norm_all_np, zero, args.sign_eps),
            "teacher_inference_norm_delta_abs_mean": _joint_table(
                np.mean(np.abs(np.asarray(teacher_vs_infer_norm_all, dtype=np.float32).reshape(-1, len(mins))), axis=0)
            ),
            "teacher_prefix_inference_norm_delta_abs_mean": (
                _joint_table(
                    np.mean(
                        np.abs(
                            np.asarray(teacher_vs_teacher_prefix_infer_norm_all, dtype=np.float32).reshape(-1, len(mins))
                        ),
                        axis=0,
                    )
                )
                if teacher_vs_teacher_prefix_infer_norm_all
                else None
            ),
        },
        "records": rows_out,
    }


def main() -> None:
    args = _parse_args()
    root = step_eval._repo_root()
    os.chdir(root)
    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = step_eval._load_positronic_config(cfg_path)
    work_dir = batch_eval._configure_env(root, cfg)
    mins, maxs = _parse_stats_file(work_dir / "min_max_action.txt", int(cfg["action_dim"]))

    args.checkpoint = (root / args.checkpoint.expanduser()).resolve() if not args.checkpoint.is_absolute() else args.checkpoint.expanduser().resolve()
    args.record = (root / args.record.expanduser()).resolve() if not args.record.is_absolute() else args.record.expanduser().resolve()
    args.conversation = (root / args.conversation.expanduser()).resolve() if not args.conversation.is_absolute() else args.conversation.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = (
            Path(cfg.get("training_output", "eval_output")).expanduser().resolve()
            / f"{cfg['task_label']}_{cfg['robot']}"
            / "model_eval"
            / args.checkpoint.name
            / "training_eval_mismatch_audit"
        )
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.rynnvla_repo:
        rynnvla_repo = Path(args.rynnvla_repo).expanduser()
        if not rynnvla_repo.is_absolute():
            rynnvla_repo = (root / rynnvla_repo).resolve()
        step_eval._ensure_rynnvla_on_path(str(rynnvla_repo))

    records = _load_json(args.record)
    conversations = _load_json(args.conversation) if args.conversation.is_file() else None

    print(f"Loading training model for teacher-forced audit: {args.checkpoint}")
    model, checkpoint_audit = _load_training_model(args.checkpoint, args, int(cfg["action_dim"]), int(cfg["chunk_size"]))
    print(json.dumps({"checkpoint_load_audit": checkpoint_audit["groups"]}, indent=2))

    print(f"Running teacher-forced audit on {min(args.teacher_limit, len(records))} records")
    teacher_summary, teacher_rows = _teacher_forced_eval(model, records, conversations, args, mins, maxs)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    paired_summary: dict[str, Any] | None = None
    if not args.skip_paired_inference:
        print(f"Running paired episode inference on up to {args.paired_limit} records")
        paired_summary = _run_paired_inference(teacher_rows, args, cfg, root, mins, maxs)

    payload = {
        "checkpoint": str(args.checkpoint),
        "record": str(args.record),
        "conversation": str(args.conversation),
        "positronic_config": str(cfg_path),
        "work_dir": str(work_dir),
        "action_convention": "saved action chunks are target deltas for joints 0-4 and absolute gripper targets; no eval-time state subtraction",
        "preprocessing_alignment": {
            "teacher_forced_path": "pretokenized token/label pkl records -> model.forward(output_hidden_states=True, training=True, att_mask=True)",
            "episode_inference_path": "episode images/state -> eval_solver_lerobot_action_head_state -> data_lerobot ItemProcessor -> generate_action_head",
            "deterministic_crop": bool(args.deterministic_crop or cfg.get("deterministic_crop", False)),
            "his": cfg.get("his"),
            "his_mode": cfg.get("his_mode"),
            "action_steps": cfg.get("action_steps"),
            "action_stats_file": str(work_dir / "min_max_action.txt"),
            "state_stats_file": str(work_dir / "min_max_state.txt"),
        },
        "checkpoint_load_audit": checkpoint_audit,
        "teacher_forced": teacher_summary,
        "paired": paired_summary,
        "teacher_record_examples": list(teacher_rows.values())[: min(8, len(teacher_rows))],
    }
    out_path = args.output_dir / "training_eval_mismatch_audit.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nTeacher-forced summary")
    print(json.dumps(teacher_summary["summary"], indent=2))
    if paired_summary is not None:
        print("\nPaired summary")
        print(json.dumps(paired_summary.get("summary", paired_summary), indent=2))
    print(f"\nSaved audit to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tiny overfit probe for the RynnVLA action head.

This is a diagnostic, not a deployable training run. It loads an existing
checkpoint, freezes everything except model.action_head, repeatedly trains on a
small episode window, and reports whether the action head can fit the target
actions under the current input format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import episode_step_eval as step_eval
import episode_batch_eval as batch_eval

_DEFAULT_CONFIG = "models/rynnvla-002/config.yaml"
_JOINT_LABELS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit action_head on a small episode window.")
    parser.add_argument("--episode", required=True, help="Path to episode_XXXXXX directory.")
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--positronic-config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--rynnvla-repo", type=str, default=os.environ.get("RYNNVLA_REPO", ""))
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--deterministic-crop", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def _conv_from_sample(sample: dict[str, Any], prompt: str, his: int) -> dict[str, Any]:
    front_imgs = list(sample["front_history"]) + [sample["front_current"]]
    wrist_imgs = list(sample["wrist_history"]) + [sample["wrist_current"]]
    if len(front_imgs) != his or len(wrist_imgs) != his:
        raise ValueError(f"expected {his} front/wrist images, got {len(front_imgs)} and {len(wrist_imgs)}")
    gt_action = np.asarray(sample["gt_action"], dtype=np.float32)
    return {
        "conversations": [
            {
                "from": "human",
                "value": "What action should the robot take to " + prompt + "?"
                + "<|state|>"
                + "<|image|>" * (len(front_imgs) + len(wrist_imgs)),
            },
            {
                "from": "gpt",
                "value": "<|action|>" * int(gt_action.shape[0]),
            },
        ],
        "image": front_imgs + wrist_imgs,
        "action": [gt_action[i] for i in range(gt_action.shape[0])],
        "state": np.asarray(sample["state"], dtype=np.float32),
    }


def _make_solver_args(args: argparse.Namespace, cfg: dict[str, Any], ckpt_path: Path, output_dir: Path) -> argparse.Namespace:
    import argparse as _argparse

    return _argparse.Namespace(
        resume_path=str(ckpt_path),
        output_dir=str(output_dir.resolve()),
        device=args.gpu if args.gpu is not None else cfg["gpu"],
        action_dim=cfg["action_dim"],
        time_horizon=cfg["chunk_size"],
        max_seq_len=cfg["max_seq_len"],
        mask_image_logits=cfg["mask_image_logits"],
        dropout=cfg["dropout"],
        z_loss_weight=float(cfg["inference_z_loss_weight"]),
        his=cfg["his_mode"],
        action_steps=cfg["action_steps"],
        deterministic_crop=bool(args.deterministic_crop or cfg.get("deterministic_crop", False)),
    )


def _decode_targets(model, hidden_states: torch.Tensor, labels: list[list[int]]) -> torch.Tensor:
    max_tokens = max(len(x) for x in labels)
    labels_t = [x + [-100] * (max_tokens - len(x)) for x in labels]
    labels_t = torch.tensor(labels_t, dtype=torch.int64, device=model.device)
    labels_action_dis, _ = model.get_action_hs_label(hidden_states, labels_t)
    return model.decode_token_ids_to_actions(labels_action_dis)


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    pred = pred.detach().float().reshape(-1, len(_JOINT_LABELS)).cpu()
    target = target.detach().float().reshape(-1, len(_JOINT_LABELS)).cpu()
    loss = torch.nn.functional.l1_loss(pred, target).item()
    rows = {}
    for idx, name in enumerate(_JOINT_LABELS):
        p = pred[:, idx]
        g = target[:, idx]
        rows[name] = {
            "loss": torch.nn.functional.l1_loss(p, g).item(),
            "pred_mean": p.mean().item(),
            "gt_mean": g.mean().item(),
            "pred_abs_mean": p.abs().mean().item(),
            "gt_abs_mean": g.abs().mean().item(),
            "mag_ratio": (p.abs().mean() / (g.abs().mean() + 1e-8)).item(),
            "sign_agree": (p.sign() == g.sign()).float().mean().item(),
        }
    return {"loss": loss, "per_joint": rows}


def _print_metrics(step: int, metrics: dict[str, Any]) -> None:
    print(f"\nprobe_step={step} loss={metrics['loss']:.6f}", flush=True)
    for name in _JOINT_LABELS:
        row = metrics["per_joint"][name]
        print(
            f"  {name}: loss={row['loss']:.5f} mag_ratio={row['mag_ratio']:.3f} "
            f"sign_agree={row['sign_agree']:.3f} pred_mean={row['pred_mean']:.4f} gt_mean={row['gt_mean']:.4f}",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    root = step_eval._repo_root()
    os.chdir(root)

    cfg_path = Path(args.positronic_config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = step_eval._load_positronic_config(cfg_path)
    if int(cfg["his"]) != 2 or cfg["his_mode"] != "2h_1a_img_both_wrist_state":
        raise SystemExit(
            "This probe is currently intended for the existing his=2 checkpoint. "
            f"Config has his={cfg['his']} his_mode={cfg['his_mode']!r}."
        )

    work_dir = batch_eval._configure_env(root, cfg)
    ckpt_path = step_eval._resolve_checkpoint(args, cfg)
    step_eval._ensure_rynnvla_on_path(args.rynnvla_repo.strip() or None)

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        root / cfg.get("training_output", "training_output") / f"{cfg['task_label']}_{cfg['robot']}" / "overfit_probe" / ckpt_path.name
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_args = _make_solver_args(args, cfg, ckpt_path, out_dir)
    print(f"Loading Solver from {ckpt_path} ...", flush=True)
    solver = batch_eval._load_solver(args.rynnvla_repo, solver_args)
    model = solver.model

    episode_dir = Path(args.episode).expanduser().resolve()
    prompt = args.prompt if args.prompt is not None else step_eval._extract_task_from_episode(episode_dir)
    steps = list(range(args.start_step, args.start_step + args.max_steps))

    print(f"Building {len(steps)} training samples from {episode_dir.name} steps {steps[0]}..{steps[-1]}", flush=True)
    examples: list[list[int]] = []
    labels: list[list[int]] = []
    for step in steps:
        sample = step_eval._build_sample(episode_dir, step=step, his=int(cfg["his"]))
        conv = _conv_from_sample(sample, prompt=prompt, his=int(cfg["his"]))
        token, label = solver.item_processor.process_item(conv, training_mode=True)
        examples.append(token)
        labels.append(label)

    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.action_head.parameters():
        param.requires_grad_(True)

    model.eval()
    model.action_head.train()
    optimizer = torch.optim.AdamW(model.action_head.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    n = len(examples)

    def run_batch(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_examples = [examples[i] for i in indices]
        batch_labels = [labels[i] for i in indices]
        _, _, _, hidden_states, _, pred, loss_ct = model(
            input_ids=batch_examples,
            labels=batch_labels,
            output_hidden_states=True,
            training=True,
            att_mask=True,
        )
        target = _decode_targets(model, hidden_states, batch_labels)
        return pred, target, loss_ct

    def evaluate_window() -> dict[str, Any]:
        was_training = model.action_head.training
        model.action_head.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for idx in range(n):
                pred_i, target_i, _ = run_batch([idx])
                preds.append(pred_i.detach())
                targets.append(target_i.detach())
        if was_training:
            model.action_head.train()
        return _metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))

    for train_step in range(args.train_steps + 1):
        start = (train_step * args.batch_size) % n
        indices = [(start + offset) % n for offset in range(args.batch_size)]
        pred, target, loss = run_batch(indices)

        if train_step % args.log_every == 0 or train_step == args.train_steps:
            metrics = evaluate_window()
            metrics["probe_step"] = train_step
            history.append(metrics)
            _print_metrics(train_step, metrics)

        if train_step == args.train_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), args.clip_grad)
        optimizer.step()

    if args.save_json:
        payload = {
            "checkpoint": str(ckpt_path),
            "episode": str(episode_dir),
            "prompt": prompt,
            "steps": steps,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "clip_grad": args.clip_grad,
            "config": {
                "his": cfg["his"],
                "his_mode": cfg["his_mode"],
                "action_stats_file": str(work_dir / "min_max_action.txt"),
            },
            "history": history,
        }
        out_path = out_dir / f"{episode_dir.name}_steps_{steps[0]:06d}_to_{steps[-1]:06d}_overfit_probe.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()

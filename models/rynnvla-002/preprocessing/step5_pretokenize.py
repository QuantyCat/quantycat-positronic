"""
Step 5 — Pretokenize the dataset.

Converts episode frames into discrete token .pkl files using the VQ-GAN
image tokenizer and action/state discretization bins.
Pretokenizes each available split into tokens/vla_data/<split>/.

Run from repo root:
    python3 models/rynnvla-002/preprocessing/pretokenize.py
"""

import os
import sys
import subprocess
from glob import glob
import yaml

CONFIG_PATH = "models/rynnvla-002/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

work_dir    = os.path.abspath(config["work_dir"])
label       = config["task_label"]
his         = config["his"]
resolution  = config["resolution"]
deterministic_crop = bool(config.get("deterministic_crop", False))
action_stats = os.path.join(work_dir, "min_max_action.txt")
state_stats = os.path.join(work_dir, "min_max_state.txt")

rynnvla_repo = os.path.join(os.path.expanduser("~"), "RynnVLA-002", "rynnvla-002")
if not os.path.isdir(rynnvla_repo):
    print(f"ERROR: RynnVLA-002 repo not found at {rynnvla_repo}")
    sys.exit(1)

input_file  = os.path.realpath(os.path.join(
    work_dir, "conversations",
    f"libero_{label}_his_{his}_train_img_state_abs_ck_1_{resolution}.json"
))
train_override = config.get("train_conversation_override")
if train_override:
    train_override = os.path.realpath(os.path.expanduser(train_override))
output_dir  = os.path.join(work_dir, "tokens", "vla_data")
tokenizer   = os.path.join(rynnvla_repo, "ckpts", "chameleon", "base_model")
script = os.path.join(rynnvla_repo, "data_lerobot", "pretoken_lerobot_state.py")
conversation_pattern = os.path.join(
    work_dir,
    "conversations",
    f"libero_{label}_his_{his}_*_img_state_abs_ck_1_{resolution}.json",
)
conversation_files = sorted(glob(conversation_pattern))

if not conversation_files:
    if not os.path.isfile(input_file):
        print(f"ERROR: no conversation files found matching {conversation_pattern}")
        sys.exit(1)
    conversation_files = [input_file]

split_inputs = []
train_name = os.path.basename(input_file)
for path in conversation_files:
    split_name = os.path.basename(path).split(f"libero_{label}_his_{his}_", 1)[1].split("_img_state_abs_ck_1_", 1)[0]
    actual_path = train_override if train_override and os.path.basename(path) == train_name else path
    split_inputs.append((split_name, actual_path))

os.makedirs(output_dir, exist_ok=True)
env = os.environ.copy()
env["RYNNVLA_ACTION_STATS_FILE"] = action_stats
env["RYNNVLA_STATE_STATS_FILE"] = state_stats

for split_name, conv_path in split_inputs:
    split_output_dir = os.path.join(output_dir, split_name)
    if os.path.isdir(split_output_dir):
        print(f"Tokens already exist at {split_output_dir} — skipping {split_name}. Delete to rerun.")
        continue

    cmd = [sys.executable, script,
           "--input_file", conv_path,
           "--output_dir", split_output_dir,
           "--resolution", str(resolution),
           "--tokenizer_path", tokenizer]
    if deterministic_crop:
        cmd.append("--deterministic_crop")

    result = subprocess.run(
        cmd,
        cwd=os.path.join(rynnvla_repo, "data_lerobot"),
        env=env,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)

    record_shards = glob(os.path.join(split_output_dir, "*-record.jsonl"))
    if not record_shards:
        log_dir = os.path.join(split_output_dir, "logs")
        print(f"ERROR: pretokenization finished without producing any record shards for split '{split_name}'")
        print(f"Check worker logs in {log_dir}")
        sys.exit(1)

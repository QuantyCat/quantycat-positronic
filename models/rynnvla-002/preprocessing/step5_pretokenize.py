"""
Step 5 — Pretokenize the dataset.

Converts episode frames into discrete token .pkl files using the VQ-GAN
image tokenizer and action/state discretization bins.
Skips if tokens/vla_data/ already exists.

Run from repo root:
    python3 models/rynnvla-002/preprocessing/pretokenize.py
"""

import os
import sys
import subprocess
import yaml

CONFIG_PATH = "models/rynnvla-002/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

work_dir    = os.path.abspath(config["work_dir"])
label       = config["task_label"]
his         = config["his"]
resolution  = config["resolution"]

rynnvla_repo = os.path.join(os.path.expanduser("~"), "RynnVLA-002", "rynnvla-002")
if not os.path.isdir(rynnvla_repo):
    print(f"ERROR: RynnVLA-002 repo not found at {rynnvla_repo}")
    sys.exit(1)

input_file  = os.path.realpath(os.path.join(
    work_dir, "conversations",
    f"libero_{label}_his_{his}_train_img_state_abs_ck_1_{resolution}.json"
))
output_dir  = os.path.join(work_dir, "tokens", "vla_data")
tokenizer   = os.path.join(rynnvla_repo, "ckpts", "chameleon", "base_model")

if os.path.isdir(output_dir):
    print(f"Tokens already exist at {output_dir} — skipping. Delete to rerun.")
    sys.exit(0)

script = os.path.join(rynnvla_repo, "data_lerobot", "pretoken_lerobot_state.py")
result = subprocess.run(
    [sys.executable, script,
     "--input_file",    input_file,
     "--output_dir",    output_dir,
     "--resolution",    str(resolution),
     "--tokenizer_path", tokenizer],
    cwd=os.path.join(rynnvla_repo, "data_lerobot"),
)
sys.exit(result.returncode)

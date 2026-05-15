"""
Run eval-only metrics for a saved RynnVLA-002 checkpoint against split validation data.

Usage:
    python3 models/rynnvla-002/fine_tuning/evaluate.py --checkpoint /path/to/checkpoint_dir
"""

import os
import shutil
import subprocess
import sys

import yaml

CONFIG_PATH = "models/rynnvla-002/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

if "z_loss_weight" in config:
    print("ERROR: config.yaml still uses ambiguous top-level z_loss_weight. Use train_z_loss_weight/inference_z_loss_weight.")
    sys.exit(1)

checkpoint = None
if len(sys.argv) >= 3 and sys.argv[1] == "--checkpoint":
    checkpoint = os.path.realpath(sys.argv[2])
if not checkpoint:
    checkpoint = os.path.realpath(config["checkpoint"])

work_dir       = os.path.abspath(config["work_dir"])
action_dim     = config["action_dim"]
chunk_size     = config["chunk_size"]
his            = config["his"]
resolution     = config["resolution"]
num_workers    = config["num_workers"]
z_loss_weight  = config["train_z_loss_weight"]
lora_r         = config["lora_r"]
lora_alpha     = config["lora_alpha"]
action_norm_joint_scales = config.get("action_norm_joint_scales")
action_stats   = os.path.join(work_dir, "min_max_action.txt")
state_stats    = os.path.join(work_dir, "min_max_state.txt")

home         = os.path.expanduser("~")
rynnvla_repo = os.path.join(home, "RynnVLA-002", "rynnvla-002")
train_script = os.path.join(rynnvla_repo, "pretrain_solver_awm_w_ck_action_head.py")
tokenizer    = os.path.join(rynnvla_repo, "ckpts", "chameleon", "base_model")

base_name = f"his_{his}_third_view_wrist_w_state_{chunk_size}_{resolution}_pretokenize"
data_config_train = os.path.join(rynnvla_repo, "configs", "lerobot", f"{base_name}.yaml")
data_config_val_ind = os.path.join(rynnvla_repo, "configs", "lerobot", f"{base_name}_val_ind.yaml")
data_config_val_ood = os.path.join(rynnvla_repo, "configs", "lerobot", f"{base_name}_val_ood.yaml")
if not os.path.isfile(data_config_val_ind):
    print(f"ERROR: validation config not found at {data_config_val_ind}")
    print("Run preprocessing (steps 2-7) with run_validation: true first.")
    sys.exit(1)
if not os.path.isfile(data_config_val_ood):
    data_config_val_ood = data_config_val_ind

output_dir = os.path.join(work_dir, "fine_tuning_eval", os.path.basename(checkpoint))
os.makedirs(output_dir, exist_ok=True)

torchrun_bin = os.path.join(os.path.dirname(sys.executable), "torchrun")
if not os.path.isfile(torchrun_bin):
    torchrun_bin = shutil.which("torchrun") or "torchrun"

cmd = [
    torchrun_bin,
    "--nproc_per_node=1",
    "--nnodes=1",
    "--master_addr=127.0.0.1",
    "--master_port=16667",
    train_script,
    "--eval_only", "True",
    "--resume_path", checkpoint,
    "--tokenizer_path", tokenizer,
    "--ablation", "0",
    "--model_size", "7B",
    "--batch_size", "4",
    "--accum_iter", "1",
    "--epochs", "1",
    "--warmup_epochs", "0.03",
    "--lr", "0.0",
    "--min_lr", "0.0",
    "--wd", "0.00001",
    "--clip_grad", "1.0",
    "--action_dim", str(action_dim),
    "--time_horizon", str(chunk_size),
    "--data_config_train", data_config_train,
    "--data_config_val_ind", data_config_val_ind,
    "--data_config_val_ood", data_config_val_ood,
    "--num_workers", str(num_workers),
    "--output_dir", output_dir,
    "--checkpointing",
    "--max_seq_len", "4096",
    "--unmask_image_logits",
    "--dropout", "0.08",
    "--z_loss_weight", str(z_loss_weight),
    "--lora_r", str(lora_r),
    "--lora_alpha", str(lora_alpha),
]

print(f"Running eval-only metrics for {checkpoint}")
print(f"  val_ind: {data_config_val_ind}")
print(f"  val_ood: {data_config_val_ood}")
print(f"  output:  {output_dir}")

env = os.environ.copy()
env["RYNNVLA_ACTION_STATS_FILE"] = action_stats
env["RYNNVLA_STATE_STATS_FILE"] = state_stats
if action_norm_joint_scales:
    env["RYNNVLA_ACTION_NORM_SCALES"] = str(action_norm_joint_scales)
result = subprocess.run(cmd, env=env)
sys.exit(result.returncode)

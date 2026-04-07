"""
Step 5 — Fine-tune RynnVLA-002 on your dataset.

Launches torchrun training using the pretokenized record.json.
Reads all paths from models/rynnvla-002/config.yaml.

Run from repo root:
    bash models/rynnvla-002/run_scripts/finetune.sh
"""

import os
import sys
import subprocess
import yaml
import shutil

CONFIG_PATH = "models/rynnvla-002/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

work_dir     = os.path.abspath(config["work_dir"])
task_label   = config["task_label"]
robot        = config["robot"]
action_dim   = config["action_dim"]
chunk_size   = config["chunk_size"]
batch_size   = config["batch_size"]
num_workers  = config["num_workers"]
epochs       = config["epochs"]
lr           = config["lr"]
lora_r       = config["lora_r"]
lora_alpha   = config["lora_alpha"]
home         = os.path.expanduser("~")
rynnvla_repo = os.path.join(home, "RynnVLA-002", "rynnvla-002")

if not os.path.isdir(rynnvla_repo):
    print(f"ERROR: RynnVLA-002 repo not found at {rynnvla_repo}")
    sys.exit(1)

data_config  = os.path.join(rynnvla_repo, "configs", "lerobot", "his_1_third_view_wrist_w_state_20_256_pretokenize.yaml")
init_from    = os.path.join(rynnvla_repo, "ckpts", "starting_point")
tokenizer    = os.path.join(rynnvla_repo, "ckpts", "chameleon", "base_model")
train_script = os.path.join(rynnvla_repo, "pretrain_solver_awm_w_ck_action_head.py")
output_dir   = os.path.realpath(os.path.join(work_dir, "fine_tuning", f"{task_label}_{robot}"))

os.makedirs(output_dir, exist_ok=True)

print(f"Starting fine-tune")
print(f"  Output: {output_dir}")
print()

# Resolve torchrun from the same Python environment as this script
_python_bin = os.path.dirname(sys.executable)
torchrun_bin = os.path.join(_python_bin, "torchrun")
if not os.path.isfile(torchrun_bin):
    torchrun_bin = shutil.which("torchrun") or "torchrun"

cmd = [
    torchrun_bin,
    "--nproc_per_node=1", # number of GPUs to use (1 for single-node training)
    "--nnodes=1", #number of machines to use (1 for single-node training)
    "--master_addr=127.0.0.1", # (not used for single-node training)
    "--master_port=16666", # (not used for single-node training)
    train_script,
    "--train_only", "True", #skip evaluation - only train the model
    "--disable_length_clustering",
    "--init_from", init_from, # pretrained checkpoint to initialize from
    "--tokenizer_path", tokenizer, # tokenizer to use - matches the pretrained checkpoint
    "--ablation", "0", #selects the full model configuration - 0 is the full model
    "--model_size", "7B", #  RynnVLA-002 comes in 7B and 34B. 7B fits in 32GB.
    "--batch_size", str(batch_size),   # from config.yaml
    "--accum_iter", "1",
    "--epochs", str(epochs),  # from config.yaml
    "--warmup_epochs", "0.01", # from RynnVLA-002 original paper
    "--lr", str(lr),          # from config.yaml
    "--min_lr", str(lr),      # from config.yaml
    "--wd", "0.15", # weight decay from RynnVLA-002 original paper [ 0.10 to 0.18 ]
    "--clip_grad", "4", # gradient clipping
    "--action_dim", str(action_dim),   # from config.yaml
    "--time_horizon", str(chunk_size), # from config.yaml
    "--data_config_train", data_config,
    "--data_config_val_ind", data_config,
    "--data_config_val_ood", data_config,
    "--num_workers", str(num_workers), # from config.yaml
    "--output_dir", output_dir,
    "--checkpointing",
    "--max_seq_len", "4096", #max token sequence length per sample. From original script, matches the pretokenized data.
    "--unmask_image_logits",
    "--dropout", "0.08", #regularization during training - from RynnVLA-002 original paper [ 0.05 to 0.10 ]
    "--z_loss_weight", "1e-5", #small auxiliary loss to stabilize softmax - from RynnVLA-002 original paper
    "--ckpt_max_keep", "3", # keep last 3 checkpoints.
    "--lora_r", str(lora_r),         # from config.yaml
    "--lora_alpha", str(lora_alpha), # from config.yaml
]

log_path = os.path.join(output_dir, "output.log")
print(f"  Log: {log_path}")
print()

with open(log_path, "a") as log:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
    proc.wait()

sys.exit(proc.returncode)

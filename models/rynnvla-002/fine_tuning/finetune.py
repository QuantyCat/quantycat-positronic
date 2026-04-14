"""
Fine-tune RynnVLA-002 on your dataset.

Launches torchrun training using the pretokenized record.json.
All paths and training knobs are read from models/rynnvla-002/config.yaml.

Run from repo root:
    bash models/rynnvla-002/run_scripts/finetune.sh
"""

import os
import sys
import subprocess
import shutil
import yaml

CONFIG_PATH = "models/rynnvla-002/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

work_dir       = os.path.abspath(config["work_dir"])
task_label     = config["task_label"]
robot          = config["robot"]
action_dim     = config["action_dim"]
chunk_size     = config["chunk_size"]
his            = config["his"]
resolution     = config["resolution"]
batch_size     = config["batch_size"]
accum_iter     = config["accum_iter"]
num_workers    = config["num_workers"]
epochs         = config["epochs"]
lr             = config["lr"]
min_lr         = config["min_lr"]
clip_grad      = config["clip_grad"]
z_loss_weight  = config["z_loss_weight"]
lora_r         = config["lora_r"]
lora_alpha     = config["lora_alpha"]
ckpt_max_keep  = config["ckpt_max_keep"]
save_interval  = config["save_interval"]
fresh_start    = config.get("fresh_start", False)

home         = os.path.expanduser("~")
rynnvla_repo = os.path.join(home, "RynnVLA-002", "rynnvla-002")

if not os.path.isdir(rynnvla_repo):
    print(f"ERROR: RynnVLA-002 repo not found at {rynnvla_repo}")
    sys.exit(1)

# data_config is written by step7_update_train_config.py — derived from his/chunk_size/resolution
data_config_name = f"his_{his}_third_view_wrist_w_state_{chunk_size}_{resolution}_pretokenize.yaml"
data_config  = os.path.join(rynnvla_repo, "configs", "lerobot", data_config_name)
if not os.path.isfile(data_config):
    print(f"ERROR: data config not found at {data_config}")
    print("Run preprocessing (step7) first, or check that his/chunk_size/resolution in config.yaml match.")
    sys.exit(1)

init_from    = os.path.join(rynnvla_repo, "ckpts", "starting_point")
tokenizer    = os.path.join(rynnvla_repo, "ckpts", "chameleon", "base_model")
train_script = os.path.join(rynnvla_repo, "pretrain_solver_awm_w_ck_action_head.py")
output_dir   = os.path.realpath(os.path.join(work_dir, "fine_tuning", f"{task_label}_{robot}"))

# Resume / fresh-start logic
if fresh_start:
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        print(f"  Cleared output dir for fresh start: {output_dir}")
    resume_flags = ["--no_auto_resume"]
else:
    print(f"  Auto-resuming from last checkpoint in: {output_dir}")
    resume_flags = []  # auto_resume=True is the training script default

os.makedirs(output_dir, exist_ok=True)

print(f"Starting fine-tune")
print(f"  Config:    {CONFIG_PATH}")
print(f"  Data:      {data_config}")
print(f"  Output:    {output_dir}")
print(f"  fresh_start={fresh_start}  lr={lr}  epochs={epochs}  accum_iter={accum_iter}  clip_grad={clip_grad}  z_loss_weight={z_loss_weight}")
print()

# Trainable params: lora_weight_ + action_head + lm_head
# The model lacks get_trainable_params so the trainer warns and falls back to is_lora detection.
# When lora_r > 0 the is_lora check is True and requires_grad is left exactly as add_lora_to_model
# set it — correct behaviour despite the warning.
print(f"  LoRA: r={lora_r} alpha={lora_alpha}  trainable=lora_weight_+action_head+lm_head")
print()

# Resolve torchrun from the same Python environment as this script
_python_bin = os.path.dirname(sys.executable)
torchrun_bin = os.path.join(_python_bin, "torchrun")
if not os.path.isfile(torchrun_bin):
    torchrun_bin = shutil.which("torchrun") or "torchrun"

cmd = [
    torchrun_bin,
    "--nproc_per_node=1",
    "--nnodes=1",
    "--master_addr=127.0.0.1",
    "--master_port=16666",
    train_script,
    "--train_only", "True",
    "--disable_length_clustering",
    "--init_from", init_from,
    "--tokenizer_path", tokenizer,
    "--ablation", "0",
    "--model_size", "7B",
    "--batch_size", str(batch_size),
    "--accum_iter", str(accum_iter),       # from config.yaml
    "--epochs", str(epochs),
    "--warmup_epochs", "0.03",
    "--lr", str(lr),
    "--min_lr", str(min_lr),
    "--wd", "0.00001",
    "--clip_grad", str(clip_grad),         # from config.yaml
    "--action_dim", str(action_dim),
    "--time_horizon", str(chunk_size),
    "--data_config_train", data_config,
    "--data_config_val_ind", data_config,
    "--data_config_val_ood", data_config,
    "--num_workers", str(num_workers),
    "--output_dir", output_dir,
    "--checkpointing",
    "--max_seq_len", "4096",
    "--unmask_image_logits",
    "--dropout", "0.08",
    "--z_loss_weight", str(z_loss_weight), # from config.yaml
    "--ckpt_max_keep", str(ckpt_max_keep),
    "--save_iteration_interval", str(save_interval),
    "--lora_r", str(lora_r),
    "--lora_alpha", str(lora_alpha),
] + resume_flags

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

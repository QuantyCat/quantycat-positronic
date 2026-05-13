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
training_run_name = config.get("training_run_name") or f"{task_label}_{robot}"
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
z_loss_weight  = config["train_z_loss_weight"]
loss_ct_weights = config["loss_ct_weights"]
action_sign_loss_weight = config.get("action_sign_loss_weight", 0.0)
action_sign_eps = config.get("action_sign_eps", 0.03)
action_sign_margin = config.get("action_sign_margin", 0.02)
action_wrong_sign_loss_multiplier = config.get("action_wrong_sign_loss_multiplier", 1.0)
action_wrong_sign_joint_weights = config.get("action_wrong_sign_joint_weights")
action_sign_joint_weights = config.get("action_sign_joint_weights")
action_sign_horizon_weights = config.get("action_sign_horizon_weights")
action_sign_center = config.get("action_sign_center", "raw_zero")
action_quiet_loss_weight = config.get("action_quiet_loss_weight", 0.0)
action_quiet_eps = config.get("action_quiet_eps", 0.01)
action_quiet_pred_eps = config.get("action_quiet_pred_eps", 0.01)
action_quiet_joint_weights = config.get("action_quiet_joint_weights")
action_quiet_horizon_weights = config.get("action_quiet_horizon_weights")
action_motion_loss_weight = config.get("action_motion_loss_weight", 0.0)
action_motion_eps = config.get("action_motion_eps", 0.08)
action_motion_joint_weights = config.get("action_motion_joint_weights")
action_motion_horizon_weights = config.get("action_motion_horizon_weights")
action_magnitude_loss_weight = config.get("action_magnitude_loss_weight", 0.0)
action_magnitude_eps = config.get("action_magnitude_eps", 0.08)
action_magnitude_joint_weights = config.get("action_magnitude_joint_weights")
action_magnitude_horizon_weights = config.get("action_magnitude_horizon_weights")
lora_r         = config["lora_r"]
lora_alpha     = config["lora_alpha"]
ckpt_max_keep  = config["ckpt_max_keep"]
save_interval  = config["save_interval"]
fresh_start    = config.get("fresh_start", False)
resume_from_checkpoint = config.get("resume_from_checkpoint")
run_validation = bool(config.get("run_validation", False))
action_stats   = os.path.join(work_dir, "min_max_action.txt")
state_stats    = os.path.join(work_dir, "min_max_state.txt")
data_config_train_override = config.get("data_config_train_override")
data_config_val_ind_override = config.get("data_config_val_ind_override")
data_config_val_ood_override = config.get("data_config_val_ood_override")

home         = os.path.expanduser("~")
rynnvla_repo = os.path.join(home, "RynnVLA-002", "rynnvla-002")

if not os.path.isdir(rynnvla_repo):
    print(f"ERROR: RynnVLA-002 repo not found at {rynnvla_repo}")
    sys.exit(1)

# data_config is written by step7_update_train_config.py — derived from his/chunk_size/resolution
data_config_train_name = f"his_{his}_third_view_wrist_w_state_{chunk_size}_{resolution}_pretokenize.yaml"
data_config_val_ind_name = f"his_{his}_third_view_wrist_w_state_{chunk_size}_{resolution}_pretokenize_val_ind.yaml"
data_config_val_ood_name = f"his_{his}_third_view_wrist_w_state_{chunk_size}_{resolution}_pretokenize_val_ood.yaml"
data_config_train = os.path.join(rynnvla_repo, "configs", "lerobot", data_config_train_name)
data_config_val_ind = os.path.join(rynnvla_repo, "configs", "lerobot", data_config_val_ind_name)
data_config_val_ood = os.path.join(rynnvla_repo, "configs", "lerobot", data_config_val_ood_name)
if data_config_train_override:
    data_config_train = os.path.abspath(os.path.expanduser(data_config_train_override))
if data_config_val_ind_override:
    data_config_val_ind = os.path.abspath(os.path.expanduser(data_config_val_ind_override))
if data_config_val_ood_override:
    data_config_val_ood = os.path.abspath(os.path.expanduser(data_config_val_ood_override))
if not os.path.isfile(data_config_train):
    print(f"ERROR: data config not found at {data_config_train}")
    print("Run preprocessing (step7) first, or check that his/chunk_size/resolution in config.yaml match.")
    sys.exit(1)
if run_validation and not os.path.isfile(data_config_val_ind):
    print(f"ERROR: validation is enabled but val config was not found at {data_config_val_ind}")
    print("Re-run preprocessing to build split validation configs, or disable run_validation.")
    sys.exit(1)
if not os.path.isfile(data_config_val_ood):
    data_config_val_ood = data_config_val_ind if os.path.isfile(data_config_val_ind) else data_config_train

init_from    = os.path.join(rynnvla_repo, "ckpts", "starting_point")
tokenizer    = os.path.join(rynnvla_repo, "ckpts", "chameleon", "base_model")
train_script = os.path.join(rynnvla_repo, "pretrain_solver_awm_w_ck_action_head.py")
output_dir   = os.path.realpath(os.path.join(work_dir, "fine_tuning", training_run_name))

# Resume / fresh-start logic
if fresh_start:
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        print(f"  Cleared output dir for fresh start: {output_dir}")
    resume_flags = ["--no_auto_resume"]
elif resume_from_checkpoint:
    resume_from_checkpoint = os.path.abspath(os.path.expanduser(resume_from_checkpoint))
    print(f"  Resuming from configured checkpoint: {resume_from_checkpoint}")
    resume_flags = ["--resume_path", resume_from_checkpoint]
else:
    print(f"  Auto-resuming from last checkpoint in: {output_dir}")
    resume_flags = []  # auto_resume=True is the training script default

# Upstream defines `--train_only` as `type=bool`, so passing the literal string
# "False" is parsed as True. Only pass it for true train-only runs; otherwise
# rely on the upstream default False so validation actually runs.
train_only_flags = [] if run_validation else ["--train_only", "True"]

os.makedirs(output_dir, exist_ok=True)

print(f"Starting fine-tune")
print(f"  Config:    {CONFIG_PATH}")
print(f"  Train data: {data_config_train}")
if run_validation:
    print(f"  Val data:   {data_config_val_ind}")
print(f"  Output:    {output_dir}")
print(f"  fresh_start={fresh_start}  run_validation={run_validation}  lr={lr}  epochs={epochs}  accum_iter={accum_iter}  clip_grad={clip_grad}  z_loss_weight={z_loss_weight}  loss_ct_weights={loss_ct_weights}  action_sign_loss_weight={action_sign_loss_weight}  action_sign_eps={action_sign_eps}  action_sign_center={action_sign_center}  action_sign_margin={action_sign_margin}  action_wrong_sign_loss_multiplier={action_wrong_sign_loss_multiplier}  action_wrong_sign_joint_weights={action_wrong_sign_joint_weights}  action_sign_joint_weights={action_sign_joint_weights}  action_sign_horizon_weights={action_sign_horizon_weights}  action_quiet_loss_weight={action_quiet_loss_weight}  action_quiet_eps={action_quiet_eps}  action_quiet_pred_eps={action_quiet_pred_eps}  action_quiet_joint_weights={action_quiet_joint_weights}  action_quiet_horizon_weights={action_quiet_horizon_weights}  action_motion_loss_weight={action_motion_loss_weight}  action_motion_eps={action_motion_eps}  action_motion_joint_weights={action_motion_joint_weights}  action_motion_horizon_weights={action_motion_horizon_weights}  action_magnitude_loss_weight={action_magnitude_loss_weight}  action_magnitude_eps={action_magnitude_eps}  action_magnitude_joint_weights={action_magnitude_joint_weights}  action_magnitude_horizon_weights={action_magnitude_horizon_weights}")
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
    "--data_config_train", data_config_train,
    "--data_config_val_ind", data_config_val_ind if run_validation else data_config_train,
    "--data_config_val_ood", data_config_val_ood if run_validation else data_config_train,
    "--num_workers", str(num_workers),
    "--output_dir", output_dir,
    "--checkpointing",
    "--max_seq_len", "4096",
    "--unmask_image_logits",
    "--dropout", "0.08",
    "--z_loss_weight", str(z_loss_weight), # from config.yaml
    "--loss_ct_weights", str(loss_ct_weights),
    "--action_sign_loss_weight", str(action_sign_loss_weight),
    "--action_sign_eps", str(action_sign_eps),
    "--action_sign_margin", str(action_sign_margin),
    "--action_wrong_sign_loss_multiplier", str(action_wrong_sign_loss_multiplier),
    "--action_wrong_sign_joint_weights", str(action_wrong_sign_joint_weights) if action_wrong_sign_joint_weights is not None else "",
    "--action_sign_joint_weights", str(action_sign_joint_weights) if action_sign_joint_weights is not None else "",
    "--action_sign_horizon_weights", str(action_sign_horizon_weights) if action_sign_horizon_weights is not None else "",
    "--action_sign_center", str(action_sign_center),
    "--action_quiet_loss_weight", str(action_quiet_loss_weight),
    "--action_quiet_eps", str(action_quiet_eps),
    "--action_quiet_pred_eps", str(action_quiet_pred_eps),
    "--action_quiet_joint_weights", str(action_quiet_joint_weights) if action_quiet_joint_weights is not None else "",
    "--action_quiet_horizon_weights", str(action_quiet_horizon_weights) if action_quiet_horizon_weights is not None else "",
    "--action_motion_loss_weight", str(action_motion_loss_weight),
    "--action_motion_eps", str(action_motion_eps),
    "--action_motion_joint_weights", str(action_motion_joint_weights) if action_motion_joint_weights is not None else "",
    "--action_motion_horizon_weights", str(action_motion_horizon_weights) if action_motion_horizon_weights is not None else "",
    "--action_magnitude_loss_weight", str(action_magnitude_loss_weight),
    "--action_magnitude_eps", str(action_magnitude_eps),
    "--action_magnitude_joint_weights", str(action_magnitude_joint_weights) if action_magnitude_joint_weights is not None else "",
    "--action_magnitude_horizon_weights", str(action_magnitude_horizon_weights) if action_magnitude_horizon_weights is not None else "",
    "--ckpt_max_keep", str(ckpt_max_keep),
    "--save_iteration_interval", str(save_interval),
    "--lora_r", str(lora_r),
    "--lora_alpha", str(lora_alpha),
] + train_only_flags + resume_flags

log_path = os.path.join(output_dir, "output.log")
print(f"  Log: {log_path}")
print()
 
with open(log_path, "a") as log:
    env = os.environ.copy()
    env["RYNNVLA_ACTION_STATS_FILE"] = action_stats
    env["RYNNVLA_STATE_STATS_FILE"] = state_stats
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
    proc.wait()

sys.exit(proc.returncode)

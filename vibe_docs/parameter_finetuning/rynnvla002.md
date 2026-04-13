# April 13th

## Context

50-episode screwdriver dataset, SO-101 robot, RynnVLA-002 fine-tune from LIBERO starting_point. After 3 epochs closs was stuck at 14.3 (above the 11.09 random baseline), z_loss pinned at 216 throughout, all action accuracies 0.0000. Visual/contrastive loss (loss_ct) was learning fine (0.68 → 0.35).

## Changes

### config.yaml

| Parameter | From | To | Why |
|---|---|---|---|
| `lr` | `0.000002` | `0.00002` | Was 10× below the DAMO default; with grad_norm of 100–165 and clip at 4.0, effective step size was ~5e-8 per parameter |
| `min_lr` | `0.0000002` | `0.000002` | Keeps the same 10:1 ratio to lr |

### finetune.py

| Parameter | From | To | Why |
|---|---|---|---|
| `--accum_iter` | `1` | `4` | Effective batch 4→16 matches the scale the DAMO default lr was designed for, and averages gradients over more samples so each optimizer step is less noisy |
| `--clip_grad` | `4.0` | `8.0` | Pre-clip grad_norm with accum_iter=4 will likely be 300+; keeping clip at 4.0 makes the effective step smaller than before, not larger |
| `--z_loss_weight` | `1e-5` | `1e-3` | z_loss=216 throughout 3 epochs but contributes only 0.002 to total loss vs 14.3 closs; model has essentially no gradient incentive to reduce the enormous partition function that's crushing action token probabilities |

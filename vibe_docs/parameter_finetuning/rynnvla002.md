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

---

# April 14th

## Context

Fresh 4-epoch run with April 13th fixes applied. Model converged — closs dropped from 14+ to ~3.5 by end of epoch 0, action accuracies went from 0.0000 to ~0.07. However by epoch 2 of 4, lr had decayed to 0.000008 via cosine schedule and closs plateaued at ~3.54. Grad norm was consistently ~33–50 throughout but clip_grad=8.0 was always clipping (effective step = 2e-5 × 8/33 ≈ 4.8e-6). Run was stopped at epoch 2 to restart with a higher effective step size.

## Changes

### config.yaml

| Parameter | From | To | Why |
|---|---|---|---|
| `lr` | `0.00002` | `0.00005` | Grad norm consistently ~33 with clip at 8 meant effective step was only ~4.8e-6; 2.5× lr increase gives more signal per step |
| `min_lr` | `0.000002` | `0.000005` | Keeps the same 10:1 ratio to lr |
| `clip_grad` | `8.0` | `16.0` | Grad norm stable at 33–50 with no spikes — safe to give gradients more room; combined with lr bump gives 5× larger effective step (4.8e-6 → 2.4e-5) |

## Training metrics at stop point (epoch 2, step ~3790/9264)

| Metric | Epoch 0 end | Epoch 1 end | Epoch 2 (stopped) |
|---|---|---|---|
| `closs` | ~3.5 | ~3.6 | 3.54 |
| `acc_action` | ~0.068 | ~0.068 | ~0.074 |
| `l1_loss_action` | ~0.73 | ~0.66 | ~0.645 |
| `grad_norm` | ~36 | ~34 | ~33 |
| `lr` | 0.000018 | 0.000016 | 0.000008 |

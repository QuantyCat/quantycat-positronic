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

---

# April 14th (run 2 — completed)

## Context

Restarted fresh with April 14th config (lr=5e-5, clip_grad=16). Ran all 4 epochs to completion in 15h 37m. Model continued improving through epoch 3 with no plateau — cosine schedule bottomed out at min_lr=0.000005 by epoch 3, which explains the smaller gains in the final epoch. Final checkpoint: `04132026_epoch3-iter1999`.

## Final training metrics

| Metric | Epoch 0 end | Epoch 1 end | Epoch 2 end | Epoch 3 end |
|---|---|---|---|---|
| `closs` | 4.01 | 3.24 | 3.13 | 3.10 |
| `acc_action` (avg joints 5–9) | ~0.086 | ~0.157 | ~0.176 | ~0.183 |
| `l1_loss_action` (avg) | ~0.641 | ~0.578 | ~0.550 | ~0.544 |
| `grad_norm` | ~44 | ~38 | ~33 | ~32 |
| `lr` | 0.000044 | 0.000028 | 0.000012 | 0.000005 |

## Comparison vs previous run (stopped at epoch 2)

| Metric | Old run epoch 2 (stopped) | New run epoch 3 (final) |
|---|---|---|
| `closs` | 3.54 | 3.10 |
| `acc_action` | ~0.074 | ~0.183 |
| `l1_loss_action` | ~0.645 | ~0.544 |

## Observations

- Action accuracy more than doubled (0.074 → 0.183) — the higher lr + clip_grad gave significantly more signal per step
- closs still declining at epoch 3 end; model not fully converged — if inference is poor, resume with `fresh_start: false` and add epochs
- grad_norm settled at ~31–32 by epoch 3, well within clip_grad=16 headroom
- cosine schedule hits min_lr too early on a 4-epoch run; consider longer schedule (8–12 epochs) for next run

---

# April 14th (run 3 — pending)

## Context

Run 2 completed with closs 3.10 and acc_action ~0.183 but model not fully converged — still improving at epoch 3 end when cosine schedule bottomed out at min_lr. Increasing effective step size for next run to push further.

## Changes

### config.yaml

| Parameter | From | To | Why |
|---|---|---|---|
| `lr` | `0.00005` | `0.0001` | 2× bump; effective step at epoch 0 ~5.5e-5 vs ~1.8e-5 previously (grad_norm ~44, clip_grad 24) |
| `min_lr` | `0.000005` | `0.00001` | Keeps the same 10:1 ratio to lr |
| `clip_grad` | `16.0` | `24.0` | grad_norm stable at 31–44 with no spikes; more headroom increases effective step without risking instability |

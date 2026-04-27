# RynnVLA-002 Fine-tuning Log — Screwdriver Task (SO-101)

## Run Summary

Correction on 2026-04-16: the local `config.yaml` had duplicate `z_loss_weight` keys, and the later inference value (`0.0`) overrode the intended training value during the latest run. The epoch-4 run therefore launched with `z_loss_weight=0.0`, not `1e-3`.

| Metric | Run 1 (failed) | Run 2 epoch 3 final | Run 3 epoch 4 final | Run 4 epoch 3 final |
|---|---|---|---|
| `closs` | 14.3 (stuck) | 3.10 | 2.80 | 2.56 |
| `acc_action` | 0.0000 | ~0.183 | ~0.252 | ~0.270 |
| `l1_loss_action` | — | ~0.544 | ~0.433 | ~0.348 |
| `grad_norm` | ~33 | ~32 | ~30 | ~30.7 |
| `lr` (final) | 0.000002 | 0.000005 | 0.000010 | 0.000010 |
| `epochs` | 4 | 4 | 5 | 4 |

## Parameter Changes Per Run

| Parameter | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|
| `lr` | `2e-6` | `5e-5` | `1e-4` | `1e-4` |
| `min_lr` | `2e-7` | `5e-6` | `1e-5` | `1e-5` |
| `clip_grad` | `4.0` | `16.0` | `24.0` | `24.0` |
| `accum_iter` | `1` | `4` | `4` | `4` |
| `z_loss_weight` | `1e-5` | `0.0` | `0.0` | `1e-3` |
| `epochs` | `4` | `4` | `5` | `4` |

## What Changed and Why

### Current setup after low-motion debugging
- **History mode reset to an official/default path**:
  ```yaml
  his: 1
  his_mode: 1h_1a_img_only_state
  ```
  We moved back from the custom `2h_1a_img_both_wrist_state` setup to the default one-history, one-action, image+state mode. This reduces the chance that a custom context format or image-slot history mismatch is contributing to weak conditioning.
- **Important compatibility note**: checkpoints trained with `his: 2` / `2h_1a_img_both_wrist_state` should not be evaluated with the new `his: 1` config. The processed data must be regenerated and the model retrained for the new history mode.
- **Attention mask fix**: the weighted action-head training path was fixed/verified to use `att_mask=True`. The previous `att_mask=False` behavior could create a training/inference mismatch where the action head learned under a different attention pattern than the one used during inference.
- **Why this matters**: the overfit probes showed the action head can learn motion on the existing windows, so the leading concern is no longer a dead action head. The current retraining setup is intended to remove context-format risk and the attention-mask mismatch before testing whether more training or loss changes are needed.

### Run 1 → Run 2
- **lr** `2e-6 → 5e-5`: Was 10× below DAMO default; effective step ~5e-8, model had no signal
- **min_lr** `2e-7 → 5e-6`: Keeps 10:1 ratio to lr
- **clip_grad** `4.0 → 16.0`: With accum_iter=4, pre-clip grad_norm was 300+; keeping at 4.0 made effective step smaller, not larger
- **accum_iter** `1 → 4`: Effective batch 4→16, matches scale DAMO default lr was designed for
- **z_loss_weight** intended `1e-5 → 1e-3`, but the launched run still used `0.0` due to a config-key collision. Treat conclusions about z-loss from this run as invalid until rerun with the fixed config.

### Run 2 → Run 3
- **lr** `5e-5 → 1e-4`: Grad norm stable at 31–44, effective step was only ~4.8e-6; 2× bump gives more signal per step
- **min_lr** `5e-6 → 1e-5`: Keeps 10:1 ratio to lr
- **clip_grad** `16.0 → 24.0`: More headroom given stable grad_norm — effective step at epoch 0 ~5.5e-5 vs ~1.8e-5 previously
- **epochs** `4 → 5`: Model still improving at end of run 2; cosine schedule was hitting min_lr too early

### Run 3 → Run 4
- **epochs** `5 → 4`: The saved `epoch3/args.json` shows the rerun was launched as a 4-epoch job and the kept checkpoint is the epoch-3 final
- **z_loss_weight** `0.0 → 1e-3`: This rerun restored the intended training-side z-loss value in the launched args

## Observations

- Each run improved meaningfully: acc_action went 0.0000 → 0.183 → 0.252; l1_loss 0.544 → 0.433
- grad_norm settling over runs (44 → 32 → 30) — model becoming more stable
- Run 3 still not fully converged — closs 2.80 still has room, acc_action 0.252 means ~75% of action bin predictions are wrong (l1_loss is the more meaningful metric for continuous control)
- Next step: test run 3 epoch4 checkpoint on robot; if inference still degenerate, consider resuming with `fresh_start: false` and more epochs

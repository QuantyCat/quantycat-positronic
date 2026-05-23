# RynnVLA Trimmed-Active Calibration Ablation: Lessons

Date: 2026-05-23

Run:

- Config: `models/rynnvla-002/config.yaml`
- Training run: `screwdriver_so101_h1j01234_alljoints_trim_active_calib_ablation_scratch`
- Checkpoint evaluated: `my_data/training_pipeline/fine_tuning/screwdriver_so101_h1j01234_alljoints_trim_active_calib_ablation_scratch/epoch0`
- Eval output: `eval_output/screwdriver_so101/model_eval/trim_active_calib_ablation/epoch0`

## What Changed

This run used the trimmed-active dataset but removed the heavy auxiliary losses that appeared to drive the previous checkpoint into saturation:

- `loss_ct_weights: 2`
- `action_sign_loss_weight: 0.0`
- `action_quiet_loss_weight: 0.0`
- `action_magnitude_loss_weight: 0.0`
- `action_motion_loss_weight: 1.0`
- `action_head_loss_routing: default`

Training was stopped after `epoch0` was saved. Partial epoch1 progress was discarded. We do not plan to resume this run.

## Partial High-Motion Results

High-motion evals were started on `epoch0` and then terminated after enough signal was available. Completed reports: j0, j1, j2, j3. j4 did not complete.

| eval | focus | centered sign | norm slope | norm MAE | corr | pred std | GT std | pred mean | GT mean | steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `broad_j0_high_motion_top20` | j0 | 0.550 | 0.000 | 0.361 | 0.047 | 0.004 | 0.385 | 0.059 | 0.022 | 1000 |
| `broad_j1_high_motion_top20` | j1 | 0.550 | 0.058 | 0.368 | 0.861 | 0.028 | 0.413 | 0.112 | 0.090 | 1000 |
| `broad_j2_high_motion_top20` | j2 | 1.000 | -0.081 | 0.299 | -0.514 | 0.021 | 0.132 | -0.098 | -0.397 | 1000 |
| `broad_j3_high_motion_top20` | j3 | 0.162 | 0.012 | 0.366 | 0.395 | 0.008 | 0.282 | -0.213 | 0.088 | 1000 |

## Interpretation

The ablation removed the prior saturation failure: outputs were no longer clipped at normalized `+1` on j2/j3.

However, it did not restore useful slopes by epoch0. The action head mostly produced low-variance templates:

- j0 pred std `0.004` vs GT std `0.385`
- j1 pred std `0.028` vs GT std `0.413`
- j2 pred std `0.021` vs GT std `0.132`
- j3 pred std `0.008` vs GT std `0.282`

This suggests the previous problem was not only an over-weighted sign/magnitude loss issue. The cleaner objective avoids the worst clipping behavior, but the model still sees enough ambiguous low-motion examples that it learns conservative template actions rather than calibrated frame-conditioned actions.

## Updated Dataset Diagnosis

The OpenPI diagnosis focused on the frozen countdown prefix. For RynnVLA, the dataset problem appears broader:

- There is likely low-motion contamination at the beginning.
- There are pauses in the middle of demonstrations.
- There may be unnecessary low-motion tails at the end.
- The examples are not continuous action trajectories.

This means per-episode first-motion trimming is probably not sufficient. RynnVLA may need data curation that preserves only active motion segments, or explicitly models/labels pauses, before another serious scratch run.

## Recommendation

Do not resume this RynnVLA run.

Before another RynnVLA attempt, build a stronger data filter:

- detect active-motion intervals throughout each episode, not just the first-motion boundary
- trim low-motion tails
- either remove mid-episode pauses or split episodes into active segments
- recompute high-motion/active-segment statistics before retraining
- only then revisit the RynnVLA objective, starting from the non-saturating calibration ablation as the safer baseline

Short-term priority should shift back to the PI 0.5 run setup, where the active-start diagnosis already produced useful live behavior.

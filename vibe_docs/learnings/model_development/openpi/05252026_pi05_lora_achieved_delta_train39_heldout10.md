# OpenPI pi0.5 Achieved-Delta Train39/Heldout10 Evaluation

**Date:** 2026-05-25
**Question:** Did the achieved-delta pi0.5 checkpoint generalize to examples it did not train on?
**Short answer:** The previous achieved-delta checkpoint was trained on all available examples. This run created a real episode-level holdout split, trained only on the train split, and evaluated on heldout episodes.

---

## Split

Source dataset:

`/home/caroline/quantycat-positronic/my_data/clean_input_data_achieved_delta`

Train split:

`/home/caroline/quantycat-positronic/my_data/clean_input_data_achieved_delta_train39`

- 39 episodes
- 19,259 frames

Heldout split:

`/home/caroline/quantycat-positronic/my_data/clean_input_data_achieved_delta_heldout10`

- 10 episodes
- 5,031 frames
- original episode IDs: `[0, 1, 2, 3, 6, 23, 33, 38, 44, 47]`

Split report:

`/home/caroline/quantycat-positronic/my_data/clean_input_data_achieved_delta/split_report_train39_heldout10_seed20260525.json`

---

## Train-Only Checkpoint

OpenPI config:

`pi05_quantycat_lora_achieved_delta_train39`

Checkpoint:

`/home/caroline/quantycat-positronic/models/openpi/training_pipeline/checkpoints/pi05_quantycat_lora_achieved_delta_train39/05252026_pi05_lora_achieved_delta_train39/4999`

Training loss fell from `0.0823` at step 0 to `0.0091` at step 4900. Final checkpoint saved cleanly at step 4999.

---

## Heldout High-Motion Eval

Eval output:

`/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/openpi_05252026_pi05_lora_achieved_delta_train39_step4999_heldout10`

Summary:

`/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/openpi_05252026_pi05_lora_achieved_delta_train39_step4999_heldout10/all_joint_focused_summary.json`

Diagonal high-motion metrics:

| Focus window | Joint | Sign | Corr | Slope | Norm MAE |
|---|---:|---:|---:|---:|---:|
| j0 high-motion | j0 | 0.911 | 0.613 | 0.513 | 0.130 |
| j1 high-motion | j1 | 0.928 | 0.747 | 0.688 | 0.124 |
| j2 high-motion | j2 | 0.927 | 0.820 | 0.638 | 0.134 |
| j3 high-motion | j3 | 0.870 | 0.714 | 0.648 | 0.135 |
| j4 high-motion | j4 | 0.871 | 0.606 | 0.444 | 0.206 |

Interpretation:

- The heldout result is clearly weaker than the earlier training-set eval, so the earlier eval overstated certainty.
- It is not a collapse: signs are mostly strong, especially j1 and j2, and correlations are usable.
- Magnitudes are under-scaled on heldout. The worst diagonal slope is j4 at `0.444`; j0 is also low at `0.513`.
- j2, the main previous concern, is much better than the commanded-delta pipeline, but still under-amplified at `0.638`.

---

## Gain Calibration

Gain calibration output:

`/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/openpi_05252026_pi05_lora_achieved_delta_train39_step4999_heldout10_gain_calibration/gain_calibration_summary.md`

Selected conservative gains:

| Joint | Gain | Baseline slope | Calibrated slope | Baseline MAE | Calibrated MAE |
|---|---:|---:|---:|---:|---:|
| j0 | 1.00 | 0.513 | 0.513 | 0.130 | 0.130 |
| j1 | 1.00 | 0.688 | 0.688 | 0.124 | 0.124 |
| j2 | 1.20 | 0.638 | 0.766 | 0.134 | 0.128 |
| j3 | 1.10 | 0.648 | 0.713 | 0.135 | 0.137 |
| j4 | 1.00 | 0.444 | 0.444 | 0.206 | 0.206 |

The gain sweep did not recommend j4 amplification despite its low slope, because increasing j4 did not improve the heldout score.

---

## Deployment Implication

This checkpoint has a real heldout result now. It is plausible enough for cautious live testing, but it should not be treated as fully validated.

Recommended live-test posture:

- Use the achieved-delta checkpoint trained on train39 if the goal is to test heldout-vetted behavior.
- Start with conservative safety limits.
- Consider applying the heldout-selected gains for j2 and j3 only: `j2=1.20`, `j3=1.10`, `j4=1.00`.
- Watch j4 closely; offline heldout suggests it is the least reliable magnitude channel.


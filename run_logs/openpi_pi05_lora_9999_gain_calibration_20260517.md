# OpenPI pi0.5 LoRA Step 9999 Gain Calibration

Date: 2026-05-17 UTC

Input traces:

```text
/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/openpi_pi05_h20_lora_step9999/broad_j*_high_motion_top50/focused_high_motion_traces.npz
```

Calibration script:

```text
/home/caroline/quantycat-positronic/models/openpi/eval/openpi_gain_calibration.py
```

Fine sweep output:

```text
/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/openpi_pi05_h20_lora_step9999_gain_calibration_fine/gain_calibration_summary.md
/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/openpi_pi05_h20_lora_step9999_gain_calibration_fine/gain_calibration_summary.json
```

## Result

Do not add j2 gain. Every positive j2 gain tested increased normalized MAE on the j2 high-motion set.

For j4, the score-selected gain is 1.050:

| Joint | Base slope | Gain slope | Base norm MAE | Gain norm MAE |
|---|---:|---:|---:|---:|
| j4 | 0.760 | 0.797 | 0.0917 | 0.0926 |

The conservative deployment-selected gain is 1.025:

| Joint | Base slope | Gain slope | Base norm MAE | Gain norm MAE |
|---|---:|---:|---:|---:|
| j4 | 0.760 | 0.779 | 0.0917 | 0.0920 |

Because the task is moving toward a real robot rollout, use the conservative gain first:

```text
joint gain vector [j0, j1, j2, j3, j4, gripper] = [1.000, 1.000, 1.000, 1.000, 1.025, 1.000]
```

For deployment, apply gains to deltas, not absolute targets:

```text
delta = predicted_target - current_state
delta *= gain_vector
calibrated_target = current_state + delta
```

Keep final joint target clipping enabled before sending commands to SO-101.

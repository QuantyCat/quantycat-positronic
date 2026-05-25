# OpenPI pi05 LoRA Achieved-Delta Eval - 05242026

**Checkpoint:** `models/openpi/training_pipeline/checkpoints/pi05_quantycat_lora_achieved_delta/05242026_pi05_lora_achieved_delta/4999`
**Config:** `pi05_quantycat_lora_achieved_delta`
**Dataset:** `my_data/clean_input_data_achieved_delta`
**Eval:** 20 high-motion windows per joint, 50 steps each, selected from achieved-delta labels

This eval reads the achieved-delta LeRobot dataset directly. For joints 0-4, it compares:

```text
policy absolute target - current state
```

against:

```text
future achieved state - current state
```

The gripper channel remains absolute.

## Per-Joint Summary

| Joint | Sign Agreement | Slope | Correlation | Norm MAE |
|---|---:|---:|---:|---:|
| j0 shoulder_pan | 0.996 | 0.969 | 0.975 | 0.062 |
| j1 shoulder_lift | 0.996 | 0.893 | 0.934 | 0.053 |
| j2 elbow_flex | 0.997 | 0.862 | 0.967 | 0.070 |
| j3 wrist_flex | 0.933 | 0.776 | 0.859 | 0.101 |
| j4 wrist_roll | 0.988 | 0.810 | 0.907 | 0.098 |

## Key Findings

The achieved-delta label fix materially improved j2. The previous commanded-label eval had j2 slope 0.617 and correlation 0.622; this achieved-delta checkpoint has j2 slope 0.862 and correlation 0.967 on achieved-delta high-motion windows.

j0 and j1 are strong. j2 is now live-testable rather than the obvious blocker. j3 and j4 still undershoot somewhat, but their direction/correlation are usable enough for a cautious offline-to-live progression.

## Gain Calibration

Gain calibration used the saved high-motion traces and achieved-delta action bounds.

| Joint | Baseline Slope | Conservative Slope | Baseline MAE | Conservative MAE |
|---|---:|---:|---:|---:|
| j2 | 0.862 | 0.948 | 0.070 | 0.068 |
| j3 | 0.776 | 0.776 | 0.101 | 0.101 |
| j4 | 0.810 | 0.973 | 0.098 | 0.095 |

Score-selected gains: `[1.0, 1.0, 1.1, 1.1, 1.2, 1.0]`

Conservative deployment gains: `[1.0, 1.0, 1.1, 1.0, 1.2, 1.0]`

## Recommendation

This checkpoint is a substantially better offline candidate than the prior 05232026 run, especially for j2. For first live testing, use the conservative gains only if we are comfortable adding calibration before the first robot run. Otherwise, all gains at 1.0 are defensible because the raw achieved-delta eval is already strong.

If using gains, apply them to deltas, not absolute targets:

```text
target = current_state + (policy_target - current_state) * gain_vector
```

## Outputs

- High-motion eval summary: `eval_output/screwdriver_so101/model_eval/openpi_05242026_pi05_lora_achieved_delta_step4999_top20/all_joint_focused_summary.json`
- Gain calibration summary: `eval_output/screwdriver_so101/model_eval/openpi_05242026_pi05_lora_achieved_delta_step4999_top20_gain_calibration/gain_calibration_summary.md`
- Eval log: `run_logs/openpi_achieved_delta_eval_05242026.log`

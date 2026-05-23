# OpenPI pi0.5 LoRA 10k Continuation Eval Comparison

Date: 2026-05-17 UTC

Eval wrapper:

```text
/home/caroline/quantycat-positronic/models/openpi/eval/openpi_alljoints_high_motion_eval.py
```

Baseline:

```text
/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/openpi_pi05_h20_lora_step4999/all_joint_focused_summary.json
```

Continuation checkpoints evaluated:

```text
/home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/checkpoints/pi05_quantycat_lora/screwdriver_so101_pi05_h20_lora_20260516_pdt/{6000,7000,8000,9000,9999}
```

## Diagonal High-Motion Metrics

Each row is the diagonal joint for that joint-ranked high-motion window set.

| Checkpoint | Joint | Sign | Corr | Slope | Norm MAE |
|---|---:|---:|---:|---:|---:|
| 4999 | j0 | 0.988 | 0.931 | 0.867 | 0.090 |
| 4999 | j1 | 0.995 | 0.947 | 0.873 | 0.093 |
| 4999 | j2 | 1.000 | 0.521 | 0.569 | 0.103 |
| 4999 | j3 | 0.952 | 0.790 | 0.659 | 0.112 |
| 4999 | j4 | 0.963 | 0.668 | 0.651 | 0.129 |
| 6000 | j0 | 0.991 | 0.935 | 0.943 | 0.091 |
| 6000 | j1 | 0.986 | 0.948 | 1.082 | 0.109 |
| 6000 | j2 | 1.000 | 0.562 | 0.734 | 0.125 |
| 6000 | j3 | 0.957 | 0.799 | 0.749 | 0.111 |
| 6000 | j4 | 0.968 | 0.723 | 0.924 | 0.135 |
| 7000 | j0 | 0.990 | 0.936 | 0.879 | 0.091 |
| 7000 | j1 | 0.987 | 0.943 | 0.863 | 0.095 |
| 7000 | j2 | 1.000 | 0.594 | 0.635 | 0.091 |
| 7000 | j3 | 0.969 | 0.838 | 0.664 | 0.098 |
| 7000 | j4 | 0.974 | 0.700 | 0.557 | 0.118 |
| 8000 | j0 | 0.994 | 0.948 | 0.909 | 0.080 |
| 8000 | j1 | 0.981 | 0.938 | 0.837 | 0.101 |
| 8000 | j2 | 1.000 | 0.559 | 0.604 | 0.096 |
| 8000 | j3 | 0.944 | 0.813 | 0.701 | 0.118 |
| 8000 | j4 | 0.977 | 0.750 | 0.711 | 0.106 |
| 9000 | j0 | 0.993 | 0.947 | 0.858 | 0.081 |
| 9000 | j1 | 0.994 | 0.955 | 0.848 | 0.087 |
| 9000 | j2 | 1.000 | 0.630 | 0.602 | 0.093 |
| 9000 | j3 | 0.962 | 0.835 | 0.651 | 0.099 |
| 9000 | j4 | 0.983 | 0.781 | 0.685 | 0.105 |
| 9999 | j0 | 0.994 | 0.955 | 0.920 | 0.073 |
| 9999 | j1 | 0.996 | 0.966 | 0.964 | 0.075 |
| 9999 | j2 | 1.000 | 0.662 | 0.697 | 0.083 |
| 9999 | j3 | 0.977 | 0.873 | 0.788 | 0.087 |
| 9999 | j4 | 0.991 | 0.811 | 0.760 | 0.092 |

## Readout

Step 9999 is the best overall deployment candidate from this eval sweep. It improves every diagonal correlation versus 4999, improves every normalized MAE, and raises the weak-joint slopes:

- j2 slope: 0.569 -> 0.697
- j4 slope: 0.651 -> 0.760

Step 6000 has the highest j4 slope at 0.924 and strong j2 slope at 0.734, but it pays for that with worse normalized MAE on j1, j2, and j4. Step 9999 is the cleaner tradeoff.

## Next Step

Use step 9999 as the base for offline gain calibration on saved traces. Start with small j2/j4 gains rather than the full original sweep, because 9999 already improved amplitude:

```text
j2 gain: 1.00, 1.10, 1.20, 1.30
j4 gain: 1.00, 1.10, 1.20, 1.30
```

Keep the smallest gains that improve slope or normalized MAE without hurting sign/corr.

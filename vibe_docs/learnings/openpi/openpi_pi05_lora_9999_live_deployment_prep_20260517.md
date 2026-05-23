# OpenPI pi0.5 LoRA Step 9999 Live Deployment Prep

Date: 2026-05-17 UTC

Added deployment files under:

```text
/home/caroline/quantycat-positronic/models/openpi/deployment
```

Files:

- `pi05_lora_step9999_so101.json` - live rollout config for the 9999 checkpoint.
- `live_so101_openpi.py` - LeRobot SO-101 live runner for OpenPI policy inference.
- `README.md` - control-machine setup and first-rollout commands.
- `models/openpi/run_scripts/live_so101_step9999.sh` - thin launcher.

The checkpoint path in the committed config intentionally points outside the
repo:

```text
/home/caroline/openpi_checkpoints/pi05_quantycat_lora/screwdriver_so101_pi05_h20_lora_20260516_pdt/9999
```

Download/copy the checkpoint to the control computer separately. The checkpoint
directory must include:

```text
params/
assets/
```

Validated locally:

```bash
bash models/openpi/run_scripts/live_so101_step9999.sh --check-only --skip-policy-load
bash models/openpi/run_scripts/live_so101_step9999.sh --check-only \
  --checkpoint /home/caroline/quantycat-positronic/my_data/training_pipeline/openpi/checkpoints/pi05_quantycat_lora/screwdriver_so101_pi05_h20_lora_20260516_pdt/9999
```

Both checks passed. The second command loaded the OpenPI policy successfully.

Recommended first live sequence on the control computer:

```bash
cd /home/caroline/quantycat-positronic

bash models/openpi/run_scripts/live_so101_step9999.sh --check-only --skip-policy-load
bash models/openpi/run_scripts/live_so101_step9999.sh --check-only --checkpoint /path/to/9999
bash models/openpi/run_scripts/live_so101_step9999.sh --dry-run --checkpoint /path/to/9999 --max-steps 5
bash models/openpi/run_scripts/live_so101_step9999.sh --checkpoint /path/to/9999 --max-steps 10
```

Safety defaults:

- `execute_steps_per_inference`: 1
- `control_period_s`: 0.10
- `max_steps`: 20 in config, override lower for the first live test.
- max per-command delta: 4 deg for arm joints, 2 deg for gripper.
- final target limits clipped to the observed training action range plus small margins.
- gain vector: `[1.000, 1.000, 1.000, 1.000, 1.025, 1.000]`.

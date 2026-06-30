# Dacha OpenPI Training and Evaluation

This directory owns OpenPI training and evaluation wrappers for the Dacha SO-101 tasks.

Prime-radiant remains responsible for collecting/recording LeRobot episodes. Once episodes exist under
`$QUANTYCAT_DATA_HOME/datasets/dacha`, conversion, training, checkpoint evaluation, and offline diagnostics
belong here in positronic.

Common entry points:

- `train_dacha_wire_v5_h50_early_weighted.sh`
- `train_dacha_wire_v5_h50_coverage_weighted.sh`
- `eval_dacha_wire_v5_checkpoints.sh`
- `eval_dacha_wire_v5_holdout_subgroups.sh`
- `eval_dacha_wire_v5_window_strategy.sh`
- `diagnose_dacha_wire_v5_nearest_windows.sh`

Generated logs are written outside the repo by default:

`$QUANTYCAT_DATA_HOME/logs/openpi/dacha`

Checkpoints and eval outputs remain under:

- `$QUANTYCAT_DATA_HOME/checkpoints/openpi/dacha`
- `$QUANTYCAT_DATA_HOME/eval_output/openpi/dacha`

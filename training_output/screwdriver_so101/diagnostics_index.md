# Screwdriver SO-101 Diagnostics

This folder gathers the readable diagnostics for the low-motion investigation.

## Training Action Data

- `action_motion_report/index.html`  
  Visual per-episode report for all 50 demonstrations.

- `action_motion_report/overview.png`  
  Heatmap of mean absolute motion by episode and joint.

- `action_motion_report/summary.json`  
  Per-episode motion numbers.

- `action_distribution_diagnostics/summary.json`  
  Whole-dataset chunk distribution summary, including quasi-static fraction,
  repeated chunk fraction, gripper near-zero fraction, and high-motion windows.

## Model Evaluation Reports

- `model_eval_reports/epoch3/episode_000000_steps_000175_to_000224.json`  
  Current epoch3 checkpoint evaluated on a high-motion training window.

- `model_eval_reports/epoch3/episode_000025_steps_000204_to_000253.json`  
  Current epoch3 checkpoint evaluated on episode 25.

- `model_eval_reports/epoch3/episode_000025_steps_000204_to_000253_summary.png`  
  Plot summary for the episode 25 model-vs-ground-truth eval.

- `model_eval_reports/epoch3/episode_000025_steps_000204_to_000253_error_heatmap.png`  
  Per-step/per-joint error heatmap.

- `model_eval_reports/epoch3/episode_000025_steps_000204_to_000253_scatter_all_chunks.png`  
  Predicted-vs-ground-truth scatter plot.

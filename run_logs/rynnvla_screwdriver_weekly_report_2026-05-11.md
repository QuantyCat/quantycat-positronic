# RynnVLA Screwdriver Fine-Tuning Report

Date: 2026-05-11

## Objective

Create a repeatable approach for fine-tuning `rynn_vla` from teleoperation data so the SO-101 robot can execute the screwdriver pickup/place task with usable, coordinated joint motion. The immediate technical goal over the last week was to bring joints j1 and j2 up to the quality previously achieved for j3, without losing j3.

## Starting Point

Earlier scratch runs showed that j3, previously the weakest joint, could be materially improved through focused data weighting and action-loss pressure. That result suggested the training process could be guided, but it also exposed a repeatability problem: the first j1/j2 balancing attempts improved j1 and j2 while allowing j3 amplitude to collapse again.

The key question became whether we could train all three target joints together rather than trading one failure mode for another.

## Work Completed

1. Evaluated the prior high-motion checkpoints and compared the j1/j2/j3 behavior across epochs.
2. Halted runs at evaluation boundaries to avoid wasting epochs before we understood the focused metrics.
3. Ran focused high-motion evals on the balanced checkpoints, including epoch 8 and epoch 14.
4. Diagnosed the balanced recipe as under-driving j3 despite improved j1/j2 behavior.
5. Built a new `h1j123_j3restore` training split and recipe to restore j3 pressure while preserving the j1/j2 gains.
6. Trained the new scratch run through epoch 8, completed both automatic validation passes, halted epoch 9, and ran the focused high-motion eval.
7. Saved the epoch 8 checkpoint as a robot-test candidate and resumed the training recipe from epoch 8.

## Important Artifacts

Training run:
`/home/caroline/quantycat-positronic/my_data/training_pipeline/fine_tuning/screwdriver_so101_h1j123_j3restore_scratch`

Saved robot-test checkpoint:
`/home/caroline/quantycat-positronic/my_data/training_pipeline/fine_tuning/testing_checkpoints/h1j123_j3restore_epoch8_robot_test_candidate`

Focused eval report:
`/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/h1j123_j3restore_epoch8/broad_j3_high_motion_top50/focused_high_motion_joint_sign.json`

Training config:
`/home/caroline/quantycat-positronic/models/rynnvla-002/config.yaml`

Training data config:
`/home/caroline/RynnVLA-002/rynnvla-002/configs/lerobot/his_1_third_view_wrist_w_state_5_256_pretokenize_h1j123_j3restore.yaml`

## Focused High-Motion Results

The focused eval used 50 high-motion windows from the established j3 high-motion case set.

| model | joint | raw sign | corr | slope | centered sign |
|---|---:|---:|---:|---:|---:|
| h1j123 balanced e8 | j1 | 0.846 | 0.598 | 0.549 | 0.795 |
| h1j123 balanced e8 | j2 | 0.903 | 0.609 | 0.628 | 0.897 |
| h1j123 balanced e8 | j3 | 0.945 | 0.751 | 0.443 | 0.945 |
| h1j123 balanced e14 | j1 | 0.876 | 0.654 | 0.625 | 0.831 |
| h1j123 balanced e14 | j2 | 0.900 | 0.650 | 0.704 | 0.894 |
| h1j123 balanced e14 | j3 | 0.943 | 0.750 | 0.460 | 0.942 |
| j3restore e8 | j1 | 0.914 | 0.663 | 0.718 | 0.818 |
| j3restore e8 | j2 | 0.908 | 0.659 | 0.722 | 0.899 |
| j3restore e8 | j3 | 0.958 | 0.810 | 0.906 | 0.957 |
| h1j3 e14 reference | j3 | 0.951 | 0.877 | 1.009 | 0.951 |

## Interpretation

The j3restore epoch 8 result is a meaningful step forward. The balanced run had learned j1/j2 but left j3 under-driven, with j3 slope stuck around 0.44-0.46. The j3restore recipe moved j3 slope to 0.906 while improving j3 sign agreement to 0.958.

j1 and j2 did not collapse. j1 raw sign rose to 0.914 and j2 raw sign reached 0.908. Their slopes are still somewhat conservative, but both improved compared with the balanced epoch 14 checkpoint.

This is evidence that the training process can be guided in the right direction using targeted data weighting plus joint/horizon-specific action losses. It is not yet a complete recipe for all joints, but it is the best combined j1-j2-j3 checkpoint so far.

## Risks And Open Questions

- j1 and j2 slopes remain below 1.0, so physical motion may still be somewhat under-commanded in those axes.
- j3 is much better offline, but the robot may expose overshoot or timing effects not visible in offline focused eval.
- j0 and j4 are not the focus of this recipe and may still limit real task success.
- Validation metrics are less informative than the focused eval for this failure mode; physical testing is now the main missing evidence.

## Recommendation

Use `h1j123_j3restore_epoch8_robot_test_candidate` for conservative physical robot testing. Treat it as a candidate anchor, not a final model. The next decision should be based on whether the robot shows usable j1/j2/j3 authority during the screwdriver pickup/place task.

Continue the training run in parallel from epoch 8 so later checkpoints are available for comparison, but do not assume later epochs will be better without focused eval and physical behavior checks.

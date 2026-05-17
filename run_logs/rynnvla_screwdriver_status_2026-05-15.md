# RynnVLA Screwdriver Fine-Tuning Status - 2026-05-15

## Objective

Create a repeatable fine-tuning recipe for `rynn_vla` on SO-101 teleoperation data that produces usable joint-level motion for the screwdriver pickup/place task. The immediate challenge remains low motion amplitude/slope on some joints, especially `j2` and `j4`, while preserving the gains we previously saw on `j3`.

## Recent Baseline: Scaled All-Joint Run

We tested centered action-space gain/scaling to make under-used action bins more informative:

- Scales used: `[1, 1, 1.2, 1, 1.5, 1]`
- Token split: `train_h1j01234_scaled`
- Run: `screwdriver_so101_h1j01234_scaled_scratch`
- Main evaluated parent checkpoint: `epoch3`, labeled as `h1j01234_scaled_parent_epoch4`

High-motion eval results:

| Run | j0 slope | j1 slope | j2 slope | j3 slope | j4 slope |
|---|---:|---:|---:|---:|---:|
| Scaled parent epoch4 | 0.8765 | 0.5384 | 0.2634 | 0.4216 | 0.0966 |

Interpretation:

- Scaling did not look like a major unlock.
- `j0` became reasonably strong.
- `j2` remained low.
- `j4` was still very weak at the parent checkpoint.
- We downgraded scaling from "primary fix candidate" to "possible marginal preprocessing tweak."

## LoRA / lm_head Audit

We audited the custom LoRA implementation in:

- `/home/caroline/RynnVLA-002/rynnvla-002/pretrain_solver_awm_w_ck_action_head.py`

Findings:

- LoRA is custom, not PEFT.
- Adapters are inserted into `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
- Checkpoints contain `128` LoRA A tensors and `128` LoRA B tensors.
- LoRA B matrices are nonzero after training, confirming the adapters are learning.
- Checkpoints save full model state, including frozen backbone weights.
- The original trainable set was:
  - LoRA adapters
  - `action_head`
  - `lm_head`

The important finding was that `lm_head.weight` was trainable. That tensor alone is about `268M` parameters, much larger than LoRA and the action head combined. We concluded it is unnecessary for our robot-control objective and likely wastes memory/optimizer capacity.

## lm_head ON/OFF Ablation

We added an explicit `train_lm_head` switch and forked two one-epoch branches from the same scaled parent checkpoint.

Parent:

- Checkpoint: `screwdriver_so101_h1j01234_scaled_scratch/epoch3`
- Label: `h1j01234_scaled_parent_epoch4`

Branches:

- `h1j01234_scaled_e4_lmhead_on_plus1`
- `h1j01234_scaled_e4_lmhead_off_plus1`

High-motion eval results:

| Run | j0 slope | j1 slope | j2 slope | j3 slope | j4 slope |
|---|---:|---:|---:|---:|---:|
| Parent | 0.8765 | 0.5384 | 0.2634 | 0.4216 | 0.0966 |
| lm_head ON +1 | 0.9929 | 0.5258 | 0.2282 | 0.4362 | 0.1628 |
| lm_head OFF +1 | 1.0074 | 0.5356 | 0.2207 | 0.4371 | 0.1629 |

Infrastructure result:

- `lm_head ON`: `323,141,638` trainable parameters
- `lm_head OFF`: `54,706,182` trainable parameters
- `lm_head OFF` reduced GPU memory use from about `21.1GB` to about `18.7GB`
- Step time improved modestly.

Conclusion:

- Freezing `lm_head` did not hurt high-motion action eval.
- It was slightly better on `j0`, `j1`, and `j3`, matched `j4`, and was slightly worse on `j2`.
- Given the memory and optimization savings, `lm_head OFF` is now the default.

## Current Run

We started a clean run with the simplest interpretation-preserving changes:

- Run name: `screwdriver_so101_h1j01234_alljoints_lmheadoff_unscaled_scratch`
- tmux training session: `h1j01234_unscaled_lmoff_train`
- tmux supervisor session: `h1j01234_unscaled_lmoff_supervisor`
- Data split: unscaled `train_h1j01234_alljoints`
- Action scaling: disabled
- `train_lm_head: false`
- `fresh_start: false`
- Trainable params: `54,706,182`
- Supervisor target: stop and evaluate after printed `epoch4`

Active config:

- `/home/caroline/quantycat-positronic/models/rynnvla-002/config.yaml`

Expected first decision point:

- All-joint high-motion eval after printed `epoch4`
- Main watch items: `j2` slope, `j4` slope, and whether `j0` remains strong without scaling.

## Physical Intelligence Paper: Knowledge Insulation

Paper:

- Physical Intelligence, "Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better"
- arXiv: `2505.23705`

Key idea:

- A continuous action expert/head can damage or distort pretrained VLM representations if its gradients are allowed to backpropagate through the shared backbone.
- Their proposed fix is not simply freezing the whole backbone.
- They train robot-relevant shared representations through discrete action/language-style objectives while preventing continuous action-head gradients from flowing back into the VLM backbone.

Relevance to our RynnVLA setup:

- RynnVLA already has two training signals:
  - `c_loss`: discrete action-token / language-model-style loss
  - `loss_ct`: continuous action-head regression loss
- Freezing `lm_head` removes a wasteful output-head update.
- It does not implement knowledge insulation.
- In the current setup, `loss_ct` still flows from `action_head` through hidden states into LoRA adapters.

Likely next hypothesis if the current run remains weak:

> Use knowledge insulation: let `c_loss` update LoRA/backbone adapters, but detach hidden states before `action_head` so the continuous `loss_ct` updates only `action_head`.

That test would isolate whether our low-slope / unstable-joint behavior is partly caused by continuous regression gradients fighting the representation learned by the discrete action-token objective.

## Current Working Hypotheses

1. `lm_head` should stay frozen.
   - The ablation gave no reason to keep it trainable.

2. Action scaling should stay off for now.
   - It did not produce a strong enough signal to justify the extra variable.

3. If `j2` and `j4` remain poor after the current epoch-4 eval, the next infrastructure-level test should be knowledge insulation, not another immediate weight tweak.
   - Weight tuning may still be needed, but KI addresses a more basic gradient-routing question.

4. If KI improves stability or slope learning, then recipe work should proceed from:
   - unscaled actions
   - `lm_head` frozen
   - continuous action loss insulated from LoRA/backbone
   - all-joint high-motion eval at early checkpoints


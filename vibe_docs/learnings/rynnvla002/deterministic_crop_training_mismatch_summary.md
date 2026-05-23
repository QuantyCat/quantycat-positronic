# Deterministic Crop Training/Eval Mismatch Summary

Date: 2026-05-02

## Context

We were investigating why the SO-101 screwdriver policy appeared to learn bad joint directions, especially around joints 1-3. Earlier debugging suggested a sign/convergence issue, so we added stronger action-head supervision and sign-loss pressure. However, the focused train/eval audit later showed a more basic problem: the model was not always seeing the same visual input format during training and evaluation.

The final issue was not the action decode path, joint ordering, action normalization, or action/state convention. Those round-trips checked out. The major remaining mismatch was in image tokenization.

## What Was Wrong

Training examples were pretokenized ahead of time. During pretokenization, images were passed into the image processor as file paths.

Evaluation examples were built from raw episode data. During evaluation, images were loaded into memory first and passed into the image processor as NumPy arrays.

Those two input types were handled differently:

- File-path images were opened and resized to `256 x 256` before tokenization.
- NumPy/PIL/list images were not resized the same way before crop/tokenization.

So even when evaluation was using the same episode, same step, same prompt, and same robot state, the image tokens did not match the tokens that the model had trained on.

The first audit failure showed this clearly:

- Training prefix length through action start: `578`
- Eval prefix length: `578`
- Prefix mismatches: `511 / 578`
- First mismatch was inside the image-token region, not the state/action region.

After fixing image resizing parity, the mismatch dropped to a single token. That last one-token mismatch came from another subtle difference:

- Pretokenization used CUDA for the Chameleon VQ image tokenizer.
- Eval forced the item processor onto CPU.

The VQ tokenizer can differ by a token between CPU and CUDA because the neural image tokenizer does floating-point computation. Even tiny numerical differences can change which discrete image code wins at a boundary. So eval now uses CUDA tokenization like pretokenization.

## Non-Deterministic Cropping, in Plain English

The model does not train directly on the camera image as a whole. Before the image is turned into tokens, the preprocessing code chooses a crop of the image.

Previously, that crop could be random. Imagine showing the model a photo of the table, but each time you make a training example, you cut out a slightly different rectangle from the photo. Sometimes the screwdriver is a little more centered, sometimes it is shifted, sometimes the wrist view includes slightly different surrounding pixels.

That kind of randomness can be useful as data augmentation when used intentionally and consistently. But it becomes a problem when training and evaluation are supposed to be compared exactly:

- The training token file might contain image tokens from one random crop.
- The eval path might generate image tokens from a different crop of the same original frame.
- The prompt and state are identical, but the model is effectively being asked to act from a different picture.

That made our train-vs-eval comparison misleading. We were trying to judge whether the action head behaved differently at eval time, but eval was not feeding the exact same visual prefix that training used.

The fix was to enable deterministic cropping. That means the crop is chosen the same way every time, for example as a fixed center crop after resizing. Given the same source image, training and eval now produce the same image tokens.

## Fixes Made

### 1. Enabled deterministic crop in config

`models/rynnvla-002/config.yaml`

```yaml
deterministic_crop: true
```

This makes preprocessing and eval request deterministic image cropping instead of random crop behavior.

### 2. Regenerated train and validation tokens

Old main token outputs were archived here:

```text
my_data/training_pipeline/tokens/vla_data/archive_20260502T000927Z
```

New deterministic-crop tokens were generated:

```text
my_data/training_pipeline/tokens/vla_data/train/record.json
my_data/training_pipeline/tokens/vla_data/val_ind/record.json
```

Counts:

- Train: `33,824` records
- Validation: `3,269` records

### 3. Fixed deterministic crop plumbing in pretokenization

Files updated:

- `/home/caroline/RynnVLA-002/rynnvla-002/data_lerobot/pre_tokenize_action_state_local.py`
- `/home/caroline/RynnVLA-002/rynnvla-002/data_lerobot/pretoken_lerobot_state.py`
- `/home/caroline/quantycat-positronic/models/rynnvla-002/preprocessing/step5_pretokenize.py`

These changes let the `deterministic_crop` config value flow into the actual worker processes that create token `.pkl` files.

### 4. Fixed image input normalization

File updated:

```text
/home/caroline/RynnVLA-002/rynnvla-002/data_lerobot/item_processor.py
```

The shared LeRobot image processor now normalizes all image input types the same way:

- file path
- PIL image
- NumPy array
- Python list

All are converted to RGB and resized to `target_size x target_size` before crop/tokenization.

This fixed the large prefix mismatch caused by file-path images and NumPy images taking different preprocessing paths.

### 5. Fixed CPU/CUDA tokenizer mismatch in eval

File updated:

```text
/home/caroline/RynnVLA-002/rynnvla-002/eval_solver_lerobot_action_head_state.py
```

Eval now creates its item processor on the same CUDA device used by the model when CUDA is available. This matches the pretokenization path and removes the remaining one-token VQ drift.

### 6. Fixed action-head inference prefix handling

File updated:

```text
/home/caroline/RynnVLA-002/rynnvla-002/model/modeling_xllmx_chameleon_ck_action_head.py
```

When `10004` action-start is already present in `input_ids`, `generate_action_head()` now runs the full prefix through the model and passes full-prefix hidden states into the action head. This makes inference behavior match the teacher-forced training path more closely.

### 7. Added/extended the train-vs-eval mismatch audit

File updated:

```text
/home/caroline/quantycat-positronic/models/rynnvla-002/eval/model_eval/training_eval_mismatch_audit.py
```

The audit now supports:

- deterministic crop mode
- prefix alignment checks
- exact teacher-prefix inference comparison
- `--require-prefix-match` to fail loudly when eval tokens do not match training tokens

## Verification

After the fixes, the end-to-end prefix assertion passed.

Audit report:

```text
/home/caroline/quantycat-positronic/eval_output/screwdriver_so101/model_eval/deterministic_crop_regen_prefix_assert_train4/training_eval_mismatch_audit.json
```

Key results:

```json
{
  "prefix_alignment": {
    "all_match": true,
    "match_count": 2,
    "checked_count": 2,
    "mismatch_count_total": 0,
    "mismatch_count_max": 0
  },
  "deterministic_crop": true,
  "inference_vs_teacher_mae": 0.0013671874767169356,
  "teacher_prefix_vs_teacher_mae": 0.0016194661147892475,
  "teacher_label_vs_episode_gt_mae": 0.0012521430617198348
}
```

Interpretation:

- Eval and training now use the same prompt/image/state token prefix.
- Remaining prediction deltas between eval and teacher-forced paths are tiny, around `0.001-0.002` in normalized action space.
- The previous large mismatch was not a policy-learning issue; it was an input-tokenization mismatch.

## New Training Run

A new training run was started using the regenerated deterministic-crop tokens.

Output directory:

```text
my_data/training_pipeline/fine_tuning/screwdriver_so101_deterministic_crop
```

It resumes weights/optimizer state from:

```text
my_data/training_pipeline/fine_tuning/screwdriver_so101/epoch6
```

The launcher was updated to support a separate run name:

```yaml
training_run_name: screwdriver_so101_deterministic_crop
```

This keeps the previous `screwdriver_so101` checkpoints intact while training against the corrected deterministic token dataset.

## Practical Takeaway

The action-head sign issue may still need monitoring, but we found and fixed a deeper mismatch first: training and eval were not seeing the same visual tokens. With deterministic crop enabled, uniform image resizing, and CUDA tokenization parity, future sign-correlation evals should now measure the model policy rather than artifacts from mismatched image preprocessing.

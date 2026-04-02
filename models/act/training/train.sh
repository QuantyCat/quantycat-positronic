#!/usr/bin/env bash
# Stage 2 — Train ACT
#
# Required env vars:
#   WORK_DIR      pipeline working directory (contains dataset/)
#   OUTPUT_DIR    where to write checkpoints
#
# Optional env vars (defaults shown):
#   TRAIN_STEPS   100000
#   BATCH_SIZE    8
#   CHUNK_SIZE    100
#   KL_WEIGHT     10
#   SAVE_STEPS    10000

set -euo pipefail

: "${WORK_DIR:?WORK_DIR must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

TRAIN_STEPS=${TRAIN_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-8}
CHUNK_SIZE=${CHUNK_SIZE:-100}
KL_WEIGHT=${KL_WEIGHT:-10}
SAVE_STEPS=${SAVE_STEPS:-10000}

DATASET_DIR="${WORK_DIR}/dataset"

echo "Training ACT"
echo "  dataset:     ${DATASET_DIR}"
echo "  output:      ${OUTPUT_DIR}"
echo "  steps:       ${TRAIN_STEPS}"
echo "  batch size:  ${BATCH_SIZE}"
echo "  chunk size:  ${CHUNK_SIZE}"
echo "  kl weight:   ${KL_WEIGHT}"
echo "  save every:  ${SAVE_STEPS} steps"

lerobot-train \
  --policy.type=act \
  --policy.repo_id=local/act \
  --dataset.repo_id=local/dataset \
  --dataset.root="${DATASET_DIR}" \
  --steps="${TRAIN_STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.kl_weight="${KL_WEIGHT}" \
  --save_freq="${SAVE_STEPS}" \
  --output_dir="${OUTPUT_DIR}"

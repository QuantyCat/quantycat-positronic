#!/usr/bin/env bash
# Download pretrained weights for RynnVLA-002 training.
# Run from repo root: bash models/rynnvla-002/run_scripts/download_weights.sh
#
# Requires: huggingface-cli (pip install huggingface_hub)
# Token:    set HF_TOKEN in your environment or in a .env file
#           source .env && bash models/rynnvla-002/run_scripts/download_weights.sh
#
# Downloads:
#   1. Chameleon tokenizer  → models/rynnvla-002/ckpts/chameleon/tokenizer
#   2. Chameleon base model → models/rynnvla-002/ckpts/chameleon/base_model
#   3. RynnVLA-002 starting point → models/rynnvla-002/ckpts/starting_point

set -e

CKPTS_DIR="models/rynnvla-002/ckpts"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "  Set it in your environment or run: source .env && bash $0"
    exit 1
fi

if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli not found. Install it with: pip install huggingface_hub"
    exit 1
fi

echo "=== 1/3 Downloading Chameleon tokenizer (Alpha-VLLM/Lumina-mGPT-7B-768) ==="
huggingface-cli download Alpha-VLLM/Lumina-mGPT-7B-768 \
    --local-dir "$CKPTS_DIR/chameleon/tokenizer" \
    --token "$HF_TOKEN"

echo ""
echo "=== 2/3 Downloading Chameleon base model (facebook/chameleon-7b) ==="
echo "    Note: requires accepting Meta's Chameleon license on HuggingFace first."
huggingface-cli download facebook/chameleon-7b \
    --local-dir "$CKPTS_DIR/chameleon/base_model" \
    --token "$HF_TOKEN"

echo ""
echo "=== 3/3 Downloading RynnVLA-002 starting point (Alibaba-DAMO-Academy/RynnVLA-002) ==="
huggingface-cli download Alibaba-DAMO-Academy/RynnVLA-002 \
    --local-dir "$CKPTS_DIR/starting_point" \
    --token "$HF_TOKEN"

echo ""
echo "Done. Weights are under $CKPTS_DIR/"
echo "  chameleon/tokenizer  — Lumina-mGPT tokenizer"
echo "  chameleon/base_model — Chameleon 7B base weights"
echo "  starting_point       — RynnVLA-002 pretrained checkpoint"

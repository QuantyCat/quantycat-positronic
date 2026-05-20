"""
Download pretrained weights for RynnVLA-002 training.

All three come from Alibaba-DAMO-Academy/WorldVLA.

Run from repo root:
    source .env && python3 models/rynnvla-002/run_scripts/download_weights.py

Downloads:
    chameleon/tokenizer          → models/rynnvla-002/ckpts/chameleon/tokenizer
    base_model                   → models/rynnvla-002/ckpts/chameleon/base_model
    chameleon/starting_point     → models/rynnvla-002/ckpts/starting_point
"""

import os
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN is not set. Run: export HF_TOKEN=your_token")
    sys.exit(1)

REPO     = "Alibaba-DAMO-Academy/WorldVLA"
CKPTS = str((Path(__file__).resolve().parents[3] / "vendor/rynnvla-002/rynnvla-002/ckpts").resolve())

# 1. chameleon/tokenizer → ckpts/chameleon/tokenizer
print("\n=== 1/3 Downloading chameleon/tokenizer ===")
snapshot_download(
    repo_id=REPO,
    local_dir=CKPTS,
    allow_patterns=["chameleon/tokenizer/**"],
    token=token,
)
print(f"    → {CKPTS}/chameleon/tokenizer")

# 2. base_model → ckpts/chameleon/base_model
#    (repo has base_model/ at root, we want it under chameleon/)
print("\n=== 2/3 Downloading base_model ===")
snapshot_download(
    repo_id=REPO,
    local_dir=f"{CKPTS}/chameleon",
    allow_patterns=["base_model/**"],
    token=token,
)
print(f"    → {CKPTS}/chameleon/base_model")

# 3. chameleon/starting_point → ckpts/starting_point
#    (repo nests it under chameleon/, we want it directly under ckpts/)
print("\n=== 3/3 Downloading chameleon/starting_point ===")
tmp = f"{CKPTS}/_tmp_starting_point"
snapshot_download(
    repo_id=REPO,
    local_dir=tmp,
    allow_patterns=["chameleon/starting_point/**"],
    token=token,
)
shutil.move(f"{tmp}/chameleon/starting_point", f"{CKPTS}/starting_point")
shutil.rmtree(tmp)
print(f"    → {CKPTS}/starting_point")

print("\nDone.")
print(f"  {CKPTS}/chameleon/tokenizer")
print(f"  {CKPTS}/chameleon/base_model")
print(f"  {CKPTS}/starting_point")

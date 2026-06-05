#!/usr/bin/env bash
# Sync vendor_patches into vendor/openpi/src.
# Run this after editing any file under models/openpi/vendor_patches/src/.
#
# Usage from the repo root:
#   bash models/openpi/vendor_patches/sync.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/src"
DST="$(cd "$SCRIPT_DIR/../../.." && pwd)/vendor/openpi/src"

if [ ! -d "$DST" ]; then
    echo "ERROR: vendor/openpi/src not found at: $DST"
    exit 1
fi

echo "Syncing vendor_patches → vendor/openpi/src"

find "$SRC" -type f -name "*.py" | while read -r file; do
    rel="${file#$SRC/}"
    dest="$DST/$rel"
    mkdir -p "$(dirname "$dest")"
    cp "$file" "$dest"
    echo "  $rel"
done

echo "Done."

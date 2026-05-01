#!/usr/bin/env bash
# Run the evaluation container locally against test data.
#
# Expected data layout at the repository root (mirrors the GC zip extraction):
#
#   <repo_root>/
#     ground_truth/          ← real post-contrast .mha slices  (same folder name as in the zip)
#     masks/                 ← binary tumour masks .mha         (same folder name as in the zip)
#     test/
#       input/               ← predictions (flat .mha or GC predictions.json layout)
#       output/              ← written by this script
#     src/evaluation/models/ ← classifiers + nnUNet weights
#
# Docker mount strategy
# ─────────────────────
# GC extracts ground_truth.zip to /opt/ml/input/data/ground_truth/.
# The zip top-level entries are ground_truth/ and masks/, so after
# extraction the container sees:
#   /opt/ml/input/data/ground_truth/ground_truth/*.mha
#   /opt/ml/input/data/ground_truth/masks/*.mha
#
# We replicate this by mounting the REPOSITORY ROOT as
# /opt/ml/input/data/ground_truth — the local ground_truth/ and masks/
# folders then appear at exactly the paths evaluate.py expects.
#
# This container image is used unchanged for the debug, validation, and
# test phases on Grand Challenge; only the uploaded ground truth data
# differs between phases.
#
# Usage:
#   ./do_test_run.sh
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

# ---------------------------------------------------------------------------
# GPU + memory detection
# ---------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=memory.total \
       --format=csv,noheader,nounits &>/dev/null 2>&1; then
    GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total \
        --format=csv,noheader,nounits | head -1)
    # Convert MiB → GB and add 4 GB headroom for the full pipeline
    MEMORY_GB=$(( (GPU_MEM_MIB / 1024) + 4 ))
    MEMORY_LIMIT="${MEMORY_GB}g"
    DOCKER_GPU_FLAG="--gpus all"
    echo "[INFO] GPU detected: ${GPU_MEM_MIB} MiB VRAM → setting --memory=${MEMORY_LIMIT}"
else
    # No GPU: 16 GB covers nnUNet + radiomics + LPIPS on CPU
    MEMORY_LIMIT="16g"
    DOCKER_GPU_FLAG=""
    echo "[INFO] No GPU detected → setting --memory=${MEMORY_LIMIT} (CPU mode)"
fi

# Clean previous output
rm -rf "$SCRIPT_DIR/test/output"
mkdir -p "$SCRIPT_DIR/test/output"

docker run --rm \
    --memory="${MEMORY_LIMIT}" \
    ${DOCKER_GPU_FLAG} \
    -v "$SCRIPT_DIR/test/input:/input:ro" \
    -v "$SCRIPT_DIR/test/output:/output" \
    -v "$SCRIPT_DIR:/opt/ml/input/data/ground_truth:ro" \
    -v "$SCRIPT_DIR/src/evaluation/models:/opt/app/models:ro" \
    mama-synth-gc-eval

echo ""
echo "=== metrics.json ==="
cat "$SCRIPT_DIR/test/output/metrics.json"

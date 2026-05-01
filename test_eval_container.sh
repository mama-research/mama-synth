#!/usr/bin/env bash
# Quick local test for the MAMA-SYNTH evaluation container.
#
# ─── What this script tests ───────────────────────────────────────────────────
# It starts the evaluation container in "local / flat-directory" mode (no
# predictions.json), which is triggered automatically when /input does NOT
# contain a predictions.json file.  evaluate.py then reads:
#
#   Predictions  : /input/*.mha                        (flat, one file per case)
#   GT images    : /opt/ml/input/data/ground_truth/ground_truth/*.mha
#   Masks        : /opt/ml/input/data/ground_truth/masks/*.mha
#   Models       : /opt/app/models/
#   Output       : /output/metrics.json
#
# ─── Required data layout at the repo root ────────────────────────────────────
#
#   <repo_root>/
#     ground_truth/          ← post-contrast .mha slices (same name as in GC zip)
#       patient_001.mha
#       patient_002.mha
#     masks/                 ← binary tumour masks (same name as in GC zip)
#       patient_001.mha
#       patient_002.mha
#     test/
#       input/               ← predicted synthetic images (flat .mha, one per case)
#         patient_001.mha    ← filenames must match ground_truth/ and masks/
#         patient_002.mha
#     src/evaluation/models/ ← classifiers + nnUNet weights (already in repo)
#
# ─── Docker mount strategy ───────────────────────────────────────────────────
# GC extracts ground_truth.zip to /opt/ml/input/data/ground_truth/.
# The zip top-level entries are ground_truth/ and masks/, so inside the
# container the paths are:
#   /opt/ml/input/data/ground_truth/ground_truth/patient_001.mha
#   /opt/ml/input/data/ground_truth/masks/patient_001.mha
#
# We replicate this exactly by mounting the REPO ROOT as
# /opt/ml/input/data/ground_truth — the local ground_truth/ and masks/
# folders then appear at the same container paths that GC uses.
#
# ─── Usage ───────────────────────────────────────────────────────────────────
#   # Option A — use the already-built image:
#   ./test_eval_container.sh
#
#   # Option B — load from the exported .tar.gz first:
#   docker load < mama-synth-gc-eval-v1.0.0.tar.gz
#   ./test_eval_container.sh
# ──────────────────────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
IMAGE="mama-synth-gc-eval"

# ── Preflight checks ──────────────────────────────────────────────────────────
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "ERROR: Docker image '$IMAGE' not found."
    echo "  Build it:  ./do_build.sh"
    echo "  Or load:   docker load < mama-synth-gc-eval-v1.0.0.tar.gz"
    exit 1
fi

for dir in "$SCRIPT_DIR/ground_truth" "$SCRIPT_DIR/masks" "$SCRIPT_DIR/test/input"; do
    if [ -z "$(ls -A "$dir"/*.mha 2>/dev/null)" ]; then
        echo "ERROR: No .mha files found in $dir"
        echo "  Place matching patient_XXX.mha files in ground_truth/, masks/, and test/input/"
        exit 1
    fi
done

# ── GPU + memory detection ────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=memory.total \
       --format=csv,noheader,nounits &>/dev/null 2>&1; then
    GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    MEMORY_LIMIT="$(( (GPU_MEM_MIB / 1024) + 4 ))g"
    DOCKER_GPU_FLAG="--gpus all"
    echo "[INFO] GPU detected: ${GPU_MEM_MIB} MiB → --memory=${MEMORY_LIMIT}"
else
    MEMORY_LIMIT="16g"
    DOCKER_GPU_FLAG=""
    echo "[INFO] No GPU → --memory=${MEMORY_LIMIT} (CPU mode)"
fi

# ── Clean output ──────────────────────────────────────────────────────────────
rm -rf "$SCRIPT_DIR/test/output"
mkdir -p "$SCRIPT_DIR/test/output"

# ── Run ───────────────────────────────────────────────────────────────────────
docker run --rm \
    --memory="${MEMORY_LIMIT}" \
    ${DOCKER_GPU_FLAG} \
    -v "$SCRIPT_DIR/test/input:/input:ro" \
    -v "$SCRIPT_DIR/test/output:/output" \
    -v "$SCRIPT_DIR:/opt/ml/input/data/ground_truth:ro" \
    -v "$SCRIPT_DIR/src/evaluation/models:/opt/app/models:ro" \
    "$IMAGE"

# ── Result ────────────────────────────────────────────────────────────────────
echo ""
echo "=== metrics.json ==="
cat "$SCRIPT_DIR/test/output/metrics.json"

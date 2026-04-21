#!/usr/bin/env bash
# Run the evaluation container locally against test data.
#
# Before running, ensure the following directories exist:
#   test/input/          ← predictions (flat .mha) or GC predictions.json layout
#   ground_truth/images/ ← real post-contrast .mha slices
#   ground_truth/masks/  ← binary tumour masks .mha
#   ground_truth/precontrast/ ← pre-contrast .mha slices
#
# Usage:
#   ./do_test_run.sh
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

# Clean previous output
rm -rf "$SCRIPT_DIR/test/output"
mkdir -p "$SCRIPT_DIR/test/output"

docker run --rm \
    --memory=4g \
    -v "$SCRIPT_DIR/test/input:/input:ro" \
    -v "$SCRIPT_DIR/test/output:/output" \
    -v "$SCRIPT_DIR/ground_truth:/opt/ml/input/data/ground_truth:ro" \
    -v "$SCRIPT_DIR/models:/opt/app/models:ro" \
    mama-synth-gc-eval

echo ""
echo "=== metrics.json ==="
cat "$SCRIPT_DIR/test/output/metrics.json"

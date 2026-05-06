#!/usr/bin/env bash
# Run the identity-baseline container locally against a test input.
#
# Layout expected under this directory:
#
#   test/
#     input/
#       images/
#         pre-contrast-breast-mri/
#           <any-name>.mha          ← place a real pre-contrast .mha here
#     output/                       ← created by this script; check result here
#
# After a successful run the synthetic output will be at:
#   test/output/images/synthetic-post-contrast-breast-mri/output.mha
#
# Usage:
#   ./do_test_run.sh
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

# Rebuild the image before each test run
bash "$SCRIPT_DIR/do_build.sh"

# Clean previous output
rm -rf "$SCRIPT_DIR/test/output"
mkdir -p "$SCRIPT_DIR/test/output"

# Verify that a test input file exists
INPUT_DIR="$SCRIPT_DIR/test/input/images/pre-contrast-dce-mri-slice-breast"
if [ -z "$(ls -A "$INPUT_DIR"/*.mha 2>/dev/null)" ]; then
    echo ""
    echo "ERROR: No .mha file found in $INPUT_DIR"
    echo "Place a pre-contrast .mha slice there before running this script."
    echo ""
    exit 1
fi

docker run --rm \
    --network=none \
    --memory=4g \
    -v "$SCRIPT_DIR/test/input:/input:ro" \
    -v "$SCRIPT_DIR/test/output:/output" \
    mama-synth-identity-baseline

echo ""
echo "=== Output files ==="
find "$SCRIPT_DIR/test/output" -type f

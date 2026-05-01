#!/usr/bin/env bash
# Quick local test for the MAMA-SYNTH identity-baseline algorithm container.
#
# ─── What this script tests ───────────────────────────────────────────────────
# It runs the identity-baseline algorithm container against a single pre-contrast
# .mha slice and verifies that the synthetic output file appears at the correct
# path.  The I/O contract mirrors what Grand Challenge expects:
#
#   Input  : /input/images/pre-contrast-breast-mri/<any>.mha
#   Output : /output/images/synthetic-post-contrast-breast-mri/output.mha
#
# ─── Required data layout ────────────────────────────────────────────────────
#
#   src/submission/identity-baseline/
#     test/
#       input/
#         images/
#           pre-contrast-breast-mri/
#             patient_test.mha     ← place any mha file here
#
# A synthetic test file is already present from the initial test run.
# Replace it with a real preprocessed slice if you want to inspect real values.
#
# ─── Usage ───────────────────────────────────────────────────────────────────
#   # From the repo root:
#   bash src/submission/test_submission_container.sh
#
#   # Or from this directory:
#   ./test_submission_container.sh
#
#   # Load from tar.gz first if the image isn't built:
#   docker load < identity-baseline/mama-synth-identity-baseline-v1.0.0.tar.gz
# ──────────────────────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
BASELINE_DIR="$SCRIPT_DIR/identity-baseline"
IMAGE="mama-synth-identity-baseline"
INPUT_DIR="$BASELINE_DIR/test/input/images/pre-contrast-breast-mri"
OUTPUT_DIR="$BASELINE_DIR/test/output"
EXPECTED_OUTPUT="$OUTPUT_DIR/images/synthetic-post-contrast-breast-mri/output.mha"

# ── Preflight checks ──────────────────────────────────────────────────────────
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "ERROR: Docker image '$IMAGE' not found."
    echo "  Build it:  bash $BASELINE_DIR/do_build.sh"
    echo "  Or load:   docker load < $BASELINE_DIR/mama-synth-identity-baseline-v1.0.0.tar.gz"
    exit 1
fi

if [ -z "$(ls -A "$INPUT_DIR"/*.mha 2>/dev/null)" ]; then
    echo "ERROR: No .mha file found in $INPUT_DIR"
    echo "  Place a pre-contrast .mha slice there before running this test."
    exit 1
fi

# ── Clean output ──────────────────────────────────────────────────────────────
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ── Run (no network, minimal memory — this model is a trivial copy) ───────────
docker run --rm \
    --network=none \
    --memory=2g \
    -v "$BASELINE_DIR/test/input:/input:ro" \
    -v "$OUTPUT_DIR:/output" \
    "$IMAGE"

# ── Verify output ─────────────────────────────────────────────────────────────
if [ ! -f "$EXPECTED_OUTPUT" ]; then
    echo ""
    echo "FAIL: expected output not found at:"
    echo "  $EXPECTED_OUTPUT"
    echo ""
    echo "Files actually produced:"
    find "$OUTPUT_DIR" -type f
    exit 1
fi

echo ""
echo "PASS: output found at:"
echo "  $EXPECTED_OUTPUT"
echo ""
echo "=== Output files ==="
find "$OUTPUT_DIR" -type f

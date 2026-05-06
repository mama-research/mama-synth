#!/usr/bin/env bash
# Export the identity-baseline container for upload to Grand Challenge.
#
# Bump VERSION before each upload so different iterations are distinguishable.
#
# Upload instructions:
#   1. Run this script to produce the .tar.gz
#   2. On grand-challenge.org go to your Algorithm page
#      → Container Management → Upload a new container
#   3. Upload the produced .tar.gz file
#   4. Wait for GC to validate the image (~up to 24 h)
#   5. Submit the algorithm to the desired challenge phase
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

VERSION="v1.0.1"
OUT_FILE="$SCRIPT_DIR/mama-synth-identity-baseline-${VERSION}.tar.gz"

bash "$SCRIPT_DIR/do_build.sh"
docker save mama-synth-identity-baseline | gzip -c > "$OUT_FILE"
echo "Saved to $OUT_FILE"

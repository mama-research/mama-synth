#!/usr/bin/env bash
# Export the evaluation container for upload to Grand Challenge.
#
# Bump VERSION before each upload so uploads are distinguishable on GC.
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

VERSION="v1.0.0"
OUT_FILE="$SCRIPT_DIR/mama-synth-gc-eval-${VERSION}.tar.gz"

docker save mama-synth-gc-eval | gzip -c > "$OUT_FILE"
echo "Saved to $OUT_FILE"

#!/usr/bin/env bash
# Export the evaluation container for upload to Grand Challenge.
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
docker save mama-synth-gc-eval | gzip -c > "$SCRIPT_DIR/mama-synth-gc-eval.tar.gz"
echo "Saved to $SCRIPT_DIR/mama-synth-gc-eval.tar.gz"

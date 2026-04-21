#!/usr/bin/env bash
# Build the evaluation container image.
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
docker build -t mama-synth-gc-eval "$SCRIPT_DIR"

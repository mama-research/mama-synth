#!/usr/bin/env bash
# Build the evaluation container image.
#
# The build context is the repository root (this directory).
# Dockerfile COPY paths reference src/evaluation/ explicitly, so all
# evaluation code, models, and requirements.txt are found correctly.
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
docker build -t mama-synth-gc-eval "$SCRIPT_DIR"

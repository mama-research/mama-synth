#!/usr/bin/env bash
# Build the identity-baseline algorithm container image.
#
# Build context is this directory (src/submission/identity-baseline/).
# --no-cache ensures source file changes (inference.py etc.) are always
# picked up — Docker layer caching can silently reuse stale layers otherwise.
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
docker build --no-cache -t mama-synth-identity-baseline "$SCRIPT_DIR"

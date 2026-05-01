#!/usr/bin/env bash
# Build the identity-baseline algorithm container image.
#
# Build context is this directory (src/submission/identity-baseline/).
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
docker build -t mama-synth-identity-baseline "$SCRIPT_DIR"

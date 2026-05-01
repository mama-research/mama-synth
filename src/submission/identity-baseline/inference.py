#!/usr/bin/env python3
"""MAMA-SYNTH — Identity Baseline Algorithm.

This is the simplest possible submission: it copies the input pre-contrast
image directly to the output, producing no synthesis.  It serves as:

  * A working end-to-end smoke-test for the GC infrastructure.
  * A lower-bound reference on the leaderboard.
  * A template showing the correct I/O contract for MAMA-SYNTH submissions.

Grand Challenge I/O contract
-----------------------------
Input  (one file per job):
    /input/images/pre-contrast-breast-mri/<uuid>.mha

Output (one file per job):
    /output/images/synthetic-post-contrast-breast-mri/output.mha

All images are 2-D z-score-normalised float32 .mha files produced by the
MAMA-SYNTH preprocessing pipeline.  Spacing and image metadata from the
input are preserved in the output.
"""

from __future__ import annotations

import os
import shutil
import sys
from glob import glob
from pathlib import Path

import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Path constants  (override via environment for local testing)
# ---------------------------------------------------------------------------
INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))

# Interface slugs — must match the challenge phase configuration on GC.
INPUT_SLUG = os.environ.get("MAMA_INPUT_SLUG", "pre-contrast-breast-mri")
OUTPUT_SLUG = os.environ.get(
    "MAMA_PREDICTION_SLUG", "synthetic-post-contrast-breast-mri"
)


def _find_input_image() -> Path:
    """Return the single input .mha file for this job."""
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates = glob(str(search_dir / "*.mha"))
    if not candidates:
        raise FileNotFoundError(
            f"No .mha file found in {search_dir}. "
            "Check that the input interface slug matches the challenge phase."
        )
    if len(candidates) > 1:
        print(
            f"WARNING: {len(candidates)} files found in {search_dir}; "
            "using the first one.",
            file=sys.stderr,
        )
    return Path(candidates[0])


def run() -> int:
    """Main entry point."""
    print("=" * 50)
    print("MAMA-SYNTH Identity Baseline")
    print("=" * 50)

    # --- 1. Locate input ------------------------------------------------
    input_file = _find_input_image()
    print(f"Input : {input_file}")

    # --- 2. Load with SimpleITK (preserves all metadata) ----------------
    image = sitk.ReadImage(str(input_file))
    print(
        f"  Size   : {image.GetSize()}"
        f"  Spacing: {image.GetSpacing()}"
        f"  Origin : {image.GetOrigin()}"
    )

    # --- 3. Identity "synthesis" — the pre-contrast image IS the output -
    #        In a real model you would replace this with your inference call.
    output_image = image  # no-op copy; metadata preserved automatically

    # --- 4. Write output ------------------------------------------------
    output_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.mha"

    sitk.WriteImage(output_image, str(output_file), useCompression=True)
    print(f"Output: {output_file}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

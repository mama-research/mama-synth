#!/usr/bin/env python3
"""Automated test for the identity-baseline algorithm container.

Runs do_test_run.sh and verifies that the output file is produced and can
be read back as a valid 2-D image.

Prerequisites:
  - Docker installed and running
  - A test .mha file placed in:
      test/input/images/pre-contrast-breast-mri/

Run:
  pytest test_algorithm.py -v
  # or directly:
  python test_algorithm.py
"""

from __future__ import annotations

import subprocess
import sys
from glob import glob
from pathlib import Path

import SimpleITK as sitk

SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "test" / "input" / "images" / "pre-contrast-breast-mri"
OUTPUT_DIR = SCRIPT_DIR / "test" / "output" / "images" / "synthetic-post-contrast-breast-mri"


def test_output_file_created():
    """Container should produce exactly one .mha output file."""
    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "do_test_run.sh")],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
    )
    assert result.returncode == 0, (
        f"do_test_run.sh failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    output_file = OUTPUT_DIR / "output.mha"
    assert output_file.exists(), (
        f"Expected output not found at {output_file}\n"
        f"Files in output dir: {list(OUTPUT_DIR.parent.rglob('*'))}"
    )


def test_output_is_readable_2d_image():
    """Output .mha must be readable by SimpleITK and be a 2-D image."""
    output_file = OUTPUT_DIR / "output.mha"
    assert output_file.exists(), "Run test_output_file_created() first."

    img = sitk.ReadImage(str(output_file))
    arr = sitk.GetArrayFromImage(img)

    # After squeezing a size-1 leading dimension the array must be 2-D
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    assert arr.ndim == 2, f"Expected 2-D array, got shape {arr.shape}"


def test_output_matches_input_shape():
    """The identity baseline must preserve the spatial shape of the input."""
    input_files = glob(str(INPUT_DIR / "*.mha"))
    assert input_files, f"No .mha input found in {INPUT_DIR}"

    input_img = sitk.ReadImage(input_files[0])
    output_img = sitk.ReadImage(str(OUTPUT_DIR / "output.mha"))

    assert input_img.GetSize() == output_img.GetSize(), (
        f"Size mismatch: input {input_img.GetSize()} vs output {output_img.GetSize()}"
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

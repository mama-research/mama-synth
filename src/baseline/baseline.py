#  Copyright 2025 mama-synth-eval contributors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Synthesis pipeline for the MAMA-SYNTH baseline (Pix2PixHD via medigan)."""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

MEDIGAN_MODEL_ID = "00023_PIX2PIXHD_BREAST_DCEMRI"
IMAGES_SUBDIR = "images"
WORK_SUBDIR = ".synthesis_work"


def _normalize_gpu_id(gpu_id: str) -> str:
    """Convert a CLI GPU id to a PyTorch device string (e.g. '0' → 'cuda:0')."""
    gpu_id = gpu_id.strip()
    if gpu_id in ("cpu", "cuda") or gpu_id.startswith("cuda:"):
        return gpu_id
    try:
        idx = int(gpu_id)
    except ValueError:
        return gpu_id
    return "cpu" if idx < 0 else f"cuda:{idx}"


def _ensure_medigan_importable() -> None:
    """Add CWD to sys.path and create models/__init__.py for medigan model imports."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    models_pkg = Path(cwd) / "models"
    models_pkg.mkdir(exist_ok=True)
    init_file = models_pkg / "__init__.py"
    if not init_file.exists():
        init_file.touch()


def _find_generated_images(root: Path) -> list[Path]:
    """Return all image files under *root*, recursively sorted."""
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )
    if not images:
        raise FileNotFoundError(
            f"No image files found in {root} — medigan may not have produced output."
        )
    return images


def synthesize_with_medigan(
    input_dir: Path,
    output_dir: Path,
    model_id: str = MEDIGAN_MODEL_ID,
    gpu_id: str = "0",
    image_size: int = 512,
    keep_work_dir: bool = False,
    lo_percentile: float = 5.0,
    hi_percentile: float = 95.0,
) -> list[Path]:
    """Run Pix2PixHD synthesis on a directory of 2D MHA images.

    Each MHA file in *input_dir* is converted to an 8-bit grayscale PNG,
    passed through medigan, and the synthetic output is rescaled to z-score
    float space using a fully per-patient linear mapping derived from the
    pre-contrast input:

        scale = (P_hi(input) - P_lo(input)) / (P_hi(raw) - P_lo(raw))
        shift = P_lo(input) - scale * P_lo(raw)

    This maps the model's output range onto the input's z-score range so that
    background tissue is anchored to the patient's own acquisition level and
    the contrast enhancement above it is whatever the model predicted,
    expressed in the same z-score units as the input.  No external calibration
    file is required.

    A visualization PNG (0–255, per-image min–max) is also written to
    ``output_dir/viz/``.  Returns paths to all generated MHA files.
    """
    try:
        from medigan import Generators
    except ImportError:
        raise ImportError("Install the 'medigan' package to use synthesis.")

    _ensure_medigan_importable()
    device_str = _normalize_gpu_id(gpu_id)
    generators = Generators()
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    work_root = output_dir / WORK_SUBDIR
    work_root.mkdir(exist_ok=True)

    input_images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".mha"
    )
    if not input_images:
        raise FileNotFoundError(f"No .mha files found in {input_dir}.")

    logger.info(
        f"Synthesizing {len(input_images)} images "
        f"(model={model_id}, device={device_str}, size={image_size})"
    )

    generated_files: list[Path] = []

    for img_path in input_images:
        patient_id = img_path.stem
        patient_work = work_root / patient_id
        work_input = patient_work / "input"
        work_output = patient_work / "output"

        try:
            if patient_work.exists():
                shutil.rmtree(patient_work)
            work_input.mkdir(parents=True)
            work_output.mkdir(parents=True)

            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(np.float64)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim != 2:
                raise ValueError(f"Expected 2D MHA, got shape {arr.shape} for {img_path.name}")
            # pre contrast which is z-score normalized
            native_size = (arr.shape[1], arr.shape[0])  # (width, height) for PIL
            vmin, vmax = arr.min(), arr.max()
            # re  normalized to 0-255
            norm = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8) if vmax > vmin else np.zeros_like(arr, dtype=np.uint8)
            PILImage.fromarray(norm, mode="L").save(work_input / "slice_0000.png")

            # goes into the generator
            generators.generate(
                model_id=model_id,
                input_path=str(work_input),
                output_path=str(work_output),
                num_samples=1,
                save_images=True,
                image_size=str(image_size),
                gpu_id=device_str,
            )
            
            produced = sorted(_find_generated_images(work_output))
            with PILImage.open(produced[0]) as out_img:
                if out_img.size != native_size:
                    out_img = out_img.resize(native_size, PILImage.BICUBIC)
                raw = np.array(out_img.convert("L"), dtype=np.float32)

            # Per-patient linear rescaling: map model output range onto input z-score range.
            # scale stretches the output to match the input's dynamic range;
            # shift anchors the background to the patient's own pre-contrast level.
            p_lo_in  = float(np.percentile(arr, lo_percentile))
            p_hi_in  = float(np.percentile(arr, hi_percentile))
            p_lo_raw = float(np.percentile(raw, lo_percentile))
            p_hi_raw = float(np.percentile(raw, hi_percentile))
            denom = p_hi_raw - p_lo_raw
            if denom == 0.0:
                raise ValueError("Model output has zero dynamic range — cannot rescale.")
            scale = (p_hi_in - p_lo_in) / denom
            shift = p_lo_in - scale * p_lo_raw
            z_hat = (scale * raw + shift).astype(np.float32)

            # Primary output: float32 MHA
            dest_mha = output_dir / f"{patient_id}.mha"
            sitk.WriteImage(sitk.GetImageFromArray(z_hat), str(dest_mha))

            # Visualization PNG: per-image min–max → 0–255
            z_min, z_max = z_hat.min(), z_hat.max()
            if z_max > z_min:
                viz = ((z_hat - z_min) / (z_max - z_min) * 255).astype(np.uint8)
            else:
                viz = np.zeros_like(z_hat, dtype=np.uint8)
            PILImage.fromarray(viz, mode="L").save(viz_dir / f"{patient_id}.png")

            generated_files.append(dest_mha)
            logger.debug(f"Generated: {dest_mha}")

            if not keep_work_dir:
                shutil.rmtree(patient_work, ignore_errors=True)

        except Exception as e:
            logger.warning(f"Synthesis failed for {patient_id}: {e}")

    if not keep_work_dir:
        try:
            work_root.rmdir()
        except OSError:
            pass

    logger.info(f"Done: {len(generated_files)} MHA(s) → {output_dir}  |  viz PNGs → {viz_dir}")
    return generated_files


def parse_synthesize_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mamasynth-synthesize",
        description="Generate synthetic post-contrast DCE-MRI images (Pix2PixHD via medigan).",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Root dataset directory; input images loaded from <data-dir>/images/.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory of 2D pre-contrast MHA images.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save generated PNG images.")
    parser.add_argument("--model-id", type=str, default=MEDIGAN_MODEL_ID,
                        help=f"Medigan model ID. Default: {MEDIGAN_MODEL_ID}.")
    parser.add_argument("--gpu-id", type=str, default="0",
                        help="GPU device (e.g. '0', 'cuda:0', '-1' for CPU). Default: 0.")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Model input resolution. Default: 512.")
    parser.add_argument("--keep-work-dir", action="store_true",
                        help="Keep intermediate staging directories after synthesis.")
    parser.add_argument(
        "--lo-percentile",
        type=float,
        default=5.0,
        help="Lower percentile used for per-patient rescaling (default: 5.0).",
    )
    parser.add_argument(
        "--hi-percentile",
        type=float,
        default=95.0,
        help="Upper percentile used for per-patient rescaling (default: 95.0).",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging.")

    args = parser.parse_args(argv)

    if args.input_dir is None:
        if args.data_dir is not None:
            args.input_dir = args.data_dir
        else:
            parser.error("Provide --input-dir or --data-dir.")

    return args


def run_baseline(argv: Optional[list[str]] = None) -> None:
    args = parse_synthesize_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    generated = synthesize_with_medigan(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        gpu_id=args.gpu_id,
        image_size=args.image_size,
        keep_work_dir=args.keep_work_dir,
        lo_percentile=args.lo_percentile,
        hi_percentile=args.hi_percentile,
    )
    logger.info(f"Generated {len(generated)} synthetic MHA(s).")


if __name__ == "__main__":
    run_baseline()

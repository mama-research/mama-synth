"""
compute_dataset_stats.py — Dataset-level Pre-contrast Normalisation Statistics

For every patient in the DCE-MRI image directory this script:
  1. Identifies the pre-contrast phase as the volume with the lowest phase index.
  2. Loads the full 3D volume.
  3. Accumulates voxel statistics across the entire dataset using Welford's
     online algorithm, which is memory-efficient (no need to store all voxels).

The resulting mean and standard deviation are written to a JSON file that can
be passed to preprocess.py via the --global_stats argument to apply consistent,
dataset-level z-score normalisation across training and test splits.

Expected input layout
---------------------
    image_dir/<patient_id>/<patient_id>_<phase_index>.nii.gz

Usage
-----
    python compute_dataset_stats.py \
        --image_dir /path/to/images \
        --output_path /path/to/dataset_stats.json

Output JSON format
------------------
    {
        "mean": 123.45,
        "std": 67.89,
        "n_voxels": 1234567890,
        "n_patients": 105
    }
"""
import argparse
import json
import logging
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_volume(path: Path) -> np.ndarray:
    p = str(path)
    if p.endswith(('.nii.gz', '.nii')):
        return nib.load(p).get_fdata().astype(np.float32).ravel()
    return sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32).ravel()


def find_pre_contrast_file(patient_folder: Path):
    """Return the file with the lowest phase number in patient_folder."""
    candidates = []
    for f in patient_folder.iterdir():
        if not f.is_file():
            continue
        if f.suffix not in ('.nii', '.gz', '.mha'):
            continue
        stem = f.stem.replace('.nii', '')
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            try:
                candidates.append((int(parts[-1]), f))
            except ValueError:
                pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def compute_stats(image_dir: Path):
    """
    Welford online algorithm for global mean and variance.
    Iterates over pre-contrast volumes without loading all data at once.
    """
    n_total = 0      # total voxel count
    mean = 0.0
    M2 = 0.0         # running sum of squared deviations
    n_patients = 0

    patients = sorted(p for p in image_dir.iterdir() if p.is_dir())
    logger.info(f"Found {len(patients)} patient folders")

    for patient_folder in patients:
        pre_file = find_pre_contrast_file(patient_folder)
        if pre_file is None:
            logger.warning(f"No pre-contrast file found in {patient_folder.name}, skipping")
            continue

        try:
            voxels = load_volume(pre_file)
            voxels = voxels[np.isfinite(voxels)]  # drop NaN / Inf
        except Exception as e:
            logger.error(f"Could not load {pre_file}: {e}")
            continue

        # Batched Welford update
        n_batch = len(voxels)
        if n_batch == 0:
            continue

        batch_mean = float(np.mean(voxels))
        batch_var = float(np.var(voxels))

        # Parallel / batch merge
        n_new = n_total + n_batch
        delta = batch_mean - mean
        mean = (n_total * mean + n_batch * batch_mean) / n_new
        M2 = M2 + n_batch * batch_var + delta ** 2 * n_total * n_batch / n_new
        n_total = n_new
        n_patients += 1

        logger.info(f"  [{n_patients:>4d}] {patient_folder.name}  "
                    f"voxels={n_batch:,}  running mean={mean:.4f}")

    if n_total == 0:
        raise RuntimeError("No voxels accumulated — check image_dir path and file format.")

    std = math.sqrt(M2 / n_total)
    return {"mean": mean, "std": std, "n_voxels": n_total, "n_patients": n_patients}


def main():
    parser = argparse.ArgumentParser(
        description="Compute global pre-contrast normalisation stats from a DCE-MRI dataset."
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Root directory containing per-patient sub-folders with phase files."
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Path to write the output JSON stats file."
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing stats from pre-contrast volumes in: {image_dir}")
    stats = compute_stats(image_dir)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"\nDone. Processed {stats['n_patients']} patients "
        f"({stats['n_voxels']:,} voxels).\n"
        f"  mean = {stats['mean']:.6f}\n"
        f"  std  = {stats['std']:.6f}\n"
        f"Stats written to: {output_path}"
    )


if __name__ == "__main__":
    main()

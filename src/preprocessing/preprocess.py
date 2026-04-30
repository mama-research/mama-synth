
"""
preprocess.py — 3D DCE-MRI to 2D Pre/Peak Slice Preprocessing

Pipeline
--------
1. For each patient, load all 3D phase volumes and the tumour segmentation.
2. Identify the peak enhancement phase as the phase with the highest mean
   tumour intensity across the full 3D volume.
3. Select the 2D slice with the largest tumour area.
   The through-plane axis is determined via a layered heuristic (see
   ``Preprocessor.determine_slice_axis`` for full details).
4. Z-score normalise the pre-contrast and peak-enhancement 2D slices.
   Normalisation uses either:
     - Dataset-level stats from a JSON file produced by compute_dataset_stats.py
       (recommended for reproducibility across splits), or
     - Per-patient stats computed from the patient's own pre-contrast image.
5. Save results as MHA and PNG in the following folder layout::

       output_dir/
           mha/
               input/          # pre-contrast slices
               ground_truth/   # peak-enhancement slices
               mask/           # tumour segmentation slices
           png/                # same structure, visualisation only
           intensity_plots/    # per-patient intensity comparison plots

6. Write a CSV report with one row per patient.

Expected input layout
---------------------
    image_dir/<patient_id>/<patient_id>_<phase_index>.nii.gz
    segmentation_dir/<patient_id>.nii.gz

Usage
-----
    python preprocess.py \\
        --image_dir      /path/to/images \\
        --seg_dir        /path/to/segmentations \\
        --output_dir     /path/to/output \\
        --csv_name       report.csv \\
        --global_stats   /path/to/dataset_stats.json  # required
"""
import argparse
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Tuple, Dict, Optional
import SimpleITK as sitk
import logging
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; safe for headless / server use
import matplotlib.pyplot as plt
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AmbiguousFOVError(ValueError):
    """Raised when the through-plane axis cannot be determined from volume shape.

    This happens when all three spatial dimensions are different (non-square FOV),
    making it impossible to identify the in-plane vs through-plane directions
    from shape alone.
    """


class Preprocessor:
    """Preprocess 3D DCE-MRI into 2D pre/peak slices with z-score normalisation."""

    def __init__(
        self,
        image_dir: str,
        segmentation_dir: str,
        output_dir: str,
        csv_output_path: str = "report.csv",
        global_stats_path: str = None,
        skip_ambiguous_shapes: bool = False,
    ):
        """
        Args:
            image_dir: Directory containing patient folders, each with multi-phase images.
                      Expected structure: image_dir/patient_id/phase_files
            segmentation_dir: Directory containing segmentations (patient_id.nii.gz or .mha).
            output_dir: Root directory for saved 2D images.
            csv_output_path: Path to save CSV report.
            global_stats_path: Path to a JSON file produced by compute_dataset_stats.py
                               containing {"mean": ..., "std": ...}. Required for
                               z-score normalisation.
            skip_ambiguous_shapes: When True, patients whose segmentation volume has all
                                   three dimensions different (non-square FOV) are logged
                                   as warnings and silently skipped instead of raising
                                   AmbiguousFOVError.  Defaults to False (raise).
        """
        self.image_dir = Path(image_dir)
        self.segmentation_dir = Path(segmentation_dir)
        self.output_dir = Path(output_dir)
        self.csv_output_path = Path(csv_output_path)

        # Load global normalisation stats (required)
        if global_stats_path is None:
            raise ValueError(
                "global_stats_path is required. Run compute_dataset_stats.py first "
                "to generate the stats JSON and pass it via --global_stats."
            )
        with open(global_stats_path, 'r') as f:
            stats = json.load(f)
        self.global_norm_mean = float(stats['mean'])
        self.global_norm_std = float(stats['std'])
        self.skip_ambiguous_shapes = skip_ambiguous_shapes
        logger.info(
            f"Global normalisation stats loaded from {global_stats_path}: "
            f"mean={self.global_norm_mean:.4f}, std={self.global_norm_std:.4f}"
        )

        self.mha_input_dir = self.output_dir / "mha" / "input"
        self.mha_gt_dir = self.output_dir / "mha" / "ground_truth"
        self.mha_mask_dir = self.output_dir / "mha" / "mask"
        self.png_input_dir = self.output_dir / "png" / "input"
        self.png_gt_dir = self.output_dir / "png" / "ground_truth"
        self.png_mask_dir = self.output_dir / "png" / "mask"
        self.plots_dir = self.output_dir / "intensity_plots"
        for d in (
            self.mha_input_dir, self.mha_gt_dir, self.mha_mask_dir,
            self.png_input_dir, self.png_gt_dir, self.png_mask_dir,
            self.plots_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        self.results = []

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from NIfTI or MHA format."""
        image_path = str(image_path)
        if image_path.endswith(('.nii.gz', '.nii')):
            img = nib.load(image_path)
            return img.get_fdata().astype(np.float32)
        img = sitk.ReadImage(image_path)
        return sitk.GetArrayFromImage(img).astype(np.float32)

    def _load_spacing(self, image_path) -> Optional[Tuple[float, ...]]:
        """Return voxel spacing (mm) for NIfTI files, or None for other formats.

        Used by determine_slice_axis() to cross-validate the shape-based axis
        against the physical voxel spacing (largest spacing = through-plane).
        Returns None for MHA files because the spacing is not needed for the
        primary heuristic and avoids an unnecessary second I/O round-trip.
        """
        image_path = str(image_path)
        if image_path.endswith(('.nii.gz', '.nii')):
            img = nib.load(image_path)
            return tuple(float(z) for z in img.header.get_zooms()[:3])
        return None

    def save_mha(self, image_2d: np.ndarray, output_path: Path, is_label: bool = False) -> None:
        """Save a 2D array as MHA, preserving float32 for images and int16 for labels."""
        image_2d = np.nan_to_num(np.asarray(image_2d), nan=0.0, posinf=0.0, neginf=0.0)
        if is_label:
            image_2d = np.rint(image_2d).astype(np.int16)
        else:
            image_2d = image_2d.astype(np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(image_2d), str(output_path))
        logger.info(f"Saved MHA: {output_path}")

    def save_png(self, image_2d: np.ndarray, output_path: Path, hi: np.float32=None, is_label: bool = False) -> None:
        """Save a 2D array as PNG (normalised to 0-255)."""
        image_2d = np.nan_to_num(np.asarray(image_2d), nan=0.0, posinf=0.0, neginf=0.0)
        if is_label:
            image_2d = (np.rint(image_2d) > 0).astype(np.uint8) * 255
        else:
            lo = np.min(image_2d)
            if hi > lo:
                image_2d = ((image_2d - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                image_2d = image_2d.astype(np.uint8)
        Image.fromarray(image_2d, mode='L').save(str(output_path))
        logger.info(f"Saved PNG: {output_path}")

    # ------------------------------------------------------------------
    # Phase / slice selection
    # ------------------------------------------------------------------

    def determine_slice_axis(
        self,
        shape: Tuple[int, ...],
        spacing: Optional[Tuple[float, ...]] = None,
    ) -> int:
        """Determine the through-plane (slice) axis from volume shape.

        Strategy (layered, in order of priority):

        1. **Two axes share the same size, one is unique** → the unique axis is
           the through-plane direction.  The result is cross-validated against
           voxel spacing when available; a WARNING is logged if they disagree,
           but the shape-based result is used as the primary criterion.

        2. **All three axes equal (cubic volume)** → cannot distinguish axes from
           shape alone.  A WARNING is logged and axis 2 is returned as the axial
           convention fallback.

        3. **All three axes differ (non-square FOV)** → raises AmbiguousFOVError.
           Set ``skip_ambiguous_shapes=True`` on the Preprocessor to automatically
           discard these patients during pipeline execution.

        Args:
            shape: 3-D volume shape, e.g. ``(H, W, D)``.
            spacing: Voxel spacing in mm, e.g. ``(sx, sy, sz)``.  Used only for
                     cross-validation in case 1; may be ``None``.

        Returns:
            Index (0, 1, or 2) of the through-plane axis.

        Raises:
            ValueError: If ``shape`` is not 3-D.
            AmbiguousFOVError: If all three dimensions are different (case 3).
        """
        if len(shape) != 3:
            raise ValueError(f"Expected a 3-D volume shape, got {shape}")

        # Group axis indices by their size
        size_to_axes: Dict[int, list] = {}
        for ax, s in enumerate(shape):
            size_to_axes.setdefault(s, []).append(ax)

        n_unique_sizes = len(size_to_axes)

        # --- Case 2: cubic ---
        if n_unique_sizes == 1:
            logger.warning(
                "Cubic volume detected (shape %s). Cannot determine through-plane "
                "axis from shape alone. Falling back to axis 2 (axial convention).",
                shape,
            )
            return 2

        # --- Case 3: all three dimensions differ ---
        if n_unique_sizes == 3:
            raise AmbiguousFOVError(
                f"All three dimensions differ (shape {shape}). Cannot reliably "
                "determine the through-plane axis from shape alone. Pass "
                "skip_ambiguous_shapes=True to the Preprocessor (or "
                "--skip_ambiguous_shapes on the CLI) to discard these cases "
                "automatically."
            )

        # --- Case 1: exactly two distinct sizes → one unique axis ---
        shape_axis: int = next(
            axes[0]
            for axes in size_to_axes.values()
            if len(axes) == 1
        )

        # Cross-validate against voxel spacing when available
        if spacing is not None and len(spacing) >= 3:
            spacing_axis = int(np.argmax(spacing[:3]))
            if spacing_axis != shape_axis:
                logger.warning(
                    "Shape-based through-plane axis (%d, size %d) disagrees with "
                    "spacing-based axis (%d, spacing %.4f mm) for volume shape %s. "
                    "Using shape-based axis as primary criterion.",
                    shape_axis, shape[shape_axis],
                    spacing_axis, spacing[spacing_axis],
                    shape,
                )

        return shape_axis

    def find_largest_label_slice(
        self,
        segmentation: np.ndarray,
        spacing: Optional[Tuple[float, ...]] = None,
    ) -> Tuple[int, int]:
        """Return ``(slice_idx, axis)`` for the slice with the most label voxels.

        The through-plane axis is resolved via :meth:`determine_slice_axis`.

        Args:
            segmentation: 3-D binary/label volume.
            spacing: Voxel spacing in mm forwarded to :meth:`determine_slice_axis`
                     for cross-validation.  May be ``None``.

        Returns:
            A tuple ``(slice_idx, axis)`` where *slice_idx* is the 0-based index
            of the selected slice and *axis* is the through-plane axis (0, 1, or 2).
        """
        axis = self.determine_slice_axis(segmentation.shape, spacing)
        others = tuple(i for i in range(segmentation.ndim) if i != axis)
        slice_areas = np.sum(segmentation > 0, axis=others)
        return int(np.argmax(slice_areas)), axis

    def extract_slice(self, image: np.ndarray, slice_idx: int, axis: int) -> np.ndarray:
        """Extract the 2D slice at *slice_idx* along *axis*."""
        return np.take(image, slice_idx, axis=axis)

    def find_peak_phase(
        self,
        phase_volumes: Dict[int, np.ndarray],
        segmentation: np.ndarray,
    ) -> Tuple[int, float, Dict[int, float]]:
        """Return (peak_phase_num, peak_mean_intensity, per-phase intensities dict).
        Works with volumes of any dimensionality (2D or 3D)."""
        peak_phase = 0
        max_intensity = -np.inf
        phase_intensities: Dict[int, float] = {}

        for phase_num, volume in phase_volumes.items():
            tumor_pixels = volume[segmentation > 0]
            if len(tumor_pixels) > 0:
                mean_val = float(np.mean(tumor_pixels))
                phase_intensities[phase_num] = mean_val
                if mean_val > max_intensity:
                    max_intensity = mean_val
                    peak_phase = phase_num

        return peak_phase, max_intensity, phase_intensities

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def zscore_normalise(image_2d: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Z-score normalise image_2d using the provided mean and std."""
        if std == 0.0:
            return np.zeros_like(image_2d, dtype=np.float32)
        return ((image_2d - mean) / std).astype(np.float32)

    # ------------------------------------------------------------------
    # Patient discovery
    # ------------------------------------------------------------------

    def get_patient_phases(self) -> Dict[str, list]:
        """Return {patient_id: [(phase_num, file_path), ...]} sorted by phase number."""
        patient_phases: Dict[str, list] = {}

        for patient_folder in sorted(self.image_dir.iterdir()):
            if not patient_folder.is_dir():
                continue

            patient_id = patient_folder.name
            phase_files = []

            for file_path in sorted(patient_folder.iterdir()):
                if file_path.is_file() and file_path.suffix in ('.nii', '.gz', '.mha'):
                    stem = file_path.stem.replace('.nii', '')
                    parts = stem.rsplit('_', 1)
                    if len(parts) == 2:
                        try:
                            phase_num = int(parts[-1])
                            phase_files.append((phase_num, file_path))
                        except ValueError:
                            pass

            if phase_files:
                phase_files.sort(key=lambda x: x[0])
                patient_phases[patient_id] = phase_files
            else:
                logger.warning(f"No phase files found for patient: {patient_id}")

        return patient_phases

    # ------------------------------------------------------------------
    # Intensity curve plotting
    # ------------------------------------------------------------------

    def plot_intensity_curve(
        self,
        patient_id: str,
        phase_images_2d: Dict[int, np.ndarray],
        segmentation_2d: np.ndarray,
        pre_phase: int,
        peak_phase: int,
        norm_mean: float,
        norm_std: float,
    ) -> None:
        """Save two side-by-side line plots (raw | normalised), each showing
        within-tumour and outside-tumour mean intensity for pre and peak phases."""
        tumour_mask = segmentation_2d > 0
        outside_mask = ~tumour_mask
        selected = [pre_phase, peak_phase]
        x_labels = [f'Pre\n(phase {pre_phase})', f'Peak\n(phase {peak_phase})']

        def _means(images, mask):
            return [
                float(np.mean(images[p][mask])) if mask.any() else 0.0
                for p in selected
            ]

        raw_in   = _means(phase_images_2d, tumour_mask)
        raw_out  = _means(phase_images_2d, outside_mask)

        norm_images = {p: self.zscore_normalise(phase_images_2d[p], norm_mean, norm_std)
                       for p in selected}
        norm_in  = _means(norm_images, tumour_mask)
        norm_out = _means(norm_images, outside_mask)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, in_vals, out_vals, title in (
            (axes[0], raw_in,  raw_out,  'Raw intensity'),
            (axes[1], norm_in, norm_out, 'Z-score normalised'),
        ):
            ax.plot(x_labels, in_vals,  marker='o', label='Within tumour',   color='tomato')
            ax.plot(x_labels, out_vals, marker='s', label='Outside tumour',  color='steelblue')
            ax.set_ylabel('Mean intensity')
            ax.set_title(f'{patient_id} – {title}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = self.plots_dir / f"{patient_id}.png"
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        logger.info(f"Saved intensity plot: {out_path}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process(self) -> pd.DataFrame:
        """Process all patients and return a summary DataFrame."""
        patient_phases = self.get_patient_phases()
        logger.info(f"Found {len(patient_phases)} patients")

        for patient_id, phase_files in patient_phases.items():
            try:
                # --- load segmentation ---
                seg_file = self.segmentation_dir / f"{patient_id}.nii.gz"
                if not seg_file.exists():
                    candidates = list(self.segmentation_dir.glob(f"{patient_id}*"))
                    if candidates:
                        seg_file = candidates[0]
                    else:
                        logger.warning(f"Segmentation not found for: {patient_id}")
                        continue

                segmentation = self.load_image(seg_file)
                logger.info(f"Processing {patient_id}: segmentation shape {segmentation.shape}")

                # Step 1 – load all 3D phase volumes; find peak phase using full tumour region
                phase_volumes_3d: Dict[int, np.ndarray] = {}
                for phase_num, phase_file in phase_files:
                    phase_volumes_3d[phase_num] = self.load_image(phase_file)

                pre_phase = phase_files[0][0]
                peak_phase, peak_mean_intensity, phase_intensities = self.find_peak_phase(
                    phase_volumes_3d, segmentation
                )

                # Step 2 – select slice with largest tumour area, then extract 2D slices
                spacing = self._load_spacing(seg_file)
                try:
                    largest_slice, slice_axis = self.find_largest_label_slice(
                        segmentation, spacing
                    )
                except AmbiguousFOVError as exc:
                    if self.skip_ambiguous_shapes:
                        logger.warning(
                            "Skipping %s — ambiguous FOV: %s", patient_id, exc
                        )
                        continue
                    raise

                seg_2d = self.extract_slice(segmentation, largest_slice, slice_axis)

                phase_images_2d: Dict[int, np.ndarray] = {
                    phase_num: self.extract_slice(vol, largest_slice, slice_axis)
                    for phase_num, vol in phase_volumes_3d.items()
                }
                logger.info(f"  Pre phase : {pre_phase}")
                logger.info(f"  Peak phase: {peak_phase}  (mean tumour intensity: {peak_mean_intensity:.2f})")

                # Step 3 – z-score normalise using global dataset stats
                norm_mean = self.global_norm_mean
                norm_std = self.global_norm_std

                pre_norm = self.zscore_normalise(phase_images_2d[pre_phase], norm_mean, norm_std)
                peak_norm = self.zscore_normalise(phase_images_2d[peak_phase], norm_mean, norm_std)
                mask_2d = np.rint(seg_2d).astype(np.int16)

                # Step 4 – save files named patient_id.extension
                fname = patient_id

                self.save_mha(pre_norm,  self.mha_input_dir  / f"{fname}.mha")
                self.save_mha(peak_norm, self.mha_gt_dir      / f"{fname}.mha")
                self.save_mha(mask_2d,   self.mha_mask_dir    / f"{fname}.mha", is_label=True)

                # Plot mean intensity curves: raw vs normalised tumour region
                self.plot_intensity_curve(
                    patient_id, phase_images_2d, seg_2d, pre_phase, peak_phase,
                    norm_mean, norm_std
                )

                # only for visualization, use the same hi for pre and peak to show relative contrast in PNGs
                peak_hi = np.max(peak_norm)
                self.save_png(pre_norm, self.png_input_dir   / f"{fname}.png", peak_hi)
                self.save_png(peak_norm, self.png_gt_dir       / f"{fname}.png", peak_hi)
                self.save_png(mask_2d, self.png_mask_dir     / f"{fname}.png", is_label=True)

                # Step 5 – collect CSV row
                phase_intensity_str = "; ".join(
                    f"Phase {p}: {v:.2f}" for p, v in sorted(phase_intensities.items())
                )
                self.results.append({
                    'patient_id': patient_id,
                    'pre_contrast_phase': pre_phase,
                    'peak_enhancement_phase': peak_phase,
                    'selected_slice': largest_slice,
                    'peak_mean_intensity': peak_mean_intensity,
                    'normalisation_mean': norm_mean,
                    'normalisation_std': norm_std,
                    'normalisation_source': 'global',
                    'phase_intensities': phase_intensity_str,
                })

            except Exception as e:
                logger.error(f"Error processing {patient_id}: {e}")
                continue
  
        return pd.DataFrame(self.results)

    def save_report(self, df_results: pd.DataFrame) -> None:
        """Save results DataFrame to CSV."""
        df_results.to_csv(self.csv_output_path, index=False)
        logger.info(f"Report saved to {self.csv_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess 3D DCE-MRI to 2D pre/peak slices with z-score normalisation."
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Root directory containing per-patient sub-folders with phase volumes."
    )
    parser.add_argument(
        "--seg_dir", required=True,
        help="Directory containing per-patient segmentation files (patient_id.nii.gz)."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root directory where MHA, PNG, plots, and CSV will be written."
    )
    parser.add_argument(
        "--csv_name", default="report.csv",
        help="Filename for the output CSV report (default: report.csv)."
    )
    parser.add_argument(
        "--global_stats", default='./mama-synth/src/preprocessing/training_pre_stats.json', dest="global_stats_path",
        help=(
            "Path to a JSON file produced by compute_dataset_stats.py "
            "containing {\"mean\": ..., \"std\": ...} for dataset-level z-score normalisation."
        )
    )
    parser.add_argument(
        "--skip_ambiguous_shapes", action="store_true", default=False,
        help=(
            "When set, patients whose segmentation volume has all three spatial dimensions "
            "different (non-square FOV) are logged as warnings and skipped instead of "
            "raising an error.  By default the pipeline raises AmbiguousFOVError for "
            "these cases so they are not silently ignored."
        )
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    seg_dir = Path(args.seg_dir)

    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return
    if not seg_dir.exists():
        logger.error(f"Segmentation directory not found: {seg_dir}")
        return
    if args.global_stats_path and not Path(args.global_stats_path).exists():
        logger.error(f"Global stats file not found: {args.global_stats_path}")
        return

    output_dir = Path(args.output_dir)
    csv_path = output_dir / args.csv_name

    preprocessor = Preprocessor(
        image_dir=str(image_dir),
        segmentation_dir=str(seg_dir),
        output_dir=str(output_dir),
        csv_output_path=str(csv_path),
        global_stats_path=args.global_stats_path,
        skip_ambiguous_shapes=args.skip_ambiguous_shapes,
    )

    logger.info("Starting preprocessing pipeline...")
    results_df = preprocessor.process()

    if not results_df.empty:
        preprocessor.save_report(results_df)
        logger.info(f"Processed {len(results_df)} patients. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

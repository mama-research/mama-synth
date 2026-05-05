#!/usr/bin/env python3
"""MAMA-SYNTH Grand Challenge Evaluation Method.

Reads algorithm outputs (synthetic post-contrast breast DCE-MRI
slices) and evaluates them against ground truth using:

  * **Image-to-image**: MSE, LPIPS
  * **ROI-to-ROI**: SSIM (tumour mask), FRD (Fréchet Radiomics Distance)
  * **Classification**: AUROC contrast, AUROC tumour-ROI
  * **Segmentation**: Dice, HD95

This single container image is used for all GC phases (debug, validation,
test).  The evaluation logic is identical across phases; only the ground
truth data uploaded to each phase differs.

Grand Challenge directory layout
---------------------------------
GC extracts the uploaded ``ground_truth.zip`` to
``/opt/ml/input/data/ground_truth/``.  The zip must contain two top-level
folders so that after extraction the structure is::

    /opt/ml/input/data/ground_truth/
        ground_truth/    ← real post-contrast 2-D .mha slices
        masks/           ← binary tumour masks .mha

    /input/
        predictions.json
        {job_pk}/output/images/{slug}/*.mha

    /opt/app/models/
        classification/
            contrast_classifier.pkl
            tumor_roi_classifier.pkl
        segmentation/    ← nnUNet model folder (fold_0/, plans.json, …)

    /output/
        metrics.json     ← evaluation results

Local (development) mode
------------------------
The ``do_test_run.sh`` script mounts the repository root as
``/opt/ml/input/data/ground_truth/`` so that the local ``ground_truth/``
and ``masks/`` folders at the repo root are accessible at the same
container paths that GC uses.  This mirrors the GC extraction exactly.

Set environment variables to override default GC paths::

    MAMA_INPUT_DIR          (default: /input)
    MAMA_OUTPUT_DIR         (default: /output)
    MAMA_GT_DIR             (default: /opt/ml/input/data/ground_truth)
    MAMA_MODELS_DIR         (default: /opt/app/models)
    MAMA_PREDICTIONS_DIR    (flat dir of .mha predictions, local mode)
    MAMA_MASKS_DIR          (flat dir of .mha masks, local mode override)
"""

from __future__ import annotations

import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from evaluators import (
    Case,
    ClassificationEvaluator,
    ImageMetricsEvaluator,
    ROIMetricsEvaluator,
    SegmentationEvaluator,
)

# ======================================================================
# Interface slugs — customise for your GC phase configuration
# ======================================================================
PREDICTION_SLUG = os.environ.get(
    "MAMA_PREDICTION_SLUG", "synthetic-post-contrast-breast-mri"
)
INPUT_SLUG = os.environ.get(
    "MAMA_INPUT_SLUG", "pre-contrast-breast-mri"
)


# ======================================================================
# I/O helpers
# ======================================================================


def load_image(path: Path) -> np.ndarray:
    """Load a .mha image as ``float64`` (no additional normalisation).

    Images are expected to arrive **z-score normalised** using pre-contrast
    reference statistics.  No per-image min-max is applied — this avoids
    the bias that independent normalisation would introduce in MSE, LPIPS,
    and SSIM.
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask from .mha — returns a ``bool`` array."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr > 0


def write_metrics(metrics: dict, path: Path) -> None:
    """Write *metrics* as JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)


def _find_file(directory: Path, stem: str) -> Optional[Path]:
    """Find a .mha (or .nii.gz) file matching *stem* in *directory*."""
    if not directory.exists():
        return None
    for ext in (".mha", ".nii.gz", ".nii"):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ======================================================================
# Case loaders
# ======================================================================


def load_cases_gc(
    input_dir: Path,
    gt_dir: Path,
) -> list[Case]:
    """Load cases using the GC ``predictions.json`` interface."""
    predictions_file = input_dir / "predictions.json"
    with open(predictions_file) as fh:
        predictions = json.load(fh)

    cases: list[Case] = []
    for job in predictions:
        pk = job["pk"]

        # --- Locate the prediction .mha --------------------------------
        pred_dir = input_dir / pk / "output" / f"images/{PREDICTION_SLUG}"
        pred_files = glob(str(pred_dir / "*.mha"))
        if not pred_files:
            continue
        prediction = load_image(Path(pred_files[0]))

        # --- Case ID from the algorithm input image name ---------------
        case_name = _gc_input_image_name(job)
        if case_name is None:
            continue
        case_id = Path(case_name).stem

        # --- Ground truth, mask, pre-contrast --------------------------
        # GC extracts ground_truth.zip so that the post-contrast images live
        # under gt_dir/ground_truth/ and masks under gt_dir/masks/.
        gt_path = _find_file(gt_dir / "ground_truth", case_id)
        if gt_path is None:
            continue
        ground_truth = load_image(gt_path)

        mask: Optional[np.ndarray] = None
        mask_path = _find_file(gt_dir / "masks", case_id)
        if mask_path:
            mask = load_mask(mask_path)

        precontrast: Optional[np.ndarray] = None
        precon_path = _find_file(gt_dir / "precontrast", case_id)
        if precon_path:
            precontrast = load_image(precon_path)

        cases.append(
            Case(
                case_id=case_id,
                prediction=prediction,
                ground_truth=ground_truth,
                mask=mask,
                precontrast=precontrast,
                prediction_path=str(pred_files[0]),
                ground_truth_path=str(gt_path),
                mask_path=str(mask_path) if mask_path else None,
            )
        )

    return cases


def load_cases_local(
    pred_dir: Path,
    gt_dir: Path,
    masks_dir: Optional[Path] = None,
    precon_dir: Optional[Path] = None,
) -> list[Case]:
    """Load cases from flat directories (for local development)."""
    cases: list[Case] = []
    for pred_file in sorted(pred_dir.glob("*.mha")):
        case_id = pred_file.stem
        gt_path = _find_file(gt_dir, case_id)
        if gt_path is None:
            continue

        mask: Optional[np.ndarray] = None
        mask_file: Optional[Path] = None
        if masks_dir:
            mask_file = _find_file(masks_dir, case_id)
            if mask_file:
                mask = load_mask(mask_file)

        precontrast: Optional[np.ndarray] = None
        if precon_dir:
            precon_path = _find_file(precon_dir, case_id)
            if precon_path:
                precontrast = load_image(precon_path)

        prediction = load_image(pred_file)

        cases.append(
            Case(
                case_id=case_id,
                prediction=prediction,
                ground_truth=load_image(gt_path),
                mask=mask,
                precontrast=precontrast,
                prediction_path=str(pred_file),
                ground_truth_path=str(gt_path),
                mask_path=str(mask_file) if mask_file else None,
            )
        )
    return cases


def _gc_input_image_name(job: dict) -> Optional[str]:
    """Extract the original input image filename from a GC job."""
    for inp in job.get("inputs", []):
        if inp.get("interface", {}).get("slug") == INPUT_SLUG:
            return inp.get("image", {}).get("name")
    return None


# ======================================================================
# Segmentation model loader
# ======================================================================


def load_segmentation_model(
    models_dir: Optional[Path],
) -> Optional[object]:
    """Load a single-fold nnUNet segmentation model from *models_dir/segmentation*.

    Returns a callable that accepts a 2-D ``float64`` array and returns
    a binary ``bool`` mask, or ``None`` if the model directory is absent
    or nnunetv2 is not installed.
    """
    if models_dir is None:
        return None
    seg_dir = models_dir / "segmentation"
    if not seg_dir.exists():
        return None

    try:
        # Set nnUNet environment variables to dummy paths to suppress warnings
        os.environ["nnUNet_raw"] = "/tmp/nnunet_raw"
        os.environ["nnUNet_preprocessed"] = "/tmp/nnunet_preprocessed"
        os.environ["nnUNet_results"] = "/tmp/nnunet_results"
        import logging
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        print(
            "WARNING: nnunetv2 not installed — segmentation evaluator disabled.",
            file=sys.stderr,
        )
        return None

    # Suppress nnunetv2 logging
    logging.getLogger("nnunetv2").setLevel(logging.WARNING)
    logging.getLogger("batchgenerators").setLevel(logging.WARNING)
    logging.getLogger("acvl_utils").setLevel(logging.WARNING)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=False,
    )

    # Load a single fold (fold 0); override via MAMA_SEG_FOLD env var
    fold_str = os.environ.get("MAMA_SEG_FOLD", "0")
    fold = int(fold_str) if fold_str.isdigit() else fold_str
    predictor.initialize_from_trained_model_folder(
        str(seg_dir),
        use_folds=(fold,),
        checkpoint_name="checkpoint_final.pth",
    )
    print(f"  Segmentation model loaded from {seg_dir} (fold {fold}, device {device})")

    import tempfile
    import SimpleITK as sitk
    from contextlib import redirect_stdout, redirect_stderr
    from io import StringIO

    def segment_fn(image: np.ndarray) -> np.ndarray:
        """Run nnUNet inference on a single 2-D image."""
        arr = image.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]  # add Z dim for nnUNet

        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "input"
            out_dir = Path(tmpdir) / "output"
            in_dir.mkdir()
            out_dir.mkdir()

            sitk.WriteImage(
                sitk.GetImageFromArray(arr),
                str(in_dir / "case_0000.nii.gz"),
            )

            # Suppress stdout/stderr during inference
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                predictor.predict_from_files(
                    str(in_dir),
                    str(out_dir),
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=1,
                    num_processes_segmentation_export=1,
                )

            pred_path = out_dir / "case.nii.gz"
            if not pred_path.exists():
                return np.zeros(image.shape, dtype=bool)

            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))

        mask = pred > 0
        if image.ndim == 2 and mask.ndim == 3:
            mask = mask[0]
        return mask.astype(bool)

    return segment_fn


# ======================================================================
# Pipeline orchestrator
# ======================================================================


def run_evaluation(
    cases: list[Case],
    models_dir: Optional[Path] = None,
) -> dict:
    """Run all evaluators on *cases*, return the metrics dict.

    This is the main public API for programmatic use and testing.
    """
    # Check for ensemble mode via environment variable
    ensemble = os.environ.get("MAMA_ENSEMBLE", "").lower() in (
        "1", "true", "yes",
    )

    evaluators: list[tuple[str, object]] = [
        ("ImageMetrics", ImageMetricsEvaluator()),
        ("ROIMetrics", ROIMetricsEvaluator()),
        (
            "Classification",
            ClassificationEvaluator(
                contrast_model=(
                    models_dir / "classification"/ "contrast_classifier.pkl"
                    if models_dir
                    else None
                ),
                tumor_roi_model=(
                    models_dir / "classification"/ "tumor_roi_classifier.pkl"
                    if models_dir
                    else None
                ),
                models_dir=models_dir / "classification" if models_dir else None,
                ensemble=ensemble
            ),
        ),
        (
            "Segmentation",
            SegmentationEvaluator(
                segment_fn=load_segmentation_model(models_dir),
            ),
        ),
    ]

    all_per_case: dict[str, dict[str, float]] = {}
    all_aggregates: dict[str, dict[str, float]] = {}

    for name, evaluator in evaluators:
        print(f"[INFO] Running evaluator: {name}")
        try:
            result = evaluator.evaluate(cases)  # type: ignore[attr-defined]
            for cid, m in result.per_case.items():
                all_per_case.setdefault(cid, {}).update(m)
            all_aggregates.update(result.aggregates)
            n_agg = len(result.aggregates)
            print(f"  {name}: OK ({n_agg} aggregate metric(s))")
        except Exception as exc:
            print(f"  {name}: FAILED — {exc}", file=sys.stderr)

    # Clear the in-memory radiomic feature cache to free memory
    from evaluators.roi_metrics import clear_feature_cache
    clear_feature_cache()

    return {"case": all_per_case, "aggregates": all_aggregates}


# ======================================================================
# CLI entry point
# ======================================================================


def main() -> int:
    """Main entry point for the GC evaluation container."""
    print("=" * 50)
    print("MAMA-SYNTH Evaluation")
    print("=" * 50)

    # ---- Read configuration from environment -------------------------
    input_dir = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
    gt_dir = Path(
        os.environ.get(
            "MAMA_GT_DIR", "/opt/ml/input/data/ground_truth"
        )
    )
    models_dir = Path(
        os.environ.get("MAMA_MODELS_DIR", "/opt/app/models")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    # ---- Discover and load cases -------------------------------------
    if (input_dir / "predictions.json").exists():
        print("Loading cases from GC predictions.json …")
        cases = load_cases_gc(input_dir, gt_dir)
    else:
        print("Loading cases from flat directories (local mode) …")
        pred_dir = Path(
            os.environ.get("MAMA_PREDICTIONS_DIR", str(input_dir))
        )
        # GT images live under gt_dir/ground_truth/ (mirrors the GC zip structure).
        # Fall back to gt_dir itself only when the subfolder is absent.
        gt_images_dir = (
            gt_dir / "ground_truth" if (gt_dir / "ground_truth").exists() else gt_dir
        )
        masks_env = os.environ.get("MAMA_MASKS_DIR")
        masks_sub = gt_dir / "masks"
        masks_dir = (
            Path(masks_env) if masks_env
            else masks_sub if masks_sub.exists()
            else None
        )
        # No precontrast in the ground_truth.zip; kept for local-override use only.
        precon_env = os.environ.get("MAMA_PRECONTRAST_DIR")
        precon_dir = Path(precon_env) if precon_env else None
        cases = load_cases_local(
            pred_dir, gt_images_dir, masks_dir, precon_dir,
        )

    if not cases:
        print("ERROR: no valid cases found.", file=sys.stderr)
        write_metrics({"case": {}, "aggregates": {}}, metrics_path)
        return 1

    n_mask = sum(1 for c in cases if c.mask is not None)
    n_pre = sum(1 for c in cases if c.precontrast is not None)
    print(
        f"  Cases: {len(cases)}, "
        f"with masks: {n_mask}, "
        f"with pre-contrast: {n_pre}"
    )

    # ---- Run evaluation ----------------------------------------------
    metrics = run_evaluation(cases, models_dir)
    write_metrics(metrics, metrics_path)

    # ---- Summary -----------------------------------------------------
    print(f"\nMetrics written to {metrics_path}")
    agg = metrics.get("aggregates", {})
    if agg:
        print("\n--- Aggregates ---")
        for key in sorted(agg):
            val = agg[key]
            if "mean" in val:
                std = val.get("std", 0.0)
                print(f"  {key}: {val['mean']:.4f} (±{std:.4f})")
            else:
                print(f"  {key}: {val}")

    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

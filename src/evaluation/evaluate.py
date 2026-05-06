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
import logging
import os
import sys
import time
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
# Logging — single stdout stream so participants see one clean log
# ======================================================================

_LOG_FMT = "%(asctime)s  %(levelname)-8s %(message)s"
_LOG_DATE = "%H:%M:%S"


def _configure_logging() -> None:
    """Configure root logger: timestamped lines to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FMT,
        datefmt=_LOG_DATE,
        stream=sys.stdout,
        force=True,
    )
    # Silence chatty third-party libraries so only evaluation output is visible
    for _lib in (
        "nnunetv2", "batchgenerators", "acvl_utils",
        "radiomics", "torch", "PIL",
    ):
        logging.getLogger(_lib).setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger(__name__)

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

    Images are expected to arrive **z-score normalised** using the challenge
    pre-contrast reference statistics.  No per-image min-max is applied —
    this avoids bias that independent normalisation would introduce in MSE,
    LPIPS, and SSIM.
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # ---- Participant-facing sanity checks --------------------------------
    if arr.ndim != 2:
        logger.warning(
            "%s — unexpected array shape %s (expected a 2-D slice). "
            "If your algorithm outputs a 3-D volume, extract and submit "
            "only the single 2-D post-contrast slice before uploading. "
            "The middle slice is being used as a fallback.",
            path.name, arr.shape,
        )
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]

    abs_max = float(np.max(np.abs(arr)))
    if abs_max > 500:
        logger.warning(
            "%s — very large intensity values detected (|max| = %.1f). "
            "Predictions MUST be z-score normalised before submission "
            "(expected typical range \u2248 \u22123 to +10, |max| < 100). "
            "Submitting raw or HU-scaled images will directly inflate "
            "MSE/LPIPS scores and distort your ranking. "
            "Apply the same preprocessing pipeline described in the "
            "challenge documentation.",
            path.name, abs_max,
        )

    return arr


def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask from .mha — returns a ``bool`` array."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    mask = arr > 0
    if not np.any(mask):
        logger.warning(
            "Mask file %s is entirely zero — no foreground tumour voxels found. "
            "Mask-dependent metrics (SSIM-tumour, AUROC-tumour-ROI, FRD) "
            "will be skipped for this case.",
            path.name,
        )
    return mask


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

    n_jobs = len(predictions)
    logger.info("  Found %d job entr%s in predictions.json",
                n_jobs, "y" if n_jobs == 1 else "ies")

    cases: list[Case] = []
    n_skip_pred = n_skip_id = n_skip_gt = n_skip_shape = 0

    for idx, job in enumerate(predictions, 1):
        pk = job["pk"]

        # --- Locate the prediction .mha --------------------------------
        pred_dir = input_dir / pk / "output" / f"images/{PREDICTION_SLUG}"
        pred_files = glob(str(pred_dir / "*.mha"))
        if not pred_files:
            logger.warning(
                "  [%d/%d] pk=%s — no prediction file found under:\n"
                "    %s\n"
                "  Your algorithm must write its output image to the socket "
                "with slug '%s'.  Check that the output interface is "
                "configured correctly in your algorithm container.",
                idx, n_jobs, pk, pred_dir, PREDICTION_SLUG,
            )
            n_skip_pred += 1
            continue

        prediction = load_image(Path(pred_files[0]))

        # --- Case ID from the algorithm input image name ---------------
        case_name = _gc_input_image_name(job)
        if case_name is None:
            logger.warning(
                "  [%d/%d] pk=%s — cannot determine case ID: no input with "
                "slug '%s' in predictions.json.  This job will be skipped.",
                idx, n_jobs, pk, INPUT_SLUG,
            )
            n_skip_id += 1
            continue
        case_id = Path(case_name).stem

        # --- Ground truth, mask, pre-contrast --------------------------
        gt_path = _find_file(gt_dir / "ground_truth", case_id)
        if gt_path is None:
            logger.warning(
                "  [%d/%d] %s — ground-truth file not found in:\n"
                "    %s\n"
                "  Verify that ground_truth.zip was uploaded correctly and "
                "that the filename stem matches the case ID '%s' exactly.",
                idx, n_jobs, case_id, gt_dir / "ground_truth", case_id,
            )
            n_skip_gt += 1
            continue

        ground_truth = load_image(gt_path)

        if prediction.shape != ground_truth.shape:
            logger.warning(
                "  [%d/%d] %s — shape mismatch: prediction %s \u2260 GT %s. "
                "Predictions must be 2-D slices with the same (H\u00d7W) as "
                "the ground-truth image.  This case will be skipped.",
                idx, n_jobs, case_id, prediction.shape, ground_truth.shape,
            )
            n_skip_shape += 1
            continue

        mask: Optional[np.ndarray] = None
        mask_path = _find_file(gt_dir / "masks", case_id)
        if mask_path:
            mask = load_mask(mask_path)
        else:
            logger.warning(
                "  [%d/%d] %s — no mask file found; "
                "SSIM-tumour, AUROC-tumour-ROI and FRD will be skipped for this case.",
                idx, n_jobs, case_id,
            )

        precontrast: Optional[np.ndarray] = None
        precon_path = _find_file(gt_dir / "precontrast", case_id)
        if precon_path:
            precontrast = load_image(precon_path)

        logger.info(
            "  [%d/%d] %-20s  pred=%-12s  gt=%-12s  mask=%s",
            idx, n_jobs, case_id,
            str(prediction.shape), str(ground_truth.shape),
            "yes" if mask is not None else "NO",
        )

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

    skipped = n_skip_pred + n_skip_id + n_skip_gt + n_skip_shape
    if skipped:
        logger.warning(
            "  Skipped %d/%d job(s) — "
            "%d missing prediction, %d missing case-ID, "
            "%d missing GT, %d shape mismatch.",
            skipped, n_jobs,
            n_skip_pred, n_skip_id, n_skip_gt, n_skip_shape,
        )

    return cases


def load_cases_local(
    pred_dir: Path,
    gt_dir: Path,
    masks_dir: Optional[Path] = None,
    precon_dir: Optional[Path] = None,
) -> list[Case]:
    """Load cases from flat directories (for local development)."""
    pred_files = sorted(pred_dir.glob("*.mha"))
    n_pred = len(pred_files)
    logger.info("  Found %d prediction file(s) in %s", n_pred, pred_dir)

    cases: list[Case] = []
    n_skip_gt = n_skip_shape = 0

    for idx, pred_file in enumerate(pred_files, 1):
        case_id = pred_file.stem
        gt_path = _find_file(gt_dir, case_id)
        if gt_path is None:
            logger.warning(
                "  [%d/%d] %s — no matching GT file in %s, skipping.",
                idx, n_pred, case_id, gt_dir,
            )
            n_skip_gt += 1
            continue

        mask: Optional[np.ndarray] = None
        mask_file: Optional[Path] = None
        if masks_dir:
            mask_file = _find_file(masks_dir, case_id)
            if mask_file:
                mask = load_mask(mask_file)
            else:
                logger.warning(
                    "  [%d/%d] %s — no mask file in %s; "
                    "SSIM-tumour, AUROC-tumour-ROI and FRD will be skipped "
                    "for this case.",
                    idx, n_pred, case_id, masks_dir,
                )

        precontrast: Optional[np.ndarray] = None
        if precon_dir:
            precon_path = _find_file(precon_dir, case_id)
            if precon_path:
                precontrast = load_image(precon_path)

        prediction = load_image(pred_file)
        ground_truth = load_image(gt_path)

        if prediction.shape != ground_truth.shape:
            logger.warning(
                "  [%d/%d] %s — shape mismatch: prediction %s \u2260 GT %s. "
                "Predictions must have the same (H\u00d7W) dimensions as the "
                "ground-truth.  This case will be skipped.",
                idx, n_pred, case_id, prediction.shape, ground_truth.shape,
            )
            n_skip_shape += 1
            continue

        logger.info(
            "  [%d/%d] %-20s  pred=%-12s  mask=%s",
            idx, n_pred, case_id, str(prediction.shape),
            "yes" if mask is not None else "NO",
        )

        cases.append(
            Case(
                case_id=case_id,
                prediction=prediction,
                ground_truth=ground_truth,
                mask=mask,
                precontrast=precontrast,
                prediction_path=str(pred_file),
                ground_truth_path=str(gt_path),
                mask_path=str(mask_file) if mask_file else None,
            )
        )

    skipped = n_skip_gt + n_skip_shape
    if skipped:
        logger.warning(
            "  Skipped %d/%d case(s) — %d missing GT, %d shape mismatch.",
            skipped, n_pred, n_skip_gt, n_skip_shape,
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
        logger.warning(
            "nnunetv2 not installed — segmentation evaluator (Dice, HD95) "
            "will be disabled. Ensure nnunetv2 is listed in your container's "
            "requirements and installed correctly."
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
    logger.info(
        "  Segmentation model loaded from %s (fold %s, device %s)",
        seg_dir, fold, device,
    )

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

    n_evaluators = len(evaluators)
    for eval_idx, (name, evaluator) in enumerate(evaluators, 1):
        logger.info("[%d/%d] Running %s ...", eval_idx, n_evaluators, name)
        t0 = time.perf_counter()
        try:
            result = evaluator.evaluate(cases)  # type: ignore[attr-defined]
            for cid, m in result.per_case.items():
                all_per_case.setdefault(cid, {}).update(m)
            all_aggregates.update(result.aggregates)
            elapsed = time.perf_counter() - t0
            agg_keys = list(result.aggregates.keys())
            logger.info(
                "       %s: OK in %.1f s — %d aggregate metric(s): %s",
                name, elapsed,
                len(agg_keys),
                ", ".join(agg_keys) if agg_keys else "none",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error(
                "       %s FAILED after %.1f s: %s\n"
                "       This evaluator's metrics will be absent from the output. "
                "Other evaluators continue unaffected.",
                name, elapsed, exc,
            )

    # Clear the in-memory radiomic feature cache to free memory
    from evaluators.roi_metrics import clear_feature_cache
    clear_feature_cache()

    return {"case": all_per_case, "aggregates": all_aggregates}


# ======================================================================
# CLI entry point
# ======================================================================


def main() -> int:
    """Main entry point for the GC evaluation container."""
    t_start = time.perf_counter()
    _SEP = "=" * 60

    # ---- Banner ------------------------------------------------------
    logger.info(_SEP)
    logger.info("  MAMA-SYNTH Evaluation  |  Grand Challenge")
    logger.info("  Prediction slug : %s", PREDICTION_SLUG)
    logger.info(_SEP)

    # ---- Read configuration from environment -------------------------
    input_dir  = Path(os.environ.get("MAMA_INPUT_DIR",  "/input"))
    output_dir = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
    gt_dir     = Path(os.environ.get("MAMA_GT_DIR",
                                     "/opt/ml/input/data/ground_truth"))
    models_dir = Path(os.environ.get("MAMA_MODELS_DIR", "/opt/app/models"))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    # ---- Log resolved paths and verify existence ---------------------
    def _status(p: Path) -> str:
        return "EXISTS" if p.exists() else "NOT FOUND"

    logger.info("Paths:")
    logger.info("  Input  : %-50s [%s]", input_dir,  _status(input_dir))
    logger.info("  GT     : %-50s [%s]", gt_dir,     _status(gt_dir))
    logger.info("  Models : %-50s [%s]", models_dir, _status(models_dir))
    logger.info("  Output : %s", output_dir)

    if not input_dir.exists():
        logger.error(
            "Input directory not found: %s\n"
            "On Grand Challenge this should be /input. "
            "Set MAMA_INPUT_DIR to override.",
            input_dir,
        )
    if not gt_dir.exists():
        logger.error(
            "Ground-truth directory not found: %s\n"
            "Ensure ground_truth.zip was uploaded to this phase and "
            "extracted correctly by Grand Challenge.",
            gt_dir,
        )

    models_dir_arg: Optional[Path]
    if not models_dir.exists():
        logger.warning(
            "Models directory not found: %s\n"
            "Classification and segmentation evaluators will be disabled.\n"
            "Image-level metrics (MSE, LPIPS) and ROI metrics (SSIM-tumour, "
            "FRD) will still run.",
            models_dir,
        )
        models_dir_arg = None
    else:
        models_dir_arg = models_dir

    logger.info(_SEP)

    # ---- Discover and load cases -------------------------------------
    if (input_dir / "predictions.json").exists():
        logger.info("Mode: Grand Challenge  (predictions.json detected)")
        logger.info("Loading cases ...")
        cases = load_cases_gc(input_dir, gt_dir)
    else:
        logger.info("Mode: local / flat-directory")
        pred_dir = Path(os.environ.get("MAMA_PREDICTIONS_DIR", str(input_dir)))
        gt_images_dir = (
            gt_dir / "ground_truth"
            if (gt_dir / "ground_truth").exists()
            else gt_dir
        )
        masks_env = os.environ.get("MAMA_MASKS_DIR")
        masks_sub = gt_dir / "masks"
        masks_dir_local: Optional[Path] = (
            Path(masks_env) if masks_env
            else masks_sub if masks_sub.exists()
            else None
        )
        precon_env = os.environ.get("MAMA_PRECONTRAST_DIR")
        precon_dir: Optional[Path] = Path(precon_env) if precon_env else None
        logger.info(
            "  Predictions : %s\n"
            "  GT images   : %s\n"
            "  Masks       : %s",
            pred_dir, gt_images_dir,
            masks_dir_local if masks_dir_local
            else "none (SSIM-tumour / AUROC / FRD disabled)",
        )
        cases = load_cases_local(pred_dir, gt_images_dir, masks_dir_local, precon_dir)

    logger.info(_SEP)

    if not cases:
        logger.error(
            "No valid cases could be loaded.\n"
            "Common causes:\n"
            "  \u2022 Your algorithm writes to a different output socket "
            "(expected slug: '%s')\n"
            "  \u2022 Prediction filenames do not match ground-truth case IDs\n"
            "  \u2022 ground_truth.zip was not uploaded or extracted correctly\n"
            "  \u2022 All cases were skipped due to spatial shape mismatch\n"
            "Review the per-case warnings above for details.",
            PREDICTION_SLUG,
        )
        write_metrics({"case": {}, "aggregates": {}}, metrics_path)
        return 1

    n_mask   = sum(1 for c in cases if c.mask is not None)
    n_pre    = sum(1 for c in cases if c.precontrast is not None)
    n_nomask = len(cases) - n_mask
    logger.info(
        "Loaded %d case(s)  |  with mask: %d  |  without mask: %d  |  "
        "with pre-contrast: %d",
        len(cases), n_mask, n_nomask, n_pre,
    )
    if n_nomask > 0:
        logger.warning(
            "%d case(s) have no tumour mask — SSIM-tumour, AUROC-tumour-ROI "
            "and FRD will not be computed for those cases; aggregate scores "
            "will reflect fewer samples.",
            n_nomask,
        )
    logger.info(_SEP)

    # ---- Run evaluation ----------------------------------------------
    logger.info("Running evaluators ...")
    metrics = run_evaluation(cases, models_dir_arg)
    write_metrics(metrics, metrics_path)

    # ---- Final summary -----------------------------------------------
    elapsed_total = time.perf_counter() - t_start
    logger.info(_SEP)
    logger.info("Results written to: %s", metrics_path)
    agg = metrics.get("aggregates", {})
    if agg:
        logger.info("Aggregate metrics:")
        for key in sorted(agg):
            val = agg[key]
            if isinstance(val, dict) and "mean" in val:
                std = val.get("std", 0.0)
                logger.info("  %-32s %.4f  (\u00b1%.4f std)", key, val["mean"], std)
            else:
                logger.info("  %-32s %s", key, val)
    else:
        logger.warning(
            "No aggregate metrics were produced — all evaluators may have "
            "been skipped or failed. Review the warnings above."
        )
    logger.info("Total wall time: %.1f s", elapsed_total)
    logger.info(_SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

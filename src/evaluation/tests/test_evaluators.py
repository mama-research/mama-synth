"""Tests for the MAMA-SYNTH GC evaluation pipeline with mock images.

Each evaluator is tested independently, followed by an end-to-end
test that writes and reads .mha files through the full pipeline.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from evaluators import (
    Case,
    ClassificationEvaluator,
    ImageMetricsEvaluator,
    ROIMetricsEvaluator,
    SegmentationEvaluator,
)
from evaluators.segmentation import compute_dice, compute_hausdorff_95


# ======================================================================
# Helpers
# ======================================================================

def _make_case(seed: int = 42, size: int = 64) -> Case:
    """Create a deterministic mock Case with z-score-like images.

    Images are drawn from a standard normal distribution (mean≈0, std≈1)
    to match the z-score-normalised inputs expected by the GC pipeline.
    """
    rng = np.random.RandomState(seed)
    gt = rng.randn(size, size).astype(np.float64)  # z-score: mean≈0, std≈1
    pred = (gt + rng.normal(0, 0.05, gt.shape)).astype(np.float64)
    mask = np.zeros((size, size), dtype=bool)
    q = size // 4
    mask[q : 3 * q, q : 3 * q] = True
    precon = (gt * 0.5).astype(np.float64)
    return Case(
        case_id=f"mock_{seed}",
        prediction=pred,
        ground_truth=gt,
        mask=mask,
        precontrast=precon,
    )


def _write_mha(arr: np.ndarray, path: Path) -> None:
    """Write a 2-D numpy array as a .mha file."""
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, str(path))


# ======================================================================
# ImageMetricsEvaluator
# ======================================================================


class TestImageMetrics:
    def test_mse_present_for_all_cases(self) -> None:
        cases = [_make_case(i) for i in range(4)]
        result = ImageMetricsEvaluator().evaluate(cases)
        for c in cases:
            assert "mse" in result.per_case[c.case_id]
            assert result.per_case[c.case_id]["mse"] >= 0
        assert "mse" in result.aggregates
        assert "mean" in result.aggregates["mse"]

    def test_mse_zero_for_identical_images(self) -> None:
        c = _make_case()
        c.prediction = c.ground_truth.copy()
        result = ImageMetricsEvaluator().evaluate([c])
        assert abs(result.per_case[c.case_id]["mse"]) < 1e-12

    def test_lpips_present_when_available(self) -> None:
        """LPIPS should be present if torch + lpips/torchmetrics installed."""
        cases = [_make_case(0)]
        ev = ImageMetricsEvaluator()
        result = ev.evaluate(cases)
        if ev._lpips_available:
            assert "lpips" in result.per_case[cases[0].case_id]
            assert "lpips" in result.aggregates


# ======================================================================
# ROIMetricsEvaluator
# ======================================================================


class TestROIMetrics:
    def test_ssim_present_when_mask_provided(self) -> None:
        cases = [_make_case(i) for i in range(4)]
        result = ROIMetricsEvaluator().evaluate(cases)
        for c in cases:
            assert "ssim_tumor" in result.per_case[c.case_id]
            # skimage local-window SSIM is in [-1, 1]
            assert -1.0 <= result.per_case[c.case_id]["ssim_tumor"] <= 1.0
        assert "ssim_tumor" in result.aggregates

    def test_ssim_skipped_without_mask(self) -> None:
        c = _make_case()
        c.mask = None
        result = ROIMetricsEvaluator().evaluate([c])
        assert not result.per_case

    def test_frd_requires_at_least_two_cases(self) -> None:
        c = _make_case()
        result = ROIMetricsEvaluator().evaluate([c])
        # Only 1 case → FRD should not be in aggregates
        assert "frd" not in result.aggregates

    def test_frd_computed_with_enough_cases(self) -> None:
        cases = [_make_case(i) for i in range(4)]
        result = ROIMetricsEvaluator().evaluate(cases)
        if "frd" in result.aggregates:
            assert result.aggregates["frd"]["mean"] >= 0


# ======================================================================
# SegmentationEvaluator
# ======================================================================


class TestSegmentation:
    def test_dice_and_hd95(self) -> None:
        cases = [_make_case(i) for i in range(4)]
        ev = SegmentationEvaluator(segment_fn=lambda img: img > 0.5)
        result = ev.evaluate(cases)
        for cid in result.per_case:
            assert "dice" in result.per_case[cid]
            assert "hausdorff_95" in result.per_case[cid]
            assert 0.0 <= result.per_case[cid]["dice"] <= 1.0

    def test_no_fn_returns_empty(self) -> None:
        cases = [_make_case()]
        result = SegmentationEvaluator(segment_fn=None).evaluate(cases)
        assert not result.per_case
        assert not result.aggregates

    def test_perfect_dice(self) -> None:
        a = np.zeros((32, 32), dtype=bool)
        a[8:24, 8:24] = True
        assert compute_dice(a, a) == pytest.approx(1.0)

    def test_dice_empty_masks(self) -> None:
        z = np.zeros((32, 32), dtype=bool)
        assert compute_dice(z, z) == 1.0

    def test_hd95_identical_masks(self) -> None:
        m = np.zeros((32, 32), dtype=bool)
        m[8:24, 8:24] = True
        assert compute_hausdorff_95(m, m) == pytest.approx(0.0)


# ======================================================================
# ClassificationEvaluator
# ======================================================================


class TestClassification:
    def test_no_models_returns_empty(self) -> None:
        cases = [_make_case()]
        ev = ClassificationEvaluator(
            contrast_model=None, tumor_roi_model=None
        )
        result = ev.evaluate(cases)
        assert not result.aggregates

    def test_contrast_auroc_with_mock_model(self, tmp_path: Path) -> None:
        from sklearn.ensemble import RandomForestClassifier

        # Determine feature count by extracting from a sample first
        try:
            from evaluators.roi_metrics import extract_radiomic_features

            sample = _make_case(0)
            whole = np.ones(sample.prediction.shape, dtype=bool)
            n_features = extract_radiomic_features(
                sample.prediction, whole
            ).shape[0]
        except Exception:
            pytest.skip("pyradiomics unavailable")

        rng = np.random.RandomState(42)
        X = rng.rand(20, n_features)
        y = np.array([0] * 10 + [1] * 10)
        clf = RandomForestClassifier(n_estimators=2, random_state=42)
        clf.fit(X, y)
        model_path = tmp_path / "contrast_classifier.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(clf, fh)

        cases = [_make_case(i) for i in range(4)]
        ev = ClassificationEvaluator(
            contrast_model=model_path, tumor_roi_model=None
        )
        result = ev.evaluate(cases)
        assert "auroc_contrast" in result.aggregates
        auroc = result.aggregates["auroc_contrast"]["mean"]
        assert 0.0 <= auroc <= 1.0


# ======================================================================
# End-to-end pipeline with .mha files
# ======================================================================


class TestEndToEnd:
    """Full pipeline reading .mha files from disk."""

    def test_local_mode_produces_metrics_json(
        self, tmp_path: Path
    ) -> None:
        # ---- Create directories ------------------------------------
        pred_dir = tmp_path / "predictions"
        gt_dir = tmp_path / "gt"
        gt_images = gt_dir / "images"
        gt_masks = gt_dir / "masks"
        gt_precon = gt_dir / "precontrast"
        output_dir = tmp_path / "output"
        for d in (pred_dir, gt_images, gt_masks, gt_precon, output_dir):
            d.mkdir(parents=True)

        # ---- Generate mock .mha files ------------------------------
        rng = np.random.RandomState(99)
        size = 32
        for i in range(4):
            name = f"case_{i:03d}.mha"
            gt_arr = rng.randint(
                50, 200, (size, size), dtype=np.uint8
            ).astype(np.float64)
            pred_arr = np.clip(
                gt_arr + rng.normal(0, 10, gt_arr.shape), 0, 255
            ).astype(np.float64)
            precon_arr = (gt_arr * 0.5).astype(np.float64)
            mask_arr = np.zeros((size, size), dtype=np.uint8)
            mask_arr[8:24, 4:16] = 1  # left-side tumour

            _write_mha(pred_arr, pred_dir / name)
            _write_mha(gt_arr, gt_images / name)
            _write_mha(mask_arr, gt_masks / name)
            _write_mha(precon_arr, gt_precon / name)

        # ---- Run pipeline ------------------------------------------
        from evaluate import load_cases_local, run_evaluation, write_metrics

        cases = load_cases_local(
            pred_dir=pred_dir,
            gt_dir=gt_images,
            masks_dir=gt_masks,
            precon_dir=gt_precon,
        )
        assert len(cases) == 4

        metrics = run_evaluation(cases, models_dir=None)
        metrics_path = output_dir / "metrics.json"
        write_metrics(metrics, metrics_path)

        # ---- Verify output -----------------------------------------
        assert metrics_path.exists()
        with open(metrics_path) as fh:
            loaded = json.load(fh)

        assert "case" in loaded
        assert "aggregates" in loaded

        agg = loaded["aggregates"]
        # MSE must always be present
        assert "mse" in agg
        assert agg["mse"]["mean"] >= 0

        # SSIM should be present (masks were provided)
        assert "ssim_tumor" in agg

        # All 4 cases should have per-case metrics
        assert len(loaded["case"]) == 4
        for cid, cm in loaded["case"].items():
            assert "mse" in cm
            assert "ssim_tumor" in cm

    def test_gc_predictions_json_mode(self, tmp_path: Path) -> None:
        """Simulate the Grand Challenge predictions.json layout."""
        import uuid

        input_dir = tmp_path / "input"
        gt_dir = tmp_path / "ground_truth"
        gt_images = gt_dir / "images"
        gt_masks = gt_dir / "masks"
        gt_precon = gt_dir / "precontrast"
        output_dir = tmp_path / "output"
        for d in (gt_images, gt_masks, gt_precon, output_dir):
            d.mkdir(parents=True)

        rng = np.random.RandomState(77)
        size = 32
        predictions = []

        for i in range(3):
            case_name = f"case_{i:03d}.mha"
            case_stem = f"case_{i:03d}"
            job_pk = str(uuid.uuid4())

            # Create GT files
            gt_arr = rng.rand(size, size).astype(np.float64) * 200
            _write_mha(gt_arr, gt_images / case_name)
            _write_mha(
                np.zeros((size, size), dtype=np.uint8),
                gt_masks / case_name,
            )
            _write_mha(gt_arr * 0.5, gt_precon / case_name)

            # Create prediction file in GC layout
            pred_subdir = (
                input_dir
                / job_pk
                / "output"
                / "images"
                / "synthetic-post-contrast-breast-mri"
            )
            pred_subdir.mkdir(parents=True)
            pred_arr = gt_arr + rng.normal(0, 5, gt_arr.shape)
            pred_uuid = str(uuid.uuid4())
            _write_mha(pred_arr, pred_subdir / f"{pred_uuid}.mha")

            predictions.append(
                {
                    "pk": job_pk,
                    "inputs": [
                        {
                            "interface": {
                                "slug": "pre-contrast-breast-mri"
                            },
                            "image": {"name": case_name},
                        }
                    ],
                    "outputs": [
                        {
                            "interface": {
                                "slug": "synthetic-post-contrast-breast-mri",
                                "relative_path": (
                                    "images/"
                                    "synthetic-post-contrast-breast-mri"
                                ),
                            }
                        }
                    ],
                }
            )

        input_dir.mkdir(exist_ok=True)
        with open(input_dir / "predictions.json", "w") as fh:
            json.dump(predictions, fh)

        # ---- Load and evaluate -------------------------------------
        from evaluate import load_cases_gc, run_evaluation

        cases = load_cases_gc(input_dir, gt_dir)
        assert len(cases) == 3

        metrics = run_evaluation(cases)
        assert "aggregates" in metrics
        assert "mse" in metrics["aggregates"]

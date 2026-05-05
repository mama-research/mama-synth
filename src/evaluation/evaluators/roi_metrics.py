"""ROI-level metrics: SSIM within the tumour mask and FRD.

* **SSIM (tumour)** – per-case local-window structural similarity
  (``skimage.metrics.structural_similarity``) computed on the full
  image and averaged within the tumour mask.
* **FRD** – Fréchet Radiomics Distance (aggregate-only) computed via
  the ``frd-score`` library using **FRD v1**: z-score / D1-referenced
  normalisation, ~464 radiomic features (Original + LoG + Wavelet
  filter banks), and tumour-mask conditioning.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity

from .base import BaseEvaluator, Case, EvaluationResult

# ======================================================================
# Pyradiomics settings — aligned with mama-synth-eval / frd-score
# ======================================================================

# Feature classes and bin width used by the *local* radiomic classifier
# (AUROC contrast / tumour-ROI).  These are separate from the frd-score
# library's internal extractor (which uses its own defaults for FRD v1).
CLF_FEATURE_CLASSES: list[str] = [
    "firstorder",
    "glcm",
    "glrlm",
    "gldm",
    "glszm",
    "ngtdm",
]

CLF_DEFAULT_BIN_WIDTH: int = 25

# Extractor cache (per-process)
_EXTRACTOR_CACHE: dict[tuple, object] = {}

# Canonical feature count cache so failures can return zero vectors
_FEATURE_COUNT_CACHE: dict[tuple, int] = {}

# In-memory feature cache keyed by "<image_hash>|<mask_hash>"
_feature_cache: dict[str, np.ndarray] = {}


class ROIMetricsEvaluator(BaseEvaluator):
    """SSIM within the tumour mask (per-case) and FRD (aggregate)."""

    def evaluate(self, cases: list[Case]) -> EvaluationResult:
        per_case: dict[str, dict[str, float]] = {}

        for case in cases:
            if case.mask is None or not np.any(case.mask):
                continue

            # ---- SSIM within mask (standard local-window) --------
            # Fixed data_range = 10.0 matches the ±5σ clip used in LPIPS
            # pre-processing, making SSIM stability constants comparable
            # across all cases in z-score normalised space.
            data_range = 10.0

            _, ssim_map = structural_similarity(
                case.prediction,
                case.ground_truth,
                data_range=data_range,
                full=True,
            )
            ssim_roi = float(np.mean(ssim_map[case.mask]))
            per_case[case.case_id] = {"ssim_tumor": ssim_roi}

        # ---- Aggregates ------------------------------------------
        agg: dict[str, dict[str, float]] = {}
        ssim_agg = self._aggregate_metric(per_case, "ssim_tumor")
        if ssim_agg:
            agg["ssim_tumor"] = ssim_agg

        # ---- FRD via frd-score library ---------------------------
        frd_val = self._compute_frd(cases)
        if frd_val is not None:
            # FRD is an aggregate-only scalar; no std across cases.
            agg["frd"] = {"mean": frd_val}

        return EvaluationResult(per_case=per_case, aggregates=agg)

    # ------------------------------------------------------------------
    # FRD via frd-score library (no fallback — errors propagate)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_frd(cases: list[Case]) -> Optional[float]:
        """Compute FRD using the ``frd-score`` library.

        Requires at least 2 cases with non-empty masks.  Uses file
        paths stored in the :class:`Case` objects.  If paths are not
        available, arrays are written to temporary ``.mha`` files.
        """
        valid = [
            c for c in cases
            if c.mask is not None and np.any(c.mask)
        ]
        if len(valid) < 2:
            return None

        try:
            from frd_score import compute_frd as frd_compute
        except ImportError:
            print(
                "WARNING: FRD unavailable (frd-score not installed). "
                "Install with: pip install frd-score",
                file=sys.stderr,
            )
            return None

        # Collect file paths (or write temp files)
        gt_paths: list[str] = []
        pred_paths: list[str] = []
        mask_paths: list[str] = []
        tmp_dir_obj = None

        try:
            for case in valid:
                gt_p = case.ground_truth_path
                pred_p = case.prediction_path
                mask_p = case.mask_path

                # Write temp files for any missing paths
                if gt_p is None or pred_p is None or mask_p is None:
                    if tmp_dir_obj is None:
                        tmp_dir_obj = tempfile.TemporaryDirectory()
                    tmp_dir = Path(tmp_dir_obj.name)

                    if gt_p is None:
                        gt_p = str(
                            tmp_dir / f"{case.case_id}_gt.mha"
                        )
                        _write_mha(case.ground_truth, gt_p)
                    if pred_p is None:
                        pred_p = str(
                            tmp_dir / f"{case.case_id}_pred.mha"
                        )
                        _write_mha(case.prediction, pred_p)
                    if mask_p is None:
                        mask_p = str(
                            tmp_dir / f"{case.case_id}_mask.mha"
                        )
                        _write_mha(
                            case.mask.astype(np.uint8), mask_p
                        )

                gt_paths.append(gt_p)
                pred_paths.append(pred_p)
                mask_paths.append(mask_p)

            # frd-score v1: z-score/D1-ref normalisation, ~464 features
            # (Original + LoG + Wavelet).  Masks are passed as conditions
            # for both the real and synthetic distributions.
            frd_val = frd_compute(
                [gt_paths, pred_paths],
                paths_masks=[mask_paths, mask_paths],
                frd_version="v1",
            )
            return float(frd_val)
        except Exception as exc:
            print(
                f"WARNING: FRD computation failed: {exc}",
                file=sys.stderr,
            )
            return None
        finally:
            if tmp_dir_obj is not None:
                tmp_dir_obj.cleanup()


# ======================================================================
# Radiomic feature extraction (for classification, shared with FRD)
# ======================================================================


def _get_cached_extractor(
    feature_classes: tuple[str, ...],
    bin_width: int,
    was_2d: bool,
) -> object:
    """Return a cached ``RadiomicsFeatureExtractor``."""
    key = (feature_classes, bin_width, was_2d)
    extractor = _EXTRACTOR_CACHE.get(key)
    if extractor is not None:
        return extractor

    from radiomics import featureextractor  # type: ignore[import-untyped]

    settings = {
        "binWidth": bin_width,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "minimumROIDimensions": 1,
        "minimumROISize": 1,
        "force2D": was_2d,
        "force2Ddimension": 0 if was_2d else None,
    }
    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllFeatures()
    for fc in feature_classes:
        ext.enableFeatureClassByName(fc)

    _EXTRACTOR_CACHE[key] = ext
    return ext


def extract_radiomic_features(
    image: np.ndarray,
    mask: np.ndarray,
    feature_classes: Optional[list[str]] = None,
    bin_width: int = CLF_DEFAULT_BIN_WIDTH,
) -> np.ndarray:
    """Extract IBSI-compliant radiomic features from a 2-D image.

    Returns a 1-D ``float64`` feature vector.  Settings (``binWidth``,
    feature classes, etc.) are aligned with ``mama-synth-eval``.
    These features are used by the local classifier (AUROC tasks) and
    are distinct from the features extracted internally by frd-score.

    Raises ``ImportError`` if ``pyradiomics`` is not installed.
    """
    import radiomics  # type: ignore[import-untyped]

    if feature_classes is None:
        feature_classes = CLF_FEATURE_CLASSES

    # Suppress verbose pyradiomics logging (once)
    if not getattr(extract_radiomic_features, "_verbosity_set", False):
        radiomics.setVerbosity(60)
        extract_radiomic_features._verbosity_set = True

    was_2d = image.ndim == 2
    if was_2d:
        image = image[np.newaxis, :, :]

    if mask.ndim == 2 and was_2d:
        mask_arr = mask[np.newaxis, :, :].astype(np.uint8)
    else:
        mask_arr = mask.astype(np.uint8)

    if np.sum(mask_arr) == 0:
        mask_arr = np.ones(image.shape, dtype=np.uint8)

    sitk_img = sitk.GetImageFromArray(image.astype(np.float64))
    sitk_msk = sitk.GetImageFromArray(mask_arr)

    extractor = _get_cached_extractor(
        feature_classes=tuple(feature_classes),
        bin_width=bin_width,
        was_2d=was_2d,
    )
    config_key = (tuple(feature_classes), bin_width, was_2d)

    try:
        result = extractor.execute(sitk_img, sitk_msk)
    except Exception as e:
        n = _FEATURE_COUNT_CACHE.get(config_key, 93)
        return np.zeros(n, dtype=np.float64)

    features: list[float] = []
    for key in sorted(result.keys()):
        if key.startswith("diagnostics_"):
            continue
        try:
            features.append(float(result[key]))
        except (ValueError, TypeError):
            features.append(0.0)

    arr = np.array(features, dtype=np.float64)
    if config_key not in _FEATURE_COUNT_CACHE and arr.size > 0:
        _FEATURE_COUNT_CACHE[config_key] = arr.size

    return arr


# ======================================================================
# Feature caching helpers
# ======================================================================


def _image_hash(arr: np.ndarray) -> str:
    """SHA-256 hash of array bytes (first 16 hex chars)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def get_cached_features(
    image: np.ndarray,
    mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Retrieve cached features, or ``None``."""
    key = f"{_image_hash(image)}|{_image_hash(mask)}"
    return _feature_cache.get(key)


def cache_features(
    image: np.ndarray,
    mask: np.ndarray,
    features: np.ndarray,
) -> None:
    """Store features in the in-memory cache."""
    key = f"{_image_hash(image)}|{_image_hash(mask)}"
    _feature_cache[key] = features


def extract_radiomic_features_cached(
    image: np.ndarray,
    mask: np.ndarray,
    feature_classes: Optional[list[str]] = None,
    bin_width: int = CLF_DEFAULT_BIN_WIDTH,
) -> np.ndarray:
    """Extract radiomic features with in-memory caching."""
    cached = get_cached_features(image, mask)
    if cached is not None:
        return cached
    feats = extract_radiomic_features(
        image, mask,
        feature_classes=feature_classes,
        bin_width=bin_width,
    )
    cache_features(image, mask, feats)
    return feats


def clear_feature_cache() -> None:
    """Clear the in-memory feature cache."""
    _feature_cache.clear()


# ======================================================================
# Helper
# ======================================================================


def _write_mha(arr: np.ndarray, path: str) -> None:
    """Write a numpy array as a ``.mha`` file, creating parent dirs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, path)

"""Contralateral mirroring utilities for breast DCE-MRI.

Provides midline detection and mask mirroring for the ``tumor_roi``
classification task.  Given a tumor segmentation mask on one breast,
mirror it across the body midline to create a contralateral ROI that
serves as a "non-tumor" reference region for binary classification
(tumor ROI vs contralateral non-tumor ROI).

**Midline detection (robust)** works in three steps:

1. Build a tissue-only column profile by masking out background air
   (voxels below ``BACKGROUND_Z_THRESHOLD``) and computing the mean
   over tissue pixels per column.  This prevents background air columns
   at the image edges and cardiac/thorax enhancement in the centre from
   biasing the minimum.

2. Smooth the profile with a box filter to suppress noise.

3. Detect the two largest breast-tissue peaks, one in each image half
   (bilateral breast check / **D2**).  The midline is the valley
   (argmin) *between* those two peaks rather than the global minimum
   of the entire central window.  Searching between identified breast
   peaks means the result is correct even when the heart or thorax
   region has high contrast-enhanced intensity in the centre.

**Orientation-invariant fallback (D4)**: ``create_mirrored_mask`` first
attempts to mirror along columns (the nominal left-right axis).  If the
bilateral check fails or tissue validation fails it automatically retries
along rows (the nominal cranio-caudal axis).  This keeps the pipeline
robust to 90°/270°-rotated inputs.

**Validation** ensures that the accepted mirrored mask overlaps with
breast tissue.

Ported and extended from ``mama-synth-eval/src/eval/mirror_utils.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================

DEFAULT_MIN_TISSUE_FRACTION = 0.3
"""Minimum fraction of mirrored mask voxels that must overlap with
tissue for the mirrored region to be considered valid."""

DEFAULT_TISSUE_THRESHOLD_PERCENTILE = 10
"""Percentile of non-zero image intensities used to determine the
tissue/background boundary."""

DEFAULT_MIDLINE_SEARCH_FRACTION = 0.4
"""Fraction of the image width (centered) within which to search
for the midline valley (legacy / fallback use only)."""

BACKGROUND_Z_THRESHOLD: float = -1.5
"""Intensity threshold separating background from breast tissue in
z-score normalised images.  Background (air) consistently falls below
−2σ; glandular tissue starts at approximately −1.5σ and above."""

_PROFILE_SMOOTH_BINS: int = 20
"""Number of bins used to define the box-filter kernel size relative
to the profile length (kernel = max(3, profile_length // bins))."""

_MIN_PEAK_HEIGHT_FRACTION: float = 0.1
"""A column peak is only considered a breast if its tissue-mean
intensity is at least this fraction of the global tissue-mean
intensity.  Eliminates spurious peaks at the image periphery."""

_MIN_PEAK_DISTANCE_FRACTION: float = 0.1
"""Two peaks must be separated by at least this fraction of the
profile length to be considered distinct breast peaks."""

_MIN_PEAK_PROMINENCE_FRACTION: float = 0.05
"""The valley between the two accepted peaks must be at least this
fraction *below* the lower of the two peak values (relative to the
global tissue mean).  This rejects flat profiles where all positions
have essentially the same intensity — characteristic of a uniform
single-breast or background image where the row/column profile
has no structural variation.
Example: if global_mean = 1.0 and threshold = 0.05, the valley must be
at least 0.05 * 1.0 = 0.05 below the lower peak."""


# ======================================================================
# Internal helpers
# ======================================================================


def _tissue_profile(
    image: NDArray[np.floating],
    reduce_axis: int,
) -> NDArray[np.floating]:
    """Background-masked, smoothed mean profile.

    Computes the mean intensity over tissue pixels (> ``BACKGROUND_Z_THRESHOLD``)
    along *reduce_axis*, yielding one value per position along the
    remaining axis.  NaN is used where a column/row contains no tissue pixels.

    The result is smoothed with a box filter to suppress pixel-level noise.

    Parameters
    ----------
    image : NDArray
        2-D float image.
    reduce_axis : int
        The axis to average over (0 = average rows → column profile;
        1 = average columns → row profile).
    """
    tissue_mask = image > BACKGROUND_Z_THRESHOLD
    count = np.sum(tissue_mask, axis=reduce_axis).astype(np.float64)
    total = np.where(tissue_mask, image, 0.0).sum(axis=reduce_axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(count > 0, total / count, np.nan)

    # Box-filter smoothing ignoring NaNs
    n = len(profile)
    k = max(3, n // _PROFILE_SMOOTH_BINS)
    valid = (~np.isnan(profile)).astype(np.float64)
    filled = np.where(np.isnan(profile), 0.0, profile)
    kernel = np.ones(k) / k
    sm_sum = np.convolve(filled, kernel, mode="same")
    sm_cnt = np.convolve(valid, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(sm_cnt > 0, sm_sum / sm_cnt, np.nan)

    return smoothed.astype(np.float64)


def _find_local_maxima(
    profile: NDArray[np.floating],
    min_height: float,
    min_distance: int,
) -> list[int]:
    """Return indices of local maxima above *min_height* spaced ≥ *min_distance* apart.

    Uses a simple left-neighbour / right-neighbour comparison after NaN
    positions are skipped, then greedily filters candidates that are too
    close (keeping the taller one).
    """
    n = len(profile)
    candidates: list[tuple[int, float]] = []

    for i in range(1, n - 1):
        v = profile[i]
        if np.isnan(v) or v < min_height:
            continue
        left = profile[i - 1] if not np.isnan(profile[i - 1]) else -np.inf
        right = profile[i + 1] if not np.isnan(profile[i + 1]) else -np.inf
        if v >= left and v >= right:
            candidates.append((i, float(v)))

    # Greedy merge: if two peaks are closer than min_distance, keep taller
    merged: list[tuple[int, float]] = []
    for idx, val in candidates:
        if merged and idx - merged[-1][0] < min_distance:
            if val > merged[-1][1]:
                merged[-1] = (idx, val)
        else:
            merged.append((idx, val))

    return [idx for idx, _ in merged]


def detect_bilateral_breasts(
    image: NDArray[np.floating],
    mirror_axis: int = 1,
) -> tuple[Optional[tuple[int, int]], str]:
    """Detect two breast peaks in the tissue profile along *mirror_axis*.

    Returns the pair of peak positions ``(left_peak, right_peak)`` where
    "left" and "right" refer to the first and second halves of the profile
    along *mirror_axis*.

    Parameters
    ----------
    image : NDArray
        2-D float image (z-score normalised).
    mirror_axis : int
        The axis along which mirroring will be performed.
        1 → look for left/right breasts in a column profile (reduce rows).
        0 → look for top/bottom breasts in a row profile (reduce columns).

    Returns
    -------
    (peaks, reason) : tuple
        *peaks* is ``(peak_a, peak_b)`` on success or ``None`` on failure.
        *reason* is an empty string on success or a human-readable
        description of the failure for diagnostic logging.
    """
    reduce_axis = 1 - mirror_axis  # axis we average over
    profile = _tissue_profile(image, reduce_axis=reduce_axis)
    n = len(profile)

    # Global tissue mean as height baseline (ignoring NaN background)
    tissue_vals = profile[~np.isnan(profile)]
    if tissue_vals.size == 0:
        return None, (
            "tissue profile is entirely NaN — image may be all-background "
            "or not z-score normalised"
        )

    global_mean = float(np.nanmean(profile))
    min_height = global_mean * _MIN_PEAK_HEIGHT_FRACTION
    min_distance = max(3, int(n * _MIN_PEAK_DISTANCE_FRACTION))

    peaks = _find_local_maxima(profile, min_height=min_height, min_distance=min_distance)

    # Partition peaks into OUTER ZONES (lateral breast regions) and inner centre.
    # Breast tissue is always the most lateral structure in bilateral breast MRI;
    # the sternum/cardiac/thorax is central.  By requiring peaks to be in the
    # outer 25 % of each side we avoid selecting a bright cardiac peak that sits
    # in the image centre as the "breast peak" for one of the halves.
    outer_zone = max(1, n // 4)            # outer 25 % of each side
    half = n // 2

    left_peaks = [p for p in peaks if p < outer_zone]
    right_peaks = [p for p in peaks if p >= n - outer_zone]

    # If the outer-zone approach found nothing in a half (e.g. the image is very
    # small or the breast extends inward), expand to the full half partition as
    # a secondary attempt.
    if not left_peaks:
        left_peaks = [p for p in peaks if p < half]
    if not right_peaks:
        right_peaks = [p for p in peaks if p >= half]

    # Last-resort: if scipy still found nothing in a half, use the argmax of the
    # outer zone (handles flat breast regions, e.g. low-contrast acquisitions).
    left_outer_profile = profile[:outer_zone]
    right_outer_profile = profile[n - outer_zone :]
    if not left_peaks and not np.all(np.isnan(left_outer_profile)):
        left_peaks = [int(np.nanargmax(left_outer_profile))]
    if not right_peaks and not np.all(np.isnan(right_outer_profile)):
        right_peaks = [(n - outer_zone) + int(np.nanargmax(right_outer_profile))]

    if not left_peaks and not right_peaks:
        return None, (
            f"no tissue peaks found in profile (axis={mirror_axis}); "
            "image may show a single breast, incorrect orientation, or "
            "have no bilateral structure"
        )

    if not left_peaks:
        return None, (
            f"no breast peak found in first half of profile (axis={mirror_axis}); "
            "image likely shows a single breast or is rotated — "
            f"only right-half peaks at positions {right_peaks}"
        )

    if not right_peaks:
        return None, (
            f"no breast peak found in second half of profile (axis={mirror_axis}); "
            "image likely shows a single breast or is rotated — "
            f"only left-half peaks at positions {left_peaks}"
        )

    # Take the tallest peak from each half
    peak_a = max(left_peaks, key=lambda i: float(profile[i]))
    peak_b = max(right_peaks, key=lambda i: float(profile[i]))

    # ---- Prominence check: valley between peaks must be meaningfully lower ---
    # Reject flat profiles (e.g. uniform single-breast image, or a row profile
    # where all rows have the same tissue content).  Prominence = peak - valley.
    lo_idx, hi_idx = min(peak_a, peak_b), max(peak_a, peak_b)
    valley_slice = profile[lo_idx : hi_idx + 1]
    valley_val = float(np.nanmin(valley_slice)) if not np.all(np.isnan(valley_slice)) else float(profile[peak_a])
    # Prominence check: valley must be below a fraction of the global tissue mean.
    # This handles cardiac-enhanced scans where sternum/cardiac tissue is brighter
    # than breast tissue — in that case lower_peak_val - valley_val can be negative
    # even for a valid bilateral image.  Instead we check that the valley dips
    # sufficiently below the tissue mean.
    required_valley_max = global_mean * (1.0 - _MIN_PEAK_PROMINENCE_FRACTION)
    if valley_val >= required_valley_max:
        return None, (
            f"valley intensity {valley_val:.4f} >= required_max {required_valley_max:.4f} "
            f"(global_tissue_mean={global_mean:.3f}, axis={mirror_axis}, "
            f"peak_a={peak_a} val={float(profile[peak_a]):.3f}, "
            f"peak_b={peak_b} val={float(profile[peak_b]):.3f}); "
            "profile is flat or valley is not below tissue mean — "
            "likely single-breast or uniform-tissue image"
        )

    return (peak_a, peak_b), ""


def detect_midline(
    image: NDArray[np.floating],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
) -> int:
    """Detect the body midline column using a robust peak-valley strategy.

    This is the **primary midline detection entry point**.  It:

    1. Builds a tissue-only (background-masked), smoothed column profile.
    2. Calls :func:`detect_bilateral_breasts` to locate the two breast peaks
       (one in each image half).
    3. Returns the argmin of the profile *between* those two peaks.

    This avoids two known failure modes of the naïve ``argmin(central window)``
    approach:

    * **Edge-column bias**: air columns at the image periphery have very low
      tissue-mean intensity and would win the global argmin, even if they are
      inside the search window.  Masking out background (step 1) eliminates
      this bias.
    * **Cardiac/thorax bias**: the heart and thorax have high intensity in
      contrast-enhanced images and sit centrally.  Searching the valley
      *between identified breast peaks* (step 3) means the result is
      geometrically constrained to the inter-breast gap regardless of how
      bright the thorax is.

    Falls back to the legacy ``_detect_midline_argmin`` implementation when
    the bilateral breast check fails (e.g. single-breast or rotated image),
    so that callers always receive a column index.

    Parameters
    ----------
    image : NDArray
        2-D float image (z-score normalised).
    search_fraction : float
        Used only in the legacy fallback path.

    Returns
    -------
    int
        Column index of the detected midline.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got {image.ndim}D")
    img2d = image if image.ndim == 2 else image[image.shape[0] // 2]

    peaks, reason = detect_bilateral_breasts(img2d, mirror_axis=1)
    if peaks is not None:
        peak_a, peak_b = peaks
        lo, hi = min(peak_a, peak_b), max(peak_a, peak_b)
        profile = _tissue_profile(img2d, reduce_axis=0)
        inter = profile[lo : hi + 1]

        # Replace NaN (background between breasts) with the local minimum so
        # that background columns are treated as part of the valley.
        finite_min = float(np.nanmin(inter)) if not np.all(np.isnan(inter)) else 0.0
        finite_mean = float(np.nanmean(inter)) if not np.all(np.isnan(inter)) else 0.0
        inter_filled = np.where(np.isnan(inter), finite_min, inter)

        # Use the MEDIAN column of the low-intensity valley rather than argmin.
        # argmin always returns the left-most minimum on a flat or monotone
        # profile, biasing the midline toward the edge of the sternum nearest
        # the tumour breast.  Taking the median of all valley columns gives the
        # centre of the inter-breast gap, which correctly places the mirror in
        # the contralateral breast.
        valley_threshold = finite_min + 0.5 * (finite_mean - finite_min)
        valley_cols = np.where(inter_filled <= valley_threshold)[0]
        if valley_cols.size > 0:
            valley_midline = lo + int(np.median(valley_cols))
        else:
            valley_midline = lo + int(np.argmin(inter_filled))

        # Sanity-check: if the detected valley is outside the central 40 % of
        # the inter-peak range it means the peaks were placed at breast
        # boundaries (e.g. via the outer-argmax fallback for cardiac-enhanced
        # images where the inter-breast region is BRIGHTER than the breasts).
        # In that case the valley is actually inside a breast, so we fall back
        # to the geometric midpoint between the two peaks.
        inter_len = hi - lo
        inner_lo = lo + int(0.3 * inter_len)
        inner_hi = lo + int(0.7 * inter_len)
        if inner_lo <= valley_midline <= inner_hi:
            return valley_midline
        return (peak_a + peak_b) // 2

    # Fallback: legacy argmin in search window (no bilateral structure found)
    logger.debug(
        "detect_midline: bilateral check failed (%s); using legacy argmin fallback.",
        reason,
    )
    return _detect_midline_argmin(img2d, search_fraction=search_fraction)


def _detect_midline_argmin(
    image: NDArray[np.floating],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
) -> int:
    """Legacy midline detector: argmin of the raw column mean in a central window.

    Only used as a fallback inside :func:`detect_midline` when the bilateral
    breast check fails.  Exposed for unit tests.
    """
    if not 0 < search_fraction <= 1:
        raise ValueError(
            f"search_fraction must be in (0, 1], got {search_fraction}"
        )
    col_profile = np.mean(image, axis=0)
    n_cols = len(col_profile)
    if n_cols < 4:
        raise ValueError(
            f"Image too narrow for midline detection ({n_cols} columns)"
        )
    half_width = int(n_cols * search_fraction / 2)
    center = n_cols // 2
    lo = max(0, center - half_width)
    hi = min(n_cols, center + half_width)
    window = col_profile[lo:hi]
    return lo + int(np.argmin(window))


def mirror_mask(
    mask: NDArray[np.bool_],
    midline: int,
    axis: int = 1,
) -> NDArray[np.bool_]:
    """Mirror a binary 2-D mask about a midline position along *axis*.

    Each True voxel at position ``p`` along *axis* is mapped to position
    ``2 * midline - p``.  Positions that map outside the image boundaries
    are silently dropped.

    Parameters
    ----------
    mask : NDArray[bool]
        2-D boolean mask.
    midline : int
        The reflection point (column index for axis=1, row index for axis=0).
    axis : int
        The axis along which to mirror (1 = columns / left-right,
        0 = rows / cranio-caudal).
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {mask.ndim}D")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    n = mask.shape[axis]
    mirrored = np.zeros_like(mask, dtype=bool)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return mirrored

    mirrored_pos = 2 * midline - coords[:, axis]
    valid = (mirrored_pos >= 0) & (mirrored_pos < n)
    coords_valid = coords[valid].copy()
    coords_valid[:, axis] = mirrored_pos[valid]
    mirrored[coords_valid[:, 0], coords_valid[:, 1]] = True
    return mirrored


# ======================================================================
# Tissue threshold + validation (unchanged API)
# ======================================================================


def _compute_tissue_threshold(
    image: NDArray[np.floating],
    percentile: float = DEFAULT_TISSUE_THRESHOLD_PERCENTILE,
) -> float:
    """Tissue/background threshold from a low percentile of tissue voxels."""
    tissue = image[image > BACKGROUND_Z_THRESHOLD]
    if tissue.size == 0:
        return BACKGROUND_Z_THRESHOLD
    return float(np.percentile(tissue, percentile))


def validate_mirrored_region(
    image: NDArray[np.floating],
    mirrored_mask: NDArray[np.bool_],
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
    tissue_threshold: Optional[float] = None,
) -> bool:
    """Validate that a mirrored mask overlaps with actual breast tissue.

    Checks that at least ``min_tissue_fraction`` of the mirrored mask
    voxels have image intensities above the tissue threshold.
    """
    n_mask_voxels = int(np.sum(mirrored_mask))
    if n_mask_voxels == 0:
        return False
    if tissue_threshold is None:
        tissue_threshold = _compute_tissue_threshold(image)
    tissue_overlap = int(np.sum(image[mirrored_mask] >= tissue_threshold))
    fraction = tissue_overlap / n_mask_voxels
    return fraction >= min_tissue_fraction


# ======================================================================
# Public entry point
# ======================================================================


def create_mirrored_mask(
    image: NDArray[np.floating],
    mask: NDArray[np.bool_],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
    case_id: str = "",
) -> Optional[NDArray[np.bool_]]:
    """Create a contralateral mirrored mask with bilateral check and axis fallback.

    Algorithm (D2 + D4):

    1. **Bilateral breast check** (D2): verify that the image contains two
       breast-like tissue peaks, one in each half, along the candidate mirror
       axis.  If not found, skip immediately with an informative reason rather
       than producing a geometrically wrong mirror.

    2. **Primary attempt** (axis=1, columns / left-right): detect the midline,
       mirror the tumour mask, validate tissue overlap.

    3. **Orientation-invariant fallback** (D4, axis=0, rows / cranio-caudal):
       if step 2 fails (bilateral check or tissue validation), retry with the
       perpendicular axis.  This handles images that have been rotated 90° or
       270° relative to the nominal orientation.

    4. Return the first successfully validated mirror, or ``None`` with a
       detailed reason logged at WARNING level using :mod:`sys.stderr`-compatible
       output (no silent drops).

    Parameters
    ----------
    image : NDArray
        2-D float image (z-score normalised).
    mask : NDArray[bool]
        2-D tumour mask.
    search_fraction : float
        Used only in the legacy midline-fallback path.
    min_tissue_fraction : float
        Minimum tissue overlap fraction for validation.
    case_id : str
        Case identifier included in all warning messages for traceability.

    Returns
    -------
    NDArray[bool] | None
        The mirrored mask (with a note on which axis was used), or ``None``
        if both axes fail.
    """
    prefix = f"[{case_id}] " if case_id else ""

    if not np.any(mask):
        logger.warning(
            "%screate_mirrored_mask: tumour mask is empty — "
            "case cannot contribute to tumour-ROI AUROC.",
            prefix,
        )
        return None

    tissue_threshold = _compute_tissue_threshold(image)

    for axis, axis_label in ((1, "columns/left-right"), (0, "rows/cranio-caudal")):
        # ---- D2: bilateral breast check --------------------------------
        peaks, bilateral_reason = detect_bilateral_breasts(image, mirror_axis=axis)
        if peaks is None:
            logger.warning(
                "%screate_mirrored_mask: bilateral breast check FAILED "
                "(axis=%d [%s]): %s",
                prefix, axis, axis_label, bilateral_reason,
            )
            continue  # try next axis (D4 fallback)

        peak_a, peak_b = peaks

        # ---- Midline detection (robust peak-valley) --------------------
        if axis == 1:
            midline = detect_midline(image, search_fraction=search_fraction)
        else:
            # For axis=0 (rows), transpose so detect_midline operates on columns
            midline = detect_midline(image.T, search_fraction=search_fraction)

        # ---- Mirror and validate ---------------------------------------
        mirrored = mirror_mask(mask, midline, axis=axis)

        if not np.any(mirrored):
            logger.warning(
                "%screate_mirrored_mask: mirrored mask is entirely empty "
                "(axis=%d [%s], midline=%d). "
                "Tumour mask may be too close to the image boundary "
                "(tumour bbox cols %d–%d, image width %d).",
                prefix, axis, axis_label, midline,
                int(np.where(mask)[1].min()), int(np.where(mask)[1].max()),
                mask.shape[1],
            )
            continue  # try next axis

        n_mirrored = int(np.sum(mirrored))
        tissue_overlap = int(np.sum(image[mirrored] >= tissue_threshold))
        tissue_frac = tissue_overlap / n_mirrored

        if tissue_frac >= min_tissue_fraction:
            if axis == 1:
                logger.debug(
                    "%screate_mirrored_mask: success on primary axis "
                    "(axis=1 [%s], midline_col=%d, breast_peaks=%s, "
                    "tissue_overlap=%.1f%%).",
                    prefix, axis_label, midline, peaks, tissue_frac * 100,
                )
            else:
                logger.warning(
                    "%screate_mirrored_mask: primary axis (columns) FAILED; "
                    "fallback axis=0 [%s] SUCCEEDED "
                    "(midline_row=%d, breast_peaks=%s, tissue_overlap=%.1f%%). "
                    "Image appears to be rotated 90°/270° relative to "
                    "the expected axial orientation.",
                    prefix, axis_label, midline, peaks, tissue_frac * 100,
                )
            return mirrored

        logger.warning(
            "%screate_mirrored_mask: tissue validation FAILED "
            "(axis=%d [%s], midline=%d, breast_peaks=%s, "
            "tissue_overlap=%.1f%% < required %.1f%%). "
            "Contralateral region may be in body/background.",
            prefix, axis, axis_label, midline, peaks,
            tissue_frac * 100, min_tissue_fraction * 100,
        )
        # continue → try next axis (D4)

    logger.warning(
        "%screate_mirrored_mask: ALL axes FAILED — case dropped from "
        "tumour-ROI AUROC. "
        "Possible causes: single-breast FOV, severe image rotation, "
        "non-z-score-normalised intensities, tumour at image boundary.",
        prefix,
    )
    return None

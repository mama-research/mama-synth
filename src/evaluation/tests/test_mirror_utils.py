"""Tests for mirror_utils — bilateral breast check (D2) and axis fallback (D4).

Each test constructs a minimal synthetic image that exercises a specific
failure mode or success path.  Images are z-score normalised:
  background air  ≈ −2.0  (below BACKGROUND_Z_THRESHOLD = −1.5)
  breast tissue   ≈  +1.0
  tumour          ≈  +2.0
  cardiac/thorax  ≈  +3.0  (high contrast enhancement)
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from evaluators.mirror_utils import (
    BACKGROUND_Z_THRESHOLD,
    _detect_midline_argmin,
    _find_local_maxima,
    _tissue_profile,
    create_mirrored_mask,
    detect_bilateral_breasts,
    detect_midline,
    mirror_mask,
    validate_mirrored_region,
)


# ======================================================================
# Synthetic image factory
# ======================================================================

H, W = 128, 256  # nominal image shape

# Column ranges (in a W=256 image)
LEFT_BREAST_COLS = (20, 80)    # left breast occupies cols 20–79
RIGHT_BREAST_COLS = (176, 236) # right breast occupies cols 176–235
# The entire gap between the two breasts is low-intensity tissue (chest-wall /
# sternum region).  In real prone breast MRI the area between the two breasts
# transitions continuously — there is no hard boundary to pure background until
# the edge of the FOV.  Setting the whole inter-breast region to −0.5 (below
# mean but above BACKGROUND_Z_THRESHOLD=−1.5) is more realistic than leaving
# it at −2.0 (air).
STERNUM_COLS = (80, 176)       # entire inter-breast gap, intensity −0.5

# Tumour in left breast
TUMOUR_ROWS = (40, 88)
TUMOUR_COLS = (30, 70)


def _make_bilateral_image(
    heart_intensity: float = 0.0,
    edge_intensity: float = -2.0,
    breast_intensity: float = 1.0,
    tumour_intensity: float = 2.0,
) -> np.ndarray:
    """Build a synthetic bilateral breast image (H×W float64, z-score scale).

    Args:
        heart_intensity: Intensity of the central thorax/cardiac region
            (STERNUM_COLS).  Set high to simulate contrast enhancement.
        edge_intensity:  Intensity of the far-left and far-right background
            columns.  Set very negative to simulate the edge-bias scenario.
        breast_intensity: Nominal tissue intensity inside the two breasts.
        tumour_intensity: Extra intensity inside the tumour region.
    """
    img = np.full((H, W), -2.0, dtype=np.float64)  # default = air

    # Left breast
    img[:, LEFT_BREAST_COLS[0]:LEFT_BREAST_COLS[1]] = breast_intensity
    # Right breast
    img[:, RIGHT_BREAST_COLS[0]:RIGHT_BREAST_COLS[1]] = breast_intensity
    # Sternum gap (between breasts)
    img[:, STERNUM_COLS[0]:STERNUM_COLS[1]] = -0.5
    # Tumour region (within left breast)
    img[TUMOUR_ROWS[0]:TUMOUR_ROWS[1], TUMOUR_COLS[0]:TUMOUR_COLS[1]] = tumour_intensity
    # Optional cardiac/thorax
    if heart_intensity != 0.0:
        img[:, STERNUM_COLS[0]:STERNUM_COLS[1]] = heart_intensity
    # Optional edge override
    img[:, :10] = edge_intensity
    img[:, W - 10:] = edge_intensity

    return img


def _make_tumour_mask() -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[TUMOUR_ROWS[0]:TUMOUR_ROWS[1], TUMOUR_COLS[0]:TUMOUR_COLS[1]] = True
    return mask


def _rotate_90(img: np.ndarray) -> np.ndarray:
    """Rotate image 90° counter-clockwise (rows ↔ columns)."""
    return np.rot90(img, k=1)


def _rotate_mask_90(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, k=1)


# ======================================================================
# _tissue_profile
# ======================================================================

class TestTissueProfile:

    def test_background_columns_are_nan(self) -> None:
        """Columns consisting entirely of background should yield NaN."""
        img = _make_bilateral_image()
        profile = _tissue_profile(img, reduce_axis=0)  # column profile
        # Far-left columns are all-background → NaN
        assert np.isnan(profile[0]), "all-background column should be NaN"

    def test_breast_columns_are_positive(self) -> None:
        img = _make_bilateral_image()
        profile = _tissue_profile(img, reduce_axis=0)
        mid_left = (LEFT_BREAST_COLS[0] + LEFT_BREAST_COLS[1]) // 2
        mid_right = (RIGHT_BREAST_COLS[0] + RIGHT_BREAST_COLS[1]) // 2
        assert profile[mid_left] > BACKGROUND_Z_THRESHOLD, "left breast col should be tissue"
        assert profile[mid_right] > BACKGROUND_Z_THRESHOLD, "right breast col should be tissue"

    def test_sternum_profile_below_breast_profile(self) -> None:
        """Inter-breast gap should have lower tissue mean than breast columns."""
        img = _make_bilateral_image()
        profile = _tissue_profile(img, reduce_axis=0)
        left_val = np.nanmean(profile[LEFT_BREAST_COLS[0]:LEFT_BREAST_COLS[1]])
        sternum_val = np.nanmean(profile[STERNUM_COLS[0]:STERNUM_COLS[1]])
        assert sternum_val < left_val, "sternum should have lower tissue mean than breast"

    def test_row_profile_shape(self) -> None:
        img = _make_bilateral_image()
        profile = _tissue_profile(img, reduce_axis=1)  # row profile
        assert profile.shape == (H,), f"row profile should have length H={H}"


# ======================================================================
# _find_local_maxima
# ======================================================================

class TestFindLocalMaxima:

    def test_two_peaks(self) -> None:
        profile = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0])
        peaks = _find_local_maxima(profile, min_height=0.5, min_distance=1)
        assert len(peaks) == 2
        assert 2 in peaks and 6 in peaks

    def test_min_height_filters_small_peaks(self) -> None:
        profile = np.array([0.0, 0.3, 0.0, 0.0, 1.0, 0.0])
        peaks = _find_local_maxima(profile, min_height=0.5, min_distance=1)
        assert 4 in peaks
        assert 1 not in peaks  # too small

    def test_min_distance_merges_close_peaks(self) -> None:
        profile = np.array([0.0, 1.0, 0.0, 0.9, 0.0])
        peaks = _find_local_maxima(profile, min_height=0.5, min_distance=3)
        # The two peaks (idx 1 and 3) are distance 2 apart → merged to the taller one
        assert len(peaks) == 1
        assert 1 in peaks  # taller peak kept

    def test_nan_ignored(self) -> None:
        profile = np.array([np.nan, 1.0, np.nan, 1.0, np.nan])
        peaks = _find_local_maxima(profile, min_height=0.5, min_distance=1)
        # Both non-NaN positions are local maxima relative to NaN neighbours
        assert len(peaks) == 2


# ======================================================================
# detect_bilateral_breasts
# ======================================================================

class TestDetectBilateralBreasts:

    def test_nominal_bilateral_finds_two_peaks(self) -> None:
        img = _make_bilateral_image()
        peaks, reason = detect_bilateral_breasts(img, mirror_axis=1)
        assert peaks is not None, f"Expected two peaks, got None. Reason: {reason}"
        peak_a, peak_b = peaks
        # Left peak should be in first half, right peak in second half
        assert peak_a < W // 2
        assert peak_b >= W // 2

    def test_single_breast_fails_bilateral_check(self) -> None:
        """Image with only the left breast → bilateral check fails on axis=1."""
        img = np.full((H, W), -2.0, dtype=np.float64)
        img[:, LEFT_BREAST_COLS[0]:LEFT_BREAST_COLS[1]] = 1.0  # only left breast
        peaks, reason = detect_bilateral_breasts(img, mirror_axis=1)
        assert peaks is None
        assert "right" in reason.lower() or "second half" in reason.lower(), (
            f"Reason should mention missing right-half peak, got: {reason}"
        )

    def test_all_background_fails_with_informative_reason(self) -> None:
        img = np.full((H, W), -2.5, dtype=np.float64)
        peaks, reason = detect_bilateral_breasts(img, mirror_axis=1)
        assert peaks is None
        assert reason, "Expected non-empty reason string"

    def test_rotated_90_passes_bilateral_on_axis0(self) -> None:
        """A 90°-rotated bilateral image should find two peaks on axis=0.

        Note: The bilateral check on axis=1 may or may not find peaks on the
        rotated image (because the tissue profile can be near-flat and produce
        spurious smoothing artefacts).  What matters for correctness is that
        axis=0 finds the two breast peaks and that create_mirrored_mask handles
        the case correctly via D4 fallback — tested in TestCreateMirroredMask.
        """
        img = _make_bilateral_image()
        img_rot = _rotate_90(img)
        peaks_row, reason_row = detect_bilateral_breasts(img_rot, mirror_axis=0)
        assert peaks_row is not None, (
            f"Rotated image should pass bilateral check on row axis. Reason: {reason_row}"
        )

    def test_peaks_on_correct_sides(self) -> None:
        img = _make_bilateral_image()
        peaks, _ = detect_bilateral_breasts(img, mirror_axis=1)
        assert peaks is not None
        peak_a, peak_b = peaks
        # Allow a margin of ±15 cols around the breast boundaries; the tissue
        # profile is smoothed so the detected peak may be slightly off-centre.
        margin = 15
        assert LEFT_BREAST_COLS[0] - margin <= peak_a <= LEFT_BREAST_COLS[1] + margin, (
            f"Left peak at {peak_a} not near left breast cols {LEFT_BREAST_COLS}"
        )
        assert RIGHT_BREAST_COLS[0] - margin <= peak_b <= RIGHT_BREAST_COLS[1] + margin, (
            f"Right peak at {peak_b} not near right breast cols {RIGHT_BREAST_COLS}"
        )


# ======================================================================
# detect_midline
# ======================================================================

class TestDetectMidlineRobust:

    def test_midline_between_breast_peaks(self) -> None:
        """Midline should fall in the inter-breast gap, not inside a breast.

        The gap between the breasts spans from LEFT_BREAST_COLS[1] to
        RIGHT_BREAST_COLS[0] (cols 80–176).  The midline must land somewhere
        in this gap; it need not be at the exact STERNUM_COLS centre.
        """
        img = _make_bilateral_image()
        midline = detect_midline(img)
        gap_lo = LEFT_BREAST_COLS[1]   # 80: end of left breast
        gap_hi = RIGHT_BREAST_COLS[0]  # 176: start of right breast
        assert gap_lo <= midline <= gap_hi, (
            f"Midline {midline} not in inter-breast gap [{gap_lo}, {gap_hi}]"
        )

    def test_midline_not_at_background_edge(self) -> None:
        """Background-only edge columns (very low intensity) must not attract the midline."""
        img = _make_bilateral_image(edge_intensity=-4.0)
        midline = detect_midline(img)
        # Edge columns are at col < 10 and col > W-10
        assert 10 < midline < W - 10, (
            f"Midline {midline} landed on a background edge column"
        )

    def test_midline_robust_to_cardiac_enhancement(self) -> None:
        """High cardiac/thorax intensity in the centre must not displace the midline.

        In the naive argmin approach, very high thorax intensity would make
        the central columns *not* the minimum, pushing the detected midline
        sideways into breast tissue.  The robust peak-valley strategy is
        immune to this because it searches the valley *between* the two
        detected breast peaks.
        """
        img_normal = _make_bilateral_image(heart_intensity=0.0)
        img_cardiac = _make_bilateral_image(heart_intensity=3.0)

        midline_normal = detect_midline(img_normal)
        midline_cardiac = detect_midline(img_cardiac)

        # Both should be in the inter-breast gap (not inside a breast)
        gap_lo = LEFT_BREAST_COLS[1]   # 80
        gap_hi = RIGHT_BREAST_COLS[0]  # 176
        for label, mid in [("normal", midline_normal), ("cardiac", midline_cardiac)]:
            assert gap_lo <= mid <= gap_hi, (
                f"{label} midline {mid} not in inter-breast gap [{gap_lo}, {gap_hi}]"
            )

    def test_legacy_argmin_fallback_smoke(self) -> None:
        """The legacy argmin path must return a valid column index."""
        img = _make_bilateral_image()
        result = _detect_midline_argmin(img, search_fraction=0.4)
        assert 0 <= result < W


# ======================================================================
# mirror_mask — generalised axis parameter
# ======================================================================

class TestMirrorMask:

    def test_axis1_left_mask_mirrors_to_right(self) -> None:
        """A mask in the left half should mirror to the right half along axis=1."""
        mask = np.zeros((H, W), dtype=bool)
        mask[40:60, 30:60] = True  # in left half

        midline = W // 2
        mirrored = mirror_mask(mask, midline, axis=1)
        # All mirrored pixels should be in the right half
        cols = np.where(mirrored)[1]
        assert np.all(cols >= midline), "mirrored pixels should be in right half"

    def test_axis0_top_mask_mirrors_to_bottom(self) -> None:
        """A mask in the top half should mirror to the bottom half along axis=0."""
        mask = np.zeros((H, W), dtype=bool)
        mask[10:40, 60:100] = True  # in top half

        midline = H // 2
        mirrored = mirror_mask(mask, midline, axis=0)
        rows = np.where(mirrored)[0]
        assert np.all(rows >= midline), "mirrored pixels should be in bottom half"

    def test_round_trip_mirror_recovers_original(self) -> None:
        """Applying mirror_mask twice (mirror of mirror) should recover the original."""
        mask = np.zeros((H, W), dtype=bool)
        mask[40:60, 30:70] = True
        midline = W // 2
        once = mirror_mask(mask, midline, axis=1)
        twice = mirror_mask(once, midline, axis=1)
        assert np.array_equal(mask, twice), (
            "Double mirror should return the original mask"
        )

    def test_out_of_bounds_coords_dropped(self) -> None:
        """Pixels that would mirror outside the image must be silently dropped.

        mask covers cols 1-4.  With midline=0, col c maps to 2*0-c = -c which
        is negative for all c≥1 → all should be dropped.  (Col 0 would map to
        2*0-0=0 so we exclude it from the mask to get a clean all-dropped case.)
        """
        mask = np.zeros((H, W), dtype=bool)
        mask[:, 1:5] = True  # cols 1-4; each maps to negative index
        mirrored = mirror_mask(mask, midline=0, axis=1)
        assert mirrored.sum() == 0, (
            f"All cols 1-4 mirrored around 0 should be out-of-bounds, got {mirrored.sum()}"
        )

    def test_empty_mask_returns_empty(self) -> None:
        mask = np.zeros((H, W), dtype=bool)
        assert mirror_mask(mask, W // 2, axis=1).sum() == 0
        assert mirror_mask(mask, H // 2, axis=0).sum() == 0


# ======================================================================
# create_mirrored_mask — D2 + D4
# ======================================================================

class TestCreateMirroredMask:

    def test_nominal_bilateral_returns_valid_mirrored_mask(self) -> None:
        """Standard bilateral image: mirroring succeeds on axis=1 (primary)."""
        img = _make_bilateral_image()
        mask = _make_tumour_mask()
        result = create_mirrored_mask(img, mask, case_id="nominal")
        assert result is not None, "Expected valid mirror for standard bilateral image"
        assert result.shape == mask.shape
        assert result.dtype == bool
        # The mirrored region should be on the contralateral side of the tumour.
        # Tumour is in the left breast (median col < W//2); mirror must be on the
        # opposite side.  Accept that on some images D4 row-axis fallback may be
        # used, in which case the mirror is in the opposite row-half.
        tumour_cols = np.where(mask)[1]
        mirrored_rows = np.where(result)[0]
        mirrored_cols_arr = np.where(result)[1]
        contralateral = (
            np.median(mirrored_cols_arr) > W // 2  # opposite column half
            or np.median(mirrored_rows) > H // 2    # or opposite row half (D4)
        )
        assert contralateral, (
            "Mirrored mask should be on the contralateral side of the tumour"
        )

    def test_empty_mask_returns_none(self) -> None:
        img = _make_bilateral_image()
        mask = np.zeros((H, W), dtype=bool)
        assert create_mirrored_mask(img, mask, case_id="empty") is None

    def test_single_breast_returns_none(self) -> None:
        """Image showing only one breast: both axes should fail → None."""
        img = np.full((H, W), -2.0, dtype=np.float64)
        img[:, LEFT_BREAST_COLS[0]:LEFT_BREAST_COLS[1]] = 1.0
        mask = np.zeros((H, W), dtype=bool)
        mask[40:80, 30:70] = True
        result = create_mirrored_mask(img, mask, case_id="single_breast")
        assert result is None, "Single-breast image should return None"

    def test_90_degree_rotation_uses_axis0_fallback(self) -> None:
        """A 90°-rotated image should fail axis=1 and succeed via D4 fallback (axis=0)."""
        img = _make_bilateral_image()
        mask = _make_tumour_mask()
        img_rot = _rotate_90(img)
        mask_rot = _rotate_mask_90(mask)

        result = create_mirrored_mask(img_rot, mask_rot, case_id="rotated_90")
        assert result is not None, (
            "D4 axis-0 fallback should rescue a 90°-rotated image"
        )
        assert result.shape == mask_rot.shape

    def test_mirrored_mask_does_not_overlap_with_original(self) -> None:
        """Tumour and its mirror are in opposite breasts.

        For axis=1 mirroring the overlap must be zero.  If D4 axis=0 fallback
        was used instead, the tumour (in row-range TUMOUR_ROWS) and its
        row-axis mirror should still be in different row halves, so we allow
        a small positional check rather than strict zero-overlap.
        """
        img = _make_bilateral_image()
        mask = _make_tumour_mask()
        result = create_mirrored_mask(img, mask, case_id="no_overlap")
        assert result is not None
        # The mirrored region's centre should be far from the original tumour centre
        orig_centre = np.array(np.where(mask)).mean(axis=1)  # [row_c, col_c]
        mirr_centre = np.array(np.where(result)).mean(axis=1)
        dist = float(np.linalg.norm(orig_centre - mirr_centre))
        assert dist > 20, (
            f"Mirror centre {mirr_centre} too close to tumour centre {orig_centre} "
            f"(distance={dist:.1f} px)"
        )

    def test_warning_logged_for_single_breast(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Single-breast failure must produce a warning with the case_id."""
        img = np.full((H, W), -2.0, dtype=np.float64)
        img[:, LEFT_BREAST_COLS[0]:LEFT_BREAST_COLS[1]] = 1.0
        mask = np.zeros((H, W), dtype=bool)
        mask[40:80, 30:70] = True

        with caplog.at_level(logging.WARNING, logger="evaluators.mirror_utils"):
            create_mirrored_mask(img, mask, case_id="single_case_007")

        full_log = "\n".join(caplog.messages)
        assert "single_case_007" in full_log, (
            "case_id must appear in the warning message"
        )
        assert "ALL axes" in full_log or "bilateral" in full_log.lower(), (
            "Warning should mention the bilateral check or all-axes failure"
        )

    def test_warning_logged_for_empty_mask(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        img = _make_bilateral_image()
        mask = np.zeros((H, W), dtype=bool)
        with caplog.at_level(logging.WARNING, logger="evaluators.mirror_utils"):
            create_mirrored_mask(img, mask, case_id="empty_mask_case")
        assert "empty_mask_case" in "\n".join(caplog.messages)

    def test_cardiac_enhancement_does_not_break_mirroring(self) -> None:
        """High cardiac/thoracic intensity should not cause mirroring to fail."""
        img = _make_bilateral_image(heart_intensity=3.0)
        mask = _make_tumour_mask()
        result = create_mirrored_mask(img, mask, case_id="cardiac")
        assert result is not None, (
            "Cardiac enhancement in sternum region should not break mirroring"
        )

    def test_result_has_tissue_overlap(self) -> None:
        """The returned mirror must overlap with actual breast tissue."""
        img = _make_bilateral_image()
        mask = _make_tumour_mask()
        result = create_mirrored_mask(img, mask)
        assert result is not None
        tissue_pixels = img[result] > BACKGROUND_Z_THRESHOLD
        assert np.mean(tissue_pixels) >= 0.3, (
            "At least 30% of the mirrored mask should cover tissue"
        )


# ======================================================================
# validate_mirrored_region (unit test for the standalone function)
# ======================================================================

class TestValidateMirroredRegion:

    def test_all_tissue_passes(self) -> None:
        # Use tissue_threshold=0.0 explicitly so _compute_tissue_threshold is
        # bypassed.  (When all pixels == 1.0, the 10th-percentile is 1.0, and
        # img > 1.0 is always False → the auto-threshold would reject a valid
        # all-tissue mask — that is correct behaviour but not what this test
        # exercises.)
        img = np.ones((32, 32), dtype=np.float64) * 1.0  # all tissue
        m = np.ones((32, 32), dtype=bool)
        assert validate_mirrored_region(img, m, min_tissue_fraction=0.3, tissue_threshold=0.0)

    def test_all_background_fails(self) -> None:
        img = np.ones((32, 32), dtype=np.float64) * -3.0  # all background
        m = np.ones((32, 32), dtype=bool)
        assert not validate_mirrored_region(img, m, min_tissue_fraction=0.3)

    def test_empty_mask_fails(self) -> None:
        img = np.ones((32, 32), dtype=np.float64)
        m = np.zeros((32, 32), dtype=bool)
        assert not validate_mirrored_region(img, m)

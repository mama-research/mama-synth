"""Contralateral mirroring utilities for breast DCE-MRI.

Provides midline detection and mask mirroring for the ``tumor_roi``
classification task.  Given a tumor segmentation mask on one breast,
mirror it across the body midline to create a contralateral ROI that
serves as a "non-tumor" reference region for binary classification
(tumor ROI vs contralateral non-tumor ROI).

**Midline detection** works by computing the column-wise mean intensity
of the breast tissue and finding the valley (local minimum) in the
central region of the image — this valley corresponds to the gap
between the two breasts in axial breast MRI slices.

**Validation** ensures that the mirrored mask actually overlaps with
breast tissue (i.e. is not out-of-body or in the background).

Ported from ``mama-synth-eval/src/eval/mirror_utils.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_MIN_TISSUE_FRACTION = 0.3
"""Minimum fraction of mirrored mask voxels that must overlap with
tissue for the mirrored region to be considered valid."""

DEFAULT_TISSUE_THRESHOLD_PERCENTILE = 10
"""Percentile of non-zero image intensities used to determine the
tissue/background boundary."""

DEFAULT_MIDLINE_SEARCH_FRACTION = 0.4
"""Fraction of the image width (centered) within which to search
for the midline valley."""

BACKGROUND_Z_THRESHOLD: float = -1.5
"""Intensity threshold separating background from breast tissue in
z-score normalised images.  Background (air) consistently falls below
−2σ; glandular tissue starts at approximately −1.5σ and above.
Replaces the naïve ``image > 0`` filter which erroneously excluded the
∼40–50% of tissue with below-mean z-scores."""


def detect_midline(
    image: NDArray[np.floating],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
) -> int:
    """Detect the body midline in an axial breast MRI slice.

    Computes the column-wise mean intensity and finds the column with
    the minimum average intensity in the central region of the image.
    In breast MRI, this intensity valley corresponds to the gap between
    the two breasts (sternum/midline region).

    Parameters
    ----------
    image : NDArray
        2-D or 3-D image array.
    search_fraction : float
        Fraction of the image width (centered) to search in.

    Returns
    -------
    int
        Column index of the detected midline.
    """
    if not 0 < search_fraction <= 1:
        raise ValueError(
            f"search_fraction must be in (0, 1], got {search_fraction}"
        )

    if image.ndim == 3:
        col_profile = np.mean(image, axis=(0, 1))
    elif image.ndim == 2:
        col_profile = np.mean(image, axis=0)
    else:
        raise ValueError(f"image must be 2D or 3D, got {image.ndim}D")

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
    midline_col = lo + int(np.argmin(window))

    return midline_col


def mirror_mask(
    mask: NDArray[np.bool_],
    midline_col: int,
) -> NDArray[np.bool_]:
    """Mirror a binary mask horizontally about a midline column.

    Each True voxel at column ``c`` is mapped to column
    ``2 * midline_col - c``.  Voxels that map outside the image
    boundaries are silently clipped.
    """
    n_cols = mask.shape[-1]
    mirrored = np.zeros_like(mask, dtype=bool)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return mirrored

    col_idx = -1  # last axis
    mirrored_cols = 2 * midline_col - coords[:, col_idx]

    valid = (mirrored_cols >= 0) & (mirrored_cols < n_cols)
    coords_valid = coords[valid].copy()
    coords_valid[:, col_idx] = mirrored_cols[valid]

    if mask.ndim == 2:
        mirrored[coords_valid[:, 0], coords_valid[:, 1]] = True
    elif mask.ndim == 3:
        mirrored[
            coords_valid[:, 0],
            coords_valid[:, 1],
            coords_valid[:, 2],
        ] = True
    else:
        raise ValueError(f"mask must be 2D or 3D, got {mask.ndim}D")

    return mirrored


def _compute_tissue_threshold(
    image: NDArray[np.floating],
    percentile: float = DEFAULT_TISSUE_THRESHOLD_PERCENTILE,
) -> float:
    """Tissue/background threshold from a low percentile of tissue voxels.

    Filters to pixels above ``BACKGROUND_Z_THRESHOLD`` (valid for
    z-score normalised images where background air < −2σ) and returns
    the given percentile as a conservative lower bound on tissue signal.
    """
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

    tissue_overlap = int(np.sum(image[mirrored_mask] > tissue_threshold))
    fraction = tissue_overlap / n_mask_voxels

    return fraction >= min_tissue_fraction


def create_mirrored_mask(
    image: NDArray[np.floating],
    mask: NDArray[np.bool_],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
) -> Optional[NDArray[np.bool_]]:
    """Create a contralateral mirrored mask with validation.

    Detects the midline, mirrors the mask, then validates that the
    mirrored region falls on breast tissue.

    Returns
    -------
    NDArray[bool] | None
        The mirrored mask if validation passes, or ``None`` if it fails.
    """
    if not np.any(mask):
        logger.warning("Input mask is empty, cannot create mirror.")
        return None

    midline = detect_midline(image, search_fraction=search_fraction)
    mirrored = mirror_mask(mask, midline)

    if not np.any(mirrored):
        logger.warning(
            "Mirrored mask is empty (midline=%d). "
            "Mask may be too close to the image boundary.",
            midline,
        )
        return None

    if validate_mirrored_region(
        image, mirrored, min_tissue_fraction=min_tissue_fraction
    ):
        return mirrored

    logger.warning(
        "Mirrored mask failed tissue validation "
        "(midline=%d, min_tissue_fraction=%.2f).",
        midline, min_tissue_fraction,
    )
    return None

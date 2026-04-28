"""Segmentation metrics: Dice coefficient and Hausdorff distance (HD95).

A callable ``segment_fn`` must be provided at init time.  It receives
a 2-D ``float64`` image (z-score normalised) and returns a binary
``bool`` mask of the same shape.

When no ``segment_fn`` is given the evaluator returns empty results.

HD95 is computed via ``scipy.ndimage.distance_transform_edt`` with
optional ``voxel_spacing`` support — aligned with ``mama-synth-eval``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import ndimage

from .base import BaseEvaluator, Case, EvaluationResult
from tqdm import tqdm

class SegmentationEvaluator(BaseEvaluator):
    """Dice and 95th-percentile Hausdorff between predicted and GT masks."""

    def __init__(
        self,
        segment_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.segment_fn = segment_fn

    # ------------------------------------------------------------------

    def evaluate(self, cases: list[Case]) -> EvaluationResult:
        if self.segment_fn is None:
            return EvaluationResult()

        per_case: dict[str, dict[str, float]] = {}
        for case in tqdm(cases):
            if case.mask is None or not np.any(case.mask):
                continue
            try:
                pred_mask = self.segment_fn(case.prediction)
                if pred_mask is None or not np.any(pred_mask):
                    continue
                pred_mask = pred_mask.astype(bool)
                gt_mask = case.mask.astype(bool)

                dice = compute_dice(pred_mask, gt_mask)
                hd95 = compute_hausdorff_95(pred_mask, gt_mask)
                per_case[case.case_id] = {
                    "dice": dice,
                    "hausdorff_95": hd95,
                }
            except Exception:
                continue

        agg: dict[str, dict[str, float]] = {}
        dice_agg = self._aggregate_metric(per_case, "dice")
        if dice_agg:
            agg["dice"] = dice_agg
        hd_agg = self._aggregate_metric(per_case, "hausdorff_95")
        if hd_agg:
            agg["hausdorff_95"] = hd_agg

        return EvaluationResult(per_case=per_case, aggregates=agg)


# ======================================================================
# Metric implementations
# ======================================================================


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sørensen–Dice coefficient between two binary masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = int(np.logical_and(pred_b, gt_b).sum())
    total = int(pred_b.sum()) + int(gt_b.sum())
    if total == 0:
        return 1.0  # both empty → perfect agreement
    return float(2.0 * intersection / total)


def compute_hausdorff_95(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Optional[tuple[float, ...]] = None,
) -> float:
    """95th-percentile Hausdorff distance between mask surfaces.

    Uses ``scipy.ndimage.distance_transform_edt`` for physically
    correct distances (when ``voxel_spacing`` is provided).  This
    aligns with the ``mama-synth-eval`` implementation.

    Returns 0.0 when both masks are empty, ``inf`` when exactly one
    mask is empty.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    pred_sum = int(np.sum(pred_b))
    gt_sum = int(np.sum(gt_b))

    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return float("inf")

    if voxel_spacing is None:
        voxel_spacing = tuple(1.0 for _ in range(pred_b.ndim))

    # Surface extraction: border = mask XOR eroded mask
    pred_border = pred_b ^ ndimage.binary_erosion(pred_b)
    gt_border = gt_b ^ ndimage.binary_erosion(gt_b)

    # Single-voxel masks: erosion yields empty → use mask itself
    if not np.any(pred_border):
        pred_border = pred_b
    if not np.any(gt_border):
        gt_border = gt_b

    # EDT from every voxel to nearest foreground voxel
    dt_pred = ndimage.distance_transform_edt(
        ~pred_b, sampling=voxel_spacing
    )
    dt_gt = ndimage.distance_transform_edt(
        ~gt_b, sampling=voxel_spacing
    )

    # Surface distances in both directions
    surf_dist_pred_to_gt = dt_gt[pred_border]
    surf_dist_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate(
        [surf_dist_pred_to_gt, surf_dist_gt_to_pred]
    )
    return float(np.percentile(all_distances, 95))

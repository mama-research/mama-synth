"""Image-to-image metrics: MSE and LPIPS.

Both metrics operate on the full image (no masking).

* **MSE** is computed directly on z-score normalised images.
* **LPIPS** clips inputs to ``±5σ`` and linearly maps to ``[-1, 1]``
  before feeding through the perceptual network.  This accounts for
  post-contrast intensities falling in the long tail of the pre-contrast
  z-score distribution.

LPIPS backend: ``torchmetrics`` only (the ``lpips`` pip package is
not used — see requirements.txt).
The model is cached at **module level** to avoid OOM in long sessions.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from .base import BaseEvaluator, Case, EvaluationResult

logger = logging.getLogger(__name__)

# ---- Module-level LPIPS model cache ----------------------------------
_LPIPS_MODEL_CACHE: dict[str, object] = {}

# Clipping range for z-score → [-1, 1] mapping for LPIPS.
# 5σ accommodates contrast-enhanced intensities far from the pre-contrast
# reference distribution.
LPIPS_CLIP_SIGMA: float = 5.0


def _get_lpips_model(net: str = "alex"):
    """Return a cached LPIPS model and its backend name (``torchmetrics``).

    Only ``torchmetrics`` is supported.  The legacy ``lpips`` pip package
    is intentionally *not* used here — it is not listed in requirements.txt
    and has known maintenance / reproducibility issues.
    """
    if net in _LPIPS_MODEL_CACHE:
        model = _LPIPS_MODEL_CACHE[net]
        return model, "torchmetrics"

    from torchmetrics.image.lpip import (
        LearnedPerceptualImagePatchSimilarity,
    )

    model = LearnedPerceptualImagePatchSimilarity(net_type=net)
    model.eval()
    _LPIPS_MODEL_CACHE[net] = model
    return model, "torchmetrics"


class ImageMetricsEvaluator(BaseEvaluator):
    """Per-case MSE and LPIPS between prediction and ground truth."""

    def __init__(self) -> None:
        self._lpips_available = False
        try:
            _get_lpips_model("alex")
            self._lpips_available = True
        except Exception:
            logger.warning(
                "LPIPS model unavailable (torch + torchmetrics required). "
                "LPIPS metric will be absent from results. "
                "Ensure torch and torchmetrics are installed in your container."
            )

    # ------------------------------------------------------------------

    def evaluate(self, cases: list[Case]) -> EvaluationResult:
        per_case: dict[str, dict[str, float]] = {}

        for case in cases:
            metrics: dict[str, float] = {}
            pred, gt = case.prediction, case.ground_truth

            # ---- Shape guard -----------------------------------------
            if pred.shape != gt.shape:
                logger.warning(
                    "%s — skipping ImageMetrics: prediction shape %s does not "
                    "match GT shape %s. Ensure your output image has the same "
                    "spatial dimensions as the ground-truth slice (H×W).",
                    case.case_id, pred.shape, gt.shape,
                )
                continue

            # ---- Intensity range sanity check ------------------------
            pred_absmax = float(np.max(np.abs(pred)))
            if pred_absmax > 500:
                logger.warning(
                    "%s — extreme prediction intensities (|max|=%.1f). "
                    "Metrics will be computed but scores are unreliable. "
                    "Predictions must be z-score normalised (expected |max|<100).",
                    case.case_id, pred_absmax,
                )

            metrics["mse"] = float(np.mean((pred - gt) ** 2))

            if self._lpips_available:
                lpips_val = self._compute_lpips(pred, gt)
                if lpips_val is not None:
                    metrics["lpips"] = lpips_val

            logger.debug("%s  mse=%.4f", case.case_id, metrics["mse"])
            per_case[case.case_id] = metrics

        agg: dict[str, dict[str, float]] = {}
        agg["mse"] = self._aggregate_metric(per_case, "mse")
        lpips_agg = self._aggregate_metric(per_case, "lpips")
        if lpips_agg:
            agg["lpips"] = lpips_agg

        return EvaluationResult(per_case=per_case, aggregates=agg)

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_lpips(
        pred: np.ndarray, gt: np.ndarray
    ) -> float | None:
        """Compute LPIPS on z-score normalised images.

        The images are clipped to ``±LPIPS_CLIP_SIGMA`` then linearly
        mapped to ``[-1, 1]`` so that both pred and GT undergo the
        **same deterministic** transform — no per-image min-max that
        would introduce artificial differences.
        """
        try:
            import torch

            model, backend = _get_lpips_model("alex")

            def _normalize_zscore(img: np.ndarray) -> np.ndarray:
                clipped = np.clip(img, -LPIPS_CLIP_SIGMA, LPIPS_CLIP_SIGMA)
                return clipped / LPIPS_CLIP_SIGMA  # → [-1, 1]

            p_norm = _normalize_zscore(pred)
            g_norm = _normalize_zscore(gt)

            # Handle 2-D (H, W) → slice list, or 3-D (S, H, W)
            if pred.ndim == 2:
                slices_p, slices_g = [p_norm], [g_norm]
            elif pred.ndim == 3:
                slices_p = [p_norm[i] for i in range(p_norm.shape[0])]
                slices_g = [g_norm[i] for i in range(g_norm.shape[0])]
            else:
                return None

            lpips_values: list[float] = []
            with torch.no_grad():
                for sp, sg in zip(slices_p, slices_g):
                    # LPIPS expects (N, 3, H, W)
                    tp = (
                        torch.from_numpy(sp)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(-1, 3, -1, -1)
                    )
                    tg = (
                        torch.from_numpy(sg)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(-1, 3, -1, -1)
                    )

                    # torchmetrics is the only supported backend
                    model.reset()  # type: ignore[union-attr]
                    model.update(tp, tg)  # type: ignore[union-attr]
                    val = model.compute()  # type: ignore[union-attr]

                    lpips_values.append(float(val.item()))

            return float(np.mean(lpips_values))
        except Exception:
            return None

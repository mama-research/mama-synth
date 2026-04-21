"""Base classes for MAMA-SYNTH Grand Challenge evaluation.

Every evaluator inherits from :class:`BaseEvaluator` and
implements :meth:`evaluate`, which receives **all** cases at
once so that aggregate-only metrics (FRD, AUROC) can be
computed alongside per-case metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Case:
    """A single evaluation case with all required image arrays.

    All images are 2-D ``float64`` arrays.  They are expected to arrive
    **z-score normalised** using the same reference statistics (computed
    from pre-contrast ground truth during dataset preparation).  No
    additional per-image normalisation is applied by the pipeline.

    Masks are boolean arrays of the same spatial shape.

    Optional ``*_path`` fields carry the on-disk file paths so that
    the ``frd-score`` library (which operates on file paths) can be
    called directly without writing temporary files.
    """

    case_id: str
    prediction: NDArray[np.float64]
    ground_truth: NDArray[np.float64]
    mask: Optional[NDArray[np.bool_]] = None
    precontrast: Optional[NDArray[np.float64]] = None
    # File paths for frd-score library
    prediction_path: Optional[str] = None
    ground_truth_path: Optional[str] = None
    mask_path: Optional[str] = None


@dataclass
class EvaluationResult:
    """Container returned by every evaluator.

    ``per_case``  – ``{case_id: {metric: value, …}, …}``
    ``aggregates`` – ``{metric: {"mean": …, "std": …}, …}``
    """

    per_case: dict[str, dict[str, float]] = field(default_factory=dict)
    aggregates: dict[str, dict[str, float]] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base for all metric evaluators."""

    @abstractmethod
    def evaluate(self, cases: list[Case]) -> EvaluationResult:
        """Evaluate all cases and return per-case + aggregate metrics."""
        ...

    # ---- helpers available to sub-classes ---------------------------------

    @staticmethod
    def _aggregate_metric(
        per_case: dict[str, dict[str, float]],
        metric_key: str,
    ) -> dict[str, float]:
        """Compute ``{"mean": …, "std": …}`` for *metric_key* across cases."""
        vals = [m[metric_key] for m in per_case.values() if metric_key in m]
        if not vals:
            return {}
        arr = np.array(vals, dtype=np.float64)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

"""Classification metrics: AUROC for contrast and tumour-ROI tasks.

Both metrics are **aggregate-only** (no meaningful per-case result).

* **AUROC contrast** – how well a pre-trained classifier can
  distinguish synthetic post-contrast images from real pre-contrast
  images based on radiomic features.
* **AUROC tumour-ROI** – how well a pre-trained classifier can
  distinguish the tumour region from a contralateral mirrored region
  (detected via anatomical midline) in the synthetic image.

Supports three classifier types:
  * ``RadiomicsClassifier`` – scikit-learn model on radiomic features
  * ``CNNClassifier`` – EfficientNet-based on raw images
  * ``EnsembleClassifier`` – arithmetic mean of multiple models

Aligned with ``mama-synth-eval/src/eval/classification.py``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np
from sklearn.metrics import roc_auc_score

from .base import BaseEvaluator, Case, EvaluationResult
from .mirror_utils import create_mirrored_mask
from .roi_metrics import extract_radiomic_features_cached


# ======================================================================
# Classifier protocol
# ======================================================================


class Classifier(Protocol):
    """Protocol for models compatible with evaluation."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


# ======================================================================
# RadiomicsClassifier
# ======================================================================


class RadiomicsClassifier:
    """Scikit-learn classifier operating on radiomic features.

    Loads a pre-trained ``.pkl`` model with ``predict_proba``.
    """

    def __init__(
        self,
        task: str = "contrast",
        model: Optional[Any] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        valid_tasks = {"tnbc", "luminal", "contrast", "tumor_roi"}
        if task not in valid_tasks:
            raise ValueError(
                f"task must be one of {valid_tasks}, got '{task}'"
            )
        self.task = task

        if model_path is not None:
            with open(Path(model_path), "rb") as f:
                self.model = pickle.load(f)  # noqa: S301
        elif model is not None:
            self.model = model
        else:
            self.model = self._create_default_model()

    @staticmethod
    def _create_default_model() -> Any:
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=100,
                max_depth=5,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
        except Exception:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42,
            )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.predict_proba(features)
        return proba[:, 1] if proba.ndim == 2 else proba

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)


# ======================================================================
# CNNClassifier
# ======================================================================


class CNNClassifier:
    """EfficientNet-based classifier from a ``.pt`` checkpoint."""

    def __init__(
        self,
        task: str = "contrast",
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        valid_tasks = {"tnbc", "luminal", "contrast", "tumor_roi"}
        if task not in valid_tasks:
            raise ValueError(
                f"task must be one of {valid_tasks}, got '{task}'"
            )
        self.task = task
        if model_path is None:
            raise ValueError("CNNClassifier requires a model_path.")
        self._load_model(Path(model_path))

    def _load_model(self, path: Path) -> None:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        self.image_size = config.get("image_size", 224)
        self.model_name = config.get("model_name", "efficientnet_b0")
        self.use_mask_channel = config.get("use_mask_channel", False)
        self.in_chans = config.get("in_chans", 3)

        import timm

        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=1,
            in_chans=self.in_chans,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_proba_from_images(
        self,
        images: list[np.ndarray],
        masks: Optional[list[Optional[np.ndarray]]] = None,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.model.eval()
        device = next(self.model.parameters()).device
        probabilities: list[float] = []

        for idx, img in enumerate(images):
            mask = (
                masks[idx]
                if masks is not None and idx < len(masks)
                else None
            )
            slice_2d = (
                img if img.ndim == 2 else img[img.shape[0] // 2]
            )
            tensor = self._preprocess_slice(slice_2d)
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logit = self.model(tensor).squeeze()
                prob = torch.sigmoid(logit).item()
            probabilities.append(prob)

        return np.array(probabilities, dtype=np.float64)

    def _preprocess_slice(self, slice_2d: np.ndarray):
        import torch
        import torch.nn.functional as F

        # Input is z-score normalised; pass through without rescaling
        # so that the classifier operates on the same intensity scale
        # as during training.
        img = slice_2d.astype(np.float32)

        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tensor = tensor.repeat(3, 1, 1)  # (3, H, W)
        return tensor


# ======================================================================
# EnsembleClassifier
# ======================================================================


class EnsembleClassifier:
    """Ensemble that averages probabilities from multiple models."""

    def __init__(self, task: str) -> None:
        valid_tasks = {"tnbc", "luminal", "contrast", "tumor_roi"}
        if task not in valid_tasks:
            raise ValueError(
                f"task must be one of {valid_tasks}, got '{task}'"
            )
        self.task = task
        self._radiomics_models: list[RadiomicsClassifier] = []
        self._cnn_models: list[CNNClassifier] = []

    def add_radiomics_model(
        self, clf: RadiomicsClassifier
    ) -> "EnsembleClassifier":
        self._radiomics_models.append(clf)
        return self

    def add_cnn_model(
        self, clf: CNNClassifier
    ) -> "EnsembleClassifier":
        self._cnn_models.append(clf)
        return self

    @property
    def n_models(self) -> int:
        return len(self._radiomics_models) + len(self._cnn_models)

    @property
    def has_radiomics(self) -> bool:
        return len(self._radiomics_models) > 0

    @property
    def has_cnn(self) -> bool:
        return len(self._cnn_models) > 0

    def predict_proba(
        self,
        features: Optional[np.ndarray] = None,
        images: Optional[list[np.ndarray]] = None,
        masks: Optional[list[Optional[np.ndarray]]] = None,
    ) -> np.ndarray:
        all_probs: list[np.ndarray] = []

        for clf in self._radiomics_models:
            if features is None:
                raise ValueError(
                    "Ensemble contains radiomics models but no "
                    "features provided."
                )
            all_probs.append(np.asarray(clf.predict_proba(features)))

        for clf in self._cnn_models:
            if images is None:
                raise ValueError(
                    "Ensemble contains CNN models but no images provided."
                )
            all_probs.append(
                np.asarray(clf.predict_proba_from_images(images, masks))
            )

        if not all_probs:
            raise ValueError("Ensemble has no models.")

        stacked = np.stack(all_probs, axis=0)
        return np.mean(stacked, axis=0)

    @staticmethod
    def discover_models(
        task: str,
        model_dir: Path,
    ) -> "EnsembleClassifier":
        """Auto-discover model files for *task* in *model_dir*."""
        ensemble = EnsembleClassifier(task=task)
        model_dir = Path(model_dir)

        for pkl_path in sorted(model_dir.glob(f"{task}_classifier*.pkl")):
            try:
                clf = RadiomicsClassifier(
                    task=task, model_path=pkl_path
                )
                ensemble.add_radiomics_model(clf)
            except Exception as e:
                print(
                    f"WARNING: Failed to load {pkl_path.name}: {e}",
                    file=sys.stderr,
                )

        for pt_path in sorted(model_dir.glob(f"{task}_classifier*.pt")):
            try:
                clf = CNNClassifier(task=task, model_path=pt_path)
                ensemble.add_cnn_model(clf)
            except (ImportError, Exception) as e:
                print(
                    f"WARNING: Failed to load {pt_path.name}: {e}",
                    file=sys.stderr,
                )

        return ensemble


# ======================================================================
# ClassificationEvaluator
# ======================================================================


class ClassificationEvaluator(BaseEvaluator):
    """AUROC for contrast and tumour-ROI classification.

    Supports single models (``.pkl``) and ensemble mode (auto-discovers
    all ``{task}_classifier*.pkl`` / ``*.pt`` in ``models_dir``).
    """

    def __init__(
        self,
        contrast_model: Optional[Path] = None,
        tumor_roi_model: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        ensemble: bool = False,
    ) -> None:
        self.ensemble = ensemble
        self.models_dir = models_dir

        # -- Contrast classifier(s) -----------------------------------
        self.contrast_clf: Optional[
            Union[RadiomicsClassifier, EnsembleClassifier]
        ] = None
        if ensemble and models_dir is not None:
            ens = EnsembleClassifier.discover_models(
                "contrast", models_dir
            )
            if ens.n_models > 0:
                self.contrast_clf = ens
        elif contrast_model is not None and Path(contrast_model).exists():
            self.contrast_clf = RadiomicsClassifier(
                task="contrast", model_path=contrast_model
            )

        # -- Tumor ROI classifier(s) ----------------------------------
        self.tumor_roi_clf: Optional[
            Union[RadiomicsClassifier, EnsembleClassifier]
        ] = None
        if ensemble and models_dir is not None:
            ens = EnsembleClassifier.discover_models(
                "tumor_roi", models_dir
            )
            if ens.n_models > 0:
                self.tumor_roi_clf = ens
        elif tumor_roi_model is not None and Path(tumor_roi_model).exists():
            self.tumor_roi_clf = RadiomicsClassifier(
                task="tumor_roi", model_path=tumor_roi_model
            )

    # ------------------------------------------------------------------

    def evaluate(self, cases: list[Case]) -> EvaluationResult:
        agg: dict[str, dict[str, float]] = {}

        if self.contrast_clf is not None:
            auroc = self._auroc_contrast(cases)
            if auroc is not None:
                # AUROC is an aggregate-only scalar; no std across cases.
                agg["auroc_contrast"] = {"mean": auroc}

        if self.tumor_roi_clf is not None:
            auroc = self._auroc_tumor_roi(cases)
            if auroc is not None:
                # AUROC is an aggregate-only scalar; no std across cases.
                agg["auroc_tumor_roi"] = {"mean": auroc}

        return EvaluationResult(per_case={}, aggregates=agg)

    # ---- contrast ----------------------------------------------------

    def _auroc_contrast(
        self, cases: list[Case]
    ) -> Optional[float]:
        """AUROC: synthetic-post (label 1) vs real-precontrast (label 0).

        Radiomic features are extracted from the **tumour ROI** defined by
        ``case.mask``, matching the region used during classifier training
        (see ``create_contrast_dataset`` in ``mama-synth-eval``).  When no
        mask is available for a case, the whole image is used as a fallback
        and a warning is emitted — absent masks degrade AUROC reliability.

        Returns ``None`` if fewer than 4 feature vectors are available
        (minimum required for a meaningful binary AUROC: at least one
        sample per class with one extra each to avoid degenerate ROC).
        """
        feats: list[np.ndarray] = []
        labels: list[int] = []

        for case in cases:
            if case.precontrast is None:
                continue
            # Use the tumour mask if available — this is the ROI the
            # classifier was trained on.  Fall back to whole image with a
            # warning so that cases without masks are not silently dropped.
            if case.mask is not None and np.any(case.mask):
                roi_mask = case.mask
            else:
                roi_mask = np.ones(case.prediction.shape, dtype=bool)
                print(
                    f"WARNING: no tumour mask for {case.case_id}; "
                    "using whole-image features for contrast AUROC. "
                    "Results may be unreliable.",
                    file=sys.stderr,
                )
            try:
                sf = extract_radiomic_features_cached(
                    case.prediction, roi_mask
                )
                pf = extract_radiomic_features_cached(
                    case.precontrast, roi_mask
                )
                if sf.size == 0 or pf.size == 0:
                    continue
                if sf.shape != pf.shape:
                    continue
                feats.extend([sf, pf])
                labels.extend([1, 0])
            except Exception as exc:
                print(
                    f"WARNING: contrast feature extraction failed "
                    f"for {case.case_id}: {exc}",
                    file=sys.stderr,
                )

        if len(feats) < 4:
            return None

        X = np.nan_to_num(
            np.vstack(feats), nan=0.0, posinf=0.0, neginf=0.0
        )
        y = np.array(labels, dtype=np.int64)
        try:
            if isinstance(self.contrast_clf, EnsembleClassifier):
                probs = self.contrast_clf.predict_proba(features=X)
            else:
                probs = self.contrast_clf.predict_proba(X)  # type: ignore[union-attr]
            return float(roc_auc_score(y, probs))
        except Exception as exc:
            print(
                f"WARNING: contrast AUROC failed: {exc}",
                file=sys.stderr,
            )
            return None

    # ---- tumour ROI --------------------------------------------------

    def _auroc_tumor_roi(
        self, cases: list[Case]
    ) -> Optional[float]:
        """AUROC: tumour ROI (label 1) vs mirrored contralateral (label 0).

        Uses anatomical midline detection to create a physiologically
        meaningful contralateral region.

        Returns ``None`` if fewer than 4 feature vectors are available
        (minimum required for a meaningful binary AUROC: at least one
        sample per class with one extra each to avoid degenerate ROC).
        """
        feats: list[np.ndarray] = []
        labels: list[int] = []

        for case in cases:
            if case.mask is None or not np.any(case.mask):
                continue

            # Anatomical midline mirroring
            mirrored = create_mirrored_mask(
                case.prediction, case.mask
            )
            if mirrored is None:
                continue
            if np.array_equal(case.mask, mirrored):
                continue

            try:
                tf = extract_radiomic_features_cached(
                    case.prediction, case.mask
                )
                mf = extract_radiomic_features_cached(
                    case.prediction, mirrored
                )
                if tf.size == 0 or mf.size == 0:
                    continue
                if tf.shape != mf.shape:
                    continue
                feats.extend([tf, mf])
                labels.extend([1, 0])
            except Exception as exc:
                print(
                    f"WARNING: tumour-ROI feature extraction failed "
                    f"for {case.case_id}: {exc}",
                    file=sys.stderr,
                )

        if len(feats) < 4:
            return None

        X = np.nan_to_num(
            np.vstack(feats), nan=0.0, posinf=0.0, neginf=0.0
        )
        y = np.array(labels, dtype=np.int64)
        try:
            if isinstance(self.tumor_roi_clf, EnsembleClassifier):
                probs = self.tumor_roi_clf.predict_proba(features=X)
            else:
                probs = self.tumor_roi_clf.predict_proba(X)  # type: ignore[union-attr]
            return float(roc_auc_score(y, probs))
        except Exception as exc:
            print(
                f"WARNING: tumour-ROI AUROC failed: {exc}",
                file=sys.stderr,
            )
            return None

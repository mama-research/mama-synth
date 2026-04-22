
# MAMA-SYNTH 2026

Synthesizing Virtual Contrast-Enhancement in Breast MRI

MAMA-SYNTH is a challenge focused on synthesizing virtual post-contrast breast MRI from pre-contrast T1-weighted MRI. The benchmark is designed to support the development of clinically meaningful contrast-reduced and contrast-free breast MRI workflows.

🔗 Official website: MAMA-SYNTH Challenge

Dynamic contrast-enhanced MRI (DCE-MRI) plays a central role in breast cancer diagnosis, treatment planning, and disease monitoring. However, the use of gadolinium-based contrast agents introduces important concerns related to patient safety, environmental contamination, and accessibility of advanced imaging workflows. MAMA-SYNTH provides a standardized evaluation framework for generative models that aim to recover diagnostically relevant post-contrast information from non-contrast acquisitions.

This challenge is intended to support fair and open comparison of synthesis methods across institutions, imaging settings, and downstream clinical objectives.

---

## Project Structure & Main Files

- **src/evaluation/evaluate.py**: Main evaluation pipeline. Loads predictions and ground truth, runs all evaluators (image, ROI, classification, segmentation), and writes metrics. Integrates nnUNetv2 for segmentation inference with robust log/warning suppression.
- **src/preprocessing/compute_dataset_stats.py**: Computes global mean and standard deviation for pre-contrast normalization using Welford's algorithm. Outputs a JSON file for reproducible normalization.
- **src/preprocessing/preprocess.py**: Preprocessing pipeline. Extracts 2D pre-contrast and peak-enhancement slices, applies z-score normalization, saves MHA/PNG outputs, and generates a CSV report.
- **src/evaluation/models/**: Contains pre-trained classifier models and (optionally) nnUNet segmentation weights.


## Getting Started

1. **Preprocessing**: Use `preprocess.py` to extract and normalize 2D slices from 3D DCE-MRI volumes.
2. **Evaluation**: Run `evaluate.py` to score predictions against ground truth using the standardized metrics.

See the `src/evaluation/README.md` for detailed usage, directory layout, and metric descriptions.

---

# MAMA-SYNTH Grand Challenge Evaluation Container

Minimal, modular evaluation pipeline for the MAMA-SYNTH challenge on
[Grand Challenge](https://grand-challenge.org/).  Evaluates synthetic
post-contrast breast DCE-MRI slices against real ground truth across
four metric groups.

## Metric Groups

| Group | Metric | Scope | Description |
|-------|--------|-------|-------------|
| **Image-to-Image** | MSE | per-case | Mean Squared Error |
| | LPIPS | per-case | Learned Perceptual Image Patch Similarity |
| **ROI-to-ROI** | SSIM (tumour) | per-case | Structural Similarity within the tumour mask |
| | FRD | aggregate | Fréchet Radiomics Distance (mask-region features) |
| **Classification** | AUROC contrast | aggregate | Pre-vs-post contrast classifier AUROC |
| | AUROC tumour-ROI | aggregate | Tumour ROI vs mirrored-ROI classifier AUROC |
| **Segmentation** | Dice | per-case | Sørensen–Dice coefficient |
| | HD95 | per-case | 95th-percentile Hausdorff distance |

## Directory Layout (Grand Challenge)

```
/input/
    predictions.json                        ← algorithm job manifest
    {job_pk}/output/images/{slug}/*.mha     ← one synthetic slice per job

/opt/ml/input/data/ground_truth/
    images/          ← real post-contrast 2-D .mha slices
    masks/           ← binary tumour segmentation masks .mha
    precontrast/     ← real pre-contrast 2-D .mha slices

/opt/app/models/
    contrast_classifier.pkl                 ← pre-trained contrast CLF
    tumor_roi_classifier.pkl                ← pre-trained tumour-ROI CLF
    segmentation/                           ← (optional) nnUNet weights

/output/
    metrics.json                            ← evaluation output
```

## Local Development

Set environment variables to override GC paths:

```bash
export MAMA_PREDICTIONS_DIR=/path/to/predictions
export MAMA_GT_DIR=/path/to/ground_truth          # contains images/, masks/, precontrast/
export MAMA_MASKS_DIR=/path/to/masks               # or auto-detected from GT_DIR/masks
export MAMA_PRECONTRAST_DIR=/path/to/precontrast   # or auto-detected from GT_DIR/precontrast
export MAMA_OUTPUT_DIR=/path/to/output
export MAMA_MODELS_DIR=/path/to/models

python evaluate.py
```

## Docker

```bash
./do_build.sh          # Build the container
./do_test_run.sh       # Run against test/ data
./do_save.sh           # Export .tar.gz for GC upload
```

## Testing

```bash
python3 -m pytest tests/ -v
```

## Architecture

```
evaluate.py              ← GC entry point + case discovery
evaluators/
    base.py              ← Case, EvaluationResult, BaseEvaluator ABC
    image_metrics.py     ← ImageMetricsEvaluator   (MSE, LPIPS)
    roi_metrics.py       ← ROIMetricsEvaluator      (SSIM-ROI, FRD)
    classification.py    ← ClassificationEvaluator   (AUROC × 2)
    segmentation.py      ← SegmentationEvaluator     (Dice, HD95)
    mirror_utils.py      ← midline detection + contralateral mask mirroring
models/                  ← bundled classifier .pkl files
ground_truth/            ← GT data for Docker test runs
```

## GC Leaderboard JSON Paths

Configure in **Admin → Phase → Scoring**:

| Metric | Score jsonpath | Error jsonpath |
|--------|---------------|----------------|
| MSE | `aggregates.mse.mean` | `aggregates.mse.std` |
| LPIPS | `aggregates.lpips.mean` | `aggregates.lpips.std` |
| SSIM (tumour) | `aggregates.ssim_tumor.mean` | `aggregates.ssim_tumor.std` |
| FRD | `aggregates.frd.mean` | — |
| AUROC contrast | `aggregates.auroc_contrast.mean` | — |
| AUROC tumour-ROI | `aggregates.auroc_tumor_roi.mean` | — |
| Dice | `aggregates.dice.mean` | `aggregates.dice.std` |
| HD95 | `aggregates.hausdorff_95.mean` | `aggregates.hausdorff_95.std` |

For more information, visit the official challenge website or consult the code documentation in each module.

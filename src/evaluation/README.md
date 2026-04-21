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

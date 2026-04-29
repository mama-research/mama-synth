
# 🏥 MAMA-SYNTH 2026

Synthesizing Virtual Contrast-Enhancement in Breast MRI 🎗️

MAMA-SYNTH is a challenge focused on synthesizing virtual post-contrast breast MRI from pre-contrast T1-weighted MRI. The benchmark is designed to support the development of clinically meaningful contrast-reduced and contrast-free breast MRI workflows.

Dynamic contrast-enhanced MRI (DCE-MRI) plays a central role in breast cancer diagnosis, treatment planning, and disease monitoring. However, the use of gadolinium-based contrast agents introduces important concerns related to patient safety, environmental contamination, and accessibility of advanced imaging workflows. MAMA-SYNTH provides a standardized evaluation framework for generative models that aim to recover diagnostically relevant post-contrast information from non-contrast acquisition.

### 🔗 Visit our [Website](https://www.ub.edu/mama-synth/) for more information and 📢 participate on [Grand Challenge](https://mamasynth.grand-challenge.org/).

---

## 📁 Repository Structure

```
mama-synth/
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py               ← 3D DCE-MRI → 2D slice extraction & normalisation
│   │   ├── compute_dataset_stats.py    ← Global pre-contrast normalisation statistics
│   │   └── training_pre_stats.json     ← Pre-computed dataset stats (mean, std)
│   └── evaluation/
│       ├── evaluate.py                 ← GC entry point + case discovery + pipeline orchestrator
│       ├── evaluators/
│       │   ├── base.py                 ← Case, EvaluationResult, BaseEvaluator ABC
│       │   ├── image_metrics.py        ← ImageMetricsEvaluator   (MSE, LPIPS)
│       │   ├── roi_metrics.py          ← ROIMetricsEvaluator      (SSIM-ROI, FRD)
│       │   ├── classification.py       ← ClassificationEvaluator  (AUROC × 2)
│       │   ├── segmentation.py         ← SegmentationEvaluator    (Dice, HD95)
│       │   └── mirror_utils.py         ← Midline detection + contralateral mask mirroring
│       ├── models/                     ← Bundled classifier .pkl files & nnUNet weights
│       ├── ground_truth/               ← GT data for Docker test runs
└── README.md                           ← this file
```

---
## ⚡ Quick Start

Clone the repository and install the required libraries:

```
git clone https://github.com/mama-research/mama-synth.git

pip install -r requirements.txt
```

### 1️⃣ Preprocessing

Convert 3D DCE-MRI volumes to 2D pre-contrast and peak-enhancement slices ready for the challenge.

**Step 1: Compute dataset-level normalisation statistics** (once, on training data):

This steps has already been performed and the normalization statistics can be found `/src/preprocessing/training_pre_stats.json`. To reproduce the statistics on the MAMA-MIA training dataset:

```bash
python src/preprocessing/compute_dataset_stats.py \
    --image_dir /path/to/3d_images \
    --output_path src/preprocessing/training_pre_stats.json
```

**Step 2: Preprocess images**:

```bash
python src/preprocessing/preprocess.py \
    --image_dir /path/to/3d_images \
    --seg_dir   /path/to/segmentations \
    --output_dir /path/to/output \
    --global_stats src/preprocessing/training_pre_stats.json
```

Output layout:

```
output/
    mha/
        input/          ← pre-contrast 2-D .mha slices
        ground_truth/   ← peak-enhancement 2-D .mha slices
        mask/           ← tumour segmentation 2-D .mha slices
    png/                ← visualisation (same structure)
    intensity_plots/    ← per-patient intensity comparison plots
    report.csv          ← per-patient preprocessing report
```

#### ⚠️Note: PNGs are only for visualization and are not normalized as required by the challenge evaluation.

### 2️⃣ Evaluation

Score synthetic predictions against ground truth.

**💻 Local evaluation** 

**Step 1: Add evaluation models**:

The pretrained classification and segmentation models need to be added to `/src/evaluation/models` to run the complete evaluation. **[Download Weights](https://drive.google.com/file/d/1rliFnr-mNtISkJA0etm1dDBLdbHB3d4h/view) ⬇️**. 

**Step 2: Set environment variables and run**:
```bash
export MAMA_PREDICTIONS_DIR=/path/to/predictions
export MAMA_PRECONTRAST_DIR=/path/to/input 
export MAMA_GT_DIR=/path/to/ground_truth   
export MAMA_MASKS_DIR=/path/to/mask
export MAMA_MODELS_DIR=/path/to/models
export MAMA_OUTPUT_DIR=/path/to/output
export MAMA_ENSEMBLE=True

python src/evaluation/evaluate.py
```

Note: The segmentation model is a single-fold [2D nnU-Net](https://github.com/mic-dkfz/nnunet) while the classification model is an ensemble of radiomics-based classifiers with training script available [here](https://github.com/RichardObi/MAMA-SYNTH-codebase/tree/main/mama-synth/mama-synth-eval#classifier-training). Both are trained on extracted tumor-containing DCE-MRI slices from the [mama-mia dataset](synapse.org/#!Synapse:syn60868042).

---

## 🔬 Preprocessing in Detail

`preprocess.py` implements the full 3D → 2D pipeline:

1. **Phase discovery** — identifies all DCE phases per patient from filename suffixes (`patient_id_<phase_index>.nii.gz`).
2. **Peak enhancement selection** — selects the phase with the highest mean tumour intensity in 3D.
3. **Slice selection** — picks the 2D slice with the largest tumour area along the shortest axis.
4. **Z-score normalisation** — applies dataset-level statistics (from `compute_dataset_stats.py`) to both the pre-contrast and peak-enhancement slices for reproducible, bias-free normalisation.
5. **Output** — saves `.mha` and `.png` files and an intensity comparison plot per patient.

**Expected input layout:**

```
image_dir/
    <patient_id>/
        <patient_id>_0.nii.gz   ← pre-contrast (lowest index)
        <patient_id>_1.nii.gz
        ...
segmentation_dir/
    <patient_id>.nii.gz
```

**Key arguments:**

| Argument | Description |
|---|---|
| `--image_dir` | Root directory with per-patient phase volumes |
| `--seg_dir` | Directory with per-patient segmentation files |
| `--output_dir` | Root output directory |
| `--global_stats` | Path to JSON with `mean` and `std` for z-score normalisation |
| `--csv_name` | Output CSV report filename (default: `report.csv`) |

---

## 📊 Evaluation in Detail

`evaluate.py` is the main evaluation entry point. It loads cases, runs four evaluators, and writes a `metrics.json`.

### 📈 Metric Groups

| Group | Metric | Scope | Description |
|---|---|---|---|
| **Image-to-Image** | MSE | per-case | Mean Squared Error |
| | LPIPS | per-case | Learned Perceptual Image Patch Similarity |
| **ROI-to-ROI** | SSIM (tumour) | per-case | Structural Similarity within the tumour mask |
| | FRD | aggregate | Fréchet Radiomics Distance (mask-region features) |
| **Classification** | AUROC contrast | aggregate | Pre-vs-post contrast classifier AUROC |
| | AUROC tumour-ROI | aggregate | Tumour ROI vs mirrored-ROI classifier AUROC |
| **Segmentation** | Dice | per-case | Sørensen–Dice coefficient |
| | HD95 | per-case | 95th-percentile Hausdorff distance |


Note that both classifiers use radiologist-verified **tumour segmentation mask** to extract radiomic features from the region of interest rather than full image features.

## 🙏 Acknowledgements

This challenge is supported by the [AIMED](https://www.bcn-aim.org/aimed/) and [FUTURE-ES](https://www.bcn-aim.org/future-es/) projects (Ministry of Science, Spain) and the European Union's Horizon Europe programme ([RadioVal](https://radioval.eu/) and [ODELIA](https://odelia.ai/)).

![Acknowledgements](./assets/acknowledgements_miccai.jpg)


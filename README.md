
# 🏥 MAMA-SYNTH 2026

Synthesizing Virtual Contrast-Enhancement in Breast MRI 🎗️

MAMA-SYNTH is a challenge focused on synthesizing virtual post-contrast breast MRI from pre-contrast T1-weighted MRI. The benchmark is designed to support the development of clinically meaningful contrast-reduced and contrast-free breast MRI workflows.

Dynamic contrast-enhanced MRI (DCE-MRI) plays a central role in breast cancer diagnosis, treatment planning, and disease monitoring. However, the use of gadolinium-based contrast agents introduces important concerns related to patient safety, environmental contamination, and accessibility of advanced imaging workflows. MAMA-SYNTH provides a standardized evaluation framework for generative models that aim to recover diagnostically relevant post-contrast information from non-contrast acquisition.

### 🔗 Visit our [Website](https://mama-research.github.io/mama-synth.html) for more information and 📢 participate on [Grand Challenge](https://mamasynth.grand-challenge.org/).

---

## 📁 Repository Structure

```
mama-synth/
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py               ← 3D DCE-MRI → 2D slice extraction & normalisation
│   │   ├── compute_dataset_stats.py    ← Global pre-contrast normalisation statistics
│   │   └── training_pre_stats.json     ← Pre-computed dataset stats (mean, std)
│   ├── evaluation/
│   │   ├── evaluate.py                 ← GC entry point + case discovery + pipeline orchestrator
│   │   ├── evaluators/
│   │   │   ├── base.py                 ← Case, EvaluationResult, BaseEvaluator ABC
│   │   │   ├── image_metrics.py        ← ImageMetricsEvaluator   (MSE, LPIPS)
│   │   │   ├── roi_metrics.py          ← ROIMetricsEvaluator      (SSIM-ROI, FRD)
│   │   │   ├── classification.py       ← ClassificationEvaluator  (AUROC × 2)
│   │   │   ├── segmentation.py         ← SegmentationEvaluator    (Dice, HD95)
│   │   │   └── mirror_utils.py         ← Midline detection + contralateral mask mirroring
│   │   ├── models/                     ← Bundled classifier .pkl files & nnUNet weights
│   │   └── ground_truth/               ← GT data for Docker test runs
│   └── submission/
│       ├── identity-baseline/          ← Minimal pass-through baseline (smoke-test + template)
│       └── submission-gan/             ← Pix2PixHD GAN baseline (medigan model 00023)
└── README.md                           ← this file
```

---

## ⚡ Quick Start

### Prerequisites

- Python ≥ 3.10
- **Git** must be available on your PATH (required to install `pyradiomics` directly from GitHub)
- For GPU-accelerated inference: CUDA-compatible GPU + NVIDIA drivers

Clone the repository and install the required libraries:

```bash
git clone https://github.com/mama-research/mama-synth.git
cd mama-synth
pip install -r requirements.txt
```

> **Note on `pyradiomics`** — the PyPI release is broken for Python ≥ 3.10 ([issue #903](https://github.com/AIM-Harvard/pyradiomics/issues/903)). The `requirements.txt` installs directly from the GitHub `master` branch via PEP 508 direct-reference syntax, which requires `git` to be on your PATH. On Windows, a C++ build toolchain is also needed (see [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)).

---

### 1️⃣ Preprocessing

Convert 3D DCE-MRI volumes to 2D pre-contrast and peak-enhancement slices ready for the challenge.

**Step 1: Compute dataset-level normalisation statistics** (once, on training data):

This step has already been performed and the normalisation statistics can be found in `src/preprocessing/training_pre_stats.json`. To reproduce the statistics on the MAMA-MIA training dataset:

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

> **⚠️ Note:** PNGs are for visualisation only and are **not** normalised as required by the challenge evaluation. Always use the `.mha` files for model training and submission.

---

### 2️⃣ Evaluation

Score synthetic predictions against ground truth.

**💻 Local evaluation**

**Step 1: Add evaluation models**:

The pretrained classification and segmentation models must be placed in `src/evaluation/models/` to run the complete evaluation. **[Download Weights](https://drive.google.com/file/d/1rliFnr-mNtISkJA0etm1dDBLdbHB3d4h/view) ⬇️**

Expected structure after download:

```
src/evaluation/models/
    classification/
        contrast_classifier.pkl
        tumor_roi_classifier.pkl
    segmentation/       ← nnUNet model folder (fold_0/, plans.json, …)
```

> If the models directory is absent or incomplete, the classification and segmentation evaluators are automatically disabled. Image-level metrics (MSE, LPIPS) and ROI metrics (SSIM-tumour, FRD) will still run.

**Step 2: Set environment variables and run**:

```bash
export MAMA_PREDICTIONS_DIR=/path/to/predictions   # flat dir of .mha predictions
export MAMA_PRECONTRAST_DIR=/path/to/input         # pre-contrast .mha slices (optional)
export MAMA_GT_DIR=/path/to/ground_truth           # ground-truth post-contrast slices
export MAMA_MASKS_DIR=/path/to/mask                # tumour mask .mha slices
export MAMA_MODELS_DIR=/path/to/models             # classification + segmentation weights
export MAMA_OUTPUT_DIR=/path/to/output             # where metrics.json is written
export MAMA_ENSEMBLE=True                          # enable classifier ensemble (recommended)

python src/evaluation/evaluate.py
```

Full list of supported environment variable overrides:

| Variable | Default | Description |
|---|---|---|
| `MAMA_INPUT_DIR` | `/input` | GC input directory (predictions in GC mode) |
| `MAMA_OUTPUT_DIR` | `/output` | Directory where `metrics.json` is written |
| `MAMA_GT_DIR` | `/opt/ml/input/data/ground_truth` | Ground-truth root (contains `ground_truth/` and `masks/` sub-folders) |
| `MAMA_MODELS_DIR` | `/opt/app/models` | Classification + segmentation model weights |
| `MAMA_PREDICTIONS_DIR` | — | Flat directory of `.mha` predictions (local mode only) |
| `MAMA_MASKS_DIR` | — | Flat directory of `.mha` masks (local mode override) |
| `MAMA_PRECONTRAST_DIR` | — | Flat directory of pre-contrast `.mha` slices (optional, local mode) |
| `MAMA_ENSEMBLE` | `""` | Set to `True` / `1` / `yes` to enable classifier ensemble |
| `MAMA_SEG_FOLD` | `0` | nnU-Net fold to use for the segmentation model |

Note: The segmentation model is a single-fold [2D nnU-Net](https://github.com/mic-dkfz/nnunet) while the classification model is an ensemble of radiomics-based classifiers with training scripts available [here](https://github.com/RichardObi/MAMA-SYNTH-codebase/tree/main/mama-synth/mama-synth-eval#classifier-training). Both are trained on extracted tumour-containing DCE-MRI slices from the [MAMA-MIA dataset](https://www.synapse.org/#!Synapse:syn60868042).

---

### 3️⃣ Submission Baselines

Two ready-to-containerise baselines are provided under `src/submission/`. Both follow the Grand Challenge Docker I/O contract:

| Direction | Container path | GC interface slug |
|-----------|----------------|-------------------|
| **Input** | `/input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha` | `pre-contrast-dce-mri-slice-breast` |
| **Output** | `/output/images/synthetic-contrast-dce-mri-slice-breast/output.mha` | `synthetic-contrast-dce-mri-slice-breast` |

#### Identity Baseline (`src/submission/identity-baseline/`)

Copies the pre-contrast input directly to the output without any synthesis. Use this to:
- Verify the end-to-end GC infrastructure before submitting a real model.
- Establish a lower-bound reference on the leaderboard.
- Use as a copy-paste template for building a real submission.

```bash
cd src/submission/identity-baseline/
./do_build.sh        # build Docker image
./do_test_run.sh     # run locally against test/input/
./do_save.sh         # export .tar.gz for GC upload
```

See [`src/submission/identity-baseline/README.md`](src/submission/identity-baseline/README.md) for full details.

#### Pix2PixHD GAN Baseline (`src/submission/submission-gan/`)

Uses the **Pix2PixHD** model (medigan model `00023`) to synthesise post-contrast slices from pre-contrast inputs. Requires NVIDIA GPU support and the pre-trained model weights (`30_net_G.pth`).

```bash
cd src/submission/submission-gan/
export MODEL_WEIGHTS_DIR=/path/to/00023   # directory containing 30_net_G.pth
./do_build.sh                             # stages weights + builds Docker image
./do_test_run.sh                          # GPU inference
USE_GPU=0 ./do_test_run.sh               # CPU-only inference
./do_save.sh                              # export .tar.gz for GC upload
```

See [`src/submission/submission-gan/README.md`](src/submission/submission-gan/README.md) for full details, including the normalisation strategy used to bridge z-score normalised MHA inputs with the GAN's 8-bit PNG training distribution.

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

Note that both classifiers use the radiologist-verified **tumour segmentation mask** to extract radiomic features from the region of interest rather than full-image features.

### 📄 Output format

Results are written to `metrics.json` with two top-level keys:

```json
{
  "case": {
    "<case_id>": { "mse": ..., "lpips": ..., "ssim_tumour": ..., "dice": ..., "hd95": ... }
  },
  "aggregates": {
    "frd": ...,
    "auroc_contrast": ...,
    "auroc_tumour_roi": ...,
    "mse": { "mean": ..., "std": ... }
  }
}
```

---

## 🙏 Acknowledgements

This challenge is supported by the [AIMED](https://www.bcn-aim.org/aimed/) and [FUTURE-ES](https://www.bcn-aim.org/future-es/) projects (Ministry of Science, Spain) and the European Union's Horizon Europe programme ([RadioVal](https://radioval.eu/) and [ODELIA](https://odelia.ai/)).

![Acknowledgements](./assets/acknowledgements_miccai.jpg)

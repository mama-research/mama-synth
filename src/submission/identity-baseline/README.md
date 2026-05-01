# MAMA-SYNTH — Identity Baseline Submission

This directory contains a **minimal, submission-ready algorithm** for the
[MAMA-SYNTH Grand Challenge](https://mamasynth.grand-challenge.org/).

It is an **identity baseline**: the pre-contrast input image is copied
directly to the output without any synthesis. Its purpose is to:

- Verify the end-to-end GC infrastructure before submitting a real model.
- Provide a lower-bound reference on the leaderboard.
- Serve as a copy-paste template for building a real submission.

---

## Directory structure

```
identity-baseline/
├── inference.py          ← algorithm entry point (edit this for real models)
├── Dockerfile
├── requirements.txt
├── do_build.sh           ← build the Docker image
├── do_test_run.sh        ← run locally against test/input/
├── do_save.sh            ← export .tar.gz for GC upload
├── test_algorithm.py     ← automated pytest validation
└── test/
    └── input/
        └── images/
            └── pre-contrast-breast-mri/
                └── README.txt   ← drop a .mha file here before testing
```

---

## Grand Challenge I/O contract

| Direction | Container path | GC interface slug |
|-----------|---------------|-------------------|
| **Input** | `/input/images/pre-contrast-breast-mri/<uuid>.mha` | `pre-contrast-breast-mri` |
| **Output** | `/output/images/synthetic-post-contrast-breast-mri/output.mha` | `synthetic-post-contrast-breast-mri` |

Images are 2-D z-score-normalised `float32` `.mha` files produced by the
MAMA-SYNTH preprocessing pipeline.

---

## Local test

```bash
# 1. Make scripts executable (first time only)
chmod +x do_build.sh do_test_run.sh do_save.sh

# 2. Drop a pre-contrast .mha slice in the test input folder
cp /path/to/your/patient_001.mha \
   test/input/images/pre-contrast-breast-mri/

# 3. Build and run
./do_test_run.sh

# 4. Inspect the output
ls test/output/images/synthetic-post-contrast-breast-mri/
# → output.mha

# 5. (Optional) run the automated tests
pip install pytest SimpleITK
pytest test_algorithm.py -v
```

---

## GC upload

```bash
# Bump VERSION in do_save.sh, then:
./do_save.sh
# → mama-synth-identity-baseline-v1.0.0.tar.gz

# Upload on GC:
# Algorithm page → Container Management → Upload a new container
```

---

## Developing a real model

Replace the identity copy in `inference.py` with your synthesis model:

```python
# inference.py — replace this block:
output_image = image  # identity copy

# with your model inference, e.g.:
output_array = my_model.predict(sitk.GetArrayFromImage(image))
output_image = sitk.GetImageFromArray(output_array.astype("float32"))
output_image.CopyInformation(image)   # preserve spacing / origin / direction
```

Add your dependencies to `requirements.txt` and model weights to a
`resources/` subfolder (then `COPY` them in the Dockerfile).

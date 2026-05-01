FROM --platform=linux/amd64 docker.io/library/python:3.11-slim

# Prevents Python stdout/stderr buffering (important for GC logs)
ENV PYTHONUNBUFFERED=1

# Create a non-root user (Grand Challenge requirement)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# System dependencies:
#   gcc/g++         — required to compile pyradiomics C extensions
#   git             — required because pyradiomics must be installed from GitHub
#                     (PyPI release is broken for Python >= 3.10, see issue #903)
#   libgl1          — required by OpenCV / nnUNet (libGL.so.1)
#   libglib2.0-0    — required by OpenCV (libgthread-2.0.so.0)
#   libgomp1        — required by nnUNet / scikit-learn for OpenMP parallelism
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=user:user requirements.txt /opt/app/

# Install CPU-only PyTorch first (keeps image ~1.5 GB smaller)
RUN python -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir \
    --requirement /opt/app/requirements.txt

# Copy evaluation code — build context is the repository root
COPY --chown=user:user src/evaluation/evaluators/ /opt/app/evaluators/
COPY --chown=user:user src/evaluation/evaluate.py /opt/app/

# Copy bundled models (classifiers + nnUNet segmentation weights).
# models/ already contains a .gitkeep so this COPY succeeds even when
# only the directory skeleton is present.
COPY --chown=user:user src/evaluation/models/ /opt/app/models/

# Prepare output directory
RUN mkdir -p /output && chown user:user /output

USER user

# GC evaluation containers limit max workers with this env var.
# Set to 4 as a safe default; the GC platform may override this at runtime.
ENV GRAND_CHALLENGE_MAX_WORKERS=4

ENTRYPOINT ["python", "evaluate.py"]

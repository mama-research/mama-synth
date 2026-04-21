FROM --platform=linux/amd64 docker.io/library/python:3.11-slim

# Prevents Python stdout/stderr buffering (important for GC logs)
ENV PYTHONUNBUFFERED=1

# Create a non-root user (Grand Challenge requirement)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# System dependencies:
#   gcc/g++  — required to compile pyradiomics C extensions
#   git      — required because pyradiomics must be installed from GitHub
#              (PyPI release is broken for Python >= 3.10, see issue #903)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=user:user requirements.txt /opt/app/

# Install CPU-only PyTorch first (keeps image ~1.5 GB smaller)
RUN python -m pip install --user --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --user --no-cache-dir \
    --requirement /opt/app/requirements.txt

# Copy evaluation code
COPY --chown=user:user evaluators/ /opt/app/evaluators/
COPY --chown=user:user evaluate.py /opt/app/

# Copy bundled models (classifiers, optional segmentation weights)
COPY --chown=user:user models/ /opt/app/models/

# Prepare output directory
RUN mkdir -p /output && chown user:user /output

USER user

# GC evaluation containers limit max workers with this env var
ENV GRAND_CHALLENGE_MAX_WORKERS=

ENTRYPOINT ["python", "evaluate.py"]

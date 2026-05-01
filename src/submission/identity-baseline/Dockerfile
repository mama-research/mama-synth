FROM --platform=linux/amd64 docker.io/library/python:3.11-slim

# Prevents Python stdout/stderr buffering (important for GC logs)
ENV PYTHONUNBUFFERED=1

# Create a non-root user (Grand Challenge requirement)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# System dependencies:
#   libgl1 / libglib2.0-0 — required by SimpleITK on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install --no-cache-dir \
    --requirement /opt/app/requirements.txt

# Copy algorithm code
COPY --chown=user:user inference.py /opt/app/

# Prepare output directory (GC expects /output to be writable by the user)
RUN mkdir -p /output && chown user:user /output

USER user

ENTRYPOINT ["python", "inference.py"]

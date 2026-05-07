
# Stop at first error
echo "Starting build of the Docker image. This can take a while."
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="mamasynth-gc-eval-v1.1.0"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR" 2>&1
  #--no-cache \
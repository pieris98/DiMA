#!/usr/bin/env bash
# Run a command (or an interactive shell) inside the DiMA container.
# Mounts host model caches and project volumes so weights are shared.
#
# Usage:
#   ./docker/run.sh                                   # interactive shell
#   ./docker/run.sh python example_simple.py          # unconditional generation
#   ./docker/run.sh python auto-scripts/setup_models.py --models esm2 saprot
#   ./docker/run.sh python auto-scripts/prepare_data.py --dataset afdb
#   ./docker/run.sh python auto-scripts/run_inference.py
#   ./docker/run.sh bash                              # plain bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=${DIMA_IMAGE:-dima:latest}

# Ensure cache dirs exist on the host so Docker doesn't create them as root
mkdir -p "$HOME/.cache/huggingface"
mkdir -p "$REPO_ROOT/checkpoints"
mkdir -p "$REPO_ROOT/data"

# Build the docker run command
DOCKER_ARGS=(
    --rm
    --runtime=nvidia
    --gpus all
    --ipc host           # needed for multi-GPU shared memory
    --ulimit memlock=-1  # allow locked memory for GPU
    -v "$HOME/.cache/huggingface:/workspace/DiMA/.cache/huggingface"
    -v "$REPO_ROOT/checkpoints:/workspace/DiMA/checkpoints"
    -v "$REPO_ROOT/data:/workspace/DiMA/data"
    -v "$REPO_ROOT/src:/workspace/DiMA/src"
    -v "$REPO_ROOT/auto-scripts:/workspace/DiMA/auto-scripts"
    -v "$REPO_ROOT/example_simple.py:/workspace/DiMA/example_simple.py"
    -e PROJECT_ROOT=/workspace/DiMA
    -e WANDB_MODE=offline
    -e HF_HOME=/workspace/DiMA/.cache/huggingface
    -e TRANSFORMERS_CACHE=/workspace/DiMA/.cache/huggingface/hub
    -w /workspace/DiMA
)

# Pass WANDB_API_KEY from host if set
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    DOCKER_ARGS+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
fi

# Interactive if no arguments given
if [[ $# -eq 0 ]]; then
    DOCKER_ARGS+=(-it)
    set -- bash
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE" "$@"

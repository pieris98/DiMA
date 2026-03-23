#!/bin/bash
# =============================================================================
# DiMA Conditional Training - Run Only
# 
# This script runs conditional training with ESM-3B on SwissProt.
# Assumes all setup (data, statistics, environment) is already done.
#
# Usage:
#   chmod +x run_conditional_training.sh
#   ./run_conditional_training.sh
#
# Arguments:
#   --iters N        Set number of training iterations
#   --project NAME   Set WandB project name
# =============================================================================

set -e

# Configuration - adjust these paths as needed
PROJECT_ROOT="/home/cherry/dev/phd/DiMA"
DATA_DIR="${PROJECT_ROOT}/data"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

# Training config - ESM-3B on SwissProt
ENCODER="esm2"
DATASET="swissprot"
WANDB_PROJECT="dima-conditional-esm3b"
NUM_ITERS=100000
EVAL_INTERVAL=10000
SAVE_INTERVAL=10000
BATCH_SIZE=256

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iters)
            NUM_ITERS="$2"
            shift 2
            ;;
        --project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--iters N] [--project NAME]"
            exit 1
            ;;
    esac
done

log_info "=========================================="
log_info "DiMA Conditional Training"
log_info "=========================================="
log_info "Encoder: ESM-3B"
log_info "Dataset: SwissProt"
log_info "Iterations: ${NUM_ITERS}"
log_info "WandB: ${WANDB_PROJECT}"
log_info "=========================================="

# Set wandb offline mode
export WANDB_MODE=offline

cd "${PROJECT_ROOT}"

pixi run -e dima-env python train_diffusion.py \
    --config-name config \
    encoder=${ENCODER} \
    datasets=${DATASET} \
    datasets.data_dir=${DATA_DIR}/${DATASET} \
    datasets.length_distribution=${DATA_DIR}/distributions/${DATASET}.npy \
    project.path=${PROJECT_ROOT} \
    project.checkpoints_folder=${CHECKPOINT_DIR} \
    project.statistics_folder=${CHECKPOINT_DIR}/statistics \
    project.wandb_project=${WANDB_PROJECT} \
    training.training_iters=${NUM_ITERS} \
    training.eval_interval=${EVAL_INTERVAL} \
    training.save_interval=${SAVE_INTERVAL} \
    training.batch_size=${BATCH_SIZE} \
    ddp.enabled=true

log_info "Training completed!"

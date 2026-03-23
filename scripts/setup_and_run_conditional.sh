#!/bin/bash
# =============================================================================
# DiMA Conditional Training Setup & Run Script
# 
# This script:
# 1. Clones the DiMA repository
# 2. Installs pixi and the dima environment
# 3. Downloads datasets (SwissProt)
# 4. Prepares encoder statistics
# 5. Runs conditional training with ESM-3B on SwissProt
#
# Usage:
#   chmod +x setup_and_run_conditional.sh
#   ./setup_and_run_conditional.sh
#
# Arguments:
#   --skip-setup     Skip setup, run training only
#   --setup-only     Run setup only, skip training
#   --iters N        Set number of training iterations
#   --project NAME   Set WandB project name
# =============================================================================

set -e  # Exit on error

# Configuration
REPO_URL="https://github.com/your-org/dima.git"  # TODO: Update with actual DiMA repo URL
REPO_DIR="${HOME}/projects/dima"
DATA_DIR="${REPO_DIR}/data"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints"
PIXI_PATH="${HOME}/.pixi/bin/pixi"

# Training configuration
ENCODER="esm2"           # ESM-3B (2560 dim)
DATASET="swissprot"
WANDB_PROJECT="dima-conditional-esm3b"
NUM_ITERS=100000
EVAL_INTERVAL=10000
SAVE_INTERVAL=10000
BATCH_SIZE=256

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# -----------------------------------------------------------------------------
# Step 1: Install pixi if not present
# -----------------------------------------------------------------------------
install_pixi() {
    log_info "Checking for pixi..."
    
    if command -v pixi &> /dev/null; then
        log_info "pixi is already installed: $(which pixi)"
        return 0
    fi
    
    log_info "Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | sh
    export PATH="${HOME}/.pixi/bin:$PATH"
    log_info "pixi installed successfully"
}

# -----------------------------------------------------------------------------
# Step 2: Clone or update DiMA repository
# -----------------------------------------------------------------------------
setup_repo() {
    log_info "Setting up DiMA repository..."
    
    if [ -d "${REPO_DIR}" ]; then
        log_info "Repository already exists, pulling latest changes..."
        cd "${REPO_DIR}"
        git pull
    else
        log_info "Cloning DiMA repository..."
        mkdir -p "$(dirname ${REPO_DIR})"
        git clone "${REPO_URL}" "${REPO_DIR}"
        cd "${REPO_DIR}"
    fi
    
    log_info "Repository ready at: ${REPO_DIR}"
}

# -----------------------------------------------------------------------------
# Step 3: Install pixi environment
# -----------------------------------------------------------------------------
install_env() {
    log_info "Installing pixi dima environment..."
    cd "${REPO_DIR}/.."
    pixi install --environment dima-env
    log_info "Environment installed successfully"
}

# -----------------------------------------------------------------------------
# Step 4: Install git dependencies
# -----------------------------------------------------------------------------
install_git_deps() {
    log_info "Installing git dependencies (ESM, CHEAP, etc.)..."
    cd "${REPO_DIR}"
    pixi run -e dima-env pixi run -e dima-env install-git-deps || true
    log_info "Git dependencies installed"
}

# -----------------------------------------------------------------------------
# Step 5: Download datasets
# -----------------------------------------------------------------------------
download_datasets() {
    log_info "Downloading datasets..."
    
    mkdir -p "${DATA_DIR}"
    
    if [ ! -d "${DATA_DIR}/swissprot" ]; then
        log_info "Downloading SwissProt dataset..."
        cd "${REPO_DIR}"
        pixi run -e dima-env python -m src.datasets.load_hub \
            --config_path="src/configs" \
            --load_from_hub \
            --group_name="bayes-group-diffusion"
    else
        log_info "SwissProt dataset already exists"
    fi
    
    log_info "Datasets ready"
}

# -----------------------------------------------------------------------------
# Step 6: Prepare length distributions
# -----------------------------------------------------------------------------
prepare_distributions() {
    log_info "Preparing length distributions for SwissProt..."
    
    cd "${REPO_DIR}"
    
    if [ ! -f "${DATA_DIR}/distributions/swissprot.npy" ]; then
        mkdir -p "${DATA_DIR}/distributions"
        pixi run -e dima-env python -c "
import numpy as np
from datasets import load_from_disk

ds = load_from_disk('${DATA_DIR}/swissprot')
lengths = [len(s) for s in ds['train']['sequence']]
lengths = np.array(lengths)
lengths = lengths[(lengths >= 128) & (lengths <= 254)]

bins = np.arange(128, 256)
hist, _ = np.histogram(lengths, bins=bins)
dist = hist.astype(np.float32)
dist = dist / dist.sum()
np.save('${DATA_DIR}/distributions/swissprot.npy', dist)
"
    else
        log_info "SwissProt length distribution already exists"
    fi
    
    log_info "Length distributions prepared"
}

# -----------------------------------------------------------------------------
# Step 7: Calculate encoder statistics
# -----------------------------------------------------------------------------
prepare_statistics() {
    log_info "Preparing encoder statistics for ESM-3B..."
    
    mkdir -p "${CHECKPOINT_DIR}/statistics"
    cd "${REPO_DIR}"
    
    pixi run -e dima-env python -m src.preprocessing.calculate_statistics \
        --config_path="src/configs" \
        --encoder_type="ESM2-3B" \
        --embedding_dim=2560 \
        --statistics_path="${CHECKPOINT_DIR}/statistics/encodings-ESM2-3B.pth"
    
    log_info "Encoder statistics prepared"
}

# -----------------------------------------------------------------------------
# Step 8: Run training
# -----------------------------------------------------------------------------
run_training() {
    log_info "Starting conditional training with ESM-3B on SwissProt..."
    log_info "Config:"
    log_info "  - Encoder: ESM-3B"
    log_info "  - Dataset: SwissProt"
    log_info "  - Training iterations: ${NUM_ITERS}"
    log_info "  - Eval interval: ${EVAL_INTERVAL}"
    log_info "  - Batch size: ${BATCH_SIZE}"
    log_info "  - WandB project: ${WANDB_PROJECT}"
    
    cd "${REPO_DIR}"
    export WANDB_MODE=offline
    
    pixi run -e dima-env python train_diffusion.py \
        --config-name config \
        encoder=${ENCODER} \
        datasets=${DATASET} \
        datasets.data_dir=${DATA_DIR}/${DATASET} \
        datasets.length_distribution=${DATA_DIR}/distributions/${DATASET}.npy \
        project.path=${REPO_DIR} \
        project.checkpoints_folder=${CHECKPOINT_DIR} \
        project.statistics_folder=${CHECKPOINT_DIR}/statistics \
        project.wandb_project=${WANDB_PROJECT} \
        training.training_iters=${NUM_ITERS} \
        training.eval_interval=${EVAL_INTERVAL} \
        training.save_interval=${SAVE_INTERVAL} \
        training.batch_size=${BATCH_SIZE} \
        ddp.enabled=true \
        wandb_mode=offline
    
    log_info "Training completed!"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log_info "=========================================="
    log_info "DiMA Conditional Training Setup & Run"
    log_info "=========================================="
    log_info "Encoder: ${ENCODER}"
    log_info "Dataset: ${DATASET}"
    log_info "=========================================="
    
    SKIP_SETUP=false
    SETUP_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            --setup-only)
                SETUP_ONLY=true
                shift
                ;;
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
                echo "Usage: $0 [--skip-setup] [--setup-only] [--iters N] [--project NAME]"
                exit 1
                ;;
        esac
    done
    
    if [ "$SKIP_SETUP" = true ]; then
        log_info "Skipping setup, running training only..."
        run_training
    elif [ "$SETUP_ONLY" = true ]; then
        log_info "Setup complete (--setup-only flag), skipping training..."
    else
        install_pixi
        setup_repo
        install_env
        install_git_deps
        download_datasets
        prepare_distributions
        prepare_statistics
        run_training
    fi
    
    log_info "Done!"
}

main "$@"

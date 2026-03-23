#!/bin/bash
# =============================================================================
# Download Datasets and Calculate Statistics for DiMA
#
# This script:
# 1. Downloads SwissProt and AFDB datasets from the hub
# 2. Calculates encoder statistics for ESM-3B
#
# Usage:
#   chmod +x download_data_and_stats.sh
#   ./download_data_and_stats.sh
# =============================================================================

set -e

PROJECT_ROOT="/home/cherry/dev/phd/DiMA"
DATA_DIR="${PROJECT_ROOT}/data"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

log_info "=========================================="
log_info "Downloading DiMA datasets and statistics"
log_info "=========================================="

cd "${PROJECT_ROOT}"

# Download datasets (SwissProt + AFDB)
log_info "Downloading SwissProt dataset..."
pixi run -e dima-env python -m src.datasets.load_hub \
    --config_path="src/configs" \
    --load_from_hub \
    --group_name="bayes-group-diffusion"

# Create length distributions
log_info "Creating SwissProt length distribution..."
mkdir -p "${DATA_DIR}/distributions"
pixi run -e dima-env python -c "
import numpy as np
from datasets import load_from_disk

ds = load_from_disk('${DATA_DIR}/swissprot')
lengths = np.array([len(s) for s in ds['train']['sequence']])
lengths = lengths[(lengths >= 128) & (lengths <= 254)]

bins = np.arange(128, 256)
hist, _ = np.histogram(lengths, bins=bins)
dist = hist.astype(np.float32)
dist = dist / dist.sum()
np.save('${DATA_DIR}/distributions/swissprot.npy', dist)
print(f'SwissProt: {len(lengths)} sequences, saved to swissprot.npy')
"

log_info "Creating AFDB length distribution..."
pixi run -e dima-env python -c "
import numpy as np
from datasets import load_from_disk

ds = load_from_disk('${DATA_DIR}/afdb')
lengths = np.array([len(s) for s in ds['train']['sequence']])
lengths = lengths[(lengths >= 128) & (lengths <= 254)]

bins = np.arange(128, 256)
hist, _ = np.histogram(lengths, bins=bins)
dist = hist.astype(np.float32)
dist = dist / dist.sum()
np.save('${DATA_DIR}/distributions/afdb.npy', dist)
print(f'AFDB: {len(lengths)} sequences, saved to afdb.npy')
"

# Calculate encoder statistics for ESM-3B
log_info "Calculating encoder statistics for ESM-3B..."
mkdir -p "${CHECKPOINT_DIR}/statistics"

pixi run -e dima-env python -m src.preprocessing.calculate_statistics \
    --config_path="src/configs" \
    --encoder_type="ESM2-3B" \
    --embedding_dim=2560 \
    --statistics_path="${CHECKPOINT_DIR}/statistics/encodings-ESM2-3B.pth"

log_info "=========================================="
log_info "Done! Data and statistics ready."
log_info "=========================================="

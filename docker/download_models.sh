#!/usr/bin/env bash
# Pre-download encoder models and DiMA pretrained checkpoints into the host's
# cache dirs so they are available inside the container via mounted volumes.
#
# Usage:
#   ./docker/download_models.sh                   # downloads ESM2-3B (HPC default)
#   ./docker/download_models.sh --encoder esm2-8m # downloads ESM2-8M (laptop)
#   ./docker/run.sh bash docker/download_models.sh [--encoder esm2-8m]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Parse --encoder argument (default: esm2-3b)
ENCODER="esm2-3b"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder) ENCODER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Map encoder preset to HuggingFace model name
case "$ENCODER" in
    esm2-8m)   ESM2_MODEL="facebook/esm2_t6_8M_UR50D" ;;
    esm2-35m)  ESM2_MODEL="facebook/esm2_t12_35M_UR50D" ;;
    esm2-150m) ESM2_MODEL="facebook/esm2_t30_150M_UR50D" ;;
    esm2-650m) ESM2_MODEL="facebook/esm2_t33_650M_UR50D" ;;
    esm2-3b)   ESM2_MODEL="facebook/esm2_t36_3B_UR50D" ;;
    *) echo "Unknown encoder: $ENCODER. Choose from: esm2-8m, esm2-35m, esm2-150m, esm2-650m, esm2-3b"; exit 1 ;;
esac

echo "=== Encoder: $ENCODER ($ESM2_MODEL) ==="

# Detect whether we're inside the container or on the host
if [[ -f /opt/conda/bin/conda ]]; then
    PYTHON="conda run --no-capture-output -n dima_env python"
elif [[ -f /home/cherry/miniconda3/bin/conda ]]; then
    export PATH="/home/cherry/miniconda3/bin:$PATH"
    PYTHON="conda run --no-capture-output -n dima_env python"
else
    PYTHON="python"
fi

echo ""
echo "=== Downloading ESM2 encoder weights ==="
$PYTHON auto-scripts/setup_models.py --models esm2 --esm2_model "$ESM2_MODEL"

echo ""
echo "=== Downloading DiMA pretrained checkpoints ($ENCODER) ==="
$PYTHON - <<PYEOF
import sys, os, torch
sys.path.insert(0, ".")
os.environ["WANDB_MODE"] = "offline"
from src.diffusion.dima import DiMAModel
from example_simple import ENCODER_PRESETS
encoder_type, hf_name, emb_dim = ENCODER_PRESETS["$ENCODER"]
overrides = [
    "encoder.config.encoder_type=" + encoder_type,
    "encoder.config.encoder_model_name=" + hf_name,
    "encoder.config.embedding_dim=" + str(emb_dim),
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiMAModel(config_path="src/configs", device=device, overrides=overrides)
model.load_pretrained()
print("All checkpoints downloaded.")
PYEOF

echo ""
echo "Done. Models cached and ready for offline use inside the container."

#!/usr/bin/env bash
# Pre-download all encoder models and DiMA pretrained checkpoints into
# the host's cache dirs so they are available inside the container via
# the mounted volumes.
#
# This script runs OUTSIDE the container using the host conda env,
# OR you can run it inside the container:
#   ./docker/run.sh bash docker/download_models.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Detect whether we're inside the container or on the host
if [[ -f /opt/conda/bin/conda ]]; then
    PYTHON="conda run --no-capture-output -n dima_env python"
elif [[ -f /home/cherry/miniconda3/bin/conda ]]; then
    export PATH="/home/cherry/miniconda3/bin:$PATH"
    PYTHON="conda run --no-capture-output -n dima_env python"
else
    PYTHON="python"
fi

echo "=== Downloading encoder models ==="
$PYTHON auto-scripts/setup_models.py --models esm2 saprot cheap

echo ""
echo "=== Downloading DiMA pretrained checkpoints (ESM2-8M) ==="
$PYTHON - <<'EOF'
import sys, os
sys.path.insert(0, ".")
os.environ["WANDB_MODE"] = "offline"
import torch
from src.diffusion.dima import DiMAModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiMAModel(config_path="src/configs", device=device)
model.load_pretrained()
print("All checkpoints downloaded.")
EOF

echo ""
echo "Done. Models are cached and ready for offline use inside the container."

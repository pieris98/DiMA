#!/usr/bin/env bash
# Build the DiMA Docker image.
# Usage:
#   ./docker/build.sh              # build with tag dima:latest
#   ./docker/build.sh dima:v1.0    # build with custom tag
set -euo pipefail

TAG=${1:-dima:latest}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building $TAG from $REPO_ROOT ..."
docker build \
    --tag "$TAG" \
    --file "$REPO_ROOT/Dockerfile" \
    "$REPO_ROOT"

echo ""
echo "Done. Run with:"
echo "  ./docker/run.sh                        # interactive shell"
echo "  ./docker/run.sh python example_simple.py   # generation"
echo "  ./docker/run.sh python auto-scripts/prepare_data.py  # data prep"

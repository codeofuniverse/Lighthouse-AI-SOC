#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/transfer_to_gpu.sh <dest_user>@<gpu_host> [dest_path]
# Example: ./scripts/transfer_to_gpu.sh gpu_user@gpu.example.com ~/soc_system

DEST=${1:-}
DEST_PATH=${2:-~/soc_system}

if [ -z "$DEST" ]; then
  echo "Usage: $0 <dest_user>@<gpu_host> [dest_path]" >&2
  exit 2
fi

ARCHIVE="soc_system_phase3.tar.gz"
echo "Creating archive $ARCHIVE (excluding virtualenvs, models and large files)"
tar -czf "$ARCHIVE" \
  --exclude=.venv --exclude=.venv-1 --exclude=__pycache__ \
  --exclude=.pytest_cache --exclude=detection/models --exclude=data/GeoLite2-City.mmdb \
  --exclude=docker-compose.yml .

echo "Transferring to $DEST:$DEST_PATH"
scp "$ARCHIVE" "$DEST:$DEST_PATH/"

echo "Remote extract and cleanup commands you can run on the GPU host:"
echo "  mkdir -p $DEST_PATH && tar -xzf $ARCHIVE -C $DEST_PATH && rm $ARCHIVE"

echo "Done."

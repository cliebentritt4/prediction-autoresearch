#!/usr/bin/env bash
# scripts/download_data.sh — Download and extract market data archive.
#
# Downloads data.tar.zst from Cloudflare R2 (or configured URL),
# extracts into the data/ directory at the project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

# Configure this URL to point to your data archive
DATA_URL="${DATA_ARCHIVE_URL:-https://your-r2-bucket.r2.cloudflarestorage.com/prediction-autoresearch/data.tar.zst}"

echo "=== prediction-autoresearch data download ==="
echo "Target: $DATA_DIR"

# Check for zstd
if ! command -v zstd &> /dev/null; then
    echo "Error: zstd not found. Install with: brew install zstd"
    exit 1
fi

# Download if archive doesn't exist
ARCHIVE="$PROJECT_ROOT/data.tar.zst"
if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading data archive..."
    curl -L --progress-bar -o "$ARCHIVE" "$DATA_URL"
else
    echo "Archive already exists, skipping download."
fi

# Extract
echo "Extracting to $DATA_DIR..."
mkdir -p "$DATA_DIR"
zstd -d "$ARCHIVE" --stdout | tar -xf - -C "$PROJECT_ROOT"

echo "Done. Data directory contents:"
find "$DATA_DIR" -name "*.parquet" | wc -l
echo "parquet files found."

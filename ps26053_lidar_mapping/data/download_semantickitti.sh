#!/bin/bash
# Download SemanticKITTI dataset (or prepare sample data)
# DRDO PS-26053

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR"

echo "============================================"
echo "  SemanticKITTI Dataset Downloader"
echo "============================================"

# Check if full dataset directory already exists
if [ -d "$DATA_DIR/sequences/08/velodyne" ]; then
    echo "Full dataset already exists at $DATA_DIR/sequences/"
    exit 0
fi

# Check if sample data exists
if [ -d "$DATA_DIR/sample/velodyne" ]; then
    SAMPLE_COUNT=$(ls "$DATA_DIR/sample/velodyne/"*.bin 2>/dev/null | wc -l)
    echo "Sample data exists: $SAMPLE_COUNT scans"
fi

echo ""
echo "Full SemanticKITTI download requires ~80GB and academic registration."
echo ""
echo "To download the full dataset:"
echo "  1. Go to http://www.cvlibs.net/datasets/kitti/eval_odometry.php"
echo "     - Register and download: data_odometry_velodyne.zip (~80GB)"
echo "  2. Go to http://www.semantic-kitti.org/dataset.html"
echo "     - Download: data_odometry_labels.zip (~180MB)"
echo "  3. Extract both into $DATA_DIR/:"
echo "     unzip data_odometry_velodyne.zip -d $DATA_DIR/"
echo "     unzip data_odometry_labels.zip -d $DATA_DIR/"
echo ""
echo "Expected structure after extraction:"
echo "  $DATA_DIR/sequences/08/velodyne/*.bin"
echo "  $DATA_DIR/sequences/08/labels/*.label"
echo ""

# Try to download just the labels (small, ~180MB)
echo "Attempting to download label data..."
echo "(Note: This may require manual download due to registration)"
echo ""

# Verify label format version
echo "Required: SemanticKITTI labels v1.1"
echo "Verify after download: labels should be uint32 (semantic ID in lower 16 bits)"
echo ""

# If running in an environment without the full dataset,
# generate sample data instead
echo "Generating synthetic sample data for development..."
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
fi

if [ -n "$PYTHON_CMD" ]; then
    $PYTHON_CMD "$SCRIPT_DIR/../scripts/generate_sample_data.py"
    echo "Sample data generated in $DATA_DIR/sample/"
else
    echo "WARNING: Python not found. Cannot generate sample data."
    echo "Please run: python scripts/generate_sample_data.py"
fi

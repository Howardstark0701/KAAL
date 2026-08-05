#!/bin/bash
# KAAL air-gapped executable builder
# Run from repo root: bash kaal_bundle/build.sh
#
# Produces: kaal_bundle/dist/kaal  (Linux/macOS standalone binary)
# Requirements: Python 3.10+, pip, PyInstaller 6.x
#
# Usage:
#   bash kaal_bundle/build.sh
#
# After build, copy kaal_bundle/dist/kaal to any machine and run:
#   ./kaal --help
#   ./kaal audit --model model.pt --dataset ./images/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC_FILE="$REPO_ROOT/kaal_bundle/kaal.spec"
DIST_DIR="$REPO_ROOT/kaal_bundle/dist"
BUILD_DIR="$REPO_ROOT/kaal_bundle/build"

echo "========================================"
echo "  KAAL — Building standalone executable"
echo "========================================"
echo "  Repo root : $REPO_ROOT"
echo "  Spec file : $SPEC_FILE"
echo "  Output    : $DIST_DIR/kaal"
echo ""

# Install PyInstaller if not present
if ! command -v pyinstaller &>/dev/null; then
    echo "[1/3] Installing PyInstaller..."
    pip install pyinstaller
else
    echo "[1/3] PyInstaller already installed: $(pyinstaller --version)"
fi

# Build
echo "[2/3] Running PyInstaller..."
pyinstaller \
    "$SPEC_FILE" \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --clean \
    --noconfirm

# Verify
echo "[3/3] Verifying..."
if [ -f "$DIST_DIR/kaal" ]; then
    SIZE=$(du -sh "$DIST_DIR/kaal" | cut -f1)
    echo ""
    echo "========================================"
    echo "  Build successful!"
    echo "  Executable: $DIST_DIR/kaal"
    echo "  Size:       $SIZE"
    echo "========================================"
    echo ""
    echo "  Quick test:"
    echo "    $DIST_DIR/kaal --help"
    echo ""
    echo "  Full audit:"
    echo "    $DIST_DIR/kaal audit --model model.pt --dataset ./images/"
    echo ""
else
    echo ""
    echo "ERROR: Build failed — $DIST_DIR/kaal not found."
    echo "Check the PyInstaller output above for errors."
    exit 1
fi

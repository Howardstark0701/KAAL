#!/usr/bin/env bash
# KAAL demo recorder — run from repo root in Git Bash or WSL
# Usage:  bash record_demo.sh
#
# Requirements (install once):
#   pip install asciinema       # terminal recorder
#   cargo install agg           # OR: npm install -g @asciinema/agg
#
# Output: assets/demo.gif

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAST_FILE="$REPO_ROOT/assets/demo.cast"
GIF_FILE="$REPO_ROOT/assets/demo.gif"

echo "========================================"
echo "  KAAL Demo Recorder"
echo "========================================"
echo ""

# ── Step 1: prepare model + images ───────────────────────────────────────────
echo "[1/3] Preparing demo model and images..."
source "$REPO_ROOT/.venv/Scripts/activate" 2>/dev/null \
    || source "$REPO_ROOT/.venv/bin/activate" 2>/dev/null \
    || true   # if venv activation fails, use system python

# Suppress noisy TensorFlow / torch download logs
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

python "$REPO_ROOT/generate_demo_model.py"
echo ""

# ── Step 2: record with asciinema ─────────────────────────────────────────────
echo "[2/3] Recording terminal session..."
echo "      (The audit will run automatically — just wait for the KVS score)"
echo ""
sleep 1

# Record: run the full audit inside asciinema
# --overwrite  — replace any previous cast
# --cols 90    — clean 90-column terminal width
# --rows 30    — enough rows to show the full output
# --idle-time-limit 2 — compress pauses longer than 2s
asciinema rec \
    --overwrite \
    --cols 90 \
    --rows 30 \
    --idle-time-limit 2 \
    --title "KAAL — Adversarial Audit Demo" \
    --command "bash -c 'export PYTHONWARNINGS=ignore; export TF_CPP_MIN_LOG_LEVEL=3; kaal audit --model demo_model.pt --dataset ./demo_images/ --attacks fgsm,pgd --no-gradcam --quiet; echo; echo Press Ctrl+C to stop recording; sleep 3'" \
    "$CAST_FILE"

echo ""
echo "[3/3] Converting cast to GIF..."

# Try agg (Rust binary, best quality)
if command -v agg &>/dev/null; then
    agg \
        --theme monokai \
        --cols 90 \
        --rows 30 \
        --font-size 14 \
        "$CAST_FILE" "$GIF_FILE"
    echo ""
    echo "========================================"
    echo "  Demo GIF saved → assets/demo.gif"
    echo "  Size: $(du -sh "$GIF_FILE" | cut -f1)"
    echo "========================================"

# Fallback: use asciicast2gif (Node.js)
elif command -v asciicast2gif &>/dev/null; then
    asciicast2gif "$CAST_FILE" "$GIF_FILE"
    echo "Demo GIF saved → assets/demo.gif"

else
    echo ""
    echo "WARNING: Neither 'agg' nor 'asciicast2gif' found."
    echo "  Cast saved to: $CAST_FILE"
    echo ""
    echo "  To convert manually, install agg:"
    echo "    cargo install agg"
    echo "  Then run:"
    echo "    agg --theme monokai $CAST_FILE $GIF_FILE"
    echo ""
    echo "  Or use the online converter at:"
    echo "    https://asciinema.org  (upload the .cast file)"
fi

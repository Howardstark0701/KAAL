"""Generate assets/demo.gif from real kaal audit output using PIL.

Runs a real kaal audit with --quiet flag, captures the terminal output,
then renders it frame-by-frame into an animated GIF using PIL.

Usage:
    python scripts/make_demo_gif.py

Requires: Pillow (already in venv)
Output:   assets/demo.gif
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent

# ── 1. Ensure demo model + images exist ──────────────────────────────────────
demo_model  = ROOT / "demo_model.pt"
demo_images = ROOT / "demo_images"

if not demo_model.exists() or not demo_images.exists() or not list(demo_images.glob("*.jpg")):
    print("Generating demo model and images...")
    subprocess.run(
        [sys.executable, str(ROOT / "generate_demo_model.py")],
        check=True,
        env={**os.environ, "PYTHONWARNINGS": "ignore", "TF_CPP_MIN_LOG_LEVEL": "3"},
    )

# ── 2. Run real kaal audit and capture output ─────────────────────────────────
print("Running kaal audit (capturing output)...")

import tempfile

# Write output to a temp file with UTF-8 encoding forced.
# Setting PYTHONIOENCODING + piping to a file avoids the Windows CP1252 crash
# that happens when Rich tries to write box-drawing chars to a legacy console.
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                  delete=False, encoding="utf-8") as tmp:
    tmp_path = tmp.name

env = {
    **os.environ,
    "PYTHONIOENCODING":     "utf-8",
    "PYTHONUTF8":           "1",
    "PYTHONWARNINGS":       "ignore",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "PYTHONPATH":           str(ROOT),
    # Force Rich to write plain ANSI (not Windows legacy renderer)
    "TERM":                 "xterm-256color",
}

with open(tmp_path, "w", encoding="utf-8") as out_file:
    result = subprocess.run(
        [
            sys.executable, "-m", "kaal.cli", "audit",
            "--model",   str(demo_model),
            "--dataset", str(demo_images),
            "--attacks", "fgsm,pgd",
            "--no-gradcam",
        ],
        stdout=out_file,
        stderr=out_file,
        cwd=str(ROOT),
        env=env,
    )

with open(tmp_path, encoding="utf-8", errors="replace") as f:
    raw_output = f.read()

os.unlink(tmp_path)
print(f"Captured {len(raw_output.splitlines())} lines of output.")

# ── 3. Clean up output for display ───────────────────────────────────────────
import re

def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[mGKHF]', '', text)

lines = [strip_ansi(l) for l in raw_output.splitlines()]

# Remove blank leading lines
while lines and not lines[0].strip():
    lines.pop(0)

# Keep all lines — the banner, steps, and KVS score are all we want
# Remove any lines that look like progress spinners
lines = [l for l in lines if not re.match(r'^\s*[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]\s', l)]

print(f"Cleaned to {len(lines)} lines.")

# ── 4. GIF parameters ────────────────────────────────────────────────────────
WIDTH      = 900
LINE_H     = 18
PADDING    = 20
FONT_SIZE  = 13
BG_COLOR   = (10,  10,  10)   # #0A0A0A
FG_COLOR   = (242, 242, 242)  # #F2F2F2
DIM_COLOR  = (136, 136, 136)  # #888888
RED_COLOR  = (204,   0,   0)  # #CC0000
GREEN_COLOR= ( 74, 222, 128)  # #4ADE80
RISK_RED   = (204,   0,   0)

# Try to load a monospace font; fall back to PIL default
def _load_font(size: int):
    candidates = [
        "cour.ttf",           # Windows Courier New
        "DejaVuSansMono.ttf",
        "LiberationMono-Regular.ttf",
        "Courier New.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

font   = _load_font(FONT_SIZE)
HEIGHT = 2 * PADDING + LINE_H * 36  # fixed canvas height

# ── 5. Colour each line ───────────────────────────────────────────────────────

def line_color(text: str) -> tuple:
    t = text.strip()
    if "KAAL" in t and "v1.0.0" in t:           return RED_COLOR
    if t.startswith("╔") or t.startswith("╚") or t.startswith("║"):
        return (80, 80, 80)
    if "KVS SCORE" in t:                         return RED_COLOR
    if any(x in t for x in ["CATASTROPHIC","CRITICAL"]):             return (204, 0, 0)
    if "HIGH RISK" in t:                         return (251, 146, 60)
    if "MEDIUM RISK" in t:                       return (250, 204, 21)
    if "LOW RISK" in t or "Robust" in t:         return GREEN_COLOR
    if t.startswith("[") and "/" in t[:6]:       return (200, 200, 200)
    if "Success rate:" in t:                     return GREEN_COLOR
    if "Avg" in t and "confidence" in t:         return GREEN_COLOR
    if "Audit complete" in t:                    return GREEN_COLOR
    if "Reports saved" in t:                     return (136, 136, 136)
    if t.startswith("├──") or t.startswith("└──"): return (80, 80, 80)
    if "Error" in t or "Failed" in t:            return (204, 0, 0)
    if t.startswith("──") or t == "":            return (35, 35, 35)
    if any(c in t for c in ["█", "░"]):
        if "KVS" in text or "Fgsm" in text or "Pgd" in text or "Perturb" in text:
            return RED_COLOR
        return (180, 60, 60)
    if t.startswith("  ") and ":" in t:          return (136, 136, 136)
    return (220, 220, 220)

# ── 6. Build frames ───────────────────────────────────────────────────────────
# Each frame reveals one more line; hold the last frame longer.

frames: list[Image.Image] = []
durations: list[int] = []

# How many lines to show per frame group (faster reveal = shorter gif)
LINES_PER_FRAME = 1     # add 1 line per frame
FRAME_DELAY_MS  = 60    # ms between frames during typing
HOLD_MS         = 4000  # hold on final frame (KVS score money shot)

visible: list[str] = []

for i, line in enumerate(lines):
    visible.append(line)

    # Only render a frame every N lines to keep gif size manageable
    if i % LINES_PER_FRAME != 0 and i < len(lines) - 1:
        continue

    img  = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw visible lines (show last 34 lines that fit)
    display = visible[-34:] if len(visible) > 34 else visible
    for j, text in enumerate(display):
        y = PADDING + j * LINE_H
        # Truncate to fit width
        draw.text((PADDING, y), text[:100], font=font, fill=line_color(text))

    # Cursor blink on last line
    if i < len(lines) - 1:
        last_y = PADDING + len(display) * LINE_H
        draw.rectangle([PADDING, last_y, PADDING + 8, last_y + LINE_H - 2],
                        fill=(200, 200, 200))

    frames.append(img)
    durations.append(FRAME_DELAY_MS)

# Hold on final frame
if frames:
    durations[-1] = HOLD_MS

# ── 7. Add a 1-second intro frame ────────────────────────────────────────────
intro = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
d = ImageDraw.Draw(intro)
title_font = _load_font(22)
sub_font   = _load_font(14)
d.text((WIDTH//2 - 60, HEIGHT//2 - 30), "KAAL", font=title_font, fill=RED_COLOR,
       anchor="mm" if hasattr(title_font, "getbbox") else None)
d.text((WIDTH//2, HEIGHT//2 + 10),
       "Adversarial Robustness Auditing Tool",
       font=sub_font, fill=DIM_COLOR,
       anchor="mm" if hasattr(sub_font, "getbbox") else None)
frames.insert(0, intro)
durations.insert(0, 1000)

# ── 8. Save GIF ──────────────────────────────────────────────────────────────
output_path = ROOT / "assets" / "demo.gif"
output_path.parent.mkdir(exist_ok=True)

print(f"Rendering {len(frames)} frames → assets/demo.gif ...")
frames[0].save(
    str(output_path),
    format="GIF",
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,          # loop forever
    optimize=False,
)

size_kb = output_path.stat().st_size / 1024
print(f"Done. assets/demo.gif — {size_kb:.0f} KB, {len(frames)} frames")
print(f"Total duration ≈ {sum(durations)/1000:.1f}s")

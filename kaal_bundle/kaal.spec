# -*- mode: python ; coding: utf-8 -*-
#
# KAAL PyInstaller spec — kaal_bundle/kaal.spec
#
# Packages the KAAL CLI into a single standalone executable.
# Tested with PyInstaller 6.x.
#
# Build from repo root:
#   Linux/macOS : bash  kaal_bundle/build.sh
#   Windows     : kaal_bundle\build.bat

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(SPECPATH).parent          # repo root (parent of kaal_bundle/)
ENTRY     = str(REPO_ROOT / "kaal" / "cli.py")

version = (1, 0, 0)

# ---------------------------------------------------------------------------
# Hidden imports — modules PyInstaller's static analysis misses
# ---------------------------------------------------------------------------

hidden_imports = [
    # KAAL core
    "kaal.attacks.fgsm",
    "kaal.attacks.pgd",
    "kaal.attacks.patch",
    "kaal.attacks.patch_smart",
    "kaal.attacks.gradcam",
    "kaal.attacks.blackbox",
    "kaal.attacks.physical",
    "kaal.attacks.text_attack",
    "kaal.attacks.tabular_attack",
    "kaal.attacks.audio_attack",
    "kaal.benchmark.runner",
    "kaal.benchmark.leaderboard_page",
    "kaal.defence.fingerprint",
    "kaal.defence.certification",
    "kaal.engine.loader",
    "kaal.engine.dataset",
    "kaal.engine.utils",
    "kaal.config",
    "kaal.explainability.gradcam",
    "kaal.explainability.saliency",
    "kaal.explainability.confidence",
    "kaal.fingerprint.radar",
    "kaal.reporting.pdf",
    "kaal.reporting.json_report",
    "kaal.scoring.kvs",
    # ReportLab — dynamic plugin loader needs explicit imports
    "reportlab.pdfgen",
    "reportlab.pdfgen.canvas",
    "reportlab.lib.pagesizes",
    "reportlab.lib.units",
    "reportlab.lib.utils",
    "reportlab.lib.colors",
    "reportlab.lib.styles",
    "reportlab.platypus",
    "reportlab.graphics.shapes",
    # Rich
    "rich.console",
    "rich.table",
    "rich.panel",
    "rich.progress",
    "rich.text",
    "rich.box",
    "rich.markup",
    "rich.logging",
    # Typer / Click
    "typer",
    "typer.main",
    "click",
    "click.core",
    # PyTorch — torchvision data loaders use lazy imports
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torchvision.transforms",
    "torchvision.models",
    "PIL",
    "PIL.Image",
    # Matplotlib backend — use non-interactive Agg
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.backends.backend_agg",
    # NumPy / SciPy
    "numpy",
    "scipy.ndimage",
    # Misc stdlib that PyInstaller sometimes misses
    "hashlib",
    "json",
    "pathlib",
    "importlib.metadata",
    "pkg_resources",
    "pkg_resources.py2_warn",
]

# ---------------------------------------------------------------------------
# Data files to bundle
# ---------------------------------------------------------------------------
# Format: (source_glob_or_dir, dest_folder_inside_bundle)

datas = []

# KAAL assets (screenshots, demo gif) — include if the directory exists
assets_src = str(REPO_ROOT / "assets")
if os.path.isdir(assets_src):
    datas.append((assets_src, "assets"))

# Benchmark directory (any static templates)
benchmark_src = str(REPO_ROOT / "kaal" / "benchmark")
if os.path.isdir(benchmark_src):
    datas.append((benchmark_src, "benchmark"))

# ReportLab fonts and data files (required at runtime)
try:
    import reportlab
    rl_dir = Path(reportlab.__file__).parent
    fonts_dir = str(rl_dir / "fonts")
    if os.path.isdir(fonts_dir):
        datas.append((fonts_dir, "reportlab/fonts"))
    # ReportLab graphics fonts
    graphics_dir = str(rl_dir / "graphics")
    if os.path.isdir(graphics_dir):
        datas.append((graphics_dir, "reportlab/graphics"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Excludes — keep bundle lean; these are optional / server-side only
# ---------------------------------------------------------------------------

excludes = [
    # TensorFlow is very large (~500 MB) — exclude from air-gapped bundle.
    # Users needing TF model loading should install KAAL from PyPI instead.
    "tensorflow",
    "tensorflow_core",
    "tensorflow_intel",
    "tensorboard",
    "keras",
    # ONNX runtime — large, only needed for .onnx model loading
    "onnxruntime",
    "onnx",
    # Jupyter / notebook ecosystem
    "notebook",
    "jupyter",
    "ipykernel",
    "ipython",
    "IPython",
    "nbformat",
    "nbconvert",
    # Testing
    "pytest",
    "pytest_cov",
    # Web UI dependencies (FastAPI, uvicorn, Next.js) — CLI bundle only
    "fastapi",
    "uvicorn",
    "starlette",
    "websockets",
    "aiofiles",
    # Unused ML frameworks
    "xgboost",
    "lightgbm",
    "transformers",
    # Large dev tools
    "setuptools._vendor",
    "pkg_resources._vendor",
    "distutils",
    "botocore",
    "boto3",
    "aws",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    [ENTRY],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# Single-file executable
# ---------------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="kaal",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,          # compress with UPX if available — reduces size ~30%
    upx_exclude=[
        "vcruntime140.dll",   # never compress VC runtime on Windows
        "python3*.dll",
    ],
    runtime_tmpdir=None,
    console=True,       # CLI tool — always console mode
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows version info
    version=None,       # set to a version file path for full Win32 metadata
    icon=None,          # set to path of a .ico file for Windows taskbar icon
    onefile=True,
)

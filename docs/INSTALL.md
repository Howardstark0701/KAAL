# KAAL — Installation Guide

## Requirements

- Python 3.10.x
- pip 23+
- 4GB RAM minimum (8GB recommended for large models)
- CUDA-capable GPU optional (CPU mode supported)

## Install from PyPI

```bash
pip install kaal
kaal --help
```

## Install from Source

```bash
git clone https://github.com/your-username/kaal.git
cd kaal
pip install -e .
kaal --help
```

## Install with Dev Dependencies

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Web UI Setup

Install web backend dependencies:
```bash
pip install -r requirements-web.txt
```

Install and run the frontend (requires Node.js 20.x LTS):
```bash
cd web/frontend
npm install
npm run dev
```

Start the backend:
```bash
uvicorn web.backend.main:app --port 8080
```

Visit http://localhost:3000

## Offline Operation

Once installed, KAAL runs fully offline. No network calls are made during audits.
The only internet requirement is the initial `pip install` step.

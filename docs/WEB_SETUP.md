# KAAL — Web UI Setup

## Prerequisites

- Python 3.10.x with KAAL installed
- Node.js 20.x LTS
- npm 10+

## Start the Backend

```bash
pip install -r requirements-web.txt
uvicorn web.backend.main:app --host 127.0.0.1 --port 8080
```

Backend runs on http://localhost:8080
API docs available at http://localhost:8080/docs

## Start the Frontend

```bash
cd web/frontend
npm install
npm run dev
```

Frontend runs on http://localhost:3000

## Pages

- `/`        — Landing page
- `/audit`   — Upload model + dataset, configure and run audit
- `/results` — View audit results, KVS score, radar chart, GradCAM
- `/patch`   — Adversarial patch generator
- `/compare` — Compare two audit reports side by side

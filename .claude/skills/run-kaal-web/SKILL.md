---
name: run-kaal-web
description: Build, launch, screenshot and drive the KAAL web UI (FastAPI backend + Next.js frontend) — run the app, start the server, take screenshots of the audit/results pages, drive a real end-to-end audit in a browser, or inspect the KVS radar chart geometry. Use for any UI/UX work on the KAAL web interface.
---

# Run the KAAL web UI

KAAL's web interface is a **Next.js 14 frontend** (`web/frontend`) talking to a
**FastAPI backend** (`web/backend`). The agent path is
[`driver.mjs`](driver.mjs) — it boots both servers, drives the real browser via
Playwright, screenshots each page, and reports the rendered geometry of the KVS
radar chart.

All paths below are relative to the repo root (`C:\Users\patha\KAAL`).
Verified on Windows 11, Python 3.10.5, Node v24.16.0, torch 2.13.0+cpu.

## Prerequisites

The venv and `node_modules` already exist in this checkout. Verify rather than
reinstall:

```bash
./.venv/Scripts/python.exe -c "import torch; print(torch.__version__)"   # 2.13.0+cpu
node -e "console.log(require('C:/Users/patha/KAAL/web/frontend/node_modules/playwright/package.json').version)"   # 1.62.1
```

If the venv is missing: `python -m venv .venv && ./.venv/Scripts/pip.exe install -r requirements.txt -r requirements-web.txt`.
If `web/frontend/node_modules` is missing: `cd web/frontend && npm install`.

Playwright lives in `web/frontend/node_modules`, **not** the repo root. The
driver resolves it explicitly — don't run it with a bare `require('playwright')`
from the root or you get `MODULE_NOT_FOUND`.

## Build

The driver refuses to start if `web/frontend/.next` is missing. Build once:

```bash
cd web/frontend && npm run build
```

Takes ~60s. Ends with a route table and `EXIT:0`; 7/7 static pages.

## Run (agent path)

Run from the repo root. Screenshots land in `.kaal-driver-shots/`
(override with `KAAL_SHOTS`). Add `KAAL_VERBOSE=1` to see server logs.

```bash
# Screenshot the four static pages (boots + tears down its own servers)
node .claude/skills/run-kaal-web/driver.mjs shots

# Full flow: upload demo_model.pt + 5 demo images, run the audit,
# screenshot results, dump radar geometry. Takes ~4 min.
node .claude/skills/run-kaal-web/driver.mjs audit

# Keep both servers up (Ctrl-C to stop) — needed for `radar` below
node .claude/skills/run-kaal-web/driver.mjs up

# Inspect radar geometry for a job on an ALREADY-RUNNING backend
node .claude/skills/run-kaal-web/driver.mjs radar <job_id>
```

`audit` output ends with the radar's per-axis radius as a fraction of the outer
grid ring, the results URL, any dimensions marked "not tested", and a console
error count. Exit codes: `0` ok, `1` failure, `2` bad usage.

Example from a real run:

```
[driver] largest axis reaches 83.3% of the outer ring.
[driver] dimensions reported "not tested": ["Black-Boxnot tested"]
[driver] console errors: none
```

## Run (human path)

Two terminals, and the ports are **not** optional (see Gotchas):

```bash
# terminal 1
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m uvicorn web.backend.main:app --host 127.0.0.1 --port 8080

# terminal 2
cd web/frontend && npx next start -p 3000
```

Then open <http://localhost:3000>. Backend API docs at <http://127.0.0.1:8080/docs>.

`python -m kaal.cli serve` also exists but was not used for any of the above.

## Test

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest -q   # 331 passed, ~351s
cd web/frontend && npx tsc --noEmit                              # exit 0
```

The 6 `weights_only` UserWarnings in the pytest run are expected, not failures.

## Gotchas

Everything here cost real debugging time.

- **Ports 3000 and 8080 are hardcoded — using others silently breaks uploads.**
  Two independent reasons. The backend CORS allowlist is exactly
  `["http://localhost:3000", "http://127.0.0.1:3000"]` (`web/backend/main.py`),
  and `NEXT_PUBLIC_API_URL` is **baked in at `npm run build`**, defaulting to
  `http://localhost:8080`. Setting that env var at `next start` does nothing —
  Next.js inlines `NEXT_PUBLIC_*` at build time. On the wrong port the UI looks
  fine but both upload zones show "Upload failed" and *Start Audit* never
  enables. To actually change the API URL you must rebuild.

- **Never screenshot the results page with `fullPage: true`.** A fullPage
  capture resizes the capture surface; recharts' `ResponsiveContainer`
  re-measures and **restarts the Radar animation from radius 0**, so the PNG
  shows an empty chart while the DOM holds a correct polygon. Measured: polygon
  vertical spread stays 130px across settle and viewport capture, then collapses
  to 2.46px after a fullPage capture. `driver.mjs` defaults `fullPage` to false
  and slices tall pages by scrolling instead. This is the single easiest way to
  file a false "the radar is broken" bug.

- **The radar animates for ~1.5s after mount.** Measure or screenshot too early
  and you get a collapsed polygon. The driver waits 8s after
  `.recharts-surface` appears.

- **Job IDs are in-memory only.** `web/backend/jobs/store.py` holds them in a
  dict — restarting the backend loses every job, and `/results?job_id=…` then
  404s. `driver.mjs radar` deliberately does *not* boot servers for this reason;
  it checks the job exists first and fails with a clear message.

- **Job IDs are 43-char URL-safe tokens, not UUIDs.** Anything else gets a 400
  from `_validate_job_id` before any store lookup. The compare page's
  placeholder still shows the old UUID shape — ignore it, it's stale.

- **Two concurrent jobs per IP, max.** A third `POST /api/audit/start` returns
  429. If the driver 429s, an earlier run's job is still going — wait it out.

- **Model upload is slow and synchronous.** `demo_model.pt` is 46MB and the
  backend fully loads it before responding, so *Start Audit* stays disabled for
  several seconds after the file dialog resolves. Wait on the button's
  `disabled` property, never a fixed sleep.

- **Windows console is cp1252.** Always prefix Python with
  `PYTHONIOENCODING=utf-8` or the CLI's box-drawing banner raises
  `UnicodeEncodeError`.

- **Two known UI defects the driver surfaces on purpose.** The radar has no
  `PolarRadiusAxis domain={[0,10]}`, so its outer ring is auto-scaled (measured:
  a 10.0 axis reaches 0.8333 of the ring = 10/12) and audits are **not**
  comparable by shape. And untested dimensions are plotted at radius 0 —
  visually identical to a perfect score — while the list below correctly says
  "not tested". Don't mistake either for a driver bug.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `web/frontend/.next missing` | Run the Build step. |
| Both upload zones say "Upload failed", *Start Audit* stays disabled | Wrong ports. Backend must be 8080, frontend 3000. See Gotchas. |
| `Cannot find module 'playwright'` | Resolving from the repo root. Playwright is in `web/frontend/node_modules`. |
| Radar looks empty in a screenshot | You used `fullPage: true`, or captured within ~1.5s of mount. |
| `/results?job_id=…` 404s | Backend restarted since the audit. Job store is in-memory; re-run `driver.mjs audit`. |
| `429 Too many active audits` | Two jobs already running for your IP. Wait. |
| `UnicodeEncodeError` on a `\u2500`-ish char | Missing `PYTHONIOENCODING=utf-8`. |
| Driver hangs at "frontend ready" then times out on file inputs | Frontend served a stale build. Rebuild. |

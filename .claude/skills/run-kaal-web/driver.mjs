#!/usr/bin/env node
/**
 * KAAL web UI driver.
 *
 * Boots the FastAPI backend + Next.js frontend, drives the real browser UI,
 * and reports what an agent needs to see: screenshots, console errors, and
 * the rendered geometry of the KVS radar chart.
 *
 * Run from the KAAL repo root:
 *   node .claude/skills/run-kaal-web/driver.mjs <command>
 *
 * Commands:
 *   up      Start backend+frontend, wait for both, keep running (Ctrl-C to stop)
 *   shots   Screenshot every static page (starts+stops its own servers)
 *   audit   Full upload -> audit -> results flow, screenshots + radar geometry
 *   radar <job_id>   Dump radar chart geometry for an existing completed job
 *
 * Ports are NOT configurable. See SKILL.md "Gotchas" — the backend CORS
 * allowlist and the frontend's build-time API URL both hardcode 3000/8080.
 */

import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { existsSync, mkdirSync, readdirSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

// Repo root = three levels up from .claude/skills/run-kaal-web/
const ROOT = path.resolve(import.meta.dirname, '..', '..', '..');
const FRONTEND = path.join(ROOT, 'web', 'frontend');
const PY = path.join(ROOT, '.venv', 'Scripts', 'python.exe');
const SHOTS = process.env.KAAL_SHOTS ?? path.join(ROOT, '.kaal-driver-shots');

// playwright lives in the frontend's node_modules, not the repo root
const require = createRequire(path.join(FRONTEND, 'package.json'));
const { chromium } = require('playwright');

const API = 'http://127.0.0.1:8080';
const FE = 'http://localhost:3000';

const log = (...a) => console.log(...a);
const children = [];

function die(msg) { console.error(`\n[driver] FATAL: ${msg}`); shutdown(1); }

function shutdown(code = 0) {
  for (const c of children) { try { c.kill(); } catch {} }
  process.exit(code);
}
process.on('SIGINT', () => shutdown(0));

// ── server management ──────────────────────────────────────────────────────

function startBackend() {
  if (!existsSync(PY)) die(`venv python not found at ${PY} — see SKILL.md Prerequisites`);
  log('[driver] starting backend on :8080');
  const c = spawn(PY, ['-m', 'uvicorn', 'web.backend.main:app',
                       '--host', '127.0.0.1', '--port', '8080'],
    { cwd: ROOT, env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, stdio: 'pipe' });
  c.stderr.on('data', d => { if (process.env.KAAL_VERBOSE) process.stderr.write(`[be] ${d}`); });
  children.push(c);
  return c;
}

function startFrontend() {
  // Spawn next's JS entrypoint with node rather than the .bin shim. The shim
  // is a .cmd on Windows, which needs shell:true — and that emits a DEP0190
  // deprecation warning on Node 24 and mangles arg quoting.
  const next = path.join(FRONTEND, 'node_modules', 'next', 'dist', 'bin', 'next');
  if (!existsSync(path.join(FRONTEND, '.next'))) {
    die('web/frontend/.next missing — run the Build step in SKILL.md first');
  }
  log('[driver] starting frontend on :3000');
  const c = spawn(process.execPath, [next, 'start', '-p', '3000'],
    { cwd: FRONTEND, stdio: 'pipe' });
  c.stderr.on('data', d => { if (process.env.KAAL_VERBOSE) process.stderr.write(`[fe] ${d}`); });
  children.push(c);
  return c;
}

async function waitFor(url, label, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url, { signal: AbortSignal.timeout(3000) });
      if (r.status < 500) { log(`[driver] ${label} ready`); return; }
    } catch {}
    await new Promise(r => setTimeout(r, 700));
  }
  die(`${label} did not come up within ${timeoutMs / 1000}s`);
}

async function bootBoth() {
  startBackend();
  startFrontend();
  await waitFor(`${API}/openapi.json`, 'backend');
  await waitFor(FE, 'frontend');
}

// ── radar geometry ─────────────────────────────────────────────────────────

// Reads the rendered SVG rather than the React props, so it reports what the
// user actually sees. Vertices are matched to axes BY ANGLE, not by index: an
// untested dimension renders as a break in the path, which shifts every
// subsequent vertex and silently misattributes a naive index mapping.
const RADAR_GEOMETRY = () => {
  const svg = document.querySelector('.recharts-surface');
  if (!svg) return { error: 'no radar rendered' };
  const poly = svg.querySelector('.recharts-radar-polygon polygon, .recharts-radar-polygon path');
  if (!poly) return { error: 'no radar polygon' };
  const rings = [...svg.querySelectorAll(
    '.recharts-polar-grid-concentric polygon, .recharts-polar-grid-concentric path')];
  const d = poly.getAttribute('points') || poly.getAttribute('d') || '';
  const pts = [...d.matchAll(/(-?[\d.]+),(-?[\d.]+)/g)]
    .map(m => [parseFloat(m[1]), parseFloat(m[2])]);
  const labels = [...svg.querySelectorAll('.recharts-polar-angle-axis-tick-value')]
    .map(t => t.textContent.trim());

  // centre = vertex of the degenerate innermost grid ring
  const inner = rings[0]?.getAttribute('points') || rings[0]?.getAttribute('d') || '';
  const c = [...inner.matchAll(/(-?[\d.]+),(-?[\d.]+)/g)][0];
  const cx = c ? parseFloat(c[1]) : null, cy = c ? parseFloat(c[2]) : null;
  const outerEl = rings[rings.length - 1];
  const outerD = outerEl?.getAttribute('points') || outerEl?.getAttribute('d') || '';
  const o = [...outerD.matchAll(/(-?[\d.]+),(-?[\d.]+)/g)][0];
  const outerR = o && cy !== null ? Math.abs(cy - parseFloat(o[2])) : null;

  const n = labels.length || 1;
  const slot = 360 / n;
  // Axis i sits at -90deg + i*slot, measured clockwise from 12 o'clock.
  const axisIndexOf = (x, y) => {
    let deg = Math.atan2(x - cx, cy - y) * 180 / Math.PI;   // 0 = up, cw
    if (deg < 0) deg += 360;
    return Math.round(deg / slot) % n;
  };

  const byAxis = new Map();
  for (const [x, y] of pts) {
    const r = Math.hypot(x - cx, y - cy);
    if (r < 0.01) continue;                 // path-close artefacts at centre
    const i = axisIndexOf(x, y);
    if (!byAxis.has(i) || r > byAxis.get(i)) byAxis.set(i, r);
  }

  const axes = labels.map((label, i) => {
    const r = byAxis.get(i);
    return r === undefined
      ? { axis: label, plotted: false }
      : { axis: label, plotted: true, radius: +r.toFixed(2),
          fractionOfOuter: outerR ? +(r / outerR).toFixed(4) : null };
  });
  return { centre: [cx, cy], outerRingRadius: outerR, rings: rings.length, axes };
};

async function reportRadar(page) {
  // The radar animates outward from the centre. Screenshot or measure too
  // early and it reads as an empty chart. 8s is comfortably past settle.
  await page.waitForSelector('.recharts-surface', { timeout: 60_000 });
  await page.waitForTimeout(8000);
  const g = await page.evaluate(RADAR_GEOMETRY);
  log('\n[driver] radar geometry:');
  log(JSON.stringify(g, null, 2));

  // The radius axis is pinned to [0, 10], so fractionOfOuter is score/10. Any
  // drift from that means the explicit PolarRadiusAxis domain has been lost
  // and fingerprints are no longer comparable between audits.
  const plotted = (g.axes ?? []).filter(a => a.plotted);
  const unplotted = (g.axes ?? []).filter(a => !a.plotted).map(a => a.axis);
  if (plotted.length) {
    const max = Math.max(...plotted.map(a => a.fractionOfOuter ?? 0));
    log(`\n[driver] largest plotted axis = ${(max * 100).toFixed(1)}% of the outer ring`);
    log('[driver] outer ring is pinned to KVS 10, so fraction == score/10.');
  }
  log(unplotted.length
    ? `[driver] axes left unplotted (not tested): ${JSON.stringify(unplotted)}`
    : '[driver] every axis plotted.');
  return g;
}

// ── browser helpers ────────────────────────────────────────────────────────

async function newPage(browser, errs) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  page.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));
  return page;
}

/**
 * Screenshot helper.
 *
 * fullPage defaults to FALSE on purpose. A fullPage capture resizes the
 * capture surface, which makes recharts' ResponsiveContainer re-measure and
 * restart the Radar animation from radius 0 — the resulting PNG shows an
 * empty chart even though the DOM holds a correct polygon. Only pass
 * fullPage on pages with no chart. See SKILL.md "Gotchas".
 */
async function shot(page, name, { fullPage = false } = {}) {
  mkdirSync(SHOTS, { recursive: true });
  const p = path.join(SHOTS, `${name}.png`);
  await page.screenshot({ path: p, fullPage });
  log(`[driver] shot -> ${p}`);
}

/** Capture a tall page in viewport-sized slices, avoiding fullPage. */
async function shotScrolled(page, base, slices = 2) {
  for (let i = 0; i < slices; i++) {
    await page.evaluate(y => window.scrollTo(0, y), i * 820);
    await page.waitForTimeout(700);
    await shot(page, `${base}-${i + 1}`);
  }
  await page.evaluate(() => window.scrollTo(0, 0));
}

function reportErrors(errs) {
  log('\n[driver] console errors: ' + (errs.length ? errs.length : 'none'));
  errs.slice(0, 15).forEach(e => log('  ' + e));
}

// ── commands ───────────────────────────────────────────────────────────────

async function cmdShots() {
  await bootBoth();
  const browser = await chromium.launch();
  const errs = [];
  const page = await newPage(browser, errs);
  for (const [name, url] of [
    ['01-index', FE], ['02-audit', `${FE}/audit`],
    ['03-patch', `${FE}/patch`], ['04-compare', `${FE}/compare`],
  ]) {
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForTimeout(600);
    // Safe: none of these four pages renders a chart.
    await shot(page, name, { fullPage: true });
  }
  reportErrors(errs);
  await browser.close();
  shutdown(0);
}

async function cmdAudit() {
  await bootBoth();
  const browser = await chromium.launch();
  const errs = [];
  const page = await newPage(browser, errs);

  log('[driver] driving upload -> audit -> results');
  await page.goto(`${FE}/audit`, { waitUntil: 'networkidle' });

  const inputs = await page.locator('input[type=file]').all();
  if (inputs.length < 2) die(`expected 2 file inputs on /audit, found ${inputs.length}`);

  // demo_model.pt is ~46MB; the upload round-trip includes a full model load
  // server-side, so this is genuinely slow. Wait on the UI, not a fixed sleep.
  await inputs[0].setInputFiles(path.join(ROOT, 'demo_model.pt'));
  const imgs = readdirSync(path.join(ROOT, 'demo_images'))
    .filter(f => f.endsWith('.jpg')).slice(0, 5)
    .map(f => path.join(ROOT, 'demo_images', f));
  await inputs[1].setInputFiles(imgs);

  const start = page.locator('button', { hasText: /start audit/i }).last();
  await start.waitFor({ state: 'visible' });
  // Button stays disabled until BOTH uploads resolve server-side.
  await page.waitForFunction(() => {
    const b = [...document.querySelectorAll('button')]
      .find(x => /start audit/i.test(x.textContent));
    return b && !b.disabled;
  }, { timeout: 180_000 }).catch(() => die(
    'Start Audit never enabled — uploads failed. Check ports (see Gotchas).'));
  await shot(page, '05-audit-ready', { fullPage: true });

  await start.click();
  log('[driver] audit started, waiting for results…');
  await page.waitForURL(/results/, { timeout: 60_000 });
  await page.waitForSelector('.recharts-surface', { timeout: 900_000 });

  const g = await reportRadar(page);
  // Viewport slices, never fullPage — see shot() for why.
  await shotScrolled(page, '06-results', 3);
  log(`\n[driver] results URL: ${page.url()}`);
  log('[driver] job IDs live in memory only — this ID dies with the backend.');

  const notTested = await page.evaluate(() =>
    [...document.querySelectorAll('div')]
      .map(d => d.textContent || '')
      .filter(t => /not tested/.test(t) && t.length < 60)
      .filter((v, i, a) => a.indexOf(v) === i));
  if (notTested.length) {
    log(`[driver] dimensions reported "not tested": ${JSON.stringify(notTested)}`);
    log('[driver] these must appear as unplotted axes above, never at radius 0.');
  }

  reportErrors(errs);
  await browser.close();
  shutdown(0);
}

async function cmdRadar(jobId) {
  if (!jobId) die('usage: driver.mjs radar <job_id>');
  // Deliberately does NOT boot servers. The job store is in-memory, so a
  // fresh backend has never heard of this job ID. This only works against
  // the same backend process that ran the audit — i.e. `driver.mjs up`.
  try {
    const r = await fetch(`${API}/api/audit/status/${jobId}`,
                          { signal: AbortSignal.timeout(4000) });
    if (r.status === 404) die(
      `backend has no job '${jobId}'. Job IDs are in-memory: they do not ` +
      `survive a backend restart. Re-run 'driver.mjs audit' for a fresh one.`);
    if (!r.ok) die(`backend returned ${r.status} for that job ID`);
  } catch (e) {
    if (e.message?.includes('no job')) throw e;
    die(`no backend on ${API}. Start one first:  node ${path.relative(ROOT,
      import.meta.filename)} up`);
  }
  const browser = await chromium.launch();
  const errs = [];
  const page = await newPage(browser, errs);
  await page.goto(`${FE}/results?job_id=${jobId}`, { waitUntil: 'networkidle' });
  await reportRadar(page);
  await shot(page, `radar-${jobId.slice(0, 8)}`);
  reportErrors(errs);
  await browser.close();
  shutdown(0);
}

async function cmdUp() {
  await bootBoth();
  log(`\n[driver] backend  ${API}/docs`);
  log(`[driver] frontend ${FE}`);
  log('[driver] Ctrl-C to stop both.');
  await new Promise(() => {});
}

// ── dispatch ───────────────────────────────────────────────────────────────

const [cmd, arg] = process.argv.slice(2);
const run = { up: cmdUp, shots: cmdShots, audit: cmdAudit,
              radar: () => cmdRadar(arg) }[cmd];
if (!run) {
  console.error('usage: driver.mjs <up|shots|audit|radar <job_id>>');
  process.exit(2);
}
run().catch(e => die(e.stack || e.message));

/**
 * KAAL Evidence Capture — Playwright screenshots + video
 * Run: node scripts/capture_evidence.js
 */
const { chromium } = require('../web/frontend/node_modules/playwright');
const path  = require('path');
const fs    = require('fs');
const http  = require('http');

const SS_DIR  = path.resolve(__dirname, '..', 'evidence', 'screenshots');
const VID_DIR = path.resolve(__dirname, '..', 'evidence', 'video');
fs.mkdirSync(SS_DIR,  { recursive: true });
fs.mkdirSync(VID_DIR, { recursive: true });

const API_BASE = 'http://localhost:8080';
const UI_BASE  = 'http://localhost:3000';

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function httpGet(url) {
  return new Promise((resolve, reject) => {
    http.get(url, res => {
      let d = ''; res.on('data', c => d += c); res.on('end', () => resolve(d));
    }).on('error', reject);
  });
}

function httpPost(url, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const u = new URL(url);
    const opts = { hostname: u.hostname, port: u.port, path: u.pathname, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) } };
    const req = http.request(opts, res => {
      let d = ''; res.on('data', c => d += c); res.on('end', () => resolve(JSON.parse(d)));
    });
    req.on('error', reject);
    req.write(data); req.end();
  });
}

function postMultipart(url, fieldName, filePath, filename) {
  return new Promise((resolve, reject) => {
    const boundary = '----KAALBoundary' + Date.now();
    const fileData = fs.readFileSync(filePath);
    const body = Buffer.concat([
      Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${fieldName}"; filename="${filename}"\r\nContent-Type: application/octet-stream\r\n\r\n`),
      fileData, Buffer.from(`\r\n--${boundary}--\r\n`),
    ]);
    const u = new URL(url);
    const opts = { hostname: u.hostname, port: u.port, path: u.pathname, method: 'POST',
      headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}`, 'Content-Length': body.length } };
    const req = http.request(opts, res => {
      let d = ''; res.on('data', c => d += c); res.on('end', () => { try { resolve(JSON.parse(d)); } catch(e) { reject(e); } });
    });
    req.on('error', reject); req.write(body); req.end();
  });
}

async function ss(page, name, opts = {}) {
  const p = path.join(SS_DIR, name);
  await page.screenshot({ path: p, fullPage: opts.fullPage ?? false });
  const kb = (fs.statSync(p).size / 1024).toFixed(1);
  console.log(`  [SS] ${name}  ${kb} KB`);
  return p;
}

(async () => {
  const browser = await chromium.launch({ headless: true });

  // ── VIDEO CONTEXT for recording ──────────────────────────────────────────
  const videoCtx = await browser.newContext({
    recordVideo: { dir: VID_DIR, size: { width: 1280, height: 720 } },
    viewport:    { width: 1280, height: 720 },
  });
  const vidPage = await videoCtx.newPage();

  // ── STATIC SCREENSHOT CONTEXT (no video, faster) ─────────────────────────
  const ctx  = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await ctx.newPage();

  console.log('\n[1/8] 01-dashboard.png — Landing page');
  await page.goto(UI_BASE, { waitUntil: 'networkidle', timeout: 20000 });
  await sleep(800);
  await ss(page, '01-dashboard.png');

  console.log('[2/8] 02-audit-config.png — Audit page');
  await page.goto(`${UI_BASE}/audit`, { waitUntil: 'networkidle', timeout: 20000 });
  await sleep(600);
  // Push epsilon slider to 0.105 to show warning
  await page.evaluate(() => {
    const s = document.querySelector('input[type=range]');
    if (s) { s.value = '0.105'; s.dispatchEvent(new Event('input', {bubbles:true})); }
  });
  await sleep(400);
  await ss(page, '02-audit-config.png', { fullPage: true });

  console.log('[3/8] 03-compare.png — Compare page');
  await page.goto(`${UI_BASE}/compare`, { waitUntil: 'networkidle', timeout: 20000 });
  await sleep(500);
  await ss(page, '03-compare.png');

  console.log('[4/8] 04-patch.png — Patch page');
  await page.goto(`${UI_BASE}/patch`, { waitUntil: 'networkidle', timeout: 20000 });
  await sleep(500);
  await ss(page, '04-patch.png');

  // ── Run a real audit via API for live results screenshots ─────────────────
  console.log('\n[API] Uploading demo model...');
  const modelPath = path.resolve(__dirname, '..', 'demo_model.pt');
  const mJson = await postMultipart(`${API_BASE}/api/upload/model`, 'file', modelPath, 'demo_model.pt');
  const model_id = mJson.model_id;
  console.log(`  model_id: ${model_id}  framework: ${mJson.framework}`);

  console.log('[API] Uploading demo images...');
  const imgDir = path.resolve(__dirname, '..', 'demo_images');
  const imgFiles = fs.readdirSync(imgDir).filter(f => f.endsWith('.jpg')).slice(0, 5);
  // Upload first image as dataset
  const dJson = await postMultipart(`${API_BASE}/api/upload/dataset`, 'files',
    path.join(imgDir, imgFiles[0]), imgFiles[0]);
  const dataset_id = dJson.dataset_id;
  console.log(`  dataset_id: ${dataset_id}  count: ${dJson.count}`);

  console.log('[API] Starting audit (fgsm,pgd, no-gradcam)...');
  const startJson = await httpPost(`${API_BASE}/api/audit/start`, {
    model_id, dataset_id, attacks: ['fgsm', 'pgd'], epsilon: 0.03, steps: 20,
    report_formats: ['json'], no_gradcam: true,
  });
  const job_id = startJson.job_id;
  console.log(`  job_id: ${job_id}`);

  // Navigate to results page — will show skeleton while running
  console.log('[5/8] 05-live-audit-running.png — Live audit in progress');
  await page.goto(`${UI_BASE}/results?job_id=${job_id}`, { waitUntil: 'networkidle', timeout: 20000 });
  await sleep(2000);
  await ss(page, '05-live-audit-running.png');

  // Poll until complete
  let status = 'running';
  for (let i = 0; i < 60 && status !== 'complete' && status !== 'failed'; i++) {
    await sleep(4000);
    const s = JSON.parse(await httpGet(`${API_BASE}/api/audit/status/${job_id}`));
    status = s.status;
    console.log(`  [poll] ${status} ${s.progress_pct}% — ${s.current_step}`);
  }
  console.log(`  Audit ${status}`);

  // Wait for frontend to render results
  await sleep(5000);
  await page.goto(`${UI_BASE}/results?job_id=${job_id}`, { waitUntil: 'networkidle', timeout: 20000 });
  await sleep(3000);

  console.log('[6/8] 06-kvs-results.png — KVS score + results');
  await ss(page, '06-kvs-results.png', { fullPage: true });

  // Screenshot just the radar chart area
  const radarSection = await page.$('section:has(svg)');
  if (radarSection) {
    const rp = path.join(SS_DIR, '07-vulnerability-fingerprint.png');
    await radarSection.screenshot({ path: rp });
    console.log(`  [SS] 07-vulnerability-fingerprint.png  ${(fs.statSync(rp).size/1024).toFixed(1)} KB`);
  } else {
    await page.screenshot({ path: path.join(SS_DIR, '07-vulnerability-fingerprint.png'),
      clip: { x: 0, y: 350, width: 1280, height: 420 } });
    console.log('  [SS] 07-vulnerability-fingerprint.png (fallback clip)');
  }

  // KAAL-D cert page — show the cert files we generated
  console.log('[8/8] 08-kaal-d-cert.png — KAAL-D certification output');
  // Navigate to a plain text view of the certificate JSON in the browser
  await page.setContent(`
    <html><body style="background:#0A0A0A;color:#F2F2F2;font-family:monospace;padding:32px">
    <h2 style="color:#CC0000">KAAL-D Certification Bundle</h2>
    <pre>${JSON.stringify(JSON.parse(fs.readFileSync(
      path.resolve(__dirname,'..','evidence','cert_output','certificate.json'),'utf8'
    )), null, 2)}</pre>
    </body></html>`);
  await sleep(500);
  await ss(page, '08-kaal-d-certification.png');

  // ── VIDEO recording: full workflow on video page ─────────────────────────
  console.log('\n[VIDEO] Recording live workflow...');
  await vidPage.goto(UI_BASE, { waitUntil: 'networkidle', timeout: 20000 });
  await sleep(1500);
  await vidPage.goto(`${UI_BASE}/audit`, { waitUntil: 'networkidle', timeout: 20000 });
  await sleep(1000);
  await vidPage.goto(`${UI_BASE}/results?job_id=${job_id}`, { waitUntil: 'networkidle', timeout: 20000 });
  await sleep(4000);
  await vidPage.evaluate(() => window.scrollTo({ top: 500, behavior: 'smooth' }));
  await sleep(2000);
  await vidPage.evaluate(() => window.scrollTo({ top: 1000, behavior: 'smooth' }));
  await sleep(2000);

  // Close video context — this finalises the video file
  await videoCtx.close();
  const videoFile = await vidPage.video()?.path();
  if (videoFile && fs.existsSync(videoFile)) {
    const dest = path.join(VID_DIR, 'kaal-live-audit-demo.webm');
    fs.renameSync(videoFile, dest);
    console.log(`  [VIDEO] kaal-live-audit-demo.webm  ${(fs.statSync(dest).size/1024).toFixed(0)} KB`);
  }

  await ctx.close();
  await browser.close();

  // ── Summary ────────────────────────────────────────────────────────────────
  console.log('\n========================================');
  console.log('  Evidence Capture Complete');
  console.log('========================================');
  const shots = fs.readdirSync(SS_DIR);
  shots.forEach(f => {
    const kb = (fs.statSync(path.join(SS_DIR,f)).size/1024).toFixed(1);
    console.log(`  ${f}  ${kb} KB`);
  });
  const vids = fs.readdirSync(VID_DIR);
  vids.forEach(f => {
    const kb = (fs.statSync(path.join(VID_DIR,f)).size/1024).toFixed(0);
    console.log(`  video/${f}  ${kb} KB`);
  });

})().catch(e => { console.error('CAPTURE FAILED:', e.message); process.exit(1); });

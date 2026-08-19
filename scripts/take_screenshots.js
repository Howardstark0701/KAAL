const { chromium } = require('../web/frontend/node_modules/playwright');
const path = require('path');
const fs   = require('fs');

const OUT = path.resolve(__dirname, '..', 'assets', 'screenshots');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page    = await browser.newPage();
  await page.setViewportSize({ width: 1024, height: 768 });

  // 1. web-dashboard.png — home page
  console.log('Screenshotting home page...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(OUT, 'web-dashboard.png') });
  const s1 = (fs.statSync(path.join(OUT, 'web-dashboard.png')).size / 1024).toFixed(1);
  console.log('  saved web-dashboard.png  ' + s1 + ' KB');

  // 2. audit-page.png — audit page with epsilon warning visible
  console.log('Screenshotting audit page...');
  await page.goto('http://localhost:3000/audit', { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(800);
  // Push epsilon to 0.105 to show the warning
  const slider = await page.$('input[type=range]');
  if (slider) {
    await slider.evaluate(el => {
      el.value = '0.105';
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitForTimeout(300);
  }
  await page.screenshot({ path: path.join(OUT, 'audit-page.png'), fullPage: true });
  const s2 = (fs.statSync(path.join(OUT, 'audit-page.png')).size / 1024).toFixed(1);
  console.log('  saved audit-page.png  ' + s2 + ' KB');

  // 3. results-view.png — go to results with the last known job from leaderboard results
  // We'll just capture the results page route — it shows "No job ID" state cleanly,
  // then overlay with the provided screenshot via README reference.
  // For now, screenshot /results which shows the no-job-id guidance page.
  console.log('Screenshotting results page...');
  await page.goto('http://localhost:3000/results', { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'results-view.png') });
  const s3 = (fs.statSync(path.join(OUT, 'results-view.png')).size / 1024).toFixed(1);
  console.log('  saved results-view.png  ' + s3 + ' KB');

  await browser.close();
  console.log('\nAll screenshots saved to assets/screenshots/');
})().catch(e => { console.error('FAILED:', e.message); process.exit(1); });

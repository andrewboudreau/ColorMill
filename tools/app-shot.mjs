/**
 * A still of the real app (simulator + renderer) through the headless path (Chromium +
 * SwiftShader WebGPU): the low preset, paused and stepped by hand, pigment tapped in and milled.
 *
 *   node tools/app-shot.mjs <out.png> [frames before pigment = 40] [frames after = 120]
 *
 * SHOT_W / SHOT_H set the viewport (default 720x480). Slow: the simulator runs in software.
 */
import { writeFileSync } from 'node:fs';
import { withBrowser, encodePng } from '../tests/e2e/lib.mjs';

const [out, a = '40', b = '120'] = process.argv.slice(2);
if (!out) { console.error('usage: app-shot.mjs <out.png> [framesBefore] [framesAfter]'); process.exit(1); }
const W = Number(process.env.SHOT_W || 720), H = Number(process.env.SHOT_H || 480);

await withBrowser(async ({ page, baseUrl }) => {
  await page.setViewportSize({ width: W, height: H });
  await page.goto(`${baseUrl}?preset=low&paused=1${process.env.SHOT_QUERY ? '&' + process.env.SHOT_QUERY : ''}`, { waitUntil: 'load', timeout: 60000 });
  await page.waitForFunction(() => window.__colormill?.ready, null, { timeout: 180000 });
  const step = async (n) => { for (let i = 0; i < n; i += 10) await page.evaluate((k) => window.__colormill.stepFrames(k), Math.min(10, n - i)); };
  let t0 = Date.now();
  await step(Number(a));
  await page.evaluate(() => { window.__colormill.tapPigment('cobaltBlue'); window.__colormill.tapPigment('cadmiumYellow'); });
  await step(Number(b));
  console.log(`stepped ${Number(a) + Number(b)} frames in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  const shot = await page.evaluate(async () => {
    const s = await window.__colormill.screenshot();
    let bin = '';
    for (let i = 0; i < s.data.length; i += 0x8000) bin += String.fromCharCode.apply(null, Array.from(s.data.subarray(i, i + 0x8000)));
    return { width: s.width, height: s.height, base64: btoa(bin) };
  });
  writeFileSync(out, encodePng(shot.width, shot.height, new Uint8Array(Buffer.from(shot.base64, 'base64'))));
  console.log(`${out} (${shot.width}x${shot.height})`);
});
process.exit(0);

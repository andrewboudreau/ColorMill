/**
 * Stills of the renderer harness (procedural volumes, no simulator), through the project's
 * headless path (Chromium + SwiftShader WebGPU), for before/after comparisons of shader work.
 *
 *   node tools/render-shots.mjs <outDir> <suffix> [scene ...]
 *
 * writes <outDir>/<scene>-<suffix>.png for each scene below. SHOT_W / SHOT_H set the viewport
 * (default 720x480); SHOT_QUERY appends query parameters to every scene.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { withBrowser, encodePng } from '../tests/e2e/lib.mjs';

const SCENES = {
  // the bank with pigment on the mill, default view
  bank: { query: 'quality=medium', cam: {} },
  // three-quarter view, closer: bank, sheet, bare roll end, end plate, tray
  bank34: { query: 'quality=medium', cam: { yaw: 0.75, pitch: 0.42, distance: 2.2 } },
  // thin clear sheet on the front roll (no pigment anywhere), low preset thickness
  clear: { query: 'quality=low&sheet=1.7&clear=all', cam: { yaw: -0.35, pitch: 0.5, distance: 2.3 } },
  // clear sheet with a pigmented bank at medium
  clearsheet: { query: 'quality=medium&sheet=2.5&clear=sheet', cam: { yaw: 0.3, pitch: 0.3, distance: 2.0 } },
  // close-up of the bare end of the front roll
  roll: { query: 'quality=medium', cam: { yaw: 0.9, pitch: 0.28, distance: 1.05, targetEnd: true } },
  // noisy low-preset volume (what the simulator's density looks like)
  noisy: { query: 'quality=low&sheet=1.7&noise=0.15', cam: { yaw: 0.25, pitch: 0.55, distance: 2.4 } }
};

const [outDir, suffix, ...only] = process.argv.slice(2);
if (!outDir || !suffix) { console.error('usage: render-shots.mjs <outDir> <suffix> [scene ...]'); process.exit(1); }
mkdirSync(outDir, { recursive: true });
const W = Number(process.env.SHOT_W || 720), H = Number(process.env.SHOT_H || 480);

await withBrowser(async ({ page, baseUrl }) => {
  await page.setViewportSize({ width: W, height: H });
  for (const [name, scene] of Object.entries(SCENES)) {
    if (only.length && !only.includes(name)) continue;
    await page.goto(`${baseUrl}harness/render.html?${scene.query}${process.env.SHOT_QUERY ? '&' + process.env.SHOT_QUERY : ''}`);
    await page.waitForFunction(() => window.__render && (window.__render.ready || window.__render.error), null, { timeout: 180000 });
    const st = await page.evaluate(() => ({ ready: window.__render.ready, error: window.__render.error ?? null, v: window.__render.validationError ?? null, c: (window.__render.compileMessages ?? []).filter((m) => m.type === 'error') }));
    if (!st.ready || st.v || st.c.length) { console.log(name, 'FAILED', JSON.stringify(st)); continue; }
    const px = await page.evaluate(async (cam) => {
      const r = window.__render;
      const c = r.renderer.camera;
      for (const k of ['yaw', 'pitch', 'distance', 'fovY']) if (cam[k] !== undefined) c[k] = cam[k];
      if (cam.targetEnd) c.target = [c.target[0] * 2 - 0.12, c.target[1] - 0.1, c.target[2] + 0.3];
      await r.frame();
      const ms = r.lastFrameMs();
      return { ...(await r.readPixels()), ms };
    }, scene.cam);
    const file = path.join(outDir, `${name}-${suffix}.png`);
    writeFileSync(file, encodePng(px.width, px.height, new Uint8Array(Buffer.from(px.base64, 'base64'))));
    console.log(`${name}: ${file} (frame ${px.ms.toFixed(0)} ms)`);
  }
}, { mode: 'dev' });
process.exit(0);

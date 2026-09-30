/**
 * Renderer e2e: load the harness (procedural volumes), render a frame in
 * headless Chromium (SwiftShader WebGPU), save a screenshot and check that
 * the image is a lit scene with the pigment bands where the harness put them.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { ROOT, assert, withBrowser, encodePng } from './lib.mjs';

const OUT_DIR = path.join(ROOT, 'tests', 'e2e', 'out');



function pixel(img, x, y) {
  const i = (Math.round(y) * img.width + Math.round(x)) * 4;
  return [img.data[i], img.data[i + 1], img.data[i + 2]];
}

/** Mean colour of a (2r+1)^2 patch. */
function patchMean(img, cx, cy, r) {
  let R = 0, G = 0, B = 0, n = 0;
  for (let y = cy - r; y <= cy + r; y++) {
    for (let x = cx - r; x <= cx + r; x++) {
      if (x < 0 || y < 0 || x >= img.width || y >= img.height) continue;
      const p = pixel(img, x, y);
      R += p[0]; G += p[1]; B += p[2]; n++;
    }
  }
  return [R / n, G / n, B / n];
}

function luminance(p) {
  return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
}

export default async function run() {
  await withBrowser(async ({ page, baseUrl }) => {
    await page.goto(baseUrl + 'harness/render.html');
    await page.waitForFunction(() => window.__render && (window.__render.ready || window.__render.error), null, { timeout: 90000 });
    const state = await page.evaluate(() => ({
      ready: window.__render.ready,
      error: window.__render.error ?? null,
      validationError: window.__render.validationError ?? null,
      compileErrors: (window.__render.compileMessages ?? []).filter((m) => m.type === 'error')
    }));
    assert(state.ready, 'harness initialised: ' + state.error);
    assert(!state.validationError, 'no WebGPU validation error: ' + state.validationError);
    assert(state.compileErrors.length === 0, 'WGSL compiled: ' + JSON.stringify(state.compileErrors));

    // one timed frame (the first frame above already warmed the pipeline)
    const timings = await page.evaluate(async () => {
      const out = [];
      for (let i = 0; i < 3; i++) {
        const t0 = performance.now();
        await window.__render.frame();
        out.push(performance.now() - t0);
      }
      return out;
    });
    console.log('  frame times (ms):', timings.map((t) => t.toFixed(0)).join(', '));

    // exact pixels of the last frame (rendered offscreen: see harness/render.ts), saved as PNG
    const px = await page.evaluate(() => window.__render.readPixels());
    const img = { width: px.width, height: px.height, data: new Uint8Array(Buffer.from(px.base64, 'base64')) };
    assert(img.width === 960 && img.height === 640, `frame is 960x640 (got ${img.width}x${img.height})`);
    mkdirSync(OUT_DIR, { recursive: true });
    const shot = path.join(OUT_DIR, 'render.png');
    writeFileSync(shot, encodePng(img.width, img.height, img.data));
    console.log('  screenshot:', shot);

    // background reference: top-left corner (dark gradient)
    const bg = patchMean(img, 6, 6, 4);
    const centre = patchMean(img, img.width / 2, img.height / 2, 6);
    console.log('  background', bg.map(Math.round), 'centre', centre.map(Math.round));
    assert(Math.abs(luminance(centre) - luminance(bg)) > 25, 'centre of the image is not the background colour');

    // luminance variation across the scene: a lit surface has highlights and shadows
    const lums = [];
    for (let y = 40; y < img.height - 40; y += 16) {
      for (let x = 40; x < img.width - 40; x += 16) lums.push(luminance(pixel(img, x, y)));
    }
    lums.sort((a, b) => a - b);
    const p10 = lums[Math.floor(lums.length * 0.1)];
    const p90 = lums[Math.floor(lums.length * 0.9)];
    console.log('  luminance p10/p90:', p10.toFixed(0), p90.toFixed(0));
    assert(p90 - p10 > 40, 'there is luminance variation (lit surface)');

    // band colours: sample where the harness says the yellow / blue bands land on screen
    const spots = await page.evaluate(() => {
      const sp = window.__render.samplePoints;
      const out = {};
      for (const name of ['yellow', 'blue', 'red', 'white']) out[name] = sp[name].map((p) => window.__render.project(p));
      return out;
    });
    const sampled = {};
    for (const [name, pts] of Object.entries(spots)) {
      sampled[name] = pts.map((pt) => (pt ? patchMean(img, Math.round(pt.x), Math.round(pt.y), 3) : null));
      console.log(`  ${name}:`, sampled[name].map((c) => (c ? c.map(Math.round).join(',') : 'off-screen')).join('  '));
    }
    const yellowish = (c) => c && c[0] > 120 && c[1] > 100 && c[2] < Math.min(c[0], c[1]) * 0.6;
    const bluish = (c) => c && c[2] > c[0] * 1.3 && c[2] > c[1] * 1.15 && c[2] > 40;
    assert(sampled.yellow.some(yellowish), 'the yellow band reads as yellow');
    assert(sampled.blue.some(bluish), 'the blue band reads as blue');
    const reddish = (c) => c && c[0] > c[1] * 1.35 && c[0] > c[2] * 1.35 && c[0] > 120;
    assert(sampled.red.some(reddish), 'the red band reads as red');

    // low preset with a sheet as thin as the sim's (gap / h ~ 1.6 cells) and per-node density
    // noise: the march must not step over the thin sheet on the front roll, and the block
    // skipping (coarse max-density mip) must leave the surface intact
    await page.goto(baseUrl + 'harness/render.html?quality=low&sheet=1.7&noise=0.15');
    await page.waitForFunction(() => window.__render && (window.__render.ready || window.__render.error), null, { timeout: 90000 });
    const state2 = await page.evaluate(() => ({ ready: window.__render.ready, error: window.__render.error ?? null, validationError: window.__render.validationError ?? null }));
    assert(state2.ready && !state2.validationError, 'thin-sheet harness initialised: ' + (state2.error ?? state2.validationError));
    const t0 = performance.now();
    await page.evaluate(() => window.__render.frame());
    console.log('  low / thin sheet frame (ms):', (performance.now() - t0).toFixed(0));
    const px2 = await page.evaluate(() => window.__render.readPixels());
    const img2 = { width: px2.width, height: px2.height, data: new Uint8Array(Buffer.from(px2.base64, 'base64')) };
    writeFileSync(path.join(OUT_DIR, 'render-low-thin.png'), encodePng(img2.width, img2.height, img2.data));
    const spots2 = await page.evaluate(() => {
      const sp = window.__render.samplePoints;
      const out = {};
      for (const name of ['yellow', 'blue', 'red']) out[name] = window.__render.project(sp[name][0]); // [0] = on the sheet
      return out;
    });
    const sheet = {};
    for (const [name, pt] of Object.entries(spots2)) {
      sheet[name] = pt ? patchMean(img2, Math.round(pt.x), Math.round(pt.y), 3) : null;
      console.log(`  thin sheet ${name}:`, sheet[name] ? sheet[name].map(Math.round).join(',') : 'off-screen');
    }
    assert(yellowish(sheet.yellow), 'the yellow band on the thin sheet reads as yellow');
    assert(bluish(sheet.blue), 'the blue band on the thin sheet reads as blue');
    assert(reddish(sheet.red), 'the red band on the thin sheet reads as red');

    // --- colour modes on a layered sheet: a two-cell clear skin over a two-cell pigment core ---
    // surface mode (the default) draws the top-most material, so the sheet reads clear while the
    // bank (pigment to the top) stays yellow; depth mode blends a few cells in, so the sheet
    // still reads yellow through the skin
    const layered = {};
    for (const mode of ['surface', 'depth']) {
      await page.goto(`${baseUrl}harness/render.html?quality=medium&sheet=4&skin=2&render=${mode}`);
      await page.waitForFunction(() => window.__render && (window.__render.ready || window.__render.error), null, { timeout: 90000 });
      const st = await page.evaluate(() => ({ ready: window.__render.ready, error: window.__render.error ?? null, validationError: window.__render.validationError ?? null, mode: window.__render.mode }));
      assert(st.ready && !st.validationError && st.mode === mode, `layered harness initialised in ${mode} mode: ` + (st.error ?? st.validationError));
      await page.evaluate(() => window.__render.frame());
      const px3 = await page.evaluate(() => window.__render.readPixels());
      const img3 = { width: px3.width, height: px3.height, data: new Uint8Array(Buffer.from(px3.base64, 'base64')) };
      writeFileSync(path.join(OUT_DIR, `render-layered-${mode}.png`), encodePng(img3.width, img3.height, img3.data));
      const pts = await page.evaluate(() => { const sp = window.__render.samplePoints; return { sheet: window.__render.project(sp.yellow[0]), bank: window.__render.project(sp.yellow[1]) }; });
      layered[mode] = {
        sheet: pts.sheet ? patchMean(img3, Math.round(pts.sheet.x), Math.round(pts.sheet.y), 3) : null,
        bank: pts.bank ? patchMean(img3, Math.round(pts.bank.x), Math.round(pts.bank.y), 3) : null
      };
      console.log(`  layered (${mode}): sheet ${layered[mode].sheet ? layered[mode].sheet.map(Math.round).join(',') : 'off-screen'}, bank ${layered[mode].bank ? layered[mode].bank.map(Math.round).join(',') : 'off-screen'}`);
    }
    assert(layered.surface.sheet && layered.surface.bank && layered.depth.sheet, 'layered sample points are on screen');
    assert(!yellowish(layered.surface.sheet) && layered.surface.sheet[2] > 100, `surface mode: a clear skin over yellow reads clear, not yellow (${layered.surface.sheet.map(Math.round).join(',')})`);
    assert(yellowish(layered.surface.bank), 'surface mode: the yellow bank still reads yellow');
    assert(yellowish(layered.depth.sheet), 'depth mode: the same sheet still reads yellow through the skin');

    // --- coverage on a striped sheet: the sheet's pigment alternates along x in two-cell laminae
    // (pigment / clear / ...) whose edges cut through cells, so the samples in a clear stripe pick up
    // some pigment load (about 2 per unit mass, which the load alone draws near-opaque). Surface mode
    // draws by coverage, so the pigment stripe reads as the pigment and the clear stripe next to it
    // reads clear, instead of both reading as one yellow wash
    await page.goto(`${baseUrl}harness/render.html?quality=medium&sheet=3&laminae=2&render=surface`);
    await page.waitForFunction(() => window.__render && (window.__render.ready || window.__render.error), null, { timeout: 90000 });
    const st4 = await page.evaluate(() => ({ ready: window.__render.ready, error: window.__render.error ?? null, validationError: window.__render.validationError ?? null, mode: window.__render.mode, laminae: window.__render.laminae }));
    assert(st4.ready && !st4.validationError && st4.mode === 'surface' && st4.laminae === 2, 'striped harness initialised: ' + (st4.error ?? st4.validationError));
    await page.evaluate(() => window.__render.frame());
    const px4 = await page.evaluate(() => window.__render.readPixels());
    const img4 = { width: px4.width, height: px4.height, data: new Uint8Array(Buffer.from(px4.base64, 'base64')) };
    writeFileSync(path.join(OUT_DIR, 'render-laminae.png'), encodePng(img4.width, img4.height, img4.data));
    const lp = await page.evaluate(() => { const l = window.__render.laminaePoints; return { pigment: window.__render.project(l.pigment), clear: window.__render.project(l.clear) }; });
    assert(lp.pigment && lp.clear, 'both stripe sample points are on screen');
    const dx = Math.hypot(lp.pigment.x - lp.clear.x, lp.pigment.y - lp.clear.y);
    // a 5 x 5 px patch keeps the sample inside a stripe (about 15 px wide at this view)
    const stripe = { pigment: patchMean(img4, Math.round(lp.pigment.x), Math.round(lp.pigment.y), 2), clear: patchMean(img4, Math.round(lp.clear.x), Math.round(lp.clear.y), 2) };
    console.log(`  laminae (surface, ${dx.toFixed(0)} px apart): pigment stripe ${stripe.pigment.map(Math.round).join(',')}, clear stripe ${stripe.clear.map(Math.round).join(',')}`);
    assert(yellowish(stripe.pigment), 'striped sheet: the pigment stripe reads as yellow');
    assert(!yellowish(stripe.clear), 'striped sheet: the clear stripe next to it reads clear, not a yellow wash');
  }, { mode: 'dev' }); // harness pages are dev-only, not part of the production build
}

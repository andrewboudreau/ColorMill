// Dump the solver-grid density volume the renderer finds the putty surface on, after steady milling.
// usage: node tools/perf/density.mjs <out-prefix> [low high ...] [--seconds=6] [--clear]
// writes <out-prefix>-<preset>.f32 (nx*ny*nz floats, x fastest) and <out-prefix>-<preset>.json
// (dims, cell, gap, roll poses, iso), for offline analysis of the surface (scratch scripts).
import { writeFileSync } from 'node:fs';
import { launch, openApp } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
const [prefix, ...presets] = args.filter((a) => !a.startsWith('--'));
if (!prefix) { console.error('usage: density.mjs <out-prefix> [presets] [--seconds=6] [--clear]'); process.exit(1); }
if (!presets.length) presets.push('low', 'high');
const SECONDS = Number(opt('seconds', 6));

const ctx = await launch();
try {
  for (const preset of presets) {
    const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: 640, height: 400 });
    const spf = await page.evaluate(() => window.__colormill.stats().simSecondsPerFrame);
    const step = async (seconds) => { for (let left = Math.round(seconds / spf); left > 0; left -= 100) await page.evaluate((n) => window.__colormill.stepFrames(n), Math.min(100, left)); };
    await step(0.4);
    if (!args.includes('--clear')) await page.evaluate(() => window.__colormill.applyDrops([
      { pigment: 'naphtholRed', slot: 1, size: 'm' }, { pigment: 'phthaloBlue', slot: 3, size: 'm' }, { pigment: 'hansaYellow', slot: 4, size: 'm' }]));
    await step(SECONDS);
    const out = await page.evaluate(async () => {
      const sim = window.__colormill.sim;
      const d = await sim.readDensity();
      const u8 = new Uint8Array(d.buffer);
      let bin = '';
      for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode.apply(null, Array.from(u8.subarray(i, i + 0x8000)));
      const v = sim.volumes.dims;
      return { base64: btoa(bin), meta: { nx: v.nx, ny: v.ny, nz: v.nz, h: v.h, gap: sim.params.gap, iso: window.__colormill.renderer.iso } };
    });
    writeFileSync(`${prefix}-${preset}.f32`, Buffer.from(out.base64, 'base64'));
    writeFileSync(`${prefix}-${preset}.json`, JSON.stringify({ preset, ...out.meta }));
    console.log(`${prefix}-${preset}.f32`, JSON.stringify(out.meta));
    await page.close();
  }
} finally {
  await ctx.close();
}

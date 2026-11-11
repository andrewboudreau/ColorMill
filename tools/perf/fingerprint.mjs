// Fingerprint of a deterministic run plus wall-clock sim time: hashes of the particle snapshot after pigment,
// a cut & fold, a cut & roll, and pigment chunks set down (held, fed in, then let go by an operator move).
// A chunk's seeding jitter comes from Math.random, which is replaced in the page by a fixed sequence.
// usage: node tools/perf/fingerprint.mjs low high
import { launch, openApp } from './lib.mjs';
const presets = process.argv.slice(2);
const ctx = await launch();
try {
  for (const preset of presets) {
    const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: 640, height: 480 });
    page.on('pageerror', (e) => console.log('pageerror', e.message));
    const r = await page.evaluate(async () => {
      const api = window.__colormill;
      const out = {};
      let rs = 0x9e3779b9;
      Math.random = () => { rs ^= rs << 13; rs >>>= 0; rs ^= rs >>> 17; rs ^= rs << 5; rs >>>= 0; return rs / 4294967296; };
      const mark = async (label) => {
        const s = await api.snapshot();
        const h = {};
        for (const key of ['positions', 'velocities', 'loads', 'latents', 'deformation', 'flags', 'laminae', 'mixState']) {
          const d = await crypto.subtle.digest('SHA-256', s[key]);
          h[key] = Array.from(new Uint8Array(d).slice(0, 6), (x) => x.toString(16).padStart(2, '0')).join('');
        }
        const box = await api.sim.readColourBox();
        out[label] = { ...h, box: JSON.stringify(box) };
      };
      await api.stepFrames(30);
      api.sim.addPigment([0.5, 0.95, 0.75], 0.12, [1, 0, 0, 0, 0, 0, 0]);
      api.sim.addPigment([1.0, 0.95, 0.75], 0.12, [0, 1, 0, 0, 0, 0, 0]);
      await api.stepFrames(60);
      await mark('milled');
      api.sim.cutAndFlop(0);
      await api.stepFrames(120);
      await mark('fold');
      api.sim.cutAndFold();
      await api.stepFrames(300);
      await mark('roll');
      await api.applyDrops([{ pigment: 'naphtholRed', slot: 2, size: 'm' }, { pigment: 'phthaloBlue', slot: 4, size: 'l' }, { pigment: 'hansaYellow', slot: 5, size: 's' }]);
      await api.stepFrames(40);
      await mark('chunksHeld');
      api.sim.cutAndFlop('right');
      await api.stepFrames(120);
      await mark('chunksDropped');
      await api.sim.device?.queue?.onSubmittedWorkDone?.();
      const t = performance.now();
      await api.stepFrames(60);
      await api.snapshot();
      out.msPerFrame = ((performance.now() - t) / 60).toFixed(2);
      out.count = api.stats().particleCount;
      return out;
    });
    console.log(preset, JSON.stringify(r, null, 1));
    await page.close();
  }
} finally { await ctx.browser.close(); await ctx.server.stop?.(); }

// Pigment inventory on the real GPU: drop chunks, mill, inject, fold, and print how far the total pigment
// (sum of load x mass) and its colour (sum of load x mass x latent) drift from what was put in, plus the
// mixing metrics along the way.
// usage: node tools/perf/inventory.mjs [low high ...] [--mill=600] [--k=0.02] [--every=100]
import { launch, openApp } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? Number(a.split('=')[1]) : d; };
const presets = args.filter((a) => !a.startsWith('--'));
if (!presets.length) presets.push('low');
const MILL = opt('mill', 600), K = opt('k', NaN), EVERY = opt('every', 100);

const ctx = await launch();
try {
  for (const preset of presets) {
    const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: 640, height: 480 });
    const r = await page.evaluate(async ({ MILL, K, EVERY }) => {
      const api = window.__colormill;
      if (Number.isFinite(K)) api.sim.params.dispersion = K;
      const rows = [];
      const mark = async (label) => {
        const inv = await api.inventory();
        const mix = await api.mixing();
        rows.push({
          label, particles: inv.particles, pigmented: inv.pigmented, total: inv.totalLoad, channels: inv.channels,
          maxLoad: inv.maxLoad, minLoad: inv.minLoad,
          I: mix.scales.map((s) => +s.intensity.toFixed(4)), axial: +mix.axial.toFixed(4)
        });
      };
      await api.stepFrames(30);
      await mark('rest');
      const n0 = api.stats().particleCount;
      await api.applyDrops([
        { pigment: 'naphtholRed', slot: 2, size: 'm' }, { pigment: 'phthaloBlue', slot: 4, size: 'l' }, { pigment: 'hansaYellow', slot: 5, size: 'm' }]);
      const added = api.stats().particleCount - n0;
      await mark('dropped');
      for (let f = 0; f < MILL; f += EVERY) {
        await api.stepFrames(Math.min(EVERY, MILL - f));
        await mark(`mill ${Math.min(f + EVERY, MILL)}`);
      }
      api.sim.cutAndFlop('left');
      await api.stepFrames(150);
      await mark('fold');
      api.sim.cutAndFold();
      await api.stepFrames(300);
      await mark('roll');
      return { k: api.sim.params.dispersion, added, rows };
    }, { MILL, K, EVERY });
    const base = r.rows.find((x) => x.label === 'dropped');
    const rel = (v, v0) => (v0 !== 0 ? (v - v0) / v0 : v);
    console.log(`${preset}: dispersion k=${r.k}, ${r.added} chunk particles, expected load ${12 * r.added}, read ${base.total.toFixed(3)}`);
    for (const row of r.rows) {
      const worst = Math.max(...row.channels.slice(0, 4).map((c, i) => Math.abs(c - base.channels[i]))) / Math.max(base.total, 1e-9);
      console.log(`  ${row.label.padEnd(10)} n=${row.particles} pigmented=${row.pigmented} total=${row.total.toFixed(3)} drift=${(100 * rel(row.total, base.total)).toFixed(4)}% ` +
        `colour drift=${(100 * worst).toFixed(4)}% load ${row.minLoad.toExponential(1)}..${row.maxLoad.toFixed(2)} I(h,2h,4h)=${row.I.join(', ')} axial=${row.axial}`);
    }
    await page.close();
  }
} finally { await ctx.close(); }

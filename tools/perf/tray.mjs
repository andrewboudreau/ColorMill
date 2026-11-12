// How much material, and how much of the pigment, ends up on the tray in steady milling.
// The recipe is mixcurve's (a large chunk among them: the piece that falls off the fast roll).
// usage: node tools/perf/tray.mjs [low high ...] [--seconds=60] [--every=20] [--ratios=1,1.6]
import { launch, openApp } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
const presets = args.filter((a) => !a.startsWith('--'));
if (!presets.length) presets.push('low', 'high');
const SECONDS = Number(opt('seconds', 60)), EVERY = Number(opt('every', 20));
const RATIOS = opt('ratios', '1,1.6').split(',').map(Number);

const ctx = await launch();
try {
  console.log('| preset | ratio | t (s) | particles | on tray | share of the pigment on the tray |');
  console.log('|---|---|---|---|---|---|');
  for (const preset of presets) for (const ratio of RATIOS) {
    const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: 1280, height: 800, query: `&frictionRatio=${ratio}` });
    const spf = await page.evaluate(() => window.__colormill.stats().simSecondsPerFrame);
    const step = async (seconds) => { for (let left = Math.round(seconds / spf); left > 0; left -= 100) await page.evaluate((n) => window.__colormill.stepFrames(n), Math.min(100, left)); };
    await page.evaluate(() => window.__colormill.stepFrames(30));
    await page.evaluate(() => window.__colormill.applyDrops([
      { pigment: 'naphtholRed', slot: 2, size: 'm' }, { pigment: 'phthaloBlue', slot: 4, size: 'l' }, { pigment: 'hansaYellow', slot: 5, size: 'm' }]));
    for (let t = EVERY; t <= SECONDS; t += EVERY) {
      await step(EVERY);
      const r = await page.evaluate(async () => {
        const s = window.__colormill.sim;
        const d = await window.__colormill.nipDiagnostics(2);
        const p = await s.readParticles();
        const h = s.stats.grid.h, n = p.count;
        let load = 0, low = 0;
        for (let i = 0; i < n; i++) { const w = Math.max(p.loads[i], 0); load += w; if (p.positions[3 * i + 1] < 3 * h) low += w; }
        return { n, tray: d.tray, share: low / Math.max(load, 1e-9) };
      });
      console.log(`| ${preset} | ${ratio} | ${t} | ${r.n} | ${r.tray} | ${r.share.toFixed(3)} |`);
    }
    await page.close();
  }
} finally { await ctx.close(); }

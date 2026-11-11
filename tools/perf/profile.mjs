// v3 perf review: per-pass GPU time for each preset, resting bank and milling-with-pigment.
// usage: node tools/perf/profile.mjs [low medium high ...] [--runs=2] [--mill=600] [--frames=60]
import { writeFileSync, mkdirSync } from 'node:fs';
import { launch, openApp, summarise } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? Number(a.split('=')[1]) : d; };
const presets = args.filter((a) => !a.startsWith('--'));
if (!presets.length) presets.push('low', 'medium', 'high');
const RUNS = opt('runs', 2), MILL = opt('mill', 600), FRAMES = opt('frames', 60);
const W = opt('w', 1920), H = opt('h', 1080);

const ctx = await launch();
const results = [];
try {
  for (const preset of presets) {
    for (let run = 0; run < RUNS; run++) {
      const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: W, height: H });
      const measureState = async (label) => {
        const state = await page.evaluate(() => window.__prof.state());
        // warm-up, then: split (per kernel), whole (as shipped: one pass), wall clock with and without render
        await page.evaluate(() => window.__prof.measure(8));
        const split = await page.evaluate((n) => window.__prof.measure(n, { mode: 'split' }), FRAMES);
        const whole = await page.evaluate((n) => window.__prof.measure(n, { mode: 'whole' }), FRAMES);
        const wallSim = await page.evaluate((n) => window.__prof.wall(n, { render: false }), FRAMES);
        const wallAll = await page.evaluate((n) => window.__prof.wall(n, { render: true }), FRAMES);
        const r = { preset, run, label, state, split: summarise(split), whole: summarise(whole), wallSim, wallAll };
        results.push(r);
        const c = r.split.cats;
        const f = (x) => (x?.mean ?? 0).toFixed(2);
        console.log(`${preset} run${run} ${label}: n=${state.particles} box=${state.gridBox} fineK=${state.fineK} fineRange=${state.fineRange} canvas=${state.canvas}`);
        console.log(`  split sum ${f(r.split.sum)} ms (sd ${r.split.sum.sd.toFixed(2)}) | whole sim ${f(r.whole.cats['sim-whole'])} render ${f(r.whole.cats['render.raymarch'])} | wall sim ${wallSim.msPerFrame.toFixed(2)} all ${wallAll.msPerFrame.toFixed(2)} (encode ${wallAll.encodeMsPerFrame.toFixed(2)})`);
        console.log('  ' + Object.entries(c).map(([k, v]) => `${k} ${v.mean.toFixed(2)}`).join(' | '));
      };
      await page.evaluate(() => window.__colormill.stepFrames(20));
      await measureState('rest');
      await page.evaluate(() => window.__colormill.applyDrops([
        { pigment: 'naphtholRed', slot: 2, size: 'm' }, { pigment: 'phthaloBlue', slot: 4, size: 'l' }, { pigment: 'hansaYellow', slot: 5, size: 'm' }]));
      // mill: step in slices so the page stays responsive
      for (let left = MILL; left > 0; left -= 100) await page.evaluate((n) => window.__colormill.stepFrames(n), Math.min(100, left));
      await measureState('mill');
      await page.close();
    }
  }
} finally {
  await ctx.close();
}
mkdirSync(new URL('./out/', import.meta.url), { recursive: true });
const file = new URL(`./out/profile-${presets.join('-')}-${W}x${H}.json`, import.meta.url);
writeFileSync(file, JSON.stringify(results, null, 1));
console.log('wrote', file.pathname);

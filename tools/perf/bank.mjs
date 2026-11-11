// Does the bank roll, and are dropped pigment chunks drawn into the nip? Mills each preset on the real GPU,
// drops three medium chunks on the bank at 0.4 s (as still.mjs does) and follows the chunks' own particles:
// the share that has been through the nip (seen below the roll axes) against time since the drop. Then the
// nip diagnostics, for the bank's speed and size next to the sheet's count and flux.
// usage: node tools/perf/bank.mjs [low medium high] [--seconds=8] [--query=&x=y] [--label=text]
import { launch, openApp } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
const presets = args.filter((a) => !a.startsWith('--'));
if (!presets.length) presets.push('low', 'medium', 'high');
const SECONDS = Number(opt('seconds', 8));
const QUERY = opt('query', '');
const AXIS_Y = 0.55;   // GEOMETRY.axisY

const ctx = await launch();
const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a');
try {
  console.log(`## ${opt('label', QUERY || 'defaults')}`);
  console.log('| preset | bank back / front (/U) | bank n (share) | count | flux | thick | resets/s | drawn in 25% / 50% / 75% (s) | drawn at 3 s / 6 s / end | tray |');
  for (const preset of presets) {
    const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: 640, height: 480, query: QUERY });
    const spf = await page.evaluate(() => window.__colormill.stats().simSecondsPerFrame);
    const step = async (seconds) => { for (let left = Math.round(seconds / spf); left > 0; left -= 100) await page.evaluate((n) => window.__colormill.stepFrames(n), Math.min(100, left)); };
    await step(0.4);
    const n0 = await page.evaluate(() => window.__colormill.sim.stats.particleCount);
    await page.evaluate(() => window.__colormill.applyDrops([
      { pigment: 'naphtholRed', slot: 1, size: 'm' }, { pigment: 'phthaloBlue', slot: 3, size: 'm' }, { pigment: 'hansaYellow', slot: 4, size: 'm' }]));
    const curve = [];
    const every = Math.max(1, Math.round(0.1 / spf));
    for (let t = 0; t < SECONDS; ) {
      await page.evaluate((n) => window.__colormill.stepFrames(n), every);
      t += every * spf;
      const share = await page.evaluate(async ({ n0, ay }) => {
        const s = await window.__colormill.snapshot();
        const seen = (window.__drawn ??= new Uint8Array(s.count));
        let passed = 0;
        for (let p = n0; p < s.count; p++) {
          if (s.positions[3 * p + 1] < ay) seen[p] = 1;
          passed += seen[p];
        }
        return passed / Math.max(s.count - n0, 1);
      }, { n0, ay: AXIS_Y });
      curve.push([t, share]);
    }
    const when = (q) => { const c = curve.find(([, s]) => s >= q); return c ? f(c[0], 1) : `>${SECONDS}`; };
    const at = (t) => { const c = curve.filter(([u]) => u <= t + 1e-6).pop(); return c ? f(c[1]) : 'n/a'; };
    const win = [];
    for (let i = 0; i < 3; i++) win.push(await page.evaluate(() => window.__colormill.nipDiagnostics(20)));
    const mean = (g) => win.reduce((a, w) => a + g(w), 0) / win.length;
    const n = await page.evaluate(() => window.__colormill.sim.stats.particleCount);
    const bankN = mean((w) => w.bank.n);
    console.log(`| ${preset} | ${f(mean((w) => w.bankBackSpeed))} / ${f(mean((w) => w.bankFrontSpeed))} | ${f(bankN, 0)} (${f(bankN / n)}) | ${f(mean((w) => w.sheetCount))} | ${f(mean((w) => w.flux))} | ${f(mean((w) => w.thicknessGaps))} | ${f(mean((w) => w.resetsPerSecond), 0)} | ${when(0.25)} / ${when(0.5)} / ${when(0.75)} | ${at(3)} / ${at(6)} / ${f(curve[curve.length - 1][1])} | ${Math.max(...win.map((w) => w.tray))} |`);
    await page.close();
  }
} finally {
  await ctx.close();
}

// Nip diagnostics (physics review 4.5): mill a default batch for a fixed time at each preset on the real GPU
// and print what the nip does to the material's volume. Averages `--windows` windows of `--frames` frames
// after `--settle` sim seconds; with --moves it then makes a cut & fold and a cut & roll and reports
// whether the material came through them.
// usage: node tools/perf/nip.mjs [low medium high] [--settle=5] [--windows=5] [--frames=20] [--moves] [--json=out.json] [--query=&x=y] [--dt-scale=1]
// --dt-scale multiplies the preset's time step (substeps unchanged): how far the explicit step is from its stability limit
import { writeFileSync } from 'node:fs';
import { launch, openApp } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
const presets = args.filter((a) => !a.startsWith('--'));
if (!presets.length) presets.push('low', 'medium', 'high');
const SETTLE = Number(opt('settle', 5)), WINDOWS = Number(opt('windows', 5)), FRAMES = Number(opt('frames', 20));
const QUERY = opt('query', '');
const MOVES = args.includes('--moves');
const DT_SCALE = Number(opt('dt-scale', 1));

const ctx = await launch();
const rows = [];
try {
  for (const preset of presets) {
    const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: 640, height: 480, query: QUERY });
    if (DT_SCALE !== 1) await page.evaluate((k) => { const s = window.__colormill.sim; s.quality = { ...s.quality, dt: s.quality.dt * k }; }, DT_SCALE);
    const spf = (await page.evaluate(() => window.__colormill.stats().simSecondsPerFrame)) * DT_SCALE;
    const step = async (frames) => { for (let left = frames; left > 0; left -= 100) await page.evaluate((n) => window.__colormill.stepFrames(n), Math.min(100, left)); };
    await step(Math.round(SETTLE / spf));
    const win = [];
    for (let i = 0; i < WINDOWS; i++) win.push(await page.evaluate((n) => window.__colormill.nipDiagnostics(n), FRAMES));
    const info = await page.evaluate(() => { const s = window.__colormill.sim; return { n: s.stats.particleCount, t: s.stats.simTime, gap: s.params.gap, h: s.stats.grid.h }; });
    const mean = (f) => win.reduce((a, w) => a + f(w), 0) / win.length;
    const max = (f) => Math.max(...win.map(f));
    const reg = (k) => ({
      n: mean((w) => w[k].n),
      J: [mean((w) => w[k].J.mean), mean((w) => w[k].J.p5), mean((w) => w[k].J.p95)],
      rho: [mean((w) => w[k].rho.mean), mean((w) => w[k].rho.p5), mean((w) => w[k].rho.p95)],
      rhoGhost: [mean((w) => w[k].rhoGhost.mean), mean((w) => w[k].rhoGhost.p5), mean((w) => w[k].rhoGhost.p95)]
    });
    const row = {
      preset, ...info, resets: mean((w) => w.resetsPerSecond), pushOuts: mean((w) => w.pushOutsPerSecond),
      sheet: reg('sheet'), bank: reg('bank'), sheetCount: mean((w) => w.sheetCount), bankBack: mean((w) => w.bankBackSpeed), bankFront: mean((w) => w.bankFrontSpeed), flux: mean((w) => w.flux),
      thickness: mean((w) => w.thicknessGaps), vmax: max((w) => w.maxSpeed), vOverLimit: max((w) => w.maxSpeedOverLimit),
      courant: max((w) => w.courant), tray: max((w) => w.tray), Jmin: Math.min(...win.map((w) => w.Jmin)), Jmax: max((w) => w.Jmax)
    };
    if (MOVES) {
      // a cut & fold (two passes), then a cut & roll; the fastest particle and the tray count through each,
      // and the sheet a few seconds after
      const watch = async (seconds) => {
        let vmax = 0, tray = 0, resets = 0;
        for (let t = 0; t < seconds; t += 20 * spf) {
          const d = await page.evaluate(() => window.__colormill.nipDiagnostics(20));
          vmax = Math.max(vmax, d.maxSpeedOverLimit); tray = Math.max(tray, d.tray); resets += d.resetsPerSecond * d.seconds;
        }
        return { vmax, tray, resets };
      };
      await page.evaluate(() => window.__colormill.sim.cutAndFlop('left'));
      const fold = await watch(3);
      await page.evaluate(() => window.__colormill.sim.cutAndFold());
      const roll = await watch(6);
      await step(Math.round(3 / spf));
      const after = await page.evaluate((n) => window.__colormill.nipDiagnostics(n), FRAMES);
      row.moves = { fold, roll, after: { sheetCount: after.sheetCount, flux: after.flux, vmax: after.maxSpeedOverLimit, tray: after.tray, bankN: after.bank.n } };
    }
    rows.push(row);
    await page.close();
  }
} finally {
  await ctx.close();
}
const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a');
const tri = (v, d = 2) => `${f(v[0], d)} [${f(v[1], d)}, ${f(v[2], d)}]`;
const lines = [
  ['preset', ...rows.map((r) => `${r.preset} (${r.n}, gap ${(r.gap / r.h).toFixed(2)} cells)`)],
  ['F resets /s', ...rows.map((r) => f(r.resets, 0))],
  ['push-outs /s', ...rows.map((r) => f(r.pushOuts, 0))],
  ['push-outs /s /particle', ...rows.map((r) => f(r.pushOuts / r.n, 3))],
  ['sheet J mean [p5, p95]', ...rows.map((r) => tri(r.sheet.J, 3))],
  ['sheet density (ghost)', ...rows.map((r) => tri(r.sheet.rhoGhost))],
  ['sheet density (raw gather)', ...rows.map((r) => tri(r.sheet.rho))],
  ['bank J mean [p5, p95]', ...rows.map((r) => tri(r.bank.J, 3))],
  ['bank density (ghost)', ...rows.map((r) => tri(r.bank.rhoGhost))],
  ['bank density (raw gather)', ...rows.map((r) => tri(r.bank.rho))],
  ['bank particles', ...rows.map((r) => f(r.bank.n, 0))],
  ['bank speed back / front half (/ U)', ...rows.map((r) => `${f(r.bankBack)} / ${f(r.bankFront)}`)],
  ['sheet count / gap-thick rest sheet', ...rows.map((r) => f(r.sheetCount))],
  ['nip flux / (U gap rho0 L)', ...rows.map((r) => f(r.flux))],
  ['sheet thickness p90 / gap', ...rows.map((r) => f(r.thickness))],
  ['max |v| (/ 2 omega R)', ...rows.map((r) => `${f(r.vmax)} (${f(r.vOverLimit)})`)],
  ['Courant (c + max|v|) dt / h', ...rows.map((r) => f(r.courant))],
  ['J min / max, all particles', ...rows.map((r) => `${f(r.Jmin)} / ${f(r.Jmax)}`)],
  ['on the tray', ...rows.map((r) => String(r.tray))]
];
if (MOVES) {
  const mv = (m) => `v ${f(m.vmax)}, tray ${m.tray}, resets ${f(m.resets, 0)}`;
  lines.push(['cut & fold: max v / 2 omega R, tray', ...rows.map((r) => mv(r.moves.fold))]);
  lines.push(['cut & roll: same', ...rows.map((r) => mv(r.moves.roll))]);
  lines.push(['3 s after: count, flux, tray', ...rows.map((r) => `${f(r.moves.after.sheetCount)}, ${f(r.moves.after.flux)}, ${r.moves.after.tray}`)]);
}
console.log(`milled ${SETTLE} s, then ${WINDOWS} windows of ${FRAMES} frames${QUERY ? `, query ${QUERY}` : ''}${DT_SCALE !== 1 ? `, dt x ${DT_SCALE}` : ''}`);
const w0 = Math.max(...lines.map((l) => l[0].length));
for (const l of lines) console.log('| ' + l[0].padEnd(w0) + ' | ' + l.slice(1).map((c) => c.padEnd(28)).join(' | ') + ' |');
const json = opt('json', '');
if (json) writeFileSync(json, JSON.stringify(rows, null, 1));

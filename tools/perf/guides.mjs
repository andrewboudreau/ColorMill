// The side guides sliding in on a running mill: mills each preset at full width with three pigment chunks, then
// narrows the guides through the debug API and follows the putty. Prints where the material is along the roll
// inside the plates' extent and outside it (the sheet round the front of the roll is not confined), and writes
// stills before, during and after.
// usage: node tools/perf/guides.mjs <out-prefix> [low high ...] [--width=0.6] [--before=10] [--after=10]
import { writeFileSync } from 'node:fs';
import { launch, openApp } from './lib.mjs';
import { encodePng } from '../../tests/e2e/lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
const [prefix, ...presets] = args.filter((a) => !a.startsWith('--'));
if (!prefix) { console.error('usage: guides.mjs <out-prefix> [presets] [--width=0.6] [--before=10] [--after=10]'); process.exit(1); }
if (!presets.length) presets.push('low', 'high');
const WIDTH = Number(opt('width', 0.6));
const BEFORE = Number(opt('before', 10));
const AFTER = Number(opt('after', 10));
const LENGTH = 1.5, AXIS_Y = 0.55, NIP_Z = 0.75, RADIUS = 0.32;   // GEOMETRY

const ctx = await launch();
const f = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a');
try {
  for (const preset of presets) {
    const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: 1280, height: 800, query: opt('query', '') });
    const spf = await page.evaluate(() => window.__colormill.stats().simSecondsPerFrame);
    const step = async (seconds) => { for (let left = Math.round(seconds / spf); left > 0; left -= 100) await page.evaluate((n) => window.__colormill.stepFrames(n), Math.min(100, left)); };
    const shoot = async (name) => {
      const shot = await page.evaluate(async () => {
        const s = await window.__colormill.screenshot();
        let bin = '';
        for (let i = 0; i < s.data.length; i += 0x8000) bin += String.fromCharCode.apply(null, Array.from(s.data.subarray(i, i + 0x8000)));
        return { width: s.width, height: s.height, base64: btoa(bin) };
      });
      const file = `${prefix}-${preset}-${name}.png`;
      writeFileSync(file, encodePng(shot.width, shot.height, new Uint8Array(Buffer.from(shot.base64, 'base64'))));
      console.log(file);
    };
    // the material's reach along the roll: inside the plates' extent (the wedge between the rolls) and everywhere
    const reach = (label) => page.evaluate(async ({ label, L, ay, nz, R }) => {
      const api = window.__colormill;
      const s = await api.snapshot();
      const open = api.stats().guideOpen;
      const gap = api.sim.params.gap;
      const x0 = 0.5 * L * (1 - open), x1 = 0.5 * L * (1 + open);
      const zB = nz - R - gap / 2 - 0.06, zF = nz + R + gap / 2 + 0.06;
      let inMin = Infinity, inMax = -Infinity, inN = 0, inOut = 0, allOut = 0, bad = 0;
      for (let p = 0; p < s.count; p++) {
        const x = s.positions[3 * p], y = s.positions[3 * p + 1], z = s.positions[3 * p + 2];
        if (!Number.isFinite(x + y + z)) { bad++; continue; }
        const out = x < x0 || x > x1;
        if (out) allOut++;
        if (y > ay - 0.06 && z > zB && z < zF) {
          inN++;
          if (out) inOut++;
          if (x < inMin) inMin = x;
          if (x > inMax) inMax = x;
        }
      }
      const c = await api.sim.readCounters();
      return { label, t: api.stats().simTime, open, x0, x1, inMin, inMax, inN, inOut, allOut, count: s.count, bad, resets: c.resets };
    }, { label, L: LENGTH, ay: AXIS_Y, nz: NIP_Z, R: RADIUS });
    const report = (r) => console.log(`| ${preset} | ${r.label} | ${f(r.t, 1)} | ${f(r.open, 2)} | ${f(r.x0)} .. ${f(r.x1)} | ${f(r.inMin)} .. ${f(r.inMax)} | ${r.inOut} / ${r.inN} | ${r.allOut} / ${r.count} | ${r.bad} | ${r.resets} |`);

    console.log('| preset | when | t (s) | open | plates x | material x in the plates\' extent | outside the plates there | outside anywhere | non-finite | F resets |');
    await step(0.4);
    await page.evaluate(() => window.__colormill.applyDrops([
      { pigment: 'naphtholRed', slot: 1, size: 'm' }, { pigment: 'phthaloBlue', slot: 3, size: 'm' }, { pigment: 'hansaYellow', slot: 4, size: 'm' }]));
    await step(BEFORE);
    report(await reach('full width'));
    await shoot('0-before');
    await page.evaluate((w) => { window.__colormill.sim.params.guideWidth = w; }, WIDTH);
    await step(1);
    report(await reach('closing, 1 s'));
    await shoot('1-closing');
    await step(1.5);
    report(await reach('2.5 s'));
    await shoot('2-closed');
    await step(AFTER - 2.5);
    report(await reach(`${AFTER} s`));
    await shoot('3-after');
    await page.close();
  }
} finally {
  await ctx.close();
}

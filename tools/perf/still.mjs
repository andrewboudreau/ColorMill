// A still of the real app on the real GPU after steady milling with pigment (and optionally the operator moves).
// usage: node tools/perf/still.mjs <out-prefix> [low high ...] [--seconds=6] [--moves] [--query=&x=y]
// writes <out-prefix>-<preset>.png (and -fold / -roll stills with --moves)
import { writeFileSync } from 'node:fs';
import { launch, openApp } from './lib.mjs';
import { encodePng } from '../../tests/e2e/lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
const [prefix, ...presets] = args.filter((a) => !a.startsWith('--'));
if (!prefix) { console.error('usage: still.mjs <out-prefix> [presets] [--seconds=6] [--moves]'); process.exit(1); }
if (!presets.length) presets.push('low', 'high');
const SECONDS = Number(opt('seconds', 6));

const ctx = await launch();
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
      const file = `${prefix}-${name}.png`;
      writeFileSync(file, encodePng(shot.width, shot.height, new Uint8Array(Buffer.from(shot.base64, 'base64'))));
      console.log(file);
    };
    await step(0.4);
    await page.evaluate(() => window.__colormill.applyDrops([
      { pigment: 'naphtholRed', slot: 1, size: 'm' }, { pigment: 'phthaloBlue', slot: 3, size: 'm' }, { pigment: 'hansaYellow', slot: 4, size: 'm' }]));
    await step(SECONDS);
    await shoot(preset);
    if (args.includes('--moves')) {
      await page.evaluate(() => window.__colormill.sim.cutAndFlop('left'));
      await step(0.5);
      await shoot(`${preset}-fold`);
      await step(3);
      await page.evaluate(() => window.__colormill.sim.cutAndFold());
      await step(2.2);
      await shoot(`${preset}-roll`);
      await step(6);
      await shoot(`${preset}-after`);
    }
    await page.close();
  }
} finally {
  await ctx.close();
}

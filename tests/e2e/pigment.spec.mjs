/**
 * Pigment conservation, through the app's DebugApi (`window.__colormill.inventory()`):
 *   1. a fresh batch holds no pigment at all, and milling it creates none
 *   2. a dropped chunk adds exactly its particles x PIGMENT_LOAD, in its own colour
 *   3. milling with no further drops keeps the total load and every load-weighted latent channel
 *      constant to float round-off (dispersion turned well up, a chunk still held and feeding in)
 *
 * Runs at ?preset=low on SwiftShader; PIGMENT_FRAMES overrides the milling length.
 */
import { assert, withBrowser } from './lib.mjs';

const PIGMENT_LOAD = 12; // mirrors PIGMENT_LOAD in src/sim/shaders/common.wgsl
const FRAMES = Number(process.env.PIGMENT_FRAMES) || 60;
const CHUNK = 20; // frames per stepFrames call
/** relative drift allowed: f32 round-off per particle and frame, far under any real leak (the old kernel: 1e-2 and up) */
const TOLERANCE = 2e-5;

const RED = [0.1, 0.7, 0.2, 0, 0.05, -0.04, 0.02];
const BLUE = [0, 0.2, 0, 0.8, -0.03, 0.01, 0.06];

export default async function run() {
  await withBrowser(async ({ page, baseUrl }) => {
    await page.goto(`${baseUrl}?preset=low&paused=1`, { waitUntil: 'load', timeout: 60_000 });
    await page.waitForFunction(() => window.__colormill && window.__colormill.ready, null, { timeout: 180_000 });
    const step = async (n) => { for (let left = n; left > 0; left -= CHUNK) await page.evaluate((k) => window.__colormill.stepFrames(k), Math.min(CHUNK, left)); };
    const inventory = () => page.evaluate(() => window.__colormill.inventory());

    // 1: nothing from nothing
    await step(10);
    const rest = await inventory();
    assert(rest.particles > 0, 'the batch has particles');
    assert(rest.totalLoad === 0 && rest.pigmented === 0 && rest.maxLoad === 0,
      `a clear batch holds no pigment after milling (total ${rest.totalLoad}, ${rest.pigmented} particles, max ${rest.maxLoad})`);

    // 2: a drop adds what it holds
    const added = await page.evaluate(async ({ RED, BLUE }) => {
      const sim = window.__colormill.sim;
      sim.params.dispersion = 0.3;
      const a = await sim.addPigmentChunk(0.45, 0.75, 0.07, RED);
      const b = await sim.addPigmentChunk(1.0, 0.75, 0.09, BLUE);
      return [a, b];
    }, { RED, BLUE });
    assert(added[0] > 0 && added[1] > 0, `both chunks were added (${added})`);
    const dropped = await inventory();
    const expectTotal = PIGMENT_LOAD * (added[0] + added[1]);
    const close = (a, b, tol, scale) => Math.abs(a - b) <= tol * Math.max(Math.abs(scale), 1e-9);
    assert(dropped.particles === rest.particles + added[0] + added[1], 'the chunk particles are in the snapshot');
    assert(dropped.pigmented === added[0] + added[1], `only the chunks carry pigment (${dropped.pigmented})`);
    assert(close(dropped.totalLoad, expectTotal, 1e-7, expectTotal), `a drop adds its particles x ${PIGMENT_LOAD}: ${dropped.totalLoad} vs ${expectTotal}`);
    for (let c = 0; c < 7; c++) {
      const want = PIGMENT_LOAD * (added[0] * RED[c] + added[1] * BLUE[c]);
      assert(close(dropped.channels[c], want, 1e-6, expectTotal), `channel ${c} of the drop: ${dropped.channels[c]} vs ${want}`);
    }

    // 3: milling moves pigment about and neither makes nor loses any
    let worst = 0;
    let last = dropped;
    for (let f = 0; f < FRAMES; f += CHUNK) {
      await step(Math.min(CHUNK, FRAMES - f));
      last = await inventory();
      assert(last.particles === dropped.particles, 'no particle was lost');
      const dTotal = Math.abs(last.totalLoad - dropped.totalLoad) / dropped.totalLoad;
      let dColour = 0;
      for (let c = 0; c < 7; c++) dColour = Math.max(dColour, Math.abs(last.channels[c] - dropped.channels[c]) / dropped.totalLoad);
      worst = Math.max(worst, dTotal, dColour);
      assert(dTotal <= TOLERANCE, `total load drifted ${dTotal.toExponential(2)} after ${f + CHUNK} frames (${last.totalLoad} vs ${dropped.totalLoad})`);
      assert(dColour <= TOLERANCE, `load-weighted latent drifted ${dColour.toExponential(2)} after ${f + CHUNK} frames`);
      assert(last.minLoad >= 0 && last.maxLoad <= PIGMENT_LOAD * (1 + 1e-5), `loads stay in 0..${PIGMENT_LOAD} (${last.minLoad}..${last.maxLoad})`);
    }
    // the check means something only if pigment did move
    assert(last.pigmented > dropped.pigmented, `dispersion spread the pigment to other particles (${last.pigmented} vs ${dropped.pigmented})`);
    console.log(`pigment: ${added[0]} + ${added[1]} chunk particles, load ${dropped.totalLoad}; after ${FRAMES} frames ${last.pigmented} particles carry it, worst drift ${worst.toExponential(2)}`);
  });
}

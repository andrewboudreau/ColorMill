/** Rendering invalidation and resolution controls, exercised on the real GPU app. */
import { assert, withBrowser } from './lib.mjs';

export default async function run() {
  await withBrowser(async ({ page, baseUrl }) => {
    await page.setViewportSize({ width: 400, height: 300 });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    await page.addInitScript(() => {
      window.__work = { draws: 0, mips: 0, submits: 0, textures: [] };
      const render = GPUCommandEncoder.prototype.beginRenderPass;
      GPUCommandEncoder.prototype.beginRenderPass = function (d) {
        if (d.label === 'raymarch-pass') window.__work.draws++;
        return render.call(this, d);
      };
      const compute = GPUCommandEncoder.prototype.beginComputePass;
      GPUCommandEncoder.prototype.beginComputePass = function (d) {
        if (d?.label === 'raymarch-density-mip') window.__work.mips++;
        return compute.call(this, d);
      };
      const submit = GPUQueue.prototype.submit;
      GPUQueue.prototype.submit = function (commands) {
        window.__work.submits++;
        return submit.call(this, commands);
      };
      const texture = GPUDevice.prototype.createTexture;
      GPUDevice.prototype.createTexture = function (d) {
        window.__work.textures.push(d.label);
        return texture.call(this, d);
      };
    });
    await page.goto(baseUrl + '?preset=low&paused=1');
    await page.waitForFunction(() => window.__colormill?.ready, null, { timeout: 120000 });
    const work = () => page.evaluate(() => ({ ...window.__work }));
    const settle = () => page.evaluate(async () => {
      for (let i = 0; i < 5; i++) await new Promise(requestAnimationFrame);
    });
    await settle();
    const initial = await work();
    await settle();
    const idle = await work();
    assert(idle.draws === initial.draws && idle.submits === initial.submits, 'paused app submits no unchanged frames');
    assert(!idle.textures.includes('volB') && !idle.textures.includes('volC'), 'unused coarse colour textures are not allocated');

    await page.evaluate(() => { window.__colormill.renderer.camera.yaw += 0.1; });
    await settle();
    const orbit = await work();
    assert(orbit.draws > idle.draws && orbit.mips === idle.mips, 'camera changes redraw without rebuilding density');
    await page.evaluate(() => { window.__colormill.renderer.mode = 'surface'; });
    await settle();
    const mode = await work();
    assert(mode.draws > orbit.draws && mode.mips === orbit.mips, 'colour mode changes redraw without rebuilding density');

    const before = await page.evaluate(() => {
      const api = window.__colormill;
      window.__originalSim = api.sim;
      const c = document.querySelector('canvas');
      return { width: c.width, height: c.height, count: api.stats().particleCount, time: api.stats().simTime };
    });
    await page.locator('input[name="renderScale"]').evaluate((input) => {
      input.value = '0.5';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await settle();
    const scaled = await page.evaluate(() => {
      const api = window.__colormill;
      const c = document.querySelector('canvas');
      return { width: c.width, height: c.height, count: api.stats().particleCount, time: api.stats().simTime,
        sameSim: api.sim === window.__originalSim, link: api.startLink() };
    });
    assert(scaled.width === Math.round(before.width / 2) && scaled.height === Math.round(before.height / 2), 'half resolution renders one quarter as many pixels');
    assert(scaled.sameSim && scaled.count === before.count && scaled.time === before.time, 'resolution change preserves simulation state');
    assert(new URL(scaled.link).searchParams.get('scale') === '0.5', 'start link preserves render resolution');

    const preTint = await work();
    await page.evaluate(() => window.__colormill.sim.addPigment([0.75, 0.9, 0.75], 0.3, [1, 0, 0, 0, 0, 0, 0]));
    await settle();
    const tint = await work();
    assert(tint.draws > preTint.draws && tint.mips === preTint.mips, 'paused pigment edits redraw while reusing density');
    await page.evaluate(() => window.__colormill.sim.clearPigment());
    await settle();
    const clear = await work();
    assert(clear.draws > tint.draws && clear.mips === tint.mips, 'clearing pigment redraws while reusing density');

    await page.evaluate(() => window.__colormill.sim.addPigmentChunk(0.75, 0.75, 0.06, [1, 0, 0, 0, 0, 0, 0]));
    await settle();
    const chunk = await work();
    assert(chunk.draws > clear.draws && chunk.mips > clear.mips, 'new material refreshes density while paused');
    await page.evaluate(() => window.__colormill.sim.reset());
    await settle();
    const reset = await work();
    assert(reset.draws > chunk.draws && reset.mips > chunk.mips, 'reset invalidates the density mip');
    await page.evaluate(() => window.__colormill.stepFrames(1));
    const stepped = await work();
    assert(stepped.draws > reset.draws && stepped.mips > reset.mips, 'simulation step refreshes density');
    await settle();
    const stopped = await work();
    assert(stopped.draws === stepped.draws, 'forced stepping returns to idle when paused');

    const screenshots = await page.evaluate(async () => {
      const a = await window.__colormill.screenshot();
      const b = await window.__colormill.screenshot();
      return { equal: a.data.every((v, i) => v === b.data[i]), width: b.width, height: b.height };
    });
    assert(screenshots.equal && screenshots.width === scaled.width && screenshots.height === scaled.height, 'repeated screenshots remain available and identical when paused');
    assert((await work()).mips === stepped.mips, 'screenshots reuse unchanged density');
    const beforeResize = await work();
    await page.setViewportSize({ width: 440, height: 320 });
    await settle();
    const resized = await work();
    assert(resized.draws > beforeResize.draws && resized.mips === beforeResize.mips, 'viewport resize redraws without rebuilding density');
    await page.evaluate(() => {
      const api = window.__colormill;
      api.renderer.setVolumes(api.sim.volumes);
    });
    await settle();
    const rebound = await work();
    assert(rebound.draws > resized.draws && rebound.mips > resized.mips, 'rebinding volumes invalidates both caches');
    await page.evaluate(() => {
      const api = window.__colormill;
      api.renderer.setVolumes({ ...api.sim.volumes, version: undefined, densityVersion: undefined });
    });
    await settle();
    const unversioned = await work();
    await settle();
    const refreshed = await work();
    assert(refreshed.draws > unversioned.draws && refreshed.mips > unversioned.mips, 'unversioned external volumes keep refreshing');
    await page.goto(scaled.link + '&paused=1');
    await page.waitForFunction(() => window.__colormill?.ready, null, { timeout: 120000 });
    assert(await page.evaluate(() => window.__colormill.renderer.renderScale === 0.5), 'start link restores render resolution');
    assert(errors.length === 0, 'no browser or GPU errors: ' + errors.join('; '));
    console.log('  idle submissions: 0; camera/pigment changes reuse density; 50% resolution preserves the bank');
  });
}

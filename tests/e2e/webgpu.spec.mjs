import { assert, withBrowser } from './lib.mjs';

/** Smoke: the headless browser exposes a WebGPU device. */
export default async function run() {
  await withBrowser(async ({ page, baseUrl }) => {
    await page.goto(baseUrl);
    const info = await page.evaluate(async () => {
      if (!navigator.gpu) return null;
      const a = await navigator.gpu.requestAdapter();
      if (!a) return null;
      const d = await a.requestDevice();
      return { ok: !!d, arch: a.info?.architecture ?? '' };
    });
    assert(info && info.ok, 'WebGPU device available in headless Chromium');
    console.log('  adapter architecture:', info.arch);
  });
}

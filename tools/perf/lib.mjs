// v3 perf review: shared helpers. Drives the unmodified app in system Chrome (real GPU, headless)
// and splits sim.step() into one compute pass per kernel with timestamp queries, by handing
// step() a proxy command encoder. No tracked source is modified.
import { chromium } from 'playwright';
import { startServer } from '../../tests/e2e/lib.mjs';

export const CHROME = process.env.COLORMILL_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

export async function launch({ headless = true } = {}) {
  const server = await startServer();
  const browser = await chromium.launch({
    executablePath: CHROME, headless,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features',
      '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'],
    ignoreDefaultArgs: ['--disable-gpu', '--disable-gpu-compositing']
  });
  return { server, browser, async close() { await browser.close(); await server.stop(); } };
}

export async function openApp(browser, baseUrl, preset, { width = 1920, height = 1080, query = '' } = {}) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.log('[console.error]', m.text().slice(0, 400)); });
  await page.goto(`${baseUrl}index.html?preset=${preset}&paused=1&scale=1${query}`);
  await page.waitForFunction(() => window.__colormill?.ready, null, { timeout: 180000 });
  await page.evaluate(installProfiler);
  return page;
}

/** Runs in the page. Installs window.__prof. */
export function installProfiler() {
  const api = window.__colormill;
  const QN = 4096;
  let qs = null, resolveBuf = null, device = null;
  const ensure = () => {
    const d = api.sim.device;
    if (d !== device) {
      device = d;
      qs = device.createQuerySet({ type: 'timestamp', count: QN });
      resolveBuf = device.createBuffer({ size: QN * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
  };
  const info = () => {
    const st = api.sim.stats;
    return { params: api.sim.params, rollerAngleFront: st.rollerAngleFront, rollerAngleBack: st.rollerAngleBack, timeSeconds: performance.now() / 1000 };
  };
  /**
   * Encode one frame. mode 'split': one pass per kernel dispatch; 'whole': one pass for the sim (as shipped);
   * 'none': no timestamps. `filter(label)` (optional) drops dispatches whose pipeline label it rejects
   * (isolating passes). Returns the labels of the timed passes, in query order.
   */
  const encodeFrame = (enc, q0, { mode = 'split', render = true, filter = null, subs = null } = {}) => {
    const sim = api.sim;
    // subs: { kernelLabel: variantName } -> window.__variants[variantName] = { pipeline, bindGroup?, scale? }
    const V = window.__variants || {};
    let qi = q0;
    const labels = [];
    const tw = () => { const w = { querySet: qs, beginningOfPassWriteIndex: qi, endOfPassWriteIndex: qi + 1 }; qi += 2; return w; };
    const proxy = {
      beginComputePass(desc) {
        let real = null;
        let skip = false;
        let sub = null;
        const open = (label) => {
          if (mode === 'split') { if (real) real.end(); labels.push(label); real = enc.beginComputePass({ timestampWrites: tw() }); }
          else if (!real) {
            if (mode === 'whole') { labels.push(desc?.label === 'raymarch-density-mip' ? 'raymarch-mip-pipeline' : 'sim-whole'); real = enc.beginComputePass({ timestampWrites: tw() }); }
            else real = enc.beginComputePass();
          }
        };
        return {
          setPipeline(p) {
            skip = !!(filter && !filter(p.label));
            if (skip) return;
            sub = subs && subs[p.label] ? V[subs[p.label]] : null;
            if (subs && subs[p.label] && !sub) throw new Error('no variant ' + subs[p.label]);
            open(p.label); real.setPipeline(sub ? sub.pipeline : p);
          },
          setBindGroup(i, bg, offs) { if (!skip) if (offs === undefined) real.setBindGroup(i, sub?.bindGroup ?? bg); else real.setBindGroup(i, sub?.bindGroup ?? bg, offs); },
          dispatchWorkgroups(x, y, z) {
            if (skip) return;
            if (sub?.indirect) real.dispatchWorkgroupsIndirect(sub.indirect.buffer, sub.indirect.offset);
            else real.dispatchWorkgroups(sub?.scale ? Math.ceil(x * sub.scale) : x, y, z);
          },
          dispatchWorkgroupsIndirect(...a) { if (!skip) real.dispatchWorkgroupsIndirect(...a); },
          end() { if (real) real.end(); real = null; }
        };
      },
      beginRenderPass(desc) {
        if (mode === 'none') return enc.beginRenderPass(desc);
        labels.push('render');
        return enc.beginRenderPass({ ...desc, timestampWrites: tw() });
      },
      resolveQuerySet() {}, copyBufferToBuffer() {}
    };
    const was = sim.paused;
    sim.paused = false;
    sim.step(proxy, 1 / 60);
    sim.paused = was;
    if (render) { api.renderer.resize(); api.renderer.lastFrameData = null; api.renderer.render(proxy, info()); }
    return { labels, qEnd: qi };
  };

  /** Run `frames` frames back to back in batches (so the GPU does not idle between them) and return per-frame per-label ms. */
  const measure = async (frames, opts = {}) => {
    ensure();
    const out = [];
    let left = frames;
    while (left > 0) {
      const batch = [];
      let q = 0;
      const encs = [];
      while (left > 0) {
        const enc = device.createCommandEncoder();
        const r = encodeFrame(enc, q, opts);
        batch.push({ q0: q, labels: r.labels });
        encs.push(enc);
        const per = r.qEnd - q;
        q = r.qEnd;
        left--;
        if (q + per + 8 > QN) break;
      }
      const staging = device.createBuffer({ size: Math.max(q, 2) * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      if (q > 0) {
        const last = encs[encs.length - 1];
        last.resolveQuerySet(qs, 0, q, resolveBuf, 0);
        last.copyBufferToBuffer(resolveBuf, 0, staging, 0, q * 8);
      }
      const t0 = performance.now();
      device.queue.submit(encs.map((e) => e.finish()));
      await staging.mapAsync(GPUMapMode.READ);
      const wall = performance.now() - t0;
      const ts = new BigUint64Array(staging.getMappedRange().slice(0));
      staging.unmap(); staging.destroy();
      for (const f of batch) {
        const per = {};
        let sum = 0;
        f.labels.forEach((label, i) => {
          const a = ts[f.q0 + 2 * i], b = ts[f.q0 + 2 * i + 1];
          const ms = b >= a ? Number(b - a) / 1e6 : NaN;
          per[label] = (per[label] ?? 0) + ms;
          sum += ms;
        });
        const n = f.labels.length;
        const span = n ? Number(ts[f.q0 + 2 * n - 1] - ts[f.q0]) / 1e6 : 0;
        out.push({ per, sum, span, wallBatch: wall / batch.length });
      }
    }
    return out;
  };

  /** Wall-clock throughput: n frames submitted back to back, one wait at the end. */
  const wall = async (n, opts = {}) => {
    ensure();
    await device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      const enc = device.createCommandEncoder();
      encodeFrame(enc, 0, { ...opts, mode: 'none' });
      device.queue.submit([enc.finish()]);
    }
    const tEnc = performance.now() - t0;
    await device.queue.onSubmittedWorkDone();
    return { msPerFrame: (performance.now() - t0) / n, encodeMsPerFrame: tEnc / n };
  };

  /** Read the material's box buffer (dispatch extents) and counts. */
  const state = async () => {
    ensure();
    const sim = api.sim;
    const staging = device.createBuffer({ size: 192, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(sim.bufBox, 0, staging, 0, 192);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const b = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    const d = sim.dims, fd = sim.fineDims;
    const canvas = document.querySelector('canvas');
    return {
      particles: sim.stats.particleCount, capacity: sim.stats.particleCapacity, simTime: sim.stats.simTime,
      dims: [d.nx, d.ny, d.nz], nodes: d.nodeCount, fineDims: [fd.nx, fd.ny, fd.nz], fineNodes: fd.nodeCount,
      gridBox: [...b.slice(9, 12)], gridBoxNodes: b[9] * b[10] * b[11],
      frameBox: [...b.slice(21, 24)], fineRange: [...b.slice(36, 39)], fineRangeNodes: b[36] * b[37] * b[38], fineK: b[35],
      chunksHeld: sim.chunksHeld, substeps: sim.quality.substepsPerFrame, canvas: [canvas.width, canvas.height],
      adapter: (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })).info.description
    };
  };

  window.__prof = { measure, wall, state, encodeFrame };
}

export function stats(xs) {
  const v = xs.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!v.length) return { n: 0 };
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length);
  const q = (p) => v[Math.min(v.length - 1, Math.floor(p * v.length))];
  return { n: v.length, mean, sd, min: v[0], p50: q(0.5), p95: q(0.95), max: v[v.length - 1] };
}

/** Collapse kernel labels into the report's categories. */
export const CATEGORY = {
  p2g: 'p2g', grid: 'grid', g2p: 'g2p', boxStep: 'box', boxFrame: 'box',
  clearRaster: 'rasterCoarse', raster: 'rasterCoarse', pack: 'pack(coarse)',
  clearFine: 'rasterFine.clear', rasterFine: 'rasterFine.scatter', packFine: 'rasterFine.pack',
  disperse: 'disperse', streak: 'streak', chunkStep: 'chunk', chunkSettle: 'chunk', chunkDrop: 'chunk',
  'raymarch-mip-pipeline': 'render.mip', render: 'render.raymarch', 'sim-whole': 'sim-whole'
};

export function summarise(frames) {
  const cats = {};
  const labels = new Set();
  for (const f of frames) for (const l of Object.keys(f.per)) labels.add(l);
  const byLabel = {};
  for (const l of labels) byLabel[l] = stats(frames.map((f) => f.per[l] ?? 0));
  for (const f of frames) {
    const c = {};
    for (const [l, ms] of Object.entries(f.per)) { const k = CATEGORY[l] ?? l; c[k] = (c[k] ?? 0) + ms; }
    f.cat = c;
  }
  const catNames = new Set();
  for (const f of frames) for (const k of Object.keys(f.cat)) catNames.add(k);
  for (const k of catNames) cats[k] = stats(frames.map((f) => f.cat[k] ?? 0));
  return { byLabel, cats, sum: stats(frames.map((f) => f.sum)), span: stats(frames.map((f) => f.span)) };
}

// Where the putty is stressed, on the real GPU: the distribution of the shear rate the particles carry in C and
// of the shear stress, in the bank, the converging wedge above the nip, the nip itself and the sheet on the
// front roll, at several friction ratios. What the dispersive-mixing threshold (design §5, agglomerates) is set
// from: the cohesive stress has to sit above what the bank reaches and below what the nip does.
//
// usage: node tools/perf/stress.mjs [low high ...] [--ratios=1,1.25,1.6] [--settle=6] [--windows=6] [--frames=15]
//                                    [--thresholds=1.2,1.4,1.6,1.8] [--json=<file>]
// Two stresses per particle: the flow curve's, tau_y + k (shear rate)^n with the shear rate sqrt(2 D':D') of the
// last substep's C (what agglomerate.ts / lamina.wgsl use), and the stored one, sqrt(2) mu |dev log(sigma(F))|,
// the von Mises shear stress of the elastic strain the particle holds after the plastic return.
import { writeFileSync } from 'node:fs';
import { launch, openApp } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
const presets = args.filter((a) => !a.startsWith('--'));
if (!presets.length) presets.push('low', 'high');
const RATIOS = opt('ratios', '1,1.25,1.6').split(',').map(Number);
const SETTLE = Number(opt('settle', 6)), WINDOWS = Number(opt('windows', 6)), FRAMES = Number(opt('frames', 15));
const THRESHOLDS = opt('thresholds', '1.2,1.4,1.6,1.8').split(',').map(Number);

/** Runs in the page: one snapshot's samples by region. */
async function sample() {
  const api = window.__colormill;
  const s = api.sim;
  const snap = await api.snapshot();
  const m = s.config.material;
  const mu = m.E / (2 * (1 + m.nu));
  const R = 0.32, ay = 0.55, nipZ = 0.75;
  const gap = s.params.gap, h = s.stats.grid.h;
  const azF = nipZ + R + gap / 2, azB = nipZ - R - gap / 2;
  const out = { bank: [], wedge: [], nip: [], sheet: [], all: [] };
  const X = snap.positions, C = snap.velocityGradient, F = snap.deformation;
  // eigenvalues of a symmetric 3x3 (Jacobi, enough sweeps for a diagnostic)
  const eig = (a) => {
    const A = [a[0], a[1], a[2], a[1], a[3], a[4], a[2], a[4], a[5]];
    for (let sweep = 0; sweep < 8; sweep++) {
      for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
        const apq = A[3 * p + q];
        if (Math.abs(apq) < 1e-14) continue;
        const th = (A[3 * q + q] - A[3 * p + p]) / (2 * apq);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < 3; k++) { const kp = A[3 * k + p], kq = A[3 * k + q]; A[3 * k + p] = c * kp - sn * kq; A[3 * k + q] = sn * kp + c * kq; }
        for (let k = 0; k < 3; k++) { const pk = A[3 * p + k], qk = A[3 * q + k]; A[3 * p + k] = c * pk - sn * qk; A[3 * q + k] = sn * pk + c * qk; }
      }
    }
    return [A[0], A[4], A[8]];
  };
  for (let p = 0; p < snap.count; p++) {
    if (snap.flags[p] & 1) continue;
    const x = X[3 * p], y = X[3 * p + 1], z = X[3 * p + 2];
    const c = 9 * p;
    const tr = (C[c] + C[c + 4] + C[c + 8]) / 3;
    const dxx = C[c] - tr, dyy = C[c + 4] - tr, dzz = C[c + 8] - tr;
    const dxy = 0.5 * (C[c + 1] + C[c + 3]), dxz = 0.5 * (C[c + 2] + C[c + 6]), dyz = 0.5 * (C[c + 5] + C[c + 7]);
    const rate = Math.sqrt(2 * (dxx * dxx + dyy * dyy + dzz * dzz + 2 * (dxy * dxy + dxz * dxz + dyz * dyz)));
    // F^T F, its eigenvalues are the squared stretches
    const f = [F[c], F[c + 1], F[c + 2], F[c + 3], F[c + 4], F[c + 5], F[c + 6], F[c + 7], F[c + 8]];
    const g = (i, j) => f[i] * f[j] + f[3 + i] * f[3 + j] + f[6 + i] * f[6 + j];
    const ev = eig([g(0, 0), g(0, 1), g(0, 2), g(1, 1), g(1, 2), g(2, 2)]).map((v) => 0.5 * Math.log(Math.min(Math.max(v, 0.0025), 400)));
    const mean = (ev[0] + ev[1] + ev[2]) / 3;
    const stored = Math.SQRT2 * mu * Math.hypot(ev[0] - mean, ev[1] - mean, ev[2] - mean);
    if (!Number.isFinite(rate) || !Number.isFinite(stored)) continue;
    const item = [rate, stored];
    out.all.push(item);
    const between = z > azB && z < azF;
    const dF = Math.hypot(y - ay, z - azF);
    if (between && Math.abs(y - ay) <= 0.02) out.nip.push(item);
    else if (between && y > ay + 0.02 && y <= ay + 0.06) out.wedge.push(item);
    else if (between && y > ay + 0.06) out.bank.push(item);
    else if (dF < R + gap + 4 * h) out.sheet.push(item);
    void x;
  }
  return { regions: out, material: { ty: m.yieldStress, k: m.consistency, n: m.flowIndex }, gap, h, omega: s.params.omega, ratio: s.params.frictionRatio };
}

const ctx = await launch();
const results = [];
try {
  for (const preset of presets) {
    for (const ratio of RATIOS) {
      const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: 640, height: 480, query: `&frictionRatio=${ratio}` });
      const spf = await page.evaluate(() => window.__colormill.stats().simSecondsPerFrame);
      const step = async (frames) => { for (let left = frames; left > 0; left -= 100) await page.evaluate((n) => window.__colormill.stepFrames(n), Math.min(100, left)); };
      await step(Math.round(SETTLE / spf));
      const acc = { bank: [], wedge: [], nip: [], sheet: [], all: [] };
      let info = null;
      for (let w = 0; w < WINDOWS; w++) {
        await step(FRAMES);
        const r = await page.evaluate(sample);
        info = r;
        for (const k of Object.keys(acc)) for (const it of r.regions[k]) acc[k].push(it);
      }
      const { ty, k, n } = info.material;
      const flow = (g) => ty + k * Math.pow(g, n);
      const row = { preset, ratio, gap: info.gap, nominal: (ratio - 1) * info.omega * 0.32 / info.gap, regions: {} };
      for (const key of Object.keys(acc)) {
        const items = acc[key];
        const q = (vals, f) => vals[Math.min(vals.length - 1, Math.floor(f * (vals.length - 1)))];
        const rates = items.map((i) => i[0]).sort((a, b) => a - b);
        const stored = items.map((i) => i[1]).sort((a, b) => a - b);
        const total = acc.all.length;
        row.regions[key] = {
          share: items.length / total,
          rate: [0.1, 0.5, 0.9, 0.99].map((f) => q(rates, f)),
          flow: [0.1, 0.5, 0.9, 0.99].map((f) => flow(q(rates, f))),
          stored: [0.1, 0.5, 0.9, 0.99].map((f) => q(stored, f)),
          // per threshold: the share of the region's particles over it, and this region's part of the batch's
          // mean erosion rate (shear rate x excess stress over the threshold, averaged over EVERY particle)
          over: THRESHOLDS.map((te) => {
            let cnt = 0, sum = 0;
            for (const g of rates) { const t = flow(g); if (t > te) { cnt++; sum += g * (t / te - 1); } }
            return { te, share: cnt / Math.max(items.length, 1), rate: sum / total };
          })
        };
      }
      results.push(row);
      const f = (v, d = 2) => v.toFixed(d).padStart(6);
      console.log(`\n${preset} ratio ${ratio}: gap ${info.gap}, nominal nip shear ${row.nominal.toFixed(1)} /s; flow curve ${ty} + ${k} g^${n}`);
      console.log('  region  share | shear rate p10 p50 p90 p99     | flow stress p10 p50 p90 p99    | stored stress p10 p50 p90 p99  | ' +
        THRESHOLDS.map((t) => `>${t}: share, rate`).join(' | '));
      for (const key of ['bank', 'wedge', 'nip', 'sheet', 'all']) {
        const r = row.regions[key];
        console.log(`  ${key.padEnd(6)} ${f(r.share, 3)} | ${r.rate.map((v) => f(v)).join(' ')}    | ${r.flow.map((v) => f(v)).join(' ')}    | ${r.stored.map((v) => f(v)).join(' ')}    | ` +
          r.over.map((o) => `${f(o.share, 3)} ${f(o.rate, 4)}`).join(' | '));
      }
      await page.close();
    }
  }
} finally {
  await ctx.close();
}
const json = opt('json', '');
if (json) writeFileSync(json, JSON.stringify(results, null, 1));

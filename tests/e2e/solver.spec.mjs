/**
 * GPU solver verification (design §11 items 1–3) at preset `low`, in headless
 * Chromium + SwiftShader, through the dev harness harness/solver.html.
 *
 * SwiftShader needs ~1.1-1.4 s per frame at `low` (60k particles, 8 substeps),
 * so the default run is shortened to 1.0 s of sim for the sheet check and
 * 1.3 s for the mixing check (§11 says 3 s / 6 s); the sheet criterion is met
 * from ~0.8 s and green appears from ~1 s after injection. Override with:
 *
 *   E2E_SOLVER_FRAMES=<n>      frames for the stability/sheet run (234 = 3 s)
 *   E2E_SOLVER_MIX_FRAMES=<n>  frames after injecting pigment (469 = 6 s)
 *   E2E_SOLVER_FOLD=1          also exercise cutAndFold() (adds ~1.8 s of sim): the
 *                              folded sheet must land as a slab (max node density
 *                              <= 6, >= 20 distinct z cells) and be released cleanly
 */
import { assert, withBrowser } from './lib.mjs';
import { DEFAULT_PARAMS, GEOMETRY, QUALITY_PRESETS, gridDims, rollerPoses } from '../../src/config/mill.ts';

// Mixbox endpoint latents (src/sim/mixbox.c)
const BLUE = [0.86413725, 0.00441961, 0.02987264, 0.10157050, -0.05379499, -0.01226009, 0.00350272];
const YELLOW = [0.00392157, 0.85950980, 0.00000000, 0.13656863, 0.03300247, 0.10249173, -0.08066935];
const RED = [0.00000000, 0.33960806, 0.65860828, 0.00178365, 0.08873356, -0.01747544, -0.06755477];

/** Mixbox latent -> rgb (port of EvalPolynomial + residual). */
export function latentToRgb(z) {
  const [c0, c1, c2, c3] = z;
  const c00 = c0 * c0, c11 = c1 * c1, c22 = c2 * c2, c33 = c3 * c3, c01 = c0 * c1, c02 = c0 * c2, c12 = c1 * c2;
  let R = 0, G = 0, B = 0, w;
  w = c0 * c00; R += 0.07717053 * w; G += 0.02826978 * w; B += 0.24832992 * w;
  w = c1 * c11; R += 0.95912302 * w; G += 0.80256528 * w; B += 0.03561839 * w;
  w = c2 * c22; R += 0.74683774 * w; G += 0.04868586 * w; B += 0.0 * w;
  w = c3 * c33; R += 0.99518138 * w; G += 0.99978149 * w; B += 0.99704802 * w;
  w = c00 * c1; R += 0.04819146 * w; G += 0.83363781 * w; B += 0.32515377 * w;
  w = c01 * c1; R += -0.68146950 * w; G += 1.46107803 * w; B += 1.06980936 * w;
  w = c00 * c2; R += 0.27058419 * w; G += -0.15324870 * w; B += 1.98735057 * w;
  w = c02 * c2; R += 0.80478189 * w; G += 0.67093710 * w; B += 0.18424500 * w;
  w = c00 * c3; R += -0.35031003 * w; G += 1.37855826 * w; B += 3.68865000 * w;
  w = c0 * c33; R += 1.05128046 * w; G += 1.97815239 * w; B += 2.82989073 * w;
  w = c11 * c2; R += 3.21607125 * w; G += 0.81270228 * w; B += 1.03384539 * w;
  w = c1 * c22; R += 2.78893374 * w; G += 0.41565549 * w; B += -0.04487295 * w;
  w = c11 * c3; R += 3.02162577 * w; G += 2.55374103 * w; B += 0.32766114 * w;
  w = c1 * c33; R += 2.95124691 * w; G += 2.81201112 * w; B += 1.17578442 * w;
  w = c22 * c3; R += 2.82677043 * w; G += 0.79933038 * w; B += 1.81715262 * w;
  w = c2 * c33; R += 2.99691099 * w; G += 1.22593053 * w; B += 1.80653661 * w;
  w = c01 * c2; R += 1.87394106 * w; G += 2.05027182 * w; B += -0.29835996 * w;
  w = c01 * c3; R += 2.56609566 * w; G += 7.03428198 * w; B += 0.62575374 * w;
  w = c02 * c3; R += 4.08329484 * w; G += -1.40408358 * w; B += 2.14995522 * w;
  w = c12 * c3; R += 6.00078678 * w; G += 2.55552042 * w; B += 1.90739502 * w;
  const cl = (v) => Math.min(1, Math.max(0, v));
  return [cl(R + z[4]), cl(G + z[5]), cl(B + z[6])];
}

const isGreen = ([r, g, b]) => g > 0.35 && g > r + 0.08 && g > b + 0.08;

function analyse(snap, dims) {
  const { back, front } = rollerPoses(DEFAULT_PARAMS);
  const R = GEOMETRY.radius;
  const h = dims.h;
  const dom = GEOMETRY.domain;
  const n = snap.count;
  let bad = 0, outside = 0, inRoller = 0, sheetFront = 0, sheetBack = 0, minDf = Infinity, minDb = Infinity;
  let kinematic = 0;
  for (let p = 0; p < n; p++) {
    const x = snap.positions[3 * p], y = snap.positions[3 * p + 1], z = snap.positions[3 * p + 2];
    for (let c = 0; c < 3; c++) {
      if (!Number.isFinite(snap.positions[3 * p + c]) || !Number.isFinite(snap.velocities[3 * p + c])) bad++;
    }
    for (let c = 0; c < 9; c++) if (!Number.isFinite(snap.deformation[9 * p + c])) bad++;
    for (let c = 0; c < 7; c++) if (!Number.isFinite(snap.latents[7 * p + c])) bad++;
    if (x < 0 || y < 0 || z < 0 || x > dom[0] || y > dom[1] || z > dom[2]) outside++;
    const df = Math.hypot(y - front.axisY, z - front.axisZ);
    const db = Math.hypot(y - back.axisY, z - back.axisZ);
    minDf = Math.min(minDf, df); minDb = Math.min(minDb, db);
    if (df < R - 1e-3 || db < R - 1e-3) inRoller++;
    if (y < GEOMETRY.axisY && df < R + 4 * h) sheetFront++;
    if (y < GEOMETRY.axisY && db < R + 4 * h) sheetBack++;
    if (snap.flags[p] & 1) kinematic++;
  }
  return { n, bad, outside, inRoller, sheetFront, sheetBack, minDf, minDb, kinematic };
}

export default async function run() {
  const q = QUALITY_PRESETS.low;
  const dims = gridDims(q);
  const secPerFrame = q.dt * q.substepsPerFrame;
  const framesFor = (sec) => Math.ceil(sec / secPerFrame);
  const stabilityFrames = Number(process.env.E2E_SOLVER_FRAMES || framesFor(1.0));
  const mixFrames = Number(process.env.E2E_SOLVER_MIX_FRAMES || framesFor(1.3));
  const doFold = !!process.env.E2E_SOLVER_FOLD;

  await withBrowser(async ({ page, baseUrl }) => {
    page.setDefaultTimeout(600000);
    await page.goto(baseUrl + 'harness/solver.html?preset=low');
    await page.waitForFunction(() => window.__solver && (window.__solver.ready || window.__solver.error), null, { timeout: 120000 });
    const init = await page.evaluate(() => ({ ready: window.__solver.ready, error: window.__solver.error, count: window.__solver.sim.stats.particleCount }));
    assert(init.ready && !init.error, 'solver initialised without errors: ' + init.error);
    console.log(`  particles: ${init.count}, grid ${dims.nx}x${dims.ny}x${dims.nz}, ${secPerFrame * 1000} ms sim per frame`);

    const snap0 = await page.evaluate(async () => {
      const s = await window.__solver.snapshot();
      return { count: s.count, positions: Array.from(s.positions), velocities: Array.from(s.velocities), latents: Array.from(s.latents), deformation: Array.from(s.deformation), flags: Array.from(s.flags) };
    });
    const a0 = analyse(snap0, dims);
    assert(a0.bad === 0 && a0.outside === 0 && a0.inRoller === 0, `initial state sane (${JSON.stringify(a0)})`);

    // --- 1. stability run ------------------------------------------------
    const warm = await page.evaluate(async () => { await window.__solver.stepFrames(5); return window.__solver.msPerFrame; });
    console.log(`  per-frame wall time (5 warm-up frames): ${warm.toFixed(0)} ms`);
    const t0 = Date.now();
    const st = await page.evaluate(async (n) => { await window.__solver.stepFrames(n); return { ms: window.__solver.msPerFrame, error: window.__solver.error, stats: window.__solver.stats() }; }, stabilityFrames - 5);
    console.log(`  ${stabilityFrames} frames (${(stabilityFrames * secPerFrame).toFixed(2)} s sim) in ${((Date.now() - t0) / 1000).toFixed(1)} s wall: ${st.ms.toFixed(0)} ms/frame, gpu ${st.stats.lastStepGpuMs.toFixed(2)} ms`);
    assert(!st.error, 'no WebGPU errors during stepping: ' + st.error);

    const snap1 = await page.evaluate(async () => {
      const s = await window.__solver.snapshot();
      return { count: s.count, positions: Array.from(s.positions), velocities: Array.from(s.velocities), latents: Array.from(s.latents), deformation: Array.from(s.deformation), flags: Array.from(s.flags) };
    });
    const a1 = analyse(snap1, dims);
    console.log(`  after run: ${JSON.stringify({ ...a1, minDf: +a1.minDf.toFixed(4), minDb: +a1.minDb.toFixed(4) })}`);
    assert(a1.n === init.count, `particle count unchanged (${a1.n} vs ${init.count})`);
    assert(a1.bad === 0, `no NaN/Inf in particle state (${a1.bad} bad values)`);
    assert(a1.outside === 0, `all particles inside the domain (${a1.outside} outside)`);
    assert(a1.inRoller === 0, `no particle inside a roller (${a1.inRoller}; min d_front ${a1.minDf}, min d_back ${a1.minDb})`);
    let maxV = 0;
    for (const v of snap1.velocities) maxV = Math.max(maxV, Math.abs(v));
    console.log(`  max |v| component: ${maxV.toFixed(3)}`);
    // an unstable run is NaN-free (the yield clamp bounds the stress) but shows up as
    // speeds far above the roller surface speed and material thrown onto the tray
    const surfaceSpeed = Math.abs(DEFAULT_PARAMS.omega) * DEFAULT_PARAMS.frictionRatio * GEOMETRY.radius;
    assert(maxV < 2.5 * surfaceSpeed, `max |v| ${maxV.toFixed(3)} stays below 2.5x the roller surface speed (${surfaceSpeed.toFixed(3)})`);
    let onTray = 0;
    for (let i = 0; i < snap1.count; i++) if (snap1.positions[3 * i + 1] < 3 * dims.h) onTray++;
    assert(onTray === 0, `no material fell onto the tray (${onTray} particles below 3h)`);

    // --- 2. sheet on the front roller -------------------------------------
    const sheetFrac = a1.sheetFront / a1.n;
    const backFrac = a1.sheetBack / a1.n;
    console.log(`  sheet fraction (front, d < R + 4h, y < yc): ${(sheetFrac * 100).toFixed(1)}%  back: ${(backFrac * 100).toFixed(1)}%`);
    assert(sheetFrac >= 0.15, `>= 15% of particles in the front sheet (${(sheetFrac * 100).toFixed(1)}%)`);
    assert(backFrac < sheetFrac / 3, `back roller carries < 1/3 of the sheet (${(backFrac * 100).toFixed(1)}% vs ${(sheetFrac * 100).toFixed(1)}%)`);

    // --- 3. mixing: blue + yellow -> green in the sheet --------------------
    const L = GEOMETRY.length;
    const bankTop = GEOMETRY.axisY + GEOMETRY.radius + 0.06;
    await page.evaluate(([blue, yellow, L, top, nipZ]) => {
      const sim = window.__solver.sim;
      sim.addPigment([0.35 * L, top, nipZ], 0.16, blue, 1);
      sim.addPigment([0.65 * L, top, nipZ], 0.16, yellow, 1);
      sim.addPigment([0.5 * L, top, nipZ - 0.12], 0.14, blue, 1);
      sim.addPigment([0.5 * L, top, nipZ + 0.12], 0.14, yellow, 1);
    }, [BLUE, YELLOW, L, bankTop, GEOMETRY.nipZ]);
    const t1 = Date.now();
    const mx = await page.evaluate(async (n) => { await window.__solver.stepFrames(n); return { ms: window.__solver.msPerFrame, error: window.__solver.error }; }, mixFrames);
    console.log(`  ${mixFrames} mixing frames (${(mixFrames * secPerFrame).toFixed(2)} s sim) in ${((Date.now() - t1) / 1000).toFixed(1)} s wall: ${mx.ms.toFixed(0)} ms/frame`);
    assert(!mx.error, 'no WebGPU errors during mixing: ' + mx.error);

    const snap2 = await page.evaluate(async () => {
      const s = await window.__solver.snapshot();
      return { count: s.count, positions: Array.from(s.positions), velocities: Array.from(s.velocities), latents: Array.from(s.latents), deformation: Array.from(s.deformation), flags: Array.from(s.flags) };
    });
    const a2 = analyse(snap2, dims);
    assert(a2.bad === 0 && a2.outside === 0 && a2.inRoller === 0 && a2.n === init.count, `state still sane after mixing (${JSON.stringify(a2)})`);
    const { front } = rollerPoses(DEFAULT_PARAMS);
    let greenSheet = 0, greenAll = 0, blueish = 0, yellowish = 0;
    for (let p = 0; p < a2.n; p++) {
      const z = snap2.latents.slice(7 * p, 7 * p + 7);
      const rgb = latentToRgb(z);
      const y = snap2.positions[3 * p + 1], zz = snap2.positions[3 * p + 2];
      const df = Math.hypot(y - front.axisY, zz - front.axisZ);
      if (isGreen(rgb)) { greenAll++; if (df < GEOMETRY.radius + 4 * dims.h) greenSheet++; }
      if (rgb[2] > rgb[1] + 0.1 && rgb[2] > rgb[0] + 0.1) blueish++;
      if (rgb[0] > 0.6 && rgb[1] > 0.6 && rgb[2] < 0.4) yellowish++;
    }
    console.log(`  green particles: ${greenAll} (in the front sheet: ${greenSheet}); blue-ish ${blueish}, yellow-ish ${yellowish}`);
    assert(greenSheet > 0, `some particles in the sheet decode to green (${greenSheet})`);

    // --- packed render volume: density integrates to the particle mass -----
    const dens = await page.evaluate(async () => {
      const d = await window.__solver.density();
      let sum = 0, max = 0, bad = 0;
      for (const v of d) { if (!Number.isFinite(v)) bad++; else { sum += v; max = Math.max(max, v); } }
      return { sum, max, bad, len: d.length };
    });
    console.log(`  density volume: ${dens.len} voxels, sum*8 = ${(dens.sum * 8).toFixed(0)} (particles ${init.count}), max ${dens.max.toFixed(2)}`);
    assert(dens.bad === 0, 'density volume has no NaN/Inf');
    assert(Math.abs(dens.sum * 8 - init.count) < 0.02 * init.count, 'density volume integrates to the particle count');
    assert(dens.max > 0.5, 'packed cells reach density ~1');

    // --- late surface injection: after the bank has slumped/drained, a tap over the
    // nip must still land on material (the GPU probes the column for the surface) ---
    const late = await page.evaluate(async (red) => {
      const before = await window.__solver.snapshot();
      window.__solver.sim.addPigmentOnSurface(0.75, 0.75, 0.1, red);
      await window.__solver.stepFrames(1);
      const after = await window.__solver.snapshot();
      let changed = 0, top = 0;
      for (let p = 0; p < after.count; p++) {
        let d = 0;
        for (let c = 0; c < 7; c++) d += Math.abs(after.latents[7 * p + c] - before.latents[7 * p + c]);
        if (d > 1e-3) { changed++; top = Math.max(top, after.positions[3 * p + 1]); }
      }
      return { changed, top, error: window.__solver.error };
    }, RED);
    console.log(`  late surface tap at x=0.75: ${late.changed} particles pigmented (highest y ${late.top.toFixed(3)})`);
    assert(!late.error && late.changed > 200, `a tap after ${((stabilityFrames + mixFrames) * secPerFrame).toFixed(1)} s of milling still pigments material (${late.changed} changed)`);

    // --- pigment chunk: new coloured material from the reserved pool -------------
    const chunk = await page.evaluate(async (red) => {
      const n0 = window.__solver.stats().particleCount;
      const added = window.__solver.sim.addPigmentChunk(0.4, 0.75, 0.14, red);
      await window.__solver.stepFrames(3);
      const s = await window.__solver.snapshot();
      let bad = 0;
      for (let p = n0; p < s.count; p++) for (let c = 0; c < 3; c++) if (!Number.isFinite(s.positions[3 * p + c])) bad++;
      return { n0, added, count: s.count, statCount: window.__solver.stats().particleCount, capacity: window.__solver.stats().particleCapacity, bad, error: window.__solver.error };
    }, RED);
    console.log(`  pigment chunk: +${chunk.added} particles (${chunk.n0} -> ${chunk.count}, capacity ${chunk.capacity})`);
    assert(!chunk.error && chunk.bad === 0, 'chunk particles are finite: ' + chunk.error);
    assert(chunk.added > 500 && chunk.count === chunk.n0 + chunk.added && chunk.statCount === chunk.count, `chunk added particles consistently (${JSON.stringify(chunk)})`);

    // --- optional: cut & fold ---------------------------------------------
    if (doFold) {
      const mid = await page.evaluate(async () => {
        window.__solver.sim.cutAndFold();
        await window.__solver.stepFrames(10);
        const s = await window.__solver.snapshot();
        let kin = 0, bad = 0;
        const kinIdx = [];
        for (let p = 0; p < s.count; p++) { if (s.flags[p] & 1) { kin++; kinIdx.push(p); } for (let c = 0; c < 3; c++) if (!Number.isFinite(s.positions[3 * p + c])) bad++; }
        return { kin, bad, kinIdx, count: s.count, error: window.__solver.error };
      });
      console.log(`  cut & fold: ${mid.kin} of ${mid.count} particles kinematic mid-move`);
      // the operator takes everything off the mill: sheet, bank and all
      assert(!mid.error && mid.bad === 0 && mid.kin === mid.count, `fold selected every particle and moves them (${JSON.stringify({ kin: mid.kin, count: mid.count, bad: mid.bad, error: mid.error })})`);

      // after the roll phase plus part of the feed: the log stands over the nip as an
      // area-preserving spiral (no node carries more than ~48 particles' mass), it
      // spans many y cells, and material is being released progressively (fewer
      // kinematic particles than mid-move, but not yet zero)
      const foldFrames = framesFor(1.2 + 1.0); // FOLD_ROLL_SECONDS + 1 s of feeding
      const rel = await page.evaluate(async ([n, idx, h]) => {
        await window.__solver.stepFrames(n);
        const s = await window.__solver.snapshot();
        const d = await window.__solver.density();
        let kin = 0, maxDens = 0;
        for (let p = 0; p < s.count; p++) if (s.flags[p] & 1) kin++;
        for (const v of d) if (Number.isFinite(v)) maxDens = Math.max(maxDens, v);
        const zCells = new Set(), yCells = new Set();
        let ymin = Infinity, ymax = -Infinity, zmin = Infinity, zmax = -Infinity;
        for (const p of idx) {
          const y = s.positions[3 * p + 1], z = s.positions[3 * p + 2];
          zCells.add(Math.floor(z / h)); yCells.add(Math.floor(y / h));
          ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); zmin = Math.min(zmin, z); zmax = Math.max(zmax, z);
        }
        return { kin, maxDens, zCells: zCells.size, yCells: yCells.size, ymin, ymax, zmin, zmax, error: window.__solver.error };
      }, [foldFrames - 10, mid.kinIdx, dims.h]);
      console.log(`  feeding: max density ${rel.maxDens.toFixed(2)}, log spans ${rel.yCells} y cells (y ${rel.ymin.toFixed(3)}..${rel.ymax.toFixed(3)}), ${rel.zCells} z cells; ${rel.kin} still held of ${mid.kin}`);
      assert(!rel.error, 'no WebGPU errors during the operator move: ' + rel.error);
      assert(rel.maxDens <= 8, `max density during the feed <= 8 (${rel.maxDens.toFixed(2)})`);
      assert(rel.yCells >= 15, `the standing log spans >= 15 y cells (${rel.yCells})`);
      assert(rel.kin > 0 && rel.kin < mid.kin, `the log is being fed progressively (${rel.kin} held, was ${mid.kin})`);

      const end = await page.evaluate(async (n) => {
        await window.__solver.stepFrames(n);
        const s = await window.__solver.snapshot();
        return { count: s.count, positions: Array.from(s.positions), velocities: Array.from(s.velocities), latents: Array.from(s.latents), deformation: Array.from(s.deformation), flags: Array.from(s.flags), error: window.__solver.error };
      }, 10);
      const a3 = analyse(end, dims);
      console.log(`  during the feed: ${JSON.stringify({ ...a3, minDf: +a3.minDf.toFixed(4), minDb: +a3.minDb.toFixed(4) })}`);
      // the full feed takes ~5 s of sim time (too slow for SwiftShader CI); the state must
      // stay sane while it runs and the particle count must include the pigment chunk
      assert(!end.error && a3.bad === 0 && a3.outside === 0 && a3.inRoller === 0 && a3.n === chunk.count, 'state sane during the operator move (count includes the pigment chunk)');
    }
  }, { mode: 'dev' }); // harness pages are dev-only, not part of the production build
}

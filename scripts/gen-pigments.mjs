#!/usr/bin/env node
/**
 * Generate the Mixbox latents of the ColorMill pigment palette.
 *
 * Loads the official Mixbox JS build (web/vendor/mixbox/mixbox.js, UMD, self
 * contained: it decodes its own LUT, no DOM needed) and evaluates
 * `rgbToLatent` for each named pigment. The result is written into the
 * generated block of src/color/pigments.ts (between the BEGIN/END markers).
 *
 * The four latents that the v1 CPU code hard-codes in src/sim/mixbox.c
 * (white, cadmium red, cobalt blue, cadmium yellow) are re-derived and
 * compared to the C constants. White matches to 1e-16; the chromatic three
 * match to ~4e-4 (the C constants were evidently sampled from the native
 * 256^3 LUT rather than the 64^3 trilinear LUT embedded in mixbox.js — the
 * difference is under 0.1/255 in decoded colour). The tolerance below is
 * 5e-4, and every latent must additionally decode back to its source sRGB
 * within 1/255, which is the check that matters for the renderer.
 *
 *   node scripts/gen-pigments.mjs            # regenerate + verify + write
 *   node scripts/gen-pigments.mjs --check    # verify only, print the block, no write
 *
 * Mixbox is (c) Secret Weapons, CC BY-NC 4.0 (non-commercial use).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const MIXBOX_JS = path.join(ROOT, 'web', 'vendor', 'mixbox', 'mixbox.js');
const MIXBOX_C = path.join(ROOT, 'src', 'sim', 'mixbox.c');
const OUT_TS = path.join(ROOT, 'src', 'color', 'pigments.ts');
const CHECK_ONLY = process.argv.includes('--check');

/**
 * The palette. sRGB values are the Mixbox pigment set (see the header of
 * mixbox.js) plus titanium white and ivory black. `key` is the identifier
 * used by the UI, the keyboard shortcuts and window.__colormill.tapPigment.
 */
const PALETTE = [
  { key: 'cadmiumYellow', name: 'Cadmium Yellow', hex: '#FEEC00' },
  { key: 'hansaYellow', name: 'Hansa Yellow', hex: '#FCD300' },
  { key: 'cadmiumOrange', name: 'Cadmium Orange', hex: '#FF6900' },
  { key: 'cadmiumRed', name: 'Cadmium Red', hex: '#FF2702' },
  { key: 'quinacridoneMagenta', name: 'Quinacridone Magenta', hex: '#80022E' },
  { key: 'cobaltViolet', name: 'Cobalt Violet', hex: '#4E0042' },
  { key: 'ultramarineBlue', name: 'Ultramarine Blue', hex: '#190059' },
  { key: 'cobaltBlue', name: 'Cobalt Blue', hex: '#002185' },
  { key: 'phthaloBlue', name: 'Phthalo Blue', hex: '#0D1B44' },
  { key: 'phthaloGreen', name: 'Phthalo Green', hex: '#003C32' },
  { key: 'permanentGreen', name: 'Permanent Green', hex: '#076D16' },
  { key: 'sapGreen', name: 'Sap Green', hex: '#6B9404' },
  { key: 'burntSienna', name: 'Burnt Sienna', hex: '#7B4800' },
  { key: 'titaniumWhite', name: 'Titanium White', hex: '#FFFFFF' },
  { key: 'ivoryBlack', name: 'Ivory Black', hex: '#000000' }
];

/** Which palette entries must reproduce the MB_* constants in src/sim/mixbox.c. */
const C_TOLERANCE = 5e-4;
const C_CHECKS = { MB_WHITE: 'titaniumWhite', MB_RED: 'cadmiumRed', MB_BLUE: 'cobaltBlue', MB_YELLOW: 'cadmiumYellow' };

/** Load the UMD build without a DOM: give it CommonJS-style `module`/`exports`. */
function loadMixbox() {
  const src = readFileSync(MIXBOX_JS, 'utf8');
  const mod = { exports: {} };
  // The UMD wrapper checks `typeof exports === 'object' && typeof module !== 'undefined'`
  // first, so it never touches window/self/document on this path.
  new Function('module', 'exports', 'define', src)(mod, mod.exports, undefined);
  const mixbox = mod.exports;
  if (typeof mixbox.rgbToLatent !== 'function' || typeof mixbox.latentToRgb !== 'function') {
    throw new Error('mixbox.js did not export rgbToLatent/latentToRgb');
  }
  return mixbox;
}

function hexToRgb255(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) throw new Error(`bad hex colour ${hex}`);
  const v = parseInt(m[1], 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** Parse `static const float MB_X[MB_LATENT] = { ... };` blocks from mixbox.c. */
function readCConstants() {
  const c = readFileSync(MIXBOX_C, 'utf8');
  const out = {};
  const re = /static const float (MB_[A-Z]+)\[MB_LATENT\]\s*=\s*\{([^}]*)\};/g;
  let m;
  while ((m = re.exec(c))) {
    out[m[1]] = m[2].split(',').map((s) => parseFloat(s.trim().replace(/f$/, '')));
  }
  return out;
}

function fmt(x) {
  // 8 decimals, always signed-safe, like the C file
  const s = x.toFixed(8);
  return s === '-0.00000000' ? '0.00000000' : s;
}

function main() {
  const mixbox = loadMixbox();
  const pigments = PALETTE.map((p) => {
    const rgb = hexToRgb255(p.hex);
    const latent = mixbox.rgbToLatent(rgb[0], rgb[1], rgb[2]);
    if (!Array.isArray(latent) || latent.length !== 7) throw new Error(`no latent for ${p.key}`);
    const back = mixbox.latentToRgb(latent);
    const err = Math.max(...back.map((v, i) => Math.abs(v - rgb[i])));
    return { ...p, rgb, latent: latent.map(Number), roundTripErr: err };
  });

  // --- verify against the C constants -------------------------------------
  const cConsts = readCConstants();
  let ok = true;
  for (const [cName, key] of Object.entries(C_CHECKS)) {
    const expect = cConsts[cName];
    const got = pigments.find((p) => p.key === key).latent;
    if (!expect) { console.error(`missing ${cName} in mixbox.c`); ok = false; continue; }
    const maxDiff = Math.max(...expect.map((e, i) => Math.abs(e - got[i])));
    const pass = maxDiff <= C_TOLERANCE;
    ok &&= pass;
    console.log(`${pass ? 'ok  ' : 'FAIL'} ${cName.padEnd(10)} <- ${key.padEnd(14)} max |diff| = ${maxDiff.toExponential(2)}`);
  }
  for (const p of pigments) {
    if (p.roundTripErr > 1) { console.error(`round trip error ${p.roundTripErr} for ${p.key}`); ok = false; }
  }
  if (!ok) { console.error('verification failed'); process.exit(1); }

  // --- emit the TS block ----------------------------------------------------
  const lines = [];
  lines.push('// Generated by scripts/gen-pigments.mjs from web/vendor/mixbox/mixbox.js — do not edit by hand.');
  lines.push(`export const PIGMENTS: Readonly<Record<string, Pigment>> = {`);
  pigments.forEach((p, i) => {
    const srgb = p.rgb.map((v) => (v / 255).toFixed(6)).join(', ');
    lines.push(`  ${p.key}: { // ${p.hex.toUpperCase()}`);
    lines.push(`    key: '${p.key}', name: '${p.name}', hex: '${p.hex.toLowerCase()}',`);
    lines.push(`    srgb: [${srgb}],`);
    lines.push(`    latent: [${p.latent.map(fmt).join(', ')}]`);
    lines.push(`  }${i < pigments.length - 1 ? ',' : ''}`);
  });
  lines.push('};');
  const block = lines.join('\n');

  const begin = '// BEGIN GENERATED';
  const end = '// END GENERATED';
  if (CHECK_ONLY) {
    console.log(block);
    return;
  }
  const ts = readFileSync(OUT_TS, 'utf8');
  const a = ts.indexOf(begin);
  const b = ts.indexOf(end);
  if (a < 0 || b < 0 || b < a) throw new Error(`markers ${begin}/${end} not found in ${OUT_TS}`);
  const next = ts.slice(0, a + begin.length) + '\n' + block + '\n' + ts.slice(b);
  if (next !== ts) {
    writeFileSync(OUT_TS, next);
    console.log(`wrote ${pigments.length} pigments to ${path.relative(ROOT, OUT_TS)}`);
  } else {
    console.log(`${path.relative(ROOT, OUT_TS)} already up to date (${pigments.length} pigments)`);
  }
}

main();

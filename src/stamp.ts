/**
 * ColorMill v2 — stamping a picture onto the material (pure TS, no DOM, tested).
 *
 * A stamp is a small raster of Mixbox latents with an opacity per pixel
 * (StampImage) pressed straight down onto the material's top surface over a
 * footprint on the bank (StampPlacement): a logo, a photo or a line of text
 * laid on the putty before the mill pulls it in, so the picture streaks and
 * folds as the batch mills. The GPU side (src/sim/shaders/stamp.wgsl) finds
 * the surface per column; this module sizes the raster, fits the footprint
 * and packs pixels into latents. The app (src/app.ts) draws files and text
 * onto a canvas to get the pixels.
 */
import { GEOMETRY } from './config/mill';
import type { Latent, StampImage, StampPlacement } from './sim/types';

/** Longest side of a stamp raster in pixels: finer than the colour raster can show, small enough to convert in a blink. */
export const STAMP_MAX_SIDE = 256;
/** Pixels the GPU stamp buffer holds (one STAMP_MAX_SIDE square). */
export const STAMP_MAX_PIXELS = STAMP_MAX_SIDE * STAMP_MAX_SIDE;
/** Floats per stamp pixel: the 7-float latent and the opacity. */
export const STAMP_FLOATS_PER_PIXEL = 8;
/** Longest line of text `?stamp=` takes (kept short so it stays readable across the bank). */
export const STAMP_TEXT_MAX = 40;
/** How deep a stamp goes below the surface, in cells: enough particle layers to read as solid from above. */
export const STAMP_DEPTH_CELLS = 2.5;

/** Raster size a picture is scaled to before stamping: at most `maxSide` on its longer side, never upscaled, at least 1×1. */
export function stampDims(width: number, height: number, maxSide = STAMP_MAX_SIDE): { width: number; height: number } {
  const w = Math.max(1, Math.floor(width));
  const h = Math.max(1, Math.floor(height));
  const longest = Math.max(w, h);
  if (longest <= maxSide) return { width: w, height: h };
  const k = maxSide / longest;
  return { width: Math.max(1, Math.round(w * k)), height: Math.max(1, Math.round(h * k)) };
}

/** The bank's footprint seen from above: between the end margins along the roll, the bank's depth across the nip. */
export function bankFootprint(): StampPlacement {
  return {
    x0: GEOMETRY.bankEndMargin,
    x1: GEOMETRY.length - GEOMETRY.bankEndMargin,
    z0: GEOMETRY.nipZ - GEOMETRY.bankHalfDepth,
    z1: GEOMETRY.nipZ + GEOMETRY.bankHalfDepth
  };
}

/**
 * Where a `width × height` picture lands: its aspect fitted inside the bank's
 * footprint, scaled by `scale` (0.1..1 of the fit) and centred. The top row of
 * the picture goes to the back of the bank (z0), so it reads right from the front.
 */
export function fitStamp(width: number, height: number, scale = 1): StampPlacement {
  const b = bankFootprint();
  const bw = b.x1 - b.x0;
  const bd = b.z1 - b.z0;
  const aspect = Math.max(width, 1) / Math.max(height, 1);
  const k = Math.min(Math.max(Number.isFinite(scale) ? scale : 1, 0.1), 1);
  let w = bw * k;
  let d = w / aspect;
  if (d > bd * k) { d = bd * k; w = d * aspect; }
  const cx = 0.5 * (b.x0 + b.x1);
  const cz = 0.5 * (b.z0 + b.z1);
  return { x0: cx - 0.5 * w, x1: cx + 0.5 * w, z0: cz - 0.5 * d, z1: cz + 0.5 * d };
}

/**
 * Pack RGBA pixels (0..255, row-major from the top-left, as ImageData gives
 * them) into a stamp: each pixel's latent from `rgbToLatent` (called once per
 * distinct colour; fully transparent pixels are skipped) and its alpha as the
 * opacity. Throws when the raster is bigger than the GPU buffer holds.
 */
export function rgbaToStamp(
  rgba: ArrayLike<number>, width: number, height: number,
  rgbToLatent: (r: number, g: number, b: number) => ArrayLike<number>
): StampImage {
  const n = width * height;
  if (!(width >= 1 && height >= 1) || n > STAMP_MAX_PIXELS) throw new Error(`stamp of ${width}×${height} pixels: at most ${STAMP_MAX_PIXELS} pixels`);
  if (rgba.length < 4 * n) throw new Error(`stamp of ${width}×${height} pixels needs ${4 * n} bytes of RGBA, got ${rgba.length}`);
  const data = new Float32Array(STAMP_FLOATS_PER_PIXEL * n);
  const cache = new Map<number, ArrayLike<number>>();
  for (let i = 0; i < n; i++) {
    const a = rgba[4 * i + 3] / 255;
    if (a <= 0) continue;
    const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
    const key = (r << 16) | (g << 8) | b;
    let z = cache.get(key);
    if (!z) {
      z = rgbToLatent(r / 255, g / 255, b / 255);
      if (z.length < 7) throw new Error('rgbToLatent returned no latent');
      cache.set(key, z);
    }
    const o = STAMP_FLOATS_PER_PIXEL * i;
    for (let c = 0; c < 7; c++) data[o + c] = z[c];
    data[o + 7] = a;
  }
  return { width, height, data };
}

/** A stamp straight from latents, `rows[row][col]` a latent or null for a transparent pixel (tests and scripts). */
export function stampFromLatents(rows: ReadonlyArray<ReadonlyArray<Latent | null>>): StampImage {
  const height = rows.length;
  const width = height ? rows[0].length : 0;
  if (!width || !height) throw new Error('a stamp needs at least one pixel');
  const data = new Float32Array(STAMP_FLOATS_PER_PIXEL * width * height);
  rows.forEach((row, j) => {
    if (row.length !== width) throw new Error('every row of a stamp has the same length');
    row.forEach((z, i) => {
      if (!z) return;
      const o = STAMP_FLOATS_PER_PIXEL * (j * width + i);
      for (let c = 0; c < 7; c++) data[o + c] = z[c];
      data[o + 7] = 1;
    });
  });
  return { width, height, data };
}

/** The text `?stamp=` carries, trimmed and cut to STAMP_TEXT_MAX characters; '' when there is none. */
export function parseStampText(raw: string | null | undefined): string {
  const t = (raw ?? '').replace(/\s+/g, ' ').trim();
  return t.length > STAMP_TEXT_MAX ? t.slice(0, STAMP_TEXT_MAX) : t;
}

import { describe, expect, it } from 'vitest';
import { GEOMETRY } from '../src/config/mill';
import { PIGMENTS } from '../src/color/pigments';
import {
  STAMP_DEPTH, STAMP_FLOATS_PER_PIXEL, STAMP_MAX_PIXELS, STAMP_MAX_SIDE, STAMP_TEXT_MAX, bankFootprint, fitStamp, parseStampText, rgbaToStamp, stampDepth, stampDims, stampFromLatents
} from '../src/stamp';

describe('stampDims', () => {
  it('keeps small pictures as they are and shrinks big ones to the longest side', () => {
    expect(stampDims(100, 50)).toEqual({ width: 100, height: 50 });
    expect(stampDims(1024, 512)).toEqual({ width: STAMP_MAX_SIDE, height: STAMP_MAX_SIDE / 2 });
    expect(stampDims(300, 900)).toEqual({ width: 85, height: STAMP_MAX_SIDE });
    expect(stampDims(0.5, 0)).toEqual({ width: 1, height: 1 });
    const d = stampDims(5000, 1);
    expect(d.width * d.height).toBeLessThanOrEqual(STAMP_MAX_PIXELS);
    expect(d.height).toBe(1);
  });
});

describe('fitStamp', () => {
  const b = bankFootprint();
  it('fits the picture inside the bank footprint, centred, aspect kept', () => {
    expect(b.x1 - b.x0).toBeCloseTo(GEOMETRY.length - 2 * GEOMETRY.bankEndMargin);
    expect(b.z1 - b.z0).toBeCloseTo(2 * GEOMETRY.bankHalfDepth);
    const wide = fitStamp(1000, 100);
    expect(wide.x0).toBeCloseTo(b.x0);
    expect(wide.x1).toBeCloseTo(b.x1);
    expect((wide.z1 - wide.z0) / (wide.x1 - wide.x0)).toBeCloseTo(0.1);
    expect(0.5 * (wide.z0 + wide.z1)).toBeCloseTo(GEOMETRY.nipZ);
    const square = fitStamp(64, 64);
    expect(square.z0).toBeCloseTo(b.z0);
    expect(square.z1).toBeCloseTo(b.z1);
    expect(square.x1 - square.x0).toBeCloseTo(square.z1 - square.z0);
    expect(0.5 * (square.x0 + square.x1)).toBeCloseTo(GEOMETRY.length / 2);
  });
  it('scales down inside the fit and never below a tenth', () => {
    const half = fitStamp(64, 64, 0.5);
    expect(half.x1 - half.x0).toBeCloseTo(0.5 * (b.z1 - b.z0));
    const tiny = fitStamp(64, 64, 0);
    expect(tiny.x1 - tiny.x0).toBeCloseTo(0.1 * (b.z1 - b.z0));
    const big = fitStamp(64, 64, 5);
    expect(big.x1 - big.x0).toBeCloseTo(b.z1 - b.z0);
    expect(fitStamp(64, 64, NaN)).toEqual(fitStamp(64, 64));
  });
});

describe('rgbaToStamp', () => {
  const red = PIGMENTS.naphtholRed.latent;
  const blue = PIGMENTS.phthaloBlue.latent;
  const toLatent = (r: number, _g: number, b: number) => (r > b ? red : blue);
  it('packs one latent and the alpha per pixel, skipping transparent pixels', () => {
    // 2 × 1: a red pixel at half opacity, a transparent one
    const rgba = new Uint8ClampedArray([255, 0, 0, 128, 0, 0, 255, 0]);
    const s = rgbaToStamp(rgba, 2, 1, toLatent);
    expect(s.width).toBe(2);
    expect(s.height).toBe(1);
    expect(s.data.length).toBe(2 * STAMP_FLOATS_PER_PIXEL);
    expect(Array.from(s.data.slice(0, 7))).toEqual(red.map((v) => Math.fround(v)));
    expect(s.data[7]).toBeCloseTo(128 / 255);
    expect(Array.from(s.data.slice(8, 16))).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  });
  it('converts each distinct colour once', () => {
    const n = 1000;
    const rgba = new Uint8ClampedArray(4 * n);
    for (let i = 0; i < n; i++) { rgba[4 * i] = i % 2 ? 255 : 0; rgba[4 * i + 2] = i % 2 ? 0 : 255; rgba[4 * i + 3] = 255; }
    let calls = 0;
    const s = rgbaToStamp(rgba, 40, 25, (r, g, b) => { calls++; return toLatent(r, g, b); });
    expect(calls).toBe(2);
    expect(s.data[8 * 999]).toBeCloseTo(red[0]);
    expect(s.data[8 * 998]).toBeCloseTo(blue[0]);
  });
  it('refuses rasters the GPU buffer cannot hold or short pixel data', () => {
    expect(() => rgbaToStamp(new Uint8ClampedArray(4), 1, 1, toLatent)).not.toThrow();
    expect(() => rgbaToStamp(new Uint8ClampedArray(4), 2, 1, toLatent)).toThrow(/bytes/);
    expect(() => rgbaToStamp(new Uint8ClampedArray(4), STAMP_MAX_SIDE + 1, STAMP_MAX_SIDE, toLatent)).toThrow(/at most/);
    expect(() => rgbaToStamp(new Uint8ClampedArray([9, 9, 9, 255]), 1, 1, () => [1, 2])).toThrow(/no latent/);
  });
});

describe('stampFromLatents', () => {
  it('lays rows out top to bottom with full opacity and transparent holes', () => {
    const red = PIGMENTS.naphtholRed.latent;
    const s = stampFromLatents([[red, null], [null, red]]);
    expect(s.width).toBe(2);
    expect(s.height).toBe(2);
    expect(s.data[7]).toBe(1);
    expect(s.data[15]).toBe(0);
    expect(s.data[23]).toBe(0);
    expect(s.data[31]).toBe(1);
    expect(s.data[24]).toBeCloseTo(red[0]);
    expect(() => stampFromLatents([])).toThrow();
    expect(() => stampFromLatents([[red], [red, red]])).toThrow(/same length/);
  });
});

describe('parseStampText', () => {
  it('trims, collapses whitespace and cuts long lines', () => {
    expect(parseStampText('  hello   mill \n')).toBe('hello mill');
    expect(parseStampText(null)).toBe('');
    expect(parseStampText('x'.repeat(100))).toHaveLength(STAMP_TEXT_MAX);
  });
});

describe('stampDepth', () => {
  it('is a fixed print layer, but never thinner than a cell and a half', () => {
    expect(stampDepth(1 / 64)).toBeCloseTo(STAMP_DEPTH);
    expect(stampDepth(1 / 16)).toBeCloseTo(1.5 / 16);
  });
});

import { describe, expect, it } from 'vitest';
import {
  BASE_LATENT, PALETTE_ORDER, PIGMENTS, findPigment, hexToRgb, latentToHex, latentToRgb, lerpLatent,
  mixLatents, rgbToHex
} from '../src/color/pigments';

const TOL = 2 / 255;

function expectRgbClose(got: readonly number[], want: readonly number[], tol = TOL): void {
  for (let i = 0; i < 3; i++) expect(Math.abs(got[i] - want[i])).toBeLessThanOrEqual(tol);
}

describe('pigment latents', () => {
  it('has the full palette with well-formed latents', () => {
    expect(Object.keys(PIGMENTS).length).toBe(16);
    for (const p of Object.values(PIGMENTS)) {
      expect(p.latent.length).toBe(7);
      for (const v of p.latent) expect(Number.isFinite(v)).toBe(true);
      // c0..c3 are pigment weights that sum to one (Kubelka-Munk mixture)
      expect(p.latent[0] + p.latent[1] + p.latent[2] + p.latent[3]).toBeCloseTo(1, 5);
      expect(p.hex).toMatch(/^#[0-9a-f]{6}$/);
      expectRgbClose(hexToRgb(p.hex), p.srgb, 1e-6);
    }
  });

  it('decodes cobalt blue to #002185 and cadmium yellow to #feec00', () => {
    expectRgbClose(latentToRgb(PIGMENTS.cobaltBlue.latent), hexToRgb('#002185'));
    expectRgbClose(latentToRgb(PIGMENTS.cadmiumYellow.latent), hexToRgb('#feec00'));
    expectRgbClose(latentToRgb(PIGMENTS.cadmiumRed.latent), hexToRgb('#ff2702'));
  });

  it('decodes every palette entry back to its source colour within 2/255', () => {
    for (const p of Object.values(PIGMENTS)) expectRgbClose(latentToRgb(p.latent), p.srgb);
  });

  it('decodes the white base latent to white', () => {
    expect(BASE_LATENT).toBe(PIGMENTS.titaniumWhite.latent);
    expectRgbClose(latentToRgb(BASE_LATENT), [1, 1, 1]);
    expect(latentToHex(BASE_LATENT)).toBe('#ffffff');
  });

  it('matches the v1 C constants for white / red / blue / yellow closely', () => {
    // src/sim/mixbox.c MB_* constants (native LUT); JS trilinear LUT agrees to < 5e-4
    const c = {
      white: [0, 0, 0, 1, 0.00481862, 0.00021851, 0.00295198],
      red: [0, 0.33960806, 0.65860828, 0.00178365, 0.08873356, -0.01747544, -0.06755477],
      blue: [0.86413725, 0.00441961, 0.02987264, 0.1015705, -0.05379499, -0.01226009, 0.00350272],
      yellow: [0.00392157, 0.8595098, 0, 0.13656863, 0.03300247, 0.10249173, -0.08066935]
    };
    const pairs: Array<[number[], readonly number[]]> = [
      [c.white, PIGMENTS.titaniumWhite.latent], [c.red, PIGMENTS.cadmiumRed.latent],
      [c.blue, PIGMENTS.cobaltBlue.latent], [c.yellow, PIGMENTS.cadmiumYellow.latent]
    ];
    for (const [want, got] of pairs) for (let i = 0; i < 7; i++) expect(Math.abs(want[i] - got[i])).toBeLessThan(5e-4);
  });
});

describe('latent mixing', () => {
  it('mixes cobalt blue and cadmium yellow into a green', () => {
    const z = mixLatents([{ latent: PIGMENTS.cobaltBlue.latent, weight: 1 }, { latent: PIGMENTS.cadmiumYellow.latent, weight: 1 }]);
    const [r, g, b] = latentToRgb(z);
    expect(g).toBeGreaterThan(r);
    expect(g).toBeGreaterThan(b);
    // and clearly not the grey that linear RGB averaging would give
    expect(g - Math.max(r, b)).toBeGreaterThan(0.1);
  });

  it('mixes cadmium red and cobalt blue into a purple', () => {
    const z = mixLatents([{ latent: PIGMENTS.cadmiumRed.latent, weight: 1 }, { latent: PIGMENTS.cobaltBlue.latent, weight: 1 }]);
    const [r, g, b] = latentToRgb(z);
    expect(r).toBeGreaterThan(g);
    expect(b).toBeGreaterThan(g);
  });

  it('tints white with a little pigment toward that pigment', () => {
    const z = mixLatents([{ latent: BASE_LATENT, weight: 0.9 }, { latent: PIGMENTS.cadmiumRed.latent, weight: 0.1 }]);
    const [r, g, b] = latentToRgb(z);
    expect(r).toBeGreaterThan(0.85);
    expect(r).toBeGreaterThan(g);
    expect(r).toBeGreaterThan(b);
  });

  it('normalises weights and ignores empty or non-positive entries', () => {
    const a = mixLatents([{ latent: PIGMENTS.cobaltBlue.latent, weight: 2 }, { latent: PIGMENTS.cadmiumYellow.latent, weight: 2 }]);
    const b = mixLatents([{ latent: PIGMENTS.cobaltBlue.latent, weight: 0.5 }, { latent: PIGMENTS.cadmiumYellow.latent, weight: 0.5 }]);
    for (let i = 0; i < 7; i++) expect(a[i]).toBeCloseTo(b[i], 10);
    expect(mixLatents([])).toBe(BASE_LATENT);
    expect(mixLatents([{ latent: PIGMENTS.ivoryBlack.latent, weight: 0 }])).toBe(BASE_LATENT);
    const only = mixLatents([{ latent: PIGMENTS.ivoryBlack.latent, weight: 0 }, { latent: PIGMENTS.sapGreen.latent, weight: 3 }]);
    for (let i = 0; i < 7; i++) expect(only[i]).toBeCloseTo(PIGMENTS.sapGreen.latent[i], 10);
  });

  it('lerps latents at the endpoints exactly', () => {
    const z0 = lerpLatent(PIGMENTS.cobaltBlue.latent, PIGMENTS.cadmiumYellow.latent, 0);
    const z1 = lerpLatent(PIGMENTS.cobaltBlue.latent, PIGMENTS.cadmiumYellow.latent, 1);
    for (let i = 0; i < 7; i++) {
      expect(z0[i]).toBeCloseTo(PIGMENTS.cobaltBlue.latent[i], 12);
      expect(z1[i]).toBeCloseTo(PIGMENTS.cadmiumYellow.latent[i], 12);
    }
  });

  it('clamps decoded colours to 0..1', () => {
    const rgb = latentToRgb([0, 0, 0, 1, 5, -5, 0.5]);
    expect(rgb[0]).toBe(1);
    expect(rgb[1]).toBe(0);
    expect(rgb[2]).toBe(1);
  });
});

describe('palette lookups', () => {
  it('orders the design §5 pigments first for the 1–8 shortcuts', () => {
    expect(PALETTE_ORDER.slice(0, 8)).toEqual([
      'cadmiumRed', 'cadmiumYellow', 'cobaltBlue', 'phthaloGreen', 'ultramarineBlue', 'burntSienna', 'ivoryBlack', 'titaniumWhite'
    ]);
    for (const key of PALETTE_ORDER) expect(PIGMENTS[key]?.key).toBe(key);
    expect(new Set(PALETTE_ORDER).size).toBe(Object.keys(PIGMENTS).length);
  });

  it('finds pigments by key, alias, display name and hex', () => {
    expect(findPigment('cobaltBlue')?.key).toBe('cobaltBlue');
    expect(findPigment('blue')?.key).toBe('cobaltBlue');
    expect(findPigment('white')?.key).toBe('titaniumWhite');
    expect(findPigment('Cadmium Yellow')?.key).toBe('cadmiumYellow');
    expect(findPigment('phthalo-green')?.key).toBe('phthaloGreen');
    expect(findPigment('#002185')?.key).toBe('cobaltBlue');
    expect(findPigment('nope')).toBeUndefined();
    expect(findPigment('')).toBeUndefined();
  });

  it('round-trips hex colours', () => {
    expect(rgbToHex(hexToRgb('#3C6E2A'))).toBe('#3c6e2a');
    expect(rgbToHex([1, 0.5, 0])).toBe('#ff8000');
    expect(() => hexToRgb('#12')).toThrow();
  });
});

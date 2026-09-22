import { describe, expect, it } from 'vitest';
import { BASE_LATENT, PIGMENTS, hexToRgb, latentToHex, mixLatents } from '../src/color/pigments';
import {
  CUSTOM_KEY, MAX_PARTS, REFERENCE_TERRACOTTA_HEX, STARTER_RECIPES, describeRecipe, formatMix, ladder, mixRecipe, parseMix,
  type CustomPigment, type RecipeEntry
} from '../src/mixer/mixer';

const GREEN: RecipeEntry[] = [{ key: 'cadmiumYellow', parts: 3 }, { key: 'cobaltBlue', parts: 1 }];
const CUSTOM: CustomPigment = { latent: PIGMENTS.sapGreen.latent, hex: PIGMENTS.sapGreen.hex, name: 'My Green' };

describe('mixRecipe', () => {
  it('mixes 3 yellow + 1 blue into a green using the mill mixing rule', () => {
    const res = mixRecipe(GREEN);
    const [r, g, b] = hexToRgb(res.hex);
    expect(g).toBeGreaterThan(r);
    expect(g).toBeGreaterThan(b);
    const direct = mixLatents([
      { latent: PIGMENTS.cadmiumYellow.latent, weight: 3 }, { latent: PIGMENTS.cobaltBlue.latent, weight: 1 }
    ]);
    expect(res.hex).toBe(latentToHex(direct));
    for (let i = 0; i < 7; i++) expect(res.latent[i]).toBeCloseTo(direct[i], 12);
    expect(res.totalParts).toBe(4);
  });

  it('reports a naive sRGB average that is not the Mixbox green', () => {
    const res = mixRecipe(GREEN);
    expect(res.naiveHex).not.toBe(res.hex);
    const mix = hexToRgb(res.hex);
    const naive = hexToRgb(res.naiveHex);
    // the channel average of yellow and blue is a muddy olive-yellow, far less green than the pigment mix
    expect(mix[1] - Math.max(mix[0], mix[2])).toBeGreaterThan(0.2);
    expect(naive[1] - Math.max(naive[0], naive[2])).toBeLessThan(0.05);
    // and equals the parts-weighted plain average
    const y = PIGMENTS.cadmiumYellow.srgb, c = PIGMENTS.cobaltBlue.srgb;
    for (let i = 0; i < 3; i++) expect(Math.abs(naive[i] - (3 * y[i] + c[i]) / 4)).toBeLessThanOrEqual(1 / 255);
  });

  it('shows the naive average as greyer than the Mixbox mix for equal parts yellow and blue', () => {
    const res = mixRecipe([{ key: 'cadmiumYellow', parts: 1 }, { key: 'cobaltBlue', parts: 1 }]);
    const mix = hexToRgb(res.hex);
    const naive = hexToRgb(res.naiveHex);
    const sat = (c: readonly number[]): number => (Math.max(...c) - Math.min(...c)) / Math.max(...c);
    // less saturated: min channel higher, chroma lower
    expect(Math.min(...naive)).toBeGreaterThan(Math.min(...mix));
    expect(sat(naive)).toBeLessThan(sat(mix));
  });

  it('ignores zero/negative parts and unknown keys', () => {
    const res = mixRecipe([
      { key: 'cadmiumYellow', parts: 3 }, { key: 'cobaltBlue', parts: 1 },
      { key: 'ivoryBlack', parts: 0 }, { key: 'cadmiumRed', parts: -2 }, { key: 'unobtainium', parts: 5 }
    ]);
    const clean = mixRecipe(GREEN);
    expect(res.hex).toBe(clean.hex);
    expect(res.naiveHex).toBe(clean.naiveHex);
    expect(res.totalParts).toBe(4);
  });

  it('returns the base colour for an empty recipe', () => {
    const res = mixRecipe([]);
    expect(res.latent).toBe(BASE_LATENT);
    expect(res.hex).toBe(latentToHex(BASE_LATENT));
    expect(res.naiveHex).toBe(res.hex);
    expect(res.totalParts).toBe(0);
    expect(mixRecipe([{ key: 'nope', parts: 1 }, { key: 'cobaltBlue', parts: 0 }]).totalParts).toBe(0);
  });

  it('lets a custom pigment participate, and ignores it when undefined', () => {
    const withCustom = mixRecipe([{ key: CUSTOM_KEY, parts: 1 }], CUSTOM);
    expect(withCustom.totalParts).toBe(1);
    expect(withCustom.hex).toBe(latentToHex(PIGMENTS.sapGreen.latent));
    expect(withCustom.naiveHex).toBe(PIGMENTS.sapGreen.hex);

    const mixed = mixRecipe([{ key: 'titaniumWhite', parts: 1 }, { key: CUSTOM_KEY, parts: 1 }], CUSTOM);
    expect(mixed.totalParts).toBe(2);
    const direct = mixLatents([{ latent: BASE_LATENT, weight: 1 }, { latent: CUSTOM.latent, weight: 1 }]);
    expect(mixed.hex).toBe(latentToHex(direct));

    const without = mixRecipe([{ key: 'titaniumWhite', parts: 1 }, { key: CUSTOM_KEY, parts: 1 }]);
    expect(without.totalParts).toBe(1);
    expect(without.hex).toBe('#ffffff');
    expect(mixRecipe([{ key: CUSTOM_KEY, parts: 1 }]).totalParts).toBe(0);
  });
});

describe('formatMix / parseMix', () => {
  it('formats parts with up to two decimals and skips empty entries', () => {
    expect(formatMix(GREEN)).toBe('cadmiumYellow:3,cobaltBlue:1');
    expect(formatMix([{ key: 'titaniumWhite', parts: 1.5 }, { key: 'ivoryBlack', parts: 0 }, { key: 'cadmiumRed', parts: 0.333 }]))
      .toBe('titaniumWhite:1.5,cadmiumRed:0.33');
    expect(formatMix([])).toBe('');
  });

  it('round-trips through parseMix', () => {
    const recipe: RecipeEntry[] = [
      { key: 'titaniumWhite', parts: 6 }, { key: 'cadmiumRed', parts: 1 }, { key: 'burntSienna', parts: 0.5 }, { key: CUSTOM_KEY, parts: 2 }
    ];
    expect(parseMix(formatMix(recipe))).toEqual(recipe);
    expect(parseMix(formatMix(GREEN))).toEqual(GREEN);
  });

  it('is tolerant of junk input', () => {
    expect(parseMix(null)).toEqual([]);
    expect(parseMix(undefined)).toEqual([]);
    expect(parseMix('')).toEqual([]);
    expect(parseMix('unobtainium:3,cobaltBlue:abc,cadmiumRed:-1,ivoryBlack,,:2, cadmiumYellow : 2 '))
      .toEqual([{ key: 'cadmiumYellow', parts: 2 }]);
  });

  it('resolves aliases and names through findPigment', () => {
    expect(parseMix('red:1,Cobalt Blue:2,white:0.5')).toEqual([
      { key: 'cadmiumRed', parts: 1 }, { key: 'cobaltBlue', parts: 2 }, { key: 'titaniumWhite', parts: 0.5 }
    ]);
  });

  it('clamps parts to MAX_PARTS and keeps custom as is', () => {
    expect(parseMix('cobaltBlue:99,custom:4')).toEqual([{ key: 'cobaltBlue', parts: MAX_PARTS }, { key: CUSTOM_KEY, parts: 4 }]);
  });

  it('lets the last duplicate key win', () => {
    expect(parseMix('cobaltBlue:1,cadmiumRed:2,blue:3')).toEqual([{ key: 'cobaltBlue', parts: 3 }, { key: 'cadmiumRed', parts: 2 }]);
  });
});

describe('describeRecipe', () => {
  it('uses singular and plural parts with display names', () => {
    expect(describeRecipe(GREEN)).toBe('3 parts Cadmium Yellow + 1 part Cobalt Blue');
    expect(describeRecipe([{ key: 'titaniumWhite', parts: 0.5 }, { key: 'ivoryBlack', parts: 0 }])).toBe('0.5 parts Titanium White');
  });

  it('names custom pigments and skips unusable entries', () => {
    expect(describeRecipe([{ key: CUSTOM_KEY, parts: 2 }], CUSTOM)).toBe('2 parts My Green');
    expect(describeRecipe([{ key: CUSTOM_KEY, parts: 1 }], { latent: CUSTOM.latent, hex: '#6b9404' })).toBe('1 part Custom #6b9404');
    expect(describeRecipe([{ key: CUSTOM_KEY, parts: 1 }, { key: 'nope', parts: 1 }])).toBe('Nothing yet');
  });

  it('says Nothing yet for an empty recipe', () => {
    expect(describeRecipe([])).toBe('Nothing yet');
    expect(describeRecipe([{ key: 'cobaltBlue', parts: 0 }])).toBe('Nothing yet');
  });
});

describe('ladder', () => {
  it('has `steps` entries with the endpoint colours at each end', () => {
    const a = PIGMENTS.cobaltBlue.latent, b = PIGMENTS.cadmiumYellow.latent;
    const l = ladder(a, b, 5);
    expect(l.length).toBe(5);
    expect(l[0]).toBe(latentToHex(a));
    expect(l[4]).toBe(latentToHex(b));
    for (const h of l) expect(h).toMatch(/^#[0-9a-f]{6}$/);
    // the middle of a blue–yellow ladder is a Mixbox green, not grey
    const mid = hexToRgb(l[2]);
    expect(mid[1]).toBeGreaterThan(mid[0]);
    expect(mid[1]).toBeGreaterThan(mid[2]);
    expect(ladder(a, b, 2)).toEqual([latentToHex(a), latentToHex(b)]);
  });
});

describe('STARTER_RECIPES', () => {
  it('only uses palette keys with positive parts', () => {
    const names = STARTER_RECIPES.map((s) => s.name);
    for (const want of ['Green', 'Purple', 'Skin', 'Grey', 'Olive', 'Terracotta']) expect(names).toContain(want);
    for (const s of STARTER_RECIPES) {
      expect(s.recipe.length).toBeGreaterThan(0);
      for (const e of s.recipe) {
        expect(PIGMENTS[e.key]?.key).toBe(e.key);
        expect(e.parts).toBeGreaterThan(0);
        expect(e.parts).toBeLessThanOrEqual(MAX_PARTS);
      }
    }
  });
});

describe('reference check', () => {
  it('mixes the Terracotta starter to the brown a real mill made from red, yellow and teal bands', () => {
    const starter = STARTER_RECIPES.find((s) => s.name === 'Terracotta');
    expect(starter).toBeDefined();
    const mix = hexToRgb(mixRecipe(starter!.recipe).hex);
    const ref = hexToRgb(REFERENCE_TERRACOTTA_HEX);
    // a warm brown: red over green over blue, and within ~0.1 per channel of the sampled result
    expect(mix[0]).toBeGreaterThan(mix[1]);
    expect(mix[1]).toBeGreaterThan(mix[2]);
    for (let i = 0; i < 3; i++) expect(Math.abs(mix[i] - ref[i])).toBeLessThan(0.11);
  });

  it('mixes the three sampled band colours in equal parts to a sienna close to the reference', () => {
    expect(PIGMENTS.cobaltTeal.hex).toBe('#20a4a4');
    const res = mixRecipe([{ key: 'cadmiumRed', parts: 1 }, { key: 'cadmiumYellow', parts: 1 }, { key: 'cobaltTeal', parts: 1 }]);
    const [r, g, b] = hexToRgb(res.hex);
    // equal parts of the stronger palette yellow lands on an ochre, still red over green over blue
    expect(r).toBeGreaterThan(g);
    expect(g).toBeGreaterThan(b);
  });
});

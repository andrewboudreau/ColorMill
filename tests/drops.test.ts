import { describe, expect, it } from 'vitest';
import { DROP_SLOTS } from '../src/config/mill';
import { MAX_DROPS, defaultDropSlot, formatDrops, parseDrops, recipeToDrops, type Drop } from '../src/drops';

const LIST: Drop[] = [
  { pigment: 'cadmiumYellow', slot: 3, size: 'm' },
  { pigment: 'cobaltBlue', slot: 5, size: 'l' },
  { pigment: '#ff8a00', slot: 1, size: 's' }
];

describe('formatDrops / parseDrops', () => {
  it('formats as pigment@slot.size with custom colours as bare hex digits', () => {
    expect(formatDrops(LIST)).toBe('cadmiumYellow@3.m,cobaltBlue@5.l,ff8a00@1.s');
    expect(formatDrops([])).toBe('');
  });

  it('round-trips', () => {
    expect(parseDrops(formatDrops(LIST))).toEqual(LIST);
  });

  it('resolves aliases and display names through findPigment', () => {
    expect(parseDrops('red@2.l')).toEqual([{ pigment: 'naphtholRed', slot: 2, size: 'l' }]);
    expect(parseDrops('Carbon Black@3.m')).toEqual([{ pigment: 'ivoryBlack', slot: 3, size: 'm' }]);
    expect(parseDrops('Cobalt Blue@1.s')).toEqual([{ pigment: 'cobaltBlue', slot: 1, size: 's' }]);
  });

  it('drops junk and empty entries', () => {
    expect(parseDrops('nope@1.m,,@2.m, ,cadmiumRed@1.m')).toEqual([{ pigment: 'cadmiumRed', slot: 1, size: 'm' }]);
    expect(parseDrops('')).toEqual([]);
    expect(parseDrops(null)).toEqual([]);
    expect(parseDrops(undefined)).toEqual([]);
    expect(parseDrops('zz12345@1.m')).toEqual([]);
  });

  it('clamps slots to 1..DROP_SLOTS', () => {
    expect(parseDrops('cobaltBlue@0.m')[0].slot).toBe(1);
    expect(parseDrops('cobaltBlue@-4.m')[0].slot).toBe(1);
    expect(parseDrops(`cobaltBlue@${DROP_SLOTS + 10}.m`)[0].slot).toBe(DROP_SLOTS);
  });

  it('defaults a missing slot to the middle and a missing or bad size to medium', () => {
    const mid = Math.floor((DROP_SLOTS - 1) / 2) + 1;
    expect(defaultDropSlot()).toBe(mid);
    expect(parseDrops('cobaltBlue')).toEqual([{ pigment: 'cobaltBlue', slot: mid, size: 'm' }]);
    expect(parseDrops('cobaltBlue@')).toEqual([{ pigment: 'cobaltBlue', slot: mid, size: 'm' }]);
    expect(parseDrops('cobaltBlue@x.l')).toEqual([{ pigment: 'cobaltBlue', slot: mid, size: 'l' }]);
    expect(parseDrops('cobaltBlue@4')).toEqual([{ pigment: 'cobaltBlue', slot: 4, size: 'm' }]);
    expect(parseDrops('cobaltBlue@4.huge')).toEqual([{ pigment: 'cobaltBlue', slot: 4, size: 'm' }]);
    expect(parseDrops(' cobaltBlue @ 4 . L ')).toEqual([{ pigment: 'cobaltBlue', slot: 4, size: 'l' }]);
  });

  it('accepts a custom colour as six hex digits, with or without #, normalised to lowercase', () => {
    expect(parseDrops('FF8A00@2.s')).toEqual([{ pigment: '#ff8a00', slot: 2, size: 's' }]);
    expect(parseDrops('#ff8a00@2.s')).toEqual([{ pigment: '#ff8a00', slot: 2, size: 's' }]);
    expect(parseDrops('ff8a0@2.s')).toEqual([]);
  });

  it('caps the list at MAX_DROPS', () => {
    const many = Array.from({ length: MAX_DROPS + 5 }, (_, i) => `cadmiumRed@${(i % DROP_SLOTS) + 1}.m`).join(',');
    expect(parseDrops(many).length).toBe(MAX_DROPS);
  });
});

describe('recipeToDrops', () => {
  it('gives one medium chunk per part, interleaved and spread along the roll', () => {
    const drops = recipeToDrops([{ key: 'cadmiumYellow', parts: 3 }, { key: 'cobaltBlue', parts: 1 }]);
    expect(drops).toEqual([
      { pigment: 'cadmiumYellow', slot: 1, size: 'm' },
      { pigment: 'cobaltBlue', slot: 2, size: 'm' },
      { pigment: 'cadmiumYellow', slot: 3, size: 'm' },
      { pigment: 'cadmiumYellow', slot: 4, size: 'm' }
    ]);
  });

  it('turns a half part into a small chunk', () => {
    const drops = recipeToDrops([{ key: 'cadmiumYellow', parts: 2.5 }]);
    expect(drops.map((d) => d.size)).toEqual(['m', 'm', 's']);
    expect(recipeToDrops([{ key: 'ivoryBlack', parts: 0.5 }])).toEqual([{ pigment: 'ivoryBlack', slot: 1, size: 's' }]);
  });

  it('wraps slots round-robin past DROP_SLOTS', () => {
    const drops = recipeToDrops([{ key: 'titaniumWhite', parts: DROP_SLOTS + 2 }]);
    expect(drops.length).toBe(DROP_SLOTS + 2);
    expect(drops[DROP_SLOTS].slot).toBe(1);
    expect(drops[DROP_SLOTS + 1].slot).toBe(2);
  });

  it('uses the custom hex for a custom entry and skips it without one', () => {
    const recipe = [{ key: 'custom', parts: 1 }, { key: 'cadmiumRed', parts: 1 }];
    expect(recipeToDrops(recipe, '#FF8A00')).toEqual([
      { pigment: '#ff8a00', slot: 1, size: 'm' },
      { pigment: 'cadmiumRed', slot: 2, size: 'm' }
    ]);
    expect(recipeToDrops(recipe)).toEqual([{ pigment: 'cadmiumRed', slot: 1, size: 'm' }]);
    expect(recipeToDrops(recipe, 'not a colour')).toEqual([{ pigment: 'cadmiumRed', slot: 1, size: 'm' }]);
  });

  it('skips zero parts and unknown keys, and yields nothing for an empty recipe', () => {
    expect(recipeToDrops([])).toEqual([]);
    expect(recipeToDrops([{ key: 'cadmiumRed', parts: 0 }, { key: 'nope', parts: 2 }])).toEqual([]);
  });

  it('caps at MAX_DROPS', () => {
    const drops = recipeToDrops([{ key: 'titaniumWhite', parts: 10 }, { key: 'ivoryBlack', parts: 10 }, { key: 'cadmiumRed', parts: 10 }]);
    expect(drops.length).toBe(MAX_DROPS);
    expect(formatDrops(parseDrops(formatDrops(drops)))).toBe(formatDrops(drops));
  });
});

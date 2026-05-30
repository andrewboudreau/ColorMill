import { describe, expect, it } from 'vitest';
import { clampChannel, mixColors, toCssRgb } from '../src/color';

describe('color helpers', () => {
  it('clamps channel values to the rgb byte range', () => {
    expect(clampChannel(-10)).toBe(0);
    expect(clampChannel(260)).toBe(255);
    expect(clampChannel(127.6)).toBe(128);
  });

  it('mixes two colors by ratio', () => {
    const mixed = mixColors({ r: 255, g: 0, b: 0 }, { r: 0, g: 0, b: 255 }, 0.5);

    expect(mixed).toEqual({ r: 128, g: 0, b: 128 });
  });

  it('formats css rgb colors', () => {
    expect(toCssRgb({ r: 10, g: 20, b: 30 })).toBe('rgb(10 20 30)');
  });
});

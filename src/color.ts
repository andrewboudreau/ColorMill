export interface RgbColor {
  readonly r: number;
  readonly g: number;
  readonly b: number;
}

export function clampChannel(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.min(255, Math.max(0, Math.round(value)));
}

export function mixColors(a: RgbColor, b: RgbColor, ratio: number): RgbColor {
  const t = Math.min(1, Math.max(0, ratio));
  const inverse = 1 - t;

  return {
    r: clampChannel(a.r * inverse + b.r * t),
    g: clampChannel(a.g * inverse + b.g * t),
    b: clampChannel(a.b * inverse + b.b * t)
  };
}

export function toCssRgb(color: RgbColor): string {
  return `rgb(${clampChannel(color.r)} ${clampChannel(color.g)} ${clampChannel(color.b)})`;
}

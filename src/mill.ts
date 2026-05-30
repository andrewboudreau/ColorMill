import { mixColors, type RgbColor } from './color';

export interface MaterialStrip {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly color: RgbColor;
}

export interface MillState {
  readonly leftRollerAngle: number;
  readonly rightRollerAngle: number;
  readonly mixingPass: number;
  readonly strips: readonly MaterialStrip[];
}

const redPellet: RgbColor = { r: 238, g: 80, b: 74 };
const bluePellet: RgbColor = { r: 62, g: 139, b: 230 };
const mixedPurple: RgbColor = { r: 156, g: 91, b: 184 };

function stripColor(index: number, total: number, pass: number): RgbColor {
  const leftToRight = index / Math.max(1, total - 1);
  const alternatingPellet = index % 3 === 0 ? redPellet : bluePellet;
  const rawBlend = mixColors(redPellet, bluePellet, leftToRight);
  const earlyColor = mixColors(alternatingPellet, rawBlend, 0.45);
  const blendAmount = Math.min(0.82, pass * 0.16);

  return mixColors(earlyColor, mixedPurple, blendAmount);
}

export function createInitialMillState(): MillState {
  const stripCount = 30;

  return {
    leftRollerAngle: 0,
    rightRollerAngle: 0,
    mixingPass: 0,
    strips: Array.from({ length: stripCount }, (_, index) => ({
      x: 235 + index * 14,
      y: 226,
      width: 18,
      height: 86,
      color: stripColor(index, stripCount, 0)
    }))
  };
}

export function updateMill(state: MillState, seconds: number): MillState {
  const safeSeconds = Math.max(0, Math.min(0.05, seconds));
  const nextLeftAngle = state.leftRollerAngle + safeSeconds * 1.7;
  const nextRightAngle = state.rightRollerAngle - safeSeconds * 1.7;
  const nextPass = (state.mixingPass + safeSeconds * 0.28) % 6;
  const stripCount = state.strips.length;

  return {
    leftRollerAngle: nextLeftAngle,
    rightRollerAngle: nextRightAngle,
    mixingPass: nextPass,
    strips: state.strips.map((strip, index) => {
      const distanceFromNip = Math.abs(index - (stripCount - 1) / 2) / stripCount;
      const compression = 1 - distanceFromNip * 1.35;
      const wave = Math.sin(nextLeftAngle * 2.2 + index * 0.48) * 4;

      return {
        ...strip,
        x: 235 + index * 14,
        y: 236 - compression * 26 + wave,
        height: 72 + compression * 48,
        color: stripColor(index, stripCount, nextPass)
      };
    })
  };
}

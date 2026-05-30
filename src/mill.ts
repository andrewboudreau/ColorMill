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
  readonly strips: readonly MaterialStrip[];
}

const baseA: RgbColor = { r: 238, g: 80, b: 74 };
const baseB: RgbColor = { r: 62, g: 139, b: 230 };

export function createInitialMillState(): MillState {
  return {
    leftRollerAngle: 0,
    rightRollerAngle: 0,
    strips: Array.from({ length: 18 }, (_, index) => ({
      x: 190 + index * 18,
      y: 246 + Math.sin(index * 0.85) * 22,
      width: 38,
      height: 18,
      color: mixColors(baseA, baseB, index / 17)
    }))
  };
}

export function updateMill(state: MillState, seconds: number): MillState {
  const safeSeconds = Math.max(0, Math.min(0.05, seconds));

  return {
    leftRollerAngle: state.leftRollerAngle + safeSeconds * 1.7,
    rightRollerAngle: state.rightRollerAngle - safeSeconds * 1.7,
    strips: state.strips.map((strip, index) => ({
      ...strip,
      x: 190 + index * 18 + Math.sin(state.leftRollerAngle * 1.8 + index * 0.5) * 6,
      y: 246 + Math.sin(state.rightRollerAngle * 1.4 + index * 0.85) * 24
    }))
  };
}

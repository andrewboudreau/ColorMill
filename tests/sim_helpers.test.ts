import { describe, expect, it } from 'vitest';
import { DOUBLE_SECONDS, FOLD_DURATION, FOLD_FEED_SPEED, FOLD_ROLL_SECONDS, HAND_HOLD_SECONDS, HAND_LIFT_SECONDS, HAND_RETURN, HAND_TURN_SECONDS, LIFT_SECONDS, WHITE_LATENT, handBack, handTimes, WIND_HOP_SECONDS, WIND_MAX_SECONDS, dispatchSize, foldDuration, foldProfile, halfToFloat, rollSeconds, windArc, windSeconds, windSpeed } from '../src/sim/mpm';
import { DEFAULT_PARAMS, GEOMETRY, PARAM_LIMITS } from '../src/config/mill';

describe('solver helpers', () => {
  it('splits large particle dispatches into 2D so no dimension exceeds the limit', () => {
    expect(dispatchSize(0, 65535)).toEqual([1, 1]);
    expect(dispatchSize(473, 65535)).toEqual([473, 1]);
    const [x, y] = dispatchSize(200000, 65535);
    expect(x).toBeLessThanOrEqual(65535);
    expect(x * y).toBeGreaterThanOrEqual(200000);
    expect(x * y - 200000).toBeLessThan(x);
  });

  it('fold profile is a smoothstep with zero end velocities', () => {
    expect(foldProfile(0, FOLD_DURATION)).toEqual({ s: 0, dsdt: 0 });
    expect(foldProfile(FOLD_DURATION, FOLD_DURATION).s).toBeCloseTo(1, 9);
    expect(foldProfile(2 * FOLD_DURATION, FOLD_DURATION)).toEqual({ s: 1, dsdt: 0 });
    const mid = foldProfile(FOLD_DURATION / 2, FOLD_DURATION);
    expect(mid.s).toBeCloseTo(0.5, 9);
    expect(mid.dsdt).toBeCloseTo(1.5 / FOLD_DURATION, 9);
    // derivative check by finite difference
    const t = 0.3;
    const eps = 1e-5;
    const fd = (foldProfile(t + eps, FOLD_DURATION).s - foldProfile(t - eps, FOLD_DURATION).s) / (2 * eps);
    expect(foldProfile(t, FOLD_DURATION).dsdt).toBeCloseTo(fd, 5);
  });

  it('decodes IEEE half floats', () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x0000)).toBe(0);
    expect(halfToFloat(0x3555)).toBeCloseTo(1 / 3, 3);
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    expect(Number.isNaN(halfToFloat(0x7e00))).toBe(true);
  });

  it('white base latent has unit pigment weight on the white channel', () => {
    expect(WHITE_LATENT[0] + WHITE_LATENT[1] + WHITE_LATENT[2] + WHITE_LATENT[3]).toBeCloseTo(1, 9);
  });
});

describe('foldDuration', () => {
  it('drops the log right after the roll by default, and lowers it in over several seconds otherwise', () => {
    expect(DEFAULT_PARAMS.logFeed).toBe(0);
    expect(PARAM_LIMITS.logFeed.min).toBe(0);
    expect(foldDuration(0)).toBeCloseTo(FOLD_ROLL_SECONDS + 0.02, 9);
    expect(foldDuration(-1)).toBe(foldDuration(0));
    expect(foldDuration(FOLD_FEED_SPEED)).toBe(FOLD_DURATION);
    expect(FOLD_DURATION).toBeGreaterThan(FOLD_ROLL_SECONDS + 5);
    // a faster feed is a shorter move, never shorter than the drop
    expect(foldDuration(0.6)).toBeLessThan(foldDuration(0.3));
    expect(foldDuration(0.6)).toBeGreaterThan(foldDuration(0));
  });
});

describe('peel and wind timing', () => {
  it('winds at the roll surface speed, floored so a slow roll still finishes, then lifts', () => {
    // the arc from the nip exit round to the crown of a 0.32 roll
    expect(windArc()).toBeCloseTo(1.5 * Math.PI * GEOMETRY.radius, 9);
    // at the default speed the surface is faster than the floor: one arc at omega R
    const omega = DEFAULT_PARAMS.omega;
    expect(windSpeed(omega)).toBeCloseTo(omega * GEOMETRY.radius, 9);
    expect(windSeconds(omega)).toBeCloseTo(windArc() / (omega * GEOMETRY.radius) + WIND_HOP_SECONDS, 9);
    expect(rollSeconds(omega)).toBeCloseTo(windSeconds(omega) + DOUBLE_SECONDS + LIFT_SECONDS, 9);
    expect(FOLD_ROLL_SECONDS).toBe(rollSeconds(omega));
    expect(rollSeconds(omega)).toBeGreaterThan(1.5);
    expect(rollSeconds(omega)).toBeLessThan(4);
    // a stopped or crawling roll: the operator pulls at the floor speed instead
    expect(windSeconds(0)).toBeCloseTo(WIND_MAX_SECONDS + WIND_HOP_SECONDS, 9);
    expect(windSeconds(0.1)).toBe(windSeconds(0));
    // a faster roll winds faster, and the whole move follows
    expect(rollSeconds(6)).toBeLessThan(rollSeconds(3));
    expect(foldDuration(0, 6)).toBeLessThan(foldDuration(0, 3));
    expect(foldDuration(0.3, 6)).toBeCloseTo(rollSeconds(6) + (0.75 + 0.6) / 0.3 + 0.3, 9);
    // the long roll skips the doubling and is lowered in at its full length
    expect(rollSeconds(omega, true)).toBeCloseTo(windSeconds(omega) + LIFT_SECONDS, 9);
    expect(foldDuration(0.3, omega, true)).toBeCloseTo(rollSeconds(omega, true) + (1.5 + 0.6) / 0.3 + 0.3, 9);
  });
});

describe('cut & fold timing', () => {
  it('lifts the edge, carries it back at the roll speed, turns, brings it forward over the nip, sets it down and lets go', () => {
    const omega = DEFAULT_PARAMS.omega;
    const gap = DEFAULT_PARAMS.gap;
    const t = handTimes(omega, gap);
    const D = handBack(gap);
    expect(t.speed).toBeCloseTo(windSpeed(omega), 9);
    expect(t.lift).toBe(HAND_LIFT_SECONDS);
    expect(D).toBeCloseTo(2 * GEOMETRY.radius + gap, 9);
    // the far point: a lift (half a lift's cruise), the cruise back, half the turn (0.3125 of a turn's cruise short)
    const tBack = D / t.speed - 0.5 * t.lift - 0.3125 * HAND_TURN_SECONDS;
    expect(t.back).toBeCloseTo(t.lift + tBack + 0.5 * HAND_TURN_SECONDS, 9);
    // forward again to HAND_RETURN of the way back, the set-down covering half a lift's cruise
    const tFwd = ((1 - HAND_RETURN) * D - 0.3125 * t.speed * HAND_TURN_SECONDS - 0.5 * t.speed * t.lift) / t.speed;
    expect(tFwd).toBeGreaterThan(0);
    expect(t.down).toBeCloseTo(t.lift + tBack + HAND_TURN_SECONDS + tFwd + t.lift, 9);
    expect(t.end).toBeCloseTo(t.down + HAND_HOLD_SECONDS, 9);
    expect(t.back).toBeGreaterThan(t.lift);
    expect(t.down).toBeGreaterThan(t.back);
    expect(t.end).toBeGreaterThan(1.2);
    expect(t.end).toBeLessThan(3);
    // a crawling roll is pulled at the floor speed, so the move still ends
    expect(handTimes(0).end).toBeLessThan(4);
  });
});

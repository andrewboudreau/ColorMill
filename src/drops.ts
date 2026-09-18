/**
 * ColorMill v2 — pigment drop lists (pure TS, no DOM, tested).
 *
 * A drop is one chunk of pigmented putty set down on the bank: which pigment,
 * at which of the DROP_SLOTS fixed spots along the roll (1-based, left to
 * right), and how big. Lists serialise to `pigment@slot.size,...` for the
 * `?drops=` query parameter of index.html, e.g.
 * `cadmiumYellow@3.m,cobaltBlue@5.l,ff8a00@1.s` (a custom colour is written
 * as six hex digits without the '#'). parseDrops is tolerant of junk and
 * resolves names and aliases through findPigment; recipeToDrops turns a
 * colour-mixer recipe (parts per pigment) into such a list.
 */
import { findPigment } from './color/pigments';
import { DROP_SLOTS, PIGMENT_CHUNK_SIZES } from './config/mill';

export type DropSize = 's' | 'm' | 'l';

export interface Drop {
  /** a PIGMENTS key, or '#rrggbb' (lowercase) for a custom colour */
  pigment: string;
  /** 1-based drop slot, 1..DROP_SLOTS (left to right along the roll) */
  slot: number;
  size: DropSize;
}

/** Most drops a list carries; the rest are ignored. */
export const MAX_DROPS = 24;

const CUSTOM_KEY = 'custom';
const SIZES: readonly DropSize[] = PIGMENT_CHUNK_SIZES.map((s) => s.key);
const HEX6 = /^#?([0-9a-f]{6})$/i;

/** The slot a drop lands on when the list does not say: the middle one. */
export function defaultDropSlot(): number {
  return Math.floor((DROP_SLOTS - 1) / 2) + 1;
}

function clampSlot(slot: number): number {
  if (!Number.isFinite(slot)) return defaultDropSlot();
  return Math.min(DROP_SLOTS, Math.max(1, Math.round(slot)));
}

function isSize(s: string): s is DropSize {
  return (SIZES as readonly string[]).includes(s);
}

/**
 * Resolve a pigment token to a PIGMENTS key or a '#rrggbb' custom colour;
 * undefined when it is neither.
 */
function resolvePigment(token: string): string | undefined {
  const name = token.trim();
  if (!name) return undefined;
  const p = findPigment(name);
  if (p) return p.key;
  const m = HEX6.exec(name);
  return m ? `#${m[1].toLowerCase()}` : undefined;
}

/**
 * 'cadmiumYellow@3.m,cobaltBlue@5.l,ff8a00@1.s': one `<key or rrggbb>@<slot>.<size>`
 * per drop, joined with ','. Custom colours lose their '#' so the text needs
 * no escaping in a query string. Empty list -> ''.
 */
export function formatDrops(drops: readonly Drop[]): string {
  return drops
    .map((d) => `${d.pigment.startsWith('#') ? d.pigment.slice(1) : d.pigment}@${d.slot}.${d.size}`)
    .join(',');
}

/**
 * Inverse of formatDrops, tolerant: null/empty -> []; entries are trimmed;
 * unknown pigments are dropped (names and aliases resolve through
 * findPigment; six hex digits, with or without '#', are a custom colour
 * normalised to '#rrggbb' lowercase); a missing or unparsable slot is the
 * middle slot and slots are clamped to 1..DROP_SLOTS; a missing or unknown
 * size is 'm'. At most MAX_DROPS entries are kept.
 */
export function parseDrops(text: string | null | undefined): Drop[] {
  if (!text) return [];
  const out: Drop[] = [];
  for (const piece of text.split(',')) {
    if (out.length >= MAX_DROPS) break;
    const entry = piece.trim();
    if (!entry) continue;
    const at = entry.indexOf('@');
    const pigment = resolvePigment(at >= 0 ? entry.slice(0, at) : entry);
    if (!pigment) continue;
    let slot = defaultDropSlot();
    let size: DropSize = 'm';
    if (at >= 0) {
      const rest = entry.slice(at + 1).trim();
      const dot = rest.indexOf('.');
      const slotText = (dot >= 0 ? rest.slice(0, dot) : rest).trim();
      const sizeText = (dot >= 0 ? rest.slice(dot + 1) : '').trim().toLowerCase();
      if (slotText) slot = clampSlot(parseInt(slotText, 10));
      if (isSize(sizeText)) size = sizeText;
    }
    out.push({ pigment, slot, size });
  }
  return out;
}

/**
 * Turn a colour-mixer recipe into a drop list: every whole part of a pigment
 * is one medium chunk and a leftover half part is one small chunk (2.5 parts
 * -> 2 medium + 1 small). Entries with parts <= 0 or unknown keys are skipped;
 * a 'custom' entry uses `customHex` ('#rrggbb') and is skipped when that is
 * undefined or malformed. Chunks are interleaved (one from each pigment in
 * turn until all are placed) and given slots 1..DROP_SLOTS round-robin in
 * that order, so the colours spread along the roll instead of piling up in
 * one spot. At most MAX_DROPS drops.
 */
export function recipeToDrops(recipe: readonly { key: string; parts: number }[], customHex?: string): Drop[] {
  const queues: DropSize[][] = [];
  const pigments: string[] = [];
  for (const e of recipe) {
    if (!(e.parts > 0)) continue;
    const pigment = e.key === CUSTOM_KEY ? (customHex === undefined ? undefined : resolvePigment(customHex)) : findPigment(e.key)?.key;
    if (!pigment) continue;
    const whole = Math.floor(e.parts);
    const half = e.parts - whole >= 0.5 ? 1 : 0;
    const sizes: DropSize[] = [];
    for (let i = 0; i < whole; i++) sizes.push('m');
    if (half) sizes.push('s');
    if (sizes.length === 0) continue;
    pigments.push(pigment);
    queues.push(sizes);
  }
  const out: Drop[] = [];
  let placed = true;
  while (placed && out.length < MAX_DROPS) {
    placed = false;
    for (let i = 0; i < queues.length && out.length < MAX_DROPS; i++) {
      const size = queues[i].shift();
      if (size === undefined) continue;
      placed = true;
      out.push({ pigment: pigments[i], slot: (out.length % DROP_SLOTS) + 1, size });
    }
  }
  return out;
}

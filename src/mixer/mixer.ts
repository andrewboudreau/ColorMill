/**
 * ColorMill v2 — colour mixer page logic (pure TS, tested, no DOM).
 *
 * A recipe is a list of { key, parts } entries. Mixing is the rule the mill
 * simulation uses: a parts-weighted average of the pigments' Mixbox latents
 * (mixLatents), decoded to sRGB. `naiveHex` is the parts-weighted plain sRGB
 * average of the same pigments, shown beside the Mixbox result so the page
 * can demonstrate why pigment mixing is not channel averaging.
 *
 * Recipes serialise to 'key:parts,key:parts' for URLs and inputs; parseMix is
 * tolerant of junk and resolves names/aliases through findPigment.
 */
import type { Latent } from '../color/pigments';
import { BASE_LATENT, PIGMENTS, findPigment, hexToRgb, latentToHex, lerpLatent, mixLatents, rgbToHex } from '../color/pigments';

export interface RecipeEntry {
  /** a pigment key from PIGMENTS, or CUSTOM_KEY */
  key: string;
  parts: number;
}

export interface CustomPigment {
  latent: Latent;
  /** '#rrggbb' */
  hex: string;
  name?: string;
}

export interface MixResult {
  latent: Latent;
  /** Mixbox result, '#rrggbb' */
  hex: string;
  /** parts-weighted plain sRGB average, '#rrggbb' */
  naiveHex: string;
  totalParts: number;
}

export const CUSTOM_KEY = 'custom';
export const MAX_PARTS = 10;

interface Resolved {
  latent: Latent;
  srgb: readonly [number, number, number];
  name: string;
  parts: number;
}

/** Resolve a recipe entry to its latent/srgb; undefined for unusable entries. */
function resolveEntry(e: RecipeEntry, custom?: CustomPigment): Resolved | undefined {
  if (!(e.parts > 0)) return undefined;
  if (e.key === CUSTOM_KEY) {
    if (!custom) return undefined;
    let srgb: [number, number, number];
    try {
      srgb = hexToRgb(custom.hex);
    } catch {
      return undefined;
    }
    return { latent: custom.latent, srgb, name: custom.name || `Custom ${custom.hex}`, parts: e.parts };
  }
  const p = PIGMENTS[e.key];
  if (!p) return undefined;
  return { latent: p.latent, srgb: p.srgb, name: p.name, parts: e.parts };
}

function resolveRecipe(recipe: readonly RecipeEntry[], custom?: CustomPigment): Resolved[] {
  const out: Resolved[] = [];
  for (const e of recipe) {
    const r = resolveEntry(e, custom);
    if (r) out.push(r);
  }
  return out;
}

/**
 * Mass-weighted Mixbox mix of the recipe (the rule the mill uses: a latent
 * average weighted by parts). Entries with parts <= 0 or unknown keys are
 * ignored; 'custom' uses `custom` (ignored if custom is undefined). With no
 * usable entries: latent = BASE_LATENT, hex = its colour, naiveHex = same,
 * totalParts = 0.
 */
export function mixRecipe(recipe: readonly RecipeEntry[], custom?: CustomPigment): MixResult {
  const entries = resolveRecipe(recipe, custom);
  if (entries.length === 0) {
    const hex = latentToHex(BASE_LATENT);
    return { latent: BASE_LATENT, hex, naiveHex: hex, totalParts: 0 };
  }
  const latent = mixLatents(entries.map((e) => ({ latent: e.latent, weight: e.parts })));
  let totalParts = 0;
  const acc = [0, 0, 0];
  for (const e of entries) {
    totalParts += e.parts;
    for (let i = 0; i < 3; i++) acc[i] += e.parts * e.srgb[i];
  }
  const naiveHex = rgbToHex([acc[0] / totalParts, acc[1] / totalParts, acc[2] / totalParts]);
  return { latent, hex: latentToHex(latent), naiveHex, totalParts };
}

/** Parts printed with up to 2 decimals and no trailing zeros ("3", "1.5"). */
function formatParts(parts: number): string {
  return String(Math.round(parts * 100) / 100);
}

/**
 * 'cadmiumYellow:3,cobaltBlue:1' — only entries with parts > 0, in the order
 * given. Empty recipe -> ''.
 */
export function formatMix(recipe: readonly RecipeEntry[]): string {
  return recipe
    .filter((e) => e.parts > 0)
    .map((e) => `${e.key}:${formatParts(e.parts)}`)
    .join(',');
}

/**
 * Inverse of formatMix, tolerant: null/empty -> []; unknown keys dropped
 * (names and aliases resolve through findPigment); NaN, negative or missing
 * parts dropped; parts clamped to MAX_PARTS; 'custom' kept as is; duplicate
 * keys: last wins (in the position of the first occurrence).
 */
export function parseMix(text: string | null | undefined): RecipeEntry[] {
  if (!text) return [];
  const out: RecipeEntry[] = [];
  for (const piece of text.split(',')) {
    const [rawKey, rawParts] = piece.split(':');
    const name = (rawKey ?? '').trim();
    if (!name || rawParts === undefined || rawParts.trim() === '') continue;
    const parts = Number(rawParts.trim());
    if (!Number.isFinite(parts) || parts < 0) continue;
    const key = name === CUSTOM_KEY ? CUSTOM_KEY : findPigment(name)?.key;
    if (!key) continue;
    const entry = { key, parts: Math.min(parts, MAX_PARTS) };
    const existing = out.findIndex((e) => e.key === key);
    if (existing >= 0) out[existing] = entry;
    else out.push(entry);
  }
  return out;
}

/**
 * '3 parts Cadmium Yellow + 1 part Cobalt Blue' (singular 'part' for exactly
 * 1); custom shows custom.name or 'Custom ' + custom.hex; entries with
 * parts <= 0 skipped; empty -> 'Nothing yet'.
 */
export function describeRecipe(recipe: readonly RecipeEntry[], custom?: CustomPigment): string {
  const entries = resolveRecipe(recipe, custom);
  if (entries.length === 0) return 'Nothing yet';
  return entries
    .map((e) => `${formatParts(e.parts)} ${e.parts === 1 ? 'part' : 'parts'} ${e.name}`)
    .join(' + ');
}

/** Hex colours of the Mixbox lerp from a to b at t = i/(steps-1), i = 0..steps-1 (steps >= 2). */
export function ladder(a: Latent, b: Latent, steps: number): string[] {
  const n = Math.max(2, Math.floor(steps));
  const out: string[] = [];
  for (let i = 0; i < n; i++) out.push(latentToHex(lerpLatent(a, b, i / (n - 1))));
  return out;
}

/** Starter recipes for the page's buttons (palette keys only). */
export const STARTER_RECIPES: ReadonlyArray<{ name: string; recipe: RecipeEntry[] }> = [
  { name: 'Green', recipe: [{ key: 'cadmiumYellow', parts: 3 }, { key: 'cobaltBlue', parts: 1 }] },
  { name: 'Purple', recipe: [{ key: 'cadmiumRed', parts: 1 }, { key: 'cobaltBlue', parts: 1 }] },
  {
    name: 'Skin',
    recipe: [
      { key: 'titaniumWhite', parts: 6 }, { key: 'cadmiumRed', parts: 1 },
      { key: 'cadmiumYellow', parts: 1 }, { key: 'burntSienna', parts: 0.5 }
    ]
  },
  { name: 'Grey', recipe: [{ key: 'titaniumWhite', parts: 4 }, { key: 'ivoryBlack', parts: 1 }] },
  { name: 'Olive', recipe: [{ key: 'cadmiumYellow', parts: 2 }, { key: 'ivoryBlack', parts: 0.5 }] },
  {
    // reference: red, yellow and teal bands milled to #a4634b (docs/design-v2.md §5, reference check)
    name: 'Terracotta',
    recipe: [{ key: 'naphtholRed', parts: 1.5 }, { key: 'hansaYellow', parts: 0.5 }, { key: 'cobaltTeal', parts: 1 }]
  }
];

/**
 * Colour a real two-roll mill produced from red, yellow and teal bands of
 * about equal width (reference footage; the band colours sampled from the
 * frames were #d60720, #e8ef10 and #24a2a2, and Mixbox mixes those three in
 * equal parts to #a27242). The Terracotta starter is the closest recipe from
 * the silicone family (#a15c3f).
 */
export const REFERENCE_TERRACOTTA_HEX = '#a4634b';

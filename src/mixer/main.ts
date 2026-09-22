/**
 * ColorMill colour mixer: a standalone page for sampling pigment recipes
 * outside the mill. It uses the same pigment latents and the same mixing
 * rule as the simulation (a parts-weighted average in Mixbox latent space,
 * src/color/pigments.ts), so the big swatch is the colour the mill converges
 * to for that recipe; the naive RGB average sits beside it for contrast.
 *
 * State is a recipe (parts per pigment, 0..MAX_PARTS) plus an optional custom
 * colour; `?mix=key:parts,...` in the URL restores a recipe. The page exposes
 * window.__mixer for the e2e spec.
 */
import './styles.css';
import { PALETTE_ORDER, PIGMENTS, hexToRgb, rgbToLatentAsync, type Latent } from '../color/pigments';
import { formatDrops, recipeToDrops } from '../drops';
import {
  CUSTOM_KEY, MAX_PARTS, STARTER_RECIPES, describeRecipe, formatMix, ladder, mixRecipe, parseMix,
  type CustomPigment, type MixResult, type RecipeEntry
} from './mixer';

interface MixerApi {
  readonly ready: boolean;
  setParts(key: string, parts: number): void;
  getRecipe(): RecipeEntry[];
  result(): { hex: string; naiveHex: string; latent: number[] };
  reset(): void;
}

declare global {
  interface Window {
    __mixer?: MixerApi;
  }
}

const LADDER_STEPS = 9;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function formatParts(p: number): string {
  return Number.isInteger(p) ? String(p) : p.toFixed(1);
}

function boot(): void {
  const root = document.getElementById('mixer');
  if (!root) throw new Error('mixer root element was not found');

  // --- state -------------------------------------------------------------------
  const parts = new Map<string, number>();
  for (const key of PALETTE_ORDER) parts.set(key, 0);
  parts.set(CUSTOM_KEY, 0);
  let custom: CustomPigment | undefined;
  let customHex = '#ff8a00';
  let lastResult: MixResult = mixRecipe([]);

  const recipe = (): RecipeEntry[] => [...parts.entries()].map(([key, p]) => ({ key, parts: p }));

  // --- layout ------------------------------------------------------------------
  const main = el('main', 'mx-main');
  const nav = el('nav', 'mx-nav');
  for (const [href, label] of [['index.html', 'Simulator'], ['overview.html', 'How it works'], ['project.html', 'Project notes'], ['pigment.html', 'Pigment demo'], ['mixer.html', 'Colour mixer']]) {
    const a = el('a', undefined, label);
    a.href = href;
    if (href === 'mixer.html') a.setAttribute('aria-current', 'page');
    nav.appendChild(a);
  }
  const head = el('div', 'mx-head');
  head.appendChild(el('h1', undefined, 'Colour mixer'));
  head.appendChild(el('p', undefined, 'Pick pigments in parts and see what the mill will make of them. The big swatch is mixed the way the simulation mixes, a parts-weighted average of the same Mixbox pigment latents, so it is the colour the batch converges to once it is milled through. The small one is the plain RGB average, which is what naive mixing would give.'));
  main.append(nav, head);

  const grid = el('div', 'mx-grid');
  const left = el('section', 'mx-panel');
  left.appendChild(el('h2', undefined, 'Pigments'));
  const right = el('section', 'mx-panel mx-result');
  right.appendChild(el('h2', undefined, 'Result'));
  grid.append(left, right);
  main.appendChild(grid);
  root.appendChild(main);

  // --- pigment rows ------------------------------------------------------------
  const rows = new Map<string, { row: HTMLElement; range: HTMLInputElement; count: HTMLElement }>();
  const makeRow = (key: string, name: string, swatch: HTMLElement, hint?: string): void => {
    const row = el('div', 'mixer-row');
    row.dataset.key = key;
    const label = el('div', 'mixer-name', name);
    if (hint) label.appendChild(el('small', undefined, hint));
    const range = el('input', 'mixer-parts');
    range.type = 'range';
    range.min = '0';
    range.max = String(MAX_PARTS);
    range.step = '0.5';
    range.value = '0';
    range.setAttribute('aria-label', `${name} parts`);
    const count = el('div', 'mixer-count', '0');
    range.addEventListener('input', () => setParts(key, parseFloat(range.value)));
    row.append(swatch, label, range, count);
    left.appendChild(row);
    rows.set(key, { row, range, count });
  };
  for (const key of PALETTE_ORDER) {
    const p = PIGMENTS[key];
    if (!p) continue;
    const sw = el('div', 'mixer-swatch');
    sw.style.background = p.hex;
    makeRow(key, p.name, sw, p.hex);
  }
  // custom colour: the native picker is the swatch; the latent comes from mixbox.js on demand
  const picker = el('input');
  picker.type = 'color';
  picker.value = customHex;
  picker.setAttribute('aria-label', 'Custom pigment colour');
  const pickerWrap = el('div', 'mixer-custom');
  pickerWrap.appendChild(picker);
  makeRow(CUSTOM_KEY, 'Custom colour', pickerWrap, 'from the picker');
  let customLoad = 0;
  const loadCustom = (hex: string): void => {
    customHex = hex;
    const token = ++customLoad;
    rgbToLatentAsync(hex).then((latent) => {
      if (token !== customLoad) return;
      custom = { latent, hex, name: `Custom ${hex}` };
      render();
    }).catch((e: unknown) => {
      // without the library the picker still shows something: treat the colour as its own residual
      const rgb = hexToRgb(hex);
      custom = { latent: [0, 0, 0, 1, rgb[0] - 1, rgb[1] - 1, rgb[2] - 1] as unknown as Latent, hex, name: `Custom ${hex}` };
      render();
      console.warn('mixbox.js unavailable, custom colour approximated', e);
    });
  };
  picker.addEventListener('input', () => { loadCustom(picker.value); });

  // --- result panel --------------------------------------------------------------
  const swatches = el('div', 'mx-swatches');
  const mixCol = el('div');
  const mixSwatch = el('div', 'mx-swatch is-big');
  mixSwatch.id = 'mix-swatch';
  const mixCap = el('p', 'mx-caption');
  mixCap.append(el('span', undefined, 'Mixed on the mill'), (() => { const c = el('code', undefined, '#ffffff'); c.id = 'mix-hex'; return c; })());
  mixCol.append(mixSwatch, mixCap);
  const naiveCol = el('div');
  const naiveSwatch = el('div', 'mx-swatch is-big');
  naiveSwatch.id = 'naive-swatch';
  const naiveCap = el('p', 'mx-caption');
  naiveCap.append(el('span', undefined, 'Naive RGB average'), (() => { const c = el('code', undefined, '#ffffff'); c.id = 'naive-hex'; return c; })());
  naiveCol.append(naiveSwatch, naiveCap);
  swatches.append(mixCol, naiveCol);
  right.appendChild(swatches);

  const recipeText = el('p', 'mx-recipe', 'Nothing yet');
  recipeText.id = 'recipe-text';
  right.appendChild(recipeText);

  const ladderEl = el('div', 'ladder');
  ladderEl.setAttribute('aria-label', 'Blend ladder between the two main pigments');
  const ladderCap = el('p', 'mx-ladder-caption', '');
  right.append(ladderEl, ladderCap);

  const actions = el('div', 'mx-actions');
  const share = el('a', 'mx-btn mx-btn-primary', 'Share link');
  share.id = 'share-link';
  share.href = '#';
  const copy = el('button', 'mx-btn', 'Copy recipe');
  copy.type = 'button';
  copy.addEventListener('click', () => {
    const text = `${describeRecipe(recipe(), custom)} → ${lastResult.hex}`;
    navigator.clipboard?.writeText(text).then(() => { copy.textContent = 'Copied'; setTimeout(() => { copy.textContent = 'Copy recipe'; }, 1500); }).catch(() => { copy.textContent = text; });
  });
  const clear = el('button', 'mx-btn', 'Clear');
  clear.type = 'button';
  clear.addEventListener('click', () => reset());
  // hand the recipe to the simulator: one medium chunk per part (halves as small
  // chunks), spread along the roll, via the mill's ?drops= parameter
  const openMill = el('a', 'mx-btn', 'Open in the mill');
  openMill.id = 'open-mill';
  openMill.href = 'index.html';
  openMill.title = 'Start the simulator with these pigments already dropped on the bank';
  actions.append(share, copy, clear, openMill);
  right.appendChild(actions);

  const startersTitle = el('h2', undefined, 'Starters');
  startersTitle.style.marginTop = '18px';
  const starters = el('div', 'mx-starters');
  for (const s of STARTER_RECIPES) {
    const b = el('button', 'starter');
    b.type = 'button';
    const dot = el('span', 'starter-dot');
    dot.style.background = mixRecipe(s.recipe).hex;
    b.append(dot, el('span', undefined, s.name));
    b.addEventListener('click', () => { load(s.recipe); });
    starters.appendChild(b);
  }
  right.append(startersTitle, starters);

  const note = el('p', 'mx-note');
  note.innerHTML = 'Parts are by mass of masterbatch. Mixing uses <a href="pigment.html">Mixbox</a> pigment latents, the same ones the simulator carries on every particle; drop the same pigments on the bank in the same proportions and the sheet ends up this colour once it is milled through. Checked against a real mill: red, yellow and teal bands of equal width milled to <span class="mx-ref" style="--c:#a4634b"></span> #a4634b, and Mixbox puts the same three colours in equal parts at #a27242; the Terracotta starter is the nearest palette recipe.';
  right.appendChild(note);

  // --- behaviour -----------------------------------------------------------------
  function render(): void {
    const r = recipe();
    lastResult = mixRecipe(r, custom);
    mixSwatch.style.backgroundColor = lastResult.hex;
    (document.getElementById('mix-hex') as HTMLElement).textContent = lastResult.hex;
    naiveSwatch.style.backgroundColor = lastResult.naiveHex;
    (document.getElementById('naive-hex') as HTMLElement).textContent = lastResult.naiveHex;
    recipeText.textContent = describeRecipe(r, custom);
    for (const [key, p] of parts) {
      const ui = rows.get(key);
      if (!ui) continue;
      if (parseFloat(ui.range.value) !== p) ui.range.value = String(p);
      ui.count.textContent = formatParts(p);
      ui.row.classList.toggle('is-in', p > 0);
    }
    // blend ladder between the two pigments with the most parts
    const top = r.filter((e) => e.parts > 0 && (e.key !== CUSTOM_KEY || custom)).sort((a, b) => b.parts - a.parts).slice(0, 2);
    ladderEl.replaceChildren();
    if (top.length === 2) {
      const lat = (e: RecipeEntry): Latent => (e.key === CUSTOM_KEY ? (custom as CustomPigment).latent : PIGMENTS[e.key].latent);
      const name = (e: RecipeEntry): string => (e.key === CUSTOM_KEY ? (custom?.name ?? 'Custom') : PIGMENTS[e.key].name);
      for (const hex of ladder(lat(top[0]), lat(top[1]), LADDER_STEPS)) {
        const c = el('div', 'ladder-cell');
        c.style.background = hex;
        c.title = hex;
        ladderEl.appendChild(c);
      }
      ladderCap.textContent = `${name(top[0])} to ${name(top[1])}, mixed step by step`;
    } else {
      ladderCap.textContent = top.length === 1 ? 'Add a second pigment to see the blend ladder' : '';
    }
    // the recipe reads as written in the URL: ':' and ',' are fine in a query string
    const q = formatMix(r);
    const href = `${window.location.origin}${window.location.pathname}${q ? `?mix=${q}` : ''}`;
    share.href = href;
    history.replaceState(null, '', href);
    const drops = formatDrops(recipeToDrops(r, custom?.hex));
    openMill.href = drops ? `index.html?drops=${drops}` : 'index.html';
  }

  function setParts(key: string, p: number): void {
    if (!parts.has(key)) return;
    const v = Math.min(MAX_PARTS, Math.max(0, Number.isFinite(p) ? Math.round(p * 2) / 2 : 0));
    parts.set(key, v);
    if (key === CUSTOM_KEY && v > 0 && !custom) loadCustom(customHex);
    render();
  }

  function load(entries: readonly RecipeEntry[]): void {
    for (const key of parts.keys()) parts.set(key, 0);
    for (const e of entries) if (parts.has(e.key)) parts.set(e.key, Math.min(MAX_PARTS, Math.max(0, e.parts)));
    if ((parts.get(CUSTOM_KEY) ?? 0) > 0 && !custom) loadCustom(customHex);
    render();
  }

  function reset(): void {
    load([]);
  }

  // --- start ---------------------------------------------------------------------
  const initial = parseMix(new URLSearchParams(window.location.search).get('mix'));
  load(initial);

  window.__mixer = {
    ready: true,
    setParts,
    getRecipe: () => recipe(),
    result: () => ({ hex: lastResult.hex, naiveHex: lastResult.naiveHex, latent: Array.from(lastResult.latent) }),
    reset
  };
}

boot();

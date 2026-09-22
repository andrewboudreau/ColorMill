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
import { recipeToDrops } from '../drops';
import { LINK_PARAM_KEYS, PARAM_LABELS, clampParam, formatMillQuery, isDefaultParam, parseMillStart } from '../config/link';
import { BATCH_CHOICES, DEFAULT_PARAMS, PARAM_LIMITS, QUALITY_PRESETS, type MillParams, type QualityPreset } from '../config/mill';
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
  /** mill settings forwarded to the simulator (only non-defaults reach the link) */
  setMill(key: keyof MillParams, value: number): void;
  setMillPreset(preset: QualityPreset | undefined): void;
  setMillBatch(batch: number): void;
  millQuery(): string;
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
  // mill settings forwarded to the simulator: the URL's, else the defaults
  const start = parseMillStart(window.location.search);
  const mill: MillParams = { ...DEFAULT_PARAMS, ...start.params };
  let millPreset: QualityPreset | undefined = start.preset;
  let millBatch = start.batch;
  const millQuery = (): string => formatMillQuery({ batch: millBatch, preset: millPreset, params: mill });
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
  head.appendChild(el('p', undefined, 'Pick pigments in parts, see what the mill will make of them, then open the mill with those chunks already on the bank. The big swatch is mixed the way the simulation mixes, a parts-weighted average of the same Mixbox pigment latents, so it is the colour the batch converges to once it is milled through; the small one is the plain RGB average. Mill settings live under the button and only reach the link when you change them.'));
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
  let family: string | undefined;
  for (const key of PALETTE_ORDER) {
    const p = PIGMENTS[key];
    if (!p) continue;
    if (p.family !== family) {
      family = p.family;
      const h = el('h3', 'mixer-family', family === 'silicone' ? 'Silicone pastes' : 'Artist pigments');
      h.title = family === 'silicone'
        ? 'Pigments silicone colour houses use, or stand-ins for their common paste colours'
        : 'Mixbox oil-paint pigments kept for range; cadmium and cobalt are not used in silicone';
      left.appendChild(h);
    }
    const sw = el('div', 'mixer-swatch');
    sw.style.background = p.hex;
    sw.title = p.note;
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

  // hand the recipe to the simulator: one medium chunk per part (halves as small
  // chunks), spread along the roll, via the mill's ?drops= parameter, plus any
  // mill setting changed below
  const openMill = el('a', 'mx-btn mx-btn-primary mx-btn-big', 'Open in the mill');
  openMill.id = 'open-mill';
  openMill.href = 'index.html';
  openMill.title = 'Start the simulator with these pigments already dropped on the bank';
  const openHint = el('p', 'mx-open-hint', '');
  openHint.id = 'open-hint';
  right.append(openMill, openHint);

  const actions = el('div', 'mx-actions');
  const share = el('a', 'mx-btn', 'Share link');
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
  actions.append(share, copy, clear);
  right.appendChild(actions);

  // --- mill settings (advanced): the simulator's drawer, forwarded in the link ---------
  const adv = el('details', 'mx-advanced');
  adv.id = 'mill-settings';
  const advSummary = el('summary');
  const advTitle = el('span', undefined, 'Mill settings');
  const advState = el('span', 'mx-adv-state', 'defaults');
  advState.id = 'mill-settings-state';
  advSummary.append(advTitle, advState);
  adv.appendChild(advSummary);
  const advBody = el('div', 'mx-adv-body');
  adv.appendChild(advBody);
  const selectRow = (label: string, select: HTMLSelectElement): void => {
    const row = el('label', 'mx-adv-row');
    row.append(el('span', 'mx-adv-label', label), select);
    advBody.appendChild(row);
  };
  const presetSel = el('select', 'mx-adv-select');
  presetSel.name = 'preset';
  for (const [v, label] of [['', 'Auto (by GPU)'], ...(Object.keys(QUALITY_PRESETS) as QualityPreset[]).map((p) => [p, p] as const)]) {
    const o = el('option', undefined, label);
    o.value = v;
    presetSel.appendChild(o);
  }
  presetSel.addEventListener('change', () => setMillPreset(presetSel.value ? (presetSel.value as QualityPreset) : undefined));
  selectRow('Quality', presetSel);
  const batchSel = el('select', 'mx-adv-select');
  batchSel.name = 'batch';
  const batchChoices = BATCH_CHOICES.includes(millBatch) ? BATCH_CHOICES : [...BATCH_CHOICES, millBatch].sort((a, b) => a - b);
  for (const b of batchChoices) {
    const o = el('option', undefined, `${b}×`);
    o.value = String(b);
    batchSel.appendChild(o);
  }
  batchSel.addEventListener('change', () => setMillBatch(parseFloat(batchSel.value)));
  selectRow('Batch', batchSel);
  const millRows = new Map<keyof MillParams, { range: HTMLInputElement; value: HTMLElement }>();
  for (const key of LINK_PARAM_KEYS) {
    const spec = PARAM_LABELS[key];
    const lim = PARAM_LIMITS[key];
    const row = el('label', 'mx-adv-row mx-adv-slider');
    const name = el('span', 'mx-adv-label', spec.label);
    const value = el('span', 'mx-adv-value', spec.format(mill[key]));
    const range = el('input', 'mx-adv-range');
    range.type = 'range';
    range.name = key;
    range.min = String(lim.min);
    range.max = String(lim.max);
    range.step = String(lim.step);
    range.value = String(mill[key]);
    range.addEventListener('input', () => setMill(key, parseFloat(range.value)));
    row.append(name, value, range);
    advBody.appendChild(row);
    millRows.set(key, { range, value });
  }
  const advReset = el('button', 'mx-btn mx-btn-small', 'Reset to defaults');
  advReset.type = 'button';
  advReset.addEventListener('click', () => {
    for (const key of LINK_PARAM_KEYS) mill[key] = DEFAULT_PARAMS[key];
    millPreset = undefined;
    millBatch = 1;
    render();
  });
  advBody.appendChild(advReset);
  advBody.appendChild(el('p', 'mx-adv-note', 'The same sliders as the simulator\'s drawer. Only settings that differ from the defaults are written into the links, so most links stay short.'));
  right.appendChild(adv);

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
    // mill settings rows
    presetSel.value = millPreset ?? '';
    batchSel.value = String(millBatch);
    for (const [key, ui] of millRows) {
      if (parseFloat(ui.range.value) !== mill[key]) ui.range.value = String(mill[key]);
      ui.value.textContent = PARAM_LABELS[key].format(mill[key]);
      ui.range.parentElement?.classList.toggle('is-changed', !isDefaultParam(key, mill[key]));
    }
    const changed = LINK_PARAM_KEYS.filter((k) => !isDefaultParam(k, mill[k])).length + (millPreset ? 1 : 0) + (millBatch !== 1 ? 1 : 0);
    advState.textContent = changed ? `${changed} changed` : 'defaults';
    adv.classList.toggle('is-changed', changed > 0);
    // the recipe reads as written in the URL: ':' and ',' are fine in a query string;
    // the mill settings ride along so a shared mixer link restores them too
    const settings = millQuery();
    const q = formatMix(r);
    const pieces = [q ? `mix=${q}` : '', settings].filter(Boolean);
    const href = `${window.location.origin}${window.location.pathname}${pieces.length ? `?${pieces.join('&')}` : ''}`;
    share.href = href;
    history.replaceState(null, '', href);
    const drops = recipeToDrops(r, custom?.hex);
    const millHref = formatMillQuery({ drops, batch: millBatch, preset: millPreset, params: mill });
    openMill.href = millHref ? `index.html?${millHref}` : 'index.html';
    const n = drops.length;
    openHint.textContent = n
      ? `Starts the mill with ${n} chunk${n === 1 ? '' : 's'} on the bank${changed ? ` and ${changed} setting${changed === 1 ? '' : 's'} changed` : ''}.`
      : changed ? `Starts an empty mill with ${changed} setting${changed === 1 ? '' : 's'} changed.` : 'Starts the mill with nothing dropped yet.';
  }

  function setMill(key: keyof MillParams, value: number): void {
    if (!(key in PARAM_LIMITS)) return;
    mill[key] = clampParam(key, Number.isFinite(value) ? value : DEFAULT_PARAMS[key]);
    render();
  }

  function setMillPreset(preset: QualityPreset | undefined): void {
    millPreset = preset && preset in QUALITY_PRESETS ? preset : undefined;
    render();
  }

  function setMillBatch(batch: number): void {
    millBatch = Number.isFinite(batch) && batch > 0 ? batch : 1;
    render();
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
    setMill,
    setMillPreset,
    setMillBatch,
    millQuery,
    result: () => ({ hex: lastResult.hex, naiveHex: lastResult.naiveHex, latent: Array.from(lastResult.latent) }),
    reset
  };
}

boot();

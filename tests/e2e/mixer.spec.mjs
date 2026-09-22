/**
 * End-to-end test of the pigment mixer page (mixer.html) driven through the
 * API it exposes on `window.__mixer`. No WebGPU needed. Checks:
 *   1. the page becomes ready without console/page errors
 *   2. setParts() updates the Mixbox result, swatch, hex text, recipe text,
 *      share link, range inputs and the ladder; naive mix differs
 *   3. reset() clears the recipe and shows white
 *   4. ?mix=... in the URL restores a recipe on load
 */
import { assert, withBrowser } from './lib.mjs';

const READY_TIMEOUT_MS = 60_000;

const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

/** Record console errors / page errors (favicon 404s are ignored). */
function collectErrors(page) {
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error' && !/favicon/i.test(m.text())) errors.push(`console: ${m.text()}`); });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  return errors;
}

async function open(page, url) {
  await page.goto(url, { waitUntil: 'load', timeout: 60_000 });
  await page.waitForFunction(() => window.__mixer && window.__mixer.ready, null, { timeout: READY_TIMEOUT_MS });
}

/** Read the result, DOM hexes and parts in-page; computed swatch colours are converted to '#rrggbb'. */
async function readState(page) {
  return page.evaluate(() => {
    const cssToHex = (css) => {
      const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(css);
      return m ? '#' + [m[1], m[2], m[3]].map((v) => Number(v).toString(16).padStart(2, '0')).join('') : css;
    };
    const parts = {};
    for (const row of document.querySelectorAll('.mixer-row[data-key]')) {
      const input = row.querySelector('input[type=range].mixer-parts');
      parts[row.dataset.key] = input ? input.value : null;
    }
    const r = window.__mixer.result();
    return {
      hex: r.hex, naiveHex: r.naiveHex, latent: Array.from(r.latent),
      mixHexText: (document.querySelector('#mix-hex')?.textContent || '').trim(),
      mixSwatchHex: cssToHex(getComputedStyle(document.querySelector('#mix-swatch')).backgroundColor),
      naiveHexText: (document.querySelector('#naive-hex')?.textContent || '').trim(),
      recipeText: document.querySelector('#recipe-text')?.textContent || '',
      shareHref: document.querySelector('#share-link')?.getAttribute('href') || '',
      ladderCells: document.querySelectorAll('.ladder .ladder-cell').length,
      starters: document.querySelectorAll('.starter').length,
      recipe: window.__mixer.getRecipe(),
      openMillHref: document.querySelector('#open-mill')?.getAttribute('href') || '',
      millState: document.querySelector('#mill-settings-state')?.textContent || '',
      millQuery: window.__mixer.millQuery(),
      parts
    };
  });
}

export default async function run() {
  await withBrowser(async ({ page, baseUrl }) => {
    const errors = collectErrors(page);
    await open(page, `${baseUrl}mixer.html`);
    console.log('  mixer ready');

    // --- 1. static DOM ------------------------------------------------------------------
    const s0 = await readState(page);
    assert(s0.starters >= 3, `at least 3 .starter buttons (got ${s0.starters})`);
    assert(Object.keys(s0.parts).length >= 2 && Object.values(s0.parts).every((v) => v !== null),
      `every .mixer-row[data-key] has a range input (${JSON.stringify(s0.parts)})`);

    // --- 2. yellow + blue -> green, naive differs, DOM consistent -----------------------
    await page.evaluate(() => { window.__mixer.setParts('cadmiumYellow', 3); window.__mixer.setParts('cobaltBlue', 1); });
    const s1 = await readState(page);
    console.log(`  3 cadmiumYellow + 1 cobaltBlue -> ${s1.hex} (naive ${s1.naiveHex})`);
    assert(/^#[0-9a-f]{6}$/.test(s1.hex), `result().hex is lowercase #rrggbb (got ${s1.hex})`);
    assert(s1.mixHexText === s1.hex, `#mix-hex shows result hex (${s1.mixHexText} vs ${s1.hex})`);
    assert(s1.mixSwatchHex === s1.hex, `#mix-swatch background matches result hex (${s1.mixSwatchHex} vs ${s1.hex})`);
    assert(s1.naiveHexText === s1.naiveHex, `#naive-hex shows naive hex (${s1.naiveHexText} vs ${s1.naiveHex})`);
    const [r1, g1, b1] = hexToRgb(s1.hex);
    assert(g1 > r1 && g1 > b1, `yellow + blue mixes to green: g is the largest channel (rgb ${r1},${g1},${b1})`);
    assert(s1.naiveHex !== s1.hex, `naive RGB mix differs from Mixbox result (${s1.naiveHex} vs ${s1.hex})`);
    assert(s1.latent.length === 7 && s1.latent.every(Number.isFinite), `latent has 7 finite entries (${JSON.stringify(s1.latent)})`);
    assert(s1.parts.cadmiumYellow === '3' && s1.parts.cobaltBlue === '1',
      `range inputs reflect parts (yellow ${s1.parts.cadmiumYellow}, blue ${s1.parts.cobaltBlue})`);
    assert(s1.recipeText.includes('Cadmium Yellow') && s1.recipeText.includes('Cobalt Blue'), `#recipe-text names both pigments ("${s1.recipeText.trim()}")`);
    assert(s1.shareHref.includes('mix=') && s1.shareHref.includes('cadmiumYellow:3') && s1.shareHref.includes('cobaltBlue:1'),
      `#share-link href encodes the recipe (${s1.shareHref})`);
    assert(s1.ladderCells >= 2, `ladder has at least 2 cells (got ${s1.ladderCells})`);
    assert(s1.openMillHref.includes('drops=') && s1.openMillHref.includes('cadmiumYellow@') && s1.openMillHref.includes('cobaltBlue@'),
      `#open-mill href hands the recipe to the mill as ?drops= (${s1.openMillHref})`);

    // --- 2b. mill settings ride along in both links, only when changed -----------------------
    assert(s1.millState === 'defaults' && s1.millQuery === '', `mill settings start at the defaults (${s1.millState}, "${s1.millQuery}")`);
    await page.evaluate(() => { window.__mixer.setMill('gap', 0.06); window.__mixer.setMillPreset('high'); window.__mixer.setMill('gravity', 2.0); });
    const s1b = await readState(page);
    console.log(`  mill settings: ${s1b.millQuery} (${s1b.millState})`);
    assert(s1b.millQuery === 'preset=high&gap=0.06', `only the changed settings are written, in link order (${s1b.millQuery})`);
    assert(s1b.millState === '2 changed', `summary counts the changed settings (${s1b.millState})`);
    assert(s1b.openMillHref.startsWith('index.html?drops=') && s1b.openMillHref.endsWith('&preset=high&gap=0.06'), `#open-mill carries drops then settings (${s1b.openMillHref})`);
    assert(s1b.shareHref.includes('mix=') && s1b.shareHref.endsWith('&preset=high&gap=0.06'), `#share-link carries the settings too (${s1b.shareHref})`);
    await page.evaluate(() => { window.__mixer.setMill('gap', 0.04); window.__mixer.setMillPreset(undefined); });
    const s1c = await readState(page);
    assert(s1c.millQuery === '' && !s1c.openMillHref.includes('gap='), `back at the defaults nothing is written (${s1c.openMillHref})`);

    // --- 3. reset -> empty recipe, white ---------------------------------------------------
    await page.evaluate(() => window.__mixer.reset());
    const s2 = await readState(page);
    assert(Array.isArray(s2.recipe) && !s2.recipe.some((e) => e.parts > 0), `reset() leaves no pigment with parts > 0 (${JSON.stringify(s2.recipe)})`);
    const rgb2 = hexToRgb(s2.mixHexText);
    assert(rgb2.every((v) => v > 0.9 * 255), `after reset #mix-hex is white-ish (${s2.mixHexText})`);
    console.log(`  reset -> ${s2.mixHexText}`);

    // --- 4. ?mix= restores a recipe on load -------------------------------------------------
    await open(page, `${baseUrl}mixer.html?mix=cadmiumRed:1,cobaltBlue:1&batch=1.5&logFeed=0.15`);
    const s3 = await readState(page);
    assert(s3.millQuery === 'batch=1.5&logFeed=0.15', `mill settings restored from the URL (${s3.millQuery})`);
    assert(s3.openMillHref.endsWith('&batch=1.5&logFeed=0.15'), `and forwarded to the mill (${s3.openMillHref})`);
    console.log(`  ?mix=cadmiumRed:1,cobaltBlue:1 -> ${s3.hex}`);
    assert(s3.parts.cadmiumRed === '1' && s3.parts.cobaltBlue === '1',
      `range inputs restored from URL (red ${s3.parts.cadmiumRed}, blue ${s3.parts.cobaltBlue})`);
    const [r3, g3, b3] = hexToRgb(s3.hex);
    assert(r3 > g3 && b3 > g3, `red + blue mixes to purple-ish (rgb ${r3},${g3},${b3})`);
    assert(s3.mixHexText === s3.hex, `#mix-hex shows restored result (${s3.mixHexText} vs ${s3.hex})`);

    assert(errors.length === 0, `no console/page errors:\n${errors.join('\n')}`);
  });
}

/**
 * Runs every *.spec.mjs in tests/e2e sequentially. Each spec exports
 * `export default async function run()` and throws on failure.
 *
 *   node tests/e2e/run.mjs                 # all specs
 *   node tests/e2e/run.mjs app             # specs whose file name contains "app"
 *   node tests/e2e/run.mjs render mixer    # any of several names (CI shards this way)
 */
import { readdirSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const dir = path.dirname(new URL(import.meta.url).pathname);
const only = process.argv.slice(2).filter(Boolean);
const specs = readdirSync(dir).filter((f) => f.endsWith('.spec.mjs') && (!only.length || only.some((o) => f.includes(o)))).sort();
if (!specs.length) { console.log(`no spec matches ${only.join(', ')}`); process.exit(1); }
let failed = 0;
for (const f of specs) {
  const t0 = Date.now();
  try {
    const mod = await import(pathToFileURL(path.join(dir, f)).href);
    await mod.default();
    console.log(`PASS ${f} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${f}: ${e && e.stack ? e.stack : e}`);
  }
}
if (failed) { console.log(`${failed} spec(s) failed`); process.exit(1); }
console.log(`${specs.length} spec(s) passed`);
process.exit(0);

/**
 * Shared helpers for end-to-end tests: start a Vite server and launch a
 * headless Chromium with a WebGPU adapter (SwiftShader when no GPU exists).
 *
 * Usage:
 *   import { withBrowser } from './lib.mjs';
 *   await withBrowser(async ({ page, baseUrl }) => { ... });
 */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

/** Repository root (this file lives in tests/e2e). */
export const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', '..');

const CHROMIUM_CANDIDATES = [
  process.env.COLORMILL_CHROMIUM,
  '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
].filter(Boolean);

export function chromiumExecutable() {
  for (const c of CHROMIUM_CANDIDATES) if (c && existsSync(c)) return c;
  return undefined; // fall back to playwright's default
}

/** Chromium flags that expose WebGPU headlessly (SwiftShader fallback). */
export const WEBGPU_ARGS = [
  '--headless=new',
  '--enable-unsafe-webgpu',
  '--enable-features=WebGPU,Vulkan',
  '--use-webgpu-adapter=swiftshader',
  '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist',
  '--use-angle=swiftshader'
];

export async function launchBrowser() {
  return chromium.launch({
    executablePath: chromiumExecutable(),
    headless: true,
    args: WEBGPU_ARGS,
    // playwright adds --disable-gpu by default, which removes navigator.gpu
    ignoreDefaultArgs: ['--disable-gpu', '--disable-gpu-compositing']
  });
}

/**
 * Start `vite` (dev) or `vite preview` (after a build) on a free port.
 * Resolves with { baseUrl, stop }.
 */
export async function startServer({ mode = process.env.E2E_MODE || 'dev', port = 5179 } = {}) {
  const viteBin = path.join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js');
  const args = mode === 'preview' ? [viteBin, 'preview', '--port', String(port), '--strictPort'] : [viteBin, '--port', String(port), '--strictPort'];
  const child = spawn(process.execPath, args, { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'], detached: true, env: { ...process.env, BROWSER: 'none' } });
  let out = '';
  child.stdout.on('data', (d) => { out += d.toString(); });
  child.stderr.on('data', (d) => { out += d.toString(); });
  const base = mode === 'preview' ? `http://127.0.0.1:${port}/ColorMill/` : `http://127.0.0.1:${port}/ColorMill/`;
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(base);
      if (r.ok || r.status === 404) break;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 200));
  }
  if (Date.now() >= deadline) {
    child.kill();
    throw new Error('vite did not start:\n' + out);
  }
  return {
    baseUrl: base,
    stop: () => new Promise((resolve) => {
      child.once('exit', () => resolve());
      try { process.kill(-child.pid, 'SIGTERM'); } catch { child.kill(); }
      setTimeout(resolve, 2000).unref();
    })
  };
}

/** Run fn with a page whose console/errors are echoed to the terminal. */
export async function withBrowser(fn, opts = {}) {
  const server = await startServer(opts);
  const browser = await launchBrowser();
  try {
    const page = await browser.newPage({ viewport: { width: 960, height: 640 } });
    page.on('console', (m) => { if (['error', 'warning'].includes(m.type()) || process.env.E2E_VERBOSE) console.log(`[browser:${m.type()}]`, m.text()); });
    page.on('pageerror', (e) => console.log('[pageerror]', e.message));
    return await fn({ page, browser, baseUrl: server.baseUrl });
  } finally {
    await browser.close();
    await server.stop();
  }
}

export function assert(cond, msg) {
  if (!cond) throw new Error('assertion failed: ' + msg);
}

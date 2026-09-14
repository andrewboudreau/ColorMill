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
import zlib from 'node:zlib';
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

/** Pick a free TCP port on localhost. */
export async function freePort() {
  const { createServer } = await import('node:net');
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => { const { port } = srv.address(); srv.close(() => resolve(port)); });
  });
}

/**
 * Start `vite` (dev) or `vite preview` (after a build) on a free port.
 * Resolves with { baseUrl, stop }.
 */
export async function startServer({ mode = process.env.E2E_MODE || 'dev', port = 0 } = {}) {
  if (!port) port = await freePort();
  const viteBin = path.join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js');
  // bind IPv4 explicitly: on hosts where `localhost` resolves to ::1 the 127.0.0.1 probe below would never connect
  const common = ['--host', '127.0.0.1', '--port', String(port), '--strictPort'];
  const args = mode === 'preview' ? [viteBin, 'preview', ...common] : [viteBin, ...common];
  const child = spawn(process.execPath, args, { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'], detached: true, env: { ...process.env, BROWSER: 'none', E2E_NO_HMR: '1' } });
  let out = '';
  child.stdout.on('data', (d) => { out += d.toString(); });
  child.stderr.on('data', (d) => { out += d.toString(); });
  const base = mode === 'preview' ? `http://127.0.0.1:${port}/ColorMill/` : `http://127.0.0.1:${port}/ColorMill/`;
  const deadline = Date.now() + 30000;
  let exited = false;
  child.once('exit', () => { exited = true; });
  while (Date.now() < deadline && !exited) {
    try {
      const r = await fetch(base);
      if (r.ok || r.status === 404) break;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 200));
  }
  if (exited || Date.now() >= deadline) {
    try { process.kill(-child.pid, 'SIGTERM'); } catch { /* already gone */ }
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

const CRC_TABLE = new Int32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  CRC_TABLE[n] = c;
}
function crc32(buf) {
  let c = -1;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

/** Encode tightly packed RGBA8 pixels as a PNG buffer. */
export function encodePng(width, height, rgba) {
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(td));
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 4 + 1)] = 0;
    raw.set(rgba.subarray(y * width * 4, (y + 1) * width * 4), y * (width * 4 + 1) + 1);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0))
  ]);
}

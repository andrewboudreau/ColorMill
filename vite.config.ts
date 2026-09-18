import { cpSync, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
import { defineConfig, type Plugin } from 'vitest/config';

/**
 * Static docs that ship next to the app, sourced from `web/` (the single
 * source of truth — do not add a `public/` mirror). `web/shell.html` is the
 * Emscripten shell for the legacy v1 build and is deliberately NOT copied.
 *
 * Published at the dist root so the app's nav links (`project.html`,
 * `vendor/mixbox/mixbox.js`, ...) work both from `dist/` and from the
 * legacy `dist/legacy/` copy made by `make web DIST_DIR=dist/legacy`.
 */
const WEB_DIR = 'web';
const DOC_PAGES = ['overview.html', 'project.html', 'resources.html', 'pigment.html'];
const DOC_DIRS = ['vendor'];

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.glsl': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.png': 'image/png',
  '.css': 'text/css; charset=utf-8'
};

function docsPagesPlugin(): Plugin {
  let root = process.cwd();
  let base = '/';
  let outDir = 'dist';
  let isBuild = false;
  return {
    name: 'colormill-docs-pages',
    configResolved(config) {
      root = config.root;
      base = config.base;
      outDir = config.build.outDir;
      // vitest also loads this config (with a dummy outDir) and fires closeBundle; only copy for a real `vite build`.
      isBuild = config.command === 'build' && !process.env.VITEST;
    },
    // Production build: copy web/ docs into dist/ after Vite has written the bundle.
    closeBundle() {
      if (!isBuild) return;
      const src = path.resolve(root, WEB_DIR);
      const dst = path.resolve(root, outDir);
      mkdirSync(dst, { recursive: true });
      for (const f of DOC_PAGES) {
        const from = path.join(src, f);
        if (!existsSync(from)) throw new Error(`docs page missing: ${from}`);
        cpSync(from, path.join(dst, f));
      }
      for (const d of DOC_DIRS) {
        const from = path.join(src, d);
        if (!existsSync(from)) throw new Error(`docs dir missing: ${from}`);
        cpSync(from, path.join(dst, d), { recursive: true });
      }
      console.log(`[docs-pages] copied ${DOC_PAGES.join(', ')} and ${DOC_DIRS.join(', ')}/ into ${outDir}/`);
    },
    // Dev server: serve the same files from web/ so the nav links work under `vite`.
    configureServer(server) {
      const src = path.resolve(root, WEB_DIR);
      server.middlewares.use((req, res, next) => {
        const raw = (req.url ?? '').split('?')[0];
        const rel = raw.startsWith(base) ? raw.slice(base.length) : raw.replace(/^\/+/, '');
        const allowed = DOC_PAGES.includes(rel) || DOC_DIRS.some((d) => rel.startsWith(d + '/'));
        if (!allowed || rel.includes('..')) return next();
        const file = path.join(src, rel);
        if (!existsSync(file) || !statSync(file).isFile()) return next();
        res.setHeader('Content-Type', MIME[path.extname(file)] ?? 'application/octet-stream');
        res.end(readFileSync(file));
      });
    }
  };
}

export default defineConfig({
  base: '/ColorMill/',
  // e2e runs set E2E_NO_HMR so concurrent edits cannot reload a page mid-test
  server: process.env.E2E_NO_HMR ? { hmr: false, watch: null } : undefined,
  plugins: [docsPagesPlugin()],
  build: {
    target: 'es2022',
    rollupOptions: {
      // two pages: the simulator and the colour mixer
      input: { main: path.resolve(__dirname, 'index.html'), mixer: path.resolve(__dirname, 'mixer.html') }
    }
  },
  test: {
    globals: true,
    include: ['tests/**/*.test.ts']
  }
});

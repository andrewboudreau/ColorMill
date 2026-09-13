/**
 * Renderer e2e: load the harness (procedural volumes), render a frame in
 * headless Chromium (SwiftShader WebGPU), save a screenshot and check that
 * the image is a lit scene with the pigment bands where the harness put them.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { ROOT, assert, withBrowser } from './lib.mjs';

const OUT_DIR = path.join(ROOT, 'tests', 'e2e', 'out');

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

function pixel(img, x, y) {
  const i = (Math.round(y) * img.width + Math.round(x)) * 4;
  return [img.data[i], img.data[i + 1], img.data[i + 2]];
}

/** Mean colour of a (2r+1)^2 patch. */
function patchMean(img, cx, cy, r) {
  let R = 0, G = 0, B = 0, n = 0;
  for (let y = cy - r; y <= cy + r; y++) {
    for (let x = cx - r; x <= cx + r; x++) {
      if (x < 0 || y < 0 || x >= img.width || y >= img.height) continue;
      const p = pixel(img, x, y);
      R += p[0]; G += p[1]; B += p[2]; n++;
    }
  }
  return [R / n, G / n, B / n];
}

function luminance(p) {
  return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
}

export default async function run() {
  await withBrowser(async ({ page, baseUrl }) => {
    await page.goto(baseUrl + 'harness/render.html');
    await page.waitForFunction(() => window.__render && (window.__render.ready || window.__render.error), null, { timeout: 90000 });
    const state = await page.evaluate(() => ({
      ready: window.__render.ready,
      error: window.__render.error ?? null,
      validationError: window.__render.validationError ?? null,
      compileErrors: (window.__render.compileMessages ?? []).filter((m) => m.type === 'error')
    }));
    assert(state.ready, 'harness initialised: ' + state.error);
    assert(!state.validationError, 'no WebGPU validation error: ' + state.validationError);
    assert(state.compileErrors.length === 0, 'WGSL compiled: ' + JSON.stringify(state.compileErrors));

    // one timed frame (the first frame above already warmed the pipeline)
    const timings = await page.evaluate(async () => {
      const out = [];
      for (let i = 0; i < 3; i++) {
        const t0 = performance.now();
        await window.__render.frame();
        out.push(performance.now() - t0);
      }
      return out;
    });
    console.log('  frame times (ms):', timings.map((t) => t.toFixed(0)).join(', '));

    // exact pixels of the last frame (rendered offscreen: see harness/render.ts), saved as PNG
    const px = await page.evaluate(() => window.__render.readPixels());
    const img = { width: px.width, height: px.height, data: new Uint8Array(Buffer.from(px.base64, 'base64')) };
    assert(img.width === 960 && img.height === 640, `frame is 960x640 (got ${img.width}x${img.height})`);
    mkdirSync(OUT_DIR, { recursive: true });
    const shot = path.join(OUT_DIR, 'render.png');
    writeFileSync(shot, encodePng(img.width, img.height, img.data));
    console.log('  screenshot:', shot);

    // background reference: top-left corner (dark gradient)
    const bg = patchMean(img, 6, 6, 4);
    const centre = patchMean(img, img.width / 2, img.height / 2, 6);
    console.log('  background', bg.map(Math.round), 'centre', centre.map(Math.round));
    assert(Math.abs(luminance(centre) - luminance(bg)) > 25, 'centre of the image is not the background colour');

    // luminance variation across the scene: a lit surface has highlights and shadows
    const lums = [];
    for (let y = 40; y < img.height - 40; y += 16) {
      for (let x = 40; x < img.width - 40; x += 16) lums.push(luminance(pixel(img, x, y)));
    }
    lums.sort((a, b) => a - b);
    const p10 = lums[Math.floor(lums.length * 0.1)];
    const p90 = lums[Math.floor(lums.length * 0.9)];
    console.log('  luminance p10/p90:', p10.toFixed(0), p90.toFixed(0));
    assert(p90 - p10 > 40, 'there is luminance variation (lit surface)');

    // band colours: sample where the harness says the yellow / blue bands land on screen
    const spots = await page.evaluate(() => {
      const sp = window.__render.samplePoints;
      const out = {};
      for (const name of ['yellow', 'blue', 'red', 'white']) out[name] = sp[name].map((p) => window.__render.project(p));
      return out;
    });
    const sampled = {};
    for (const [name, pts] of Object.entries(spots)) {
      sampled[name] = pts.map((pt) => (pt ? patchMean(img, Math.round(pt.x), Math.round(pt.y), 3) : null));
      console.log(`  ${name}:`, sampled[name].map((c) => (c ? c.map(Math.round).join(',') : 'off-screen')).join('  '));
    }
    const yellowish = (c) => c && c[0] > 120 && c[1] > 100 && c[2] < Math.min(c[0], c[1]) * 0.6;
    const bluish = (c) => c && c[2] > c[0] * 1.3 && c[2] > c[1] * 1.15 && c[2] > 40;
    assert(sampled.yellow.some(yellowish), 'the yellow band reads as yellow');
    assert(sampled.blue.some(bluish), 'the blue band reads as blue');
    const reddish = (c) => c && c[0] > c[1] * 1.35 && c[0] > c[2] * 1.35 && c[0] > 120;
    assert(sampled.red.some(reddish), 'the red band reads as red');
  });
}

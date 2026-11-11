// v3 review: which WebGPU adapter does a Playwright-driven browser get on this machine?
import { chromium } from 'playwright';
import { createServer } from 'node:http';
const server = createServer((req, res) => res.end('<html>GPU probe</html>'));
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
try {
  for (const [name, exe, headless] of [['chrome-headless', CHROME, true], ['edge-headless', EDGE, true], ['chrome-headed', CHROME, false]]) {
    let browser;
    try {
      browser = await chromium.launch({ executablePath: exe, headless, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'], ignoreDefaultArgs: ['--disable-gpu', '--disable-gpu-compositing'] });
      const page = await browser.newPage();
      await page.goto(`http://127.0.0.1:${server.address().port}`);
      console.log(name, browser.version(), JSON.stringify(await page.evaluate(async () => {
        const a = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
        if (!a) return null;
        const l = a.limits;
        return { vendor: a.info.vendor, architecture: a.info.architecture, device: a.info.device, description: a.info.description, isFallback: a.info.isFallbackAdapter,
          features: [...a.features].filter((f) => /timestamp|f16|subgroups|float32|chromium/.test(f)),
          maxStorageBufferBindingSize: l.maxStorageBufferBindingSize, maxBufferSize: l.maxBufferSize, maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
          maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize };
      })));
    } catch (e) { console.log(name, 'failed', e.message.split('\n')[0]); } finally { await browser?.close(); }
  }
} finally { server.close(); }

import { bootApp } from './app';
import { GpuMpm } from './sim/mpm';
import { RayMarchRenderer } from './render/renderer';
import { attachCameraControls } from './render/camera';

const root = document.getElementById('app');
if (!root) throw new Error('App root element was not found.');

bootApp({
  root,
  makeSim: (device, config) => new GpuMpm({ device, config }),
  makeRenderer: (ctx, canvas) => new RayMarchRenderer(ctx, canvas),
  attachCameraControls
}).catch((e: unknown) => {
  // bootApp has already shown the error overlay; keep the console useful.
  console.error(e);
});

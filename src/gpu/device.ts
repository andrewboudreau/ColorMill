/**
 * WebGPU device acquisition and capability reporting.
 */

export interface GpuCapabilities {
  readonly adapterDescription: string;
  readonly vendor: string;
  readonly architecture: string;
  readonly isSoftware: boolean;
  readonly timestampQuery: boolean;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeInvocationsPerWorkgroup: number;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly maxBufferSize: number;
}

export interface GpuContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly caps: GpuCapabilities;
  readonly canvasFormat: GPUTextureFormat;
}

export class WebGpuUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGpuUnavailableError';
  }
}

/**
 * Request an adapter and device with the limits the solver wants (large
 * storage buffers). Throws WebGpuUnavailableError with a user-facing message
 * when WebGPU is missing.
 */
export async function createGpuContext(): Promise<GpuContext> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new WebGpuUnavailableError(
      'This browser does not expose WebGPU. Use a current Chrome, Edge, Safari or Firefox, ' +
        'and make sure hardware acceleration is enabled.'
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new WebGpuUnavailableError('No WebGPU adapter is available (GPU acceleration may be disabled).');
  }
  const wanted: GPUFeatureName[] = [];
  if (adapter.features.has('timestamp-query')) wanted.push('timestamp-query');

  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures: wanted,
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
      maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension,
      maxStorageBuffersPerShaderStage: lim.maxStorageBuffersPerShaderStage
    }
  });

  const info = adapter.info ?? ({} as GPUAdapterInfo);
  const architecture = info.architecture ?? '';
  const vendor = info.vendor ?? '';
  const caps: GpuCapabilities = {
    adapterDescription: info.description || `${vendor} ${architecture}`.trim() || 'unknown adapter',
    vendor,
    architecture,
    isSoftware: /swiftshader|llvmpipe|software|lavapipe/i.test(`${vendor} ${architecture} ${info.description ?? ''}`),
    timestampQuery: device.features.has('timestamp-query'),
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    maxBufferSize: device.limits.maxBufferSize
  };

  device.lost.then((reason) => {
    console.error('WebGPU device lost:', reason.message);
  });

  return { adapter, device, caps, canvasFormat: navigator.gpu.getPreferredCanvasFormat() };
}

/** Configure a canvas for WebGPU presentation. */
export function configureCanvas(ctx: GpuContext, canvas: HTMLCanvasElement): GPUCanvasContext {
  const gpuCanvas = canvas.getContext('webgpu');
  if (!gpuCanvas) throw new WebGpuUnavailableError('Could not create a WebGPU canvas context.');
  gpuCanvas.configure({ device: ctx.device, format: ctx.canvasFormat, alphaMode: 'opaque' });
  return gpuCanvas;
}

// Stamp (design §5): press a picture onto the material's top surface. A raster
// of latents (8 floats per pixel: the latent and an opacity) is mapped onto a
// footprint seen from above; `probeTop` records per solver column (i, k) the
// highest particle under the footprint, then `main` gives every particle
// within `depth` of its column's top the pixel it lies under, blended by the
// pixel's opacity times the strength, and a masterbatch pigment load (as the
// inject kernel does). The column top is found fresh each stamp, so a stamp
// lands on the flat seeded bank and on whatever the material has become.
struct Stamp {
  // x0, z0, 1 / (x1 - x0), 1 / (z1 - z0)
  rect : vec4<f32>,
  // width, height (pixels), 0, 0
  size : vec4<u32>,
  // depth below the column top, strength, 0, 0
  blend : vec4<f32>,
};
@group(0) @binding(1) var<uniform> S : Stamp;
@group(0) @binding(2) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> lat : array<f32>;
// per column k * nx + i: float bits of the highest y seen (0 = nothing), zeroed on the CPU before each stamp
@group(0) @binding(4) var<storage, read_write> top : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> img : array<f32>;

/// Picture coordinates of a particle: 0..1 across the footprint, outside it otherwise.
fn stampUv(x : vec3<f32>) -> vec2<f32> {
  return vec2<f32>((x.x - S.rect.x) * S.rect.z, (x.z - S.rect.y) * S.rect.w);
}

fn insideStamp(uv : vec2<f32>) -> bool {
  return uv.x >= 0.0 && uv.x < 1.0 && uv.y >= 0.0 && uv.y < 1.0;
}

fn columnIndex(x : vec3<f32>) -> u32 {
  let i = u32(clamp(x.x * P.hdt.y, 0.0, f32(P.grid.x - 1u)));
  let k = u32(clamp(x.z * P.hdt.y, 0.0, f32(P.grid.z - 1u)));
  return k * P.grid.x + i;
}

@compute @workgroup_size(128)
fn probeTop(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let x = pos[p].xyz;
  if (!insideStamp(stampUv(x))) { return; }
  // y > 0 so the float bit pattern is monotonic and atomicMax works
  atomicMax(&top[columnIndex(x)], bitcast<u32>(max(x.y, 0.0)));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let x = pos[p].xyz;
  let uv = stampUv(x);
  if (!insideStamp(uv)) { return; }
  let surface = bitcast<f32>(atomicLoad(&top[columnIndex(x)]));
  if (x.y < surface - S.blend.x) { return; }
  let px = min(u32(uv.x * f32(S.size.x)), S.size.x - 1u);
  let py = min(u32(uv.y * f32(S.size.y)), S.size.y - 1u);
  let base = 8u * (py * S.size.x + px);
  let a = clamp(img[base + 7u] * S.blend.y, 0.0, 1.0);
  if (a <= 0.0) { return; }
  for (var c = 0u; c < 7u; c++) {
    let z = lat[7u * p + c];
    lat[7u * p + c] = z + a * (img[base + c] - z);
  }
  let w = pos[p].w;
  pos[p].w = w + a * (PIGMENT_LOAD - w);
}

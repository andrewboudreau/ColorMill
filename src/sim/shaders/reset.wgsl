// Reset particle dynamic state: v = 0, C = 0, F = I, P2G affine = 0, flags = 0.
// Positions are re-uploaded by the CPU (seed positions) and the latent is set
// by the inject kernel in "set all" mode.
@group(0) @binding(1) var<storage, read_write> vel : array<vec4<f32>>;
// affine, F, C: MAT_STRIDE vec4 per particle (common.wgsl)
@group(0) @binding(2) var<storage, read_write> mats : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> flags : array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  vel[p] = vec4<f32>(0.0);
  // F = I: its diagonal is floats 9, 13 and 17 of the particle's 28, the y of vec4 2, 3 and 4
  let b = MAT_STRIDE * p;
  mats[b] = vec4<f32>(0.0);
  mats[b + 1u] = vec4<f32>(0.0);
  mats[b + 2u] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  mats[b + 3u] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  mats[b + 4u] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  mats[b + 5u] = vec4<f32>(0.0);
  mats[b + 6u] = vec4<f32>(0.0);
  flags[p] = 0u;
}

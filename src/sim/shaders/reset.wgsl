// Reset particle dynamic state: v = 0, C = 0, F = I, stress = 0, flags = 0.
// Positions are re-uploaded by the CPU (seed positions) and the latent is set
// by the inject kernel in "set all" mode.
@group(0) @binding(1) var<storage, read_write> vel : array<f32>;
@group(0) @binding(2) var<storage, read_write> cbuf : array<f32>;
@group(0) @binding(3) var<storage, read_write> fbuf : array<f32>;
@group(0) @binding(4) var<storage, read_write> sbuf : array<f32>;
@group(0) @binding(5) var<storage, read_write> flags : array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  for (var c = 0u; c < 3u; c++) { vel[3u * p + c] = 0.0; }
  for (var c = 0u; c < 9u; c++) {
    cbuf[9u * p + c] = 0.0;
    sbuf[9u * p + c] = 0.0;
    fbuf[9u * p + c] = select(0.0, 1.0, (c % 4u) == 0u);
  }
  flags[p] = 0u;
}

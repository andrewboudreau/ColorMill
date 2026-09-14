// Clear kernels (design §3.1 / §3.5): zero the grid accumulators.
@group(0) @binding(1) var<storage, read_write> gmass : array<i32>;
@group(0) @binding(2) var<storage, read_write> gmom : array<i32>;
@group(0) @binding(3) var<storage, read_write> gvel : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> pmass : array<i32>;
@group(0) @binding(5) var<storage, read_write> plat : array<i32>;
@group(0) @binding(6) var<storage, read_write> pload : array<i32>;

@compute @workgroup_size(4, 4, 4)
fn clearGrid(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= P.grid.x || gid.y >= P.grid.y || gid.z >= P.grid.z) { return; }
  let n = nodeIndex(gid.x, gid.y, gid.z);
  gmass[n] = 0;
  gmom[3u * n] = 0;
  gmom[3u * n + 1u] = 0;
  gmom[3u * n + 2u] = 0;
  gvel[n] = vec4<f32>(0.0);
}

@compute @workgroup_size(4, 4, 4)
fn clearRaster(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= P.grid.x || gid.y >= P.grid.y || gid.z >= P.grid.z) { return; }
  let n = nodeIndex(gid.x, gid.y, gid.z);
  pmass[n] = 0;
  pload[n] = 0;
  for (var c = 0u; c < 7u; c++) { plat[7u * n + c] = 0; }
}

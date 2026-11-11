// Clear kernels (design §3.5): zero the raster accumulators. (The solver grid has no clear pass:
// the grid update zeroes the mass and momentum it reads, grid.wgsl.)
@group(0) @binding(1) var<storage, read> box : array<u32>;   // the material's box (box.wgsl)
@group(0) @binding(2) var<storage, read_write> pmass : array<i32>;
@group(0) @binding(3) var<storage, read_write> plat : array<i32>;
@group(0) @binding(4) var<storage, read_write> pload : array<i32>;
@group(0) @binding(5) var<storage, read_write> pcov : array<i32>;   // colour raster: coverage; solver grid: the dispersion exchange's capacity (disperse.wgsl)
// the dispersion exchange's lamina pools (2 per node) and its mark (disperse.wgsl); solver grid only
@group(0) @binding(6) var<storage, read_write> xq : array<i32>;
@group(0) @binding(7) var<storage, read_write> xbeta : array<u32>;

/// The solver-grid raster's accumulators over BOX_FRAME: this frame's box united with the last
/// one's, so what the material left is zeroed too. The load and latent accumulators and pcov are
/// the dispersion exchange's pools and capacity here (disperse.wgsl), zeroed for its scatter.
@compute @workgroup_size(4, 4, 4)
fn clearRaster(@builtin(global_invocation_id) lid : vec3<u32>) {
  let gid = lid + vec3<u32>(box[BOX_FRAME], box[BOX_FRAME + 1u], box[BOX_FRAME + 2u]);
  if (gid.x >= P.grid.x || gid.y >= P.grid.y || gid.z >= P.grid.z) { return; }
  let n = nodeIndex(gid.x, gid.y, gid.z);
  pmass[n] = 0;
  pload[n] = 0;
  pcov[n] = 0;
  xq[2u * n] = 0;
  xq[2u * n + 1u] = 0;
  xbeta[n] = 0u;
  for (var c = 0u; c < 7u; c++) { plat[7u * n + c] = 0; }
}

/// The colour raster's accumulators (bound in place of pmass / plat / pload, plus pcov), over
/// BOX_FINE_RANGE: the texels this frame's range and the last one's use.
@compute @workgroup_size(4, 4, 4)
fn clearFine(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= vec3<u32>(box[BOX_FINE_RANGE], box[BOX_FINE_RANGE + 1u], box[BOX_FINE_RANGE + 2u]))) { return; }
  if (gid.x >= P.fine.x || gid.y >= P.fine.y || gid.z >= P.fine.z) { return; }
  let n = (gid.z * P.fine.y + gid.y) * P.fine.x + gid.x;
  pmass[n] = 0;
  pload[n] = 0;
  pcov[n] = 0;
  for (var c = 0u; c < 7u; c++) { plat[7u * n + c] = 0; }
}

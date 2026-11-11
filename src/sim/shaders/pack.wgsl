// Pack the coarse density volume and the fine pigment volumes.
@group(0) @binding(1) var<storage, read> box : array<u32>;   // the material's box (box.wgsl)
@group(0) @binding(2) var<storage, read> pmass : array<i32>;
@group(0) @binding(3) var<storage, read> plat : array<i32>;
@group(0) @binding(6) var<storage, read> pload : array<i32>;
@group(0) @binding(4) var volA : texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var volB : texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var finC : texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var<storage, read> pcov : array<i32>;   // colour raster only: coverage and undispersed load, 2 per node (raster.wgsl)

/// Over BOX_FRAME (this frame's box united with the last one's): texels the material left are
/// written empty, so the density texture is zero everywhere outside the current box.
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) lid : vec3<u32>) {
  let gid = lid + vec3<u32>(box[BOX_FRAME], box[BOX_FRAME + 1u], box[BOX_FRAME + 2u]);
  if (gid.x >= P.grid.x || gid.y >= P.grid.y || gid.z >= P.grid.z) { return; }
  let n = nodeIndex(gid.x, gid.y, gid.z);
  let m = decodeFixed(pmass[n], MASS_SCALE);
  // Colour comes exclusively from the fine raster. The coarse texture is
  // sampled only for density (surface, shadows and empty-space skipping).
  textureStore(volA, vec3<i32>(gid), vec4<f32>(m / 8.0, 0.0, 0.0, 0.0));
}

// The colour raster's textures (design §3.5 / §8), from the fine accumulators bound in place
// of pmass / plat / pload: finA = (load, load·lat0, load·lat1, load·lat2),
// finB = (load·lat3, load·lat4, load·lat5, load·lat6), finC = (mass, coverage, undispersed load, 0), all
// normalised by (h/hF)^3 / 8 so a packed fine cell reads ~1; finA.x / finC.x is the load per unit
// mass, finC.y / finC.x is the share of the cell's mass that carries pigment, and finC.z / finA.x is the
// share of the cell's pigment still in agglomerates (design §5): an amount like the load, so the
// renderer's filter mixes it as it mixes the load, and the latent is not touched.
//
// The latent is stored PREMULTIPLIED by the load (the accumulator as it is: sum of mass · load ·
// latent), not divided by it. The renderer samples these textures trilinearly, and a filter is
// only a pigment mix when it acts on amounts: filtering the normalised latent blended a pigment
// texel with its clear neighbour's leftover latent by geometry, an opaque pastel fringe round every
// streak. The renderer divides by the latent's concentration sum (lat0..lat3 sum to 1, so the sum
// of the four premultiplied channels is the filtered load) after sampling (raymarch.wgsl).
// Over BOX_FINE_RANGE (this frame's range and the last one's), at this frame's cell
// hF = finest * FINE_K / 8.
@compute @workgroup_size(4, 4, 4)
fn mainFine(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= vec3<u32>(box[BOX_FINE_RANGE], box[BOX_FINE_RANGE + 1u], box[BOX_FINE_RANGE + 2u]))) { return; }
  if (gid.x >= P.fine.x || gid.y >= P.fine.y || gid.z >= P.fine.z) { return; }
  let n = (gid.z * P.fine.y + gid.y) * P.fine.x + gid.x;
  let ks = 8.0 / f32(box[BOX_FINE + 3u]);
  let norm = P.fineH.z * ks * ks * ks;
  let m = decodeFixed(pmass[n], MASS_SCALE);
  let w = decodeFixed(pload[n], MASS_SCALE);
  let cov = decodeFixed(pcov[2u * n], MASS_SCALE);
  let und = min(decodeFixed(pcov[2u * n + 1u], MASS_SCALE), w);
  var l : array<f32, 7>;
  if (w > 0.0) {
    for (var c = 0u; c < 7u; c++) { l[c] = decodeFixed(plat[7u * n + c], LAT_SCALE) * norm; }
  } else {
    // no pigment here (most texels: clear base and empty space): nothing to read
    for (var c = 0u; c < 7u; c++) { l[c] = 0.0; }
  }
  let coord = vec3<i32>(gid);
  textureStore(volA, coord, vec4<f32>(w * norm, l[0], l[1], l[2]));
  textureStore(volB, coord, vec4<f32>(l[3], l[4], l[5], l[6]));
  textureStore(finC, coord, vec4<f32>(m * norm, cov * norm, max(und, 0.0) * norm, 0.0));
}

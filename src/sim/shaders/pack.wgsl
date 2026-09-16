// Pack (design §3.5 / §8): write the render volume.
//   volA = (density, lat0, lat1, lat2), volB = (lat3, lat4, lat5, lat6),
//   volC = (pigment, 0, 0, 0)
//   density = pmass / 8 (a fully packed cell ~ 1), pigment = pload / 8 (the
//   pigment load in the cell, same normalisation, so pigment / density is the
//   load per unit mass: 0 for clear base, PIGMENT_LOAD for pure masterbatch),
//   latN = plat[N] / pload: the Mixbox mix of the PIGMENTS only (the clear base
//   carries no load and no colour, so it never whitens a streak).
@group(0) @binding(1) var<storage, read> pmass : array<i32>;
@group(0) @binding(2) var<storage, read> plat : array<i32>;
@group(0) @binding(5) var<storage, read> pload : array<i32>;
@group(0) @binding(3) var volA : texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var volB : texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var volC : texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= P.grid.x || gid.y >= P.grid.y || gid.z >= P.grid.z) { return; }
  let n = nodeIndex(gid.x, gid.y, gid.z);
  let m = decodeFixed(pmass[n], MASS_SCALE);
  let w = decodeFixed(pload[n], MASS_SCALE);
  var l : array<f32, 7>;
  if (m > 1e-6 && w > 1e-6) {
    for (var c = 0u; c < 7u; c++) { l[c] = decodeFixed(plat[7u * n + c], LAT_SCALE) / w; }
  } else {
    for (var c = 0u; c < 7u; c++) { l[c] = 0.0; }
  }
  let coord = vec3<i32>(gid);
  textureStore(volA, coord, vec4<f32>(m / 8.0, l[0], l[1], l[2]));
  textureStore(volB, coord, vec4<f32>(l[3], l[4], l[5], l[6]));
  textureStore(volC, coord, vec4<f32>(w / 8.0, 0.0, 0.0, 0.0));
}

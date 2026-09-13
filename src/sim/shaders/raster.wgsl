// Pigment raster (design §3.5): scatter particle mass and mass-weighted latent
// to the node accumulators used by disperse and pack. Kinematic particles are
// included (they carry pigment and must be visible).
@group(0) @binding(1) var<storage, read> pos : array<f32>;
@group(0) @binding(2) var<storage, read> lat : array<f32>;
@group(0) @binding(3) var<storage, read_write> pmass : array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> plat : array<atomic<i32>>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let invh = P.hdt.y;
  let pMass = P.part.y;
  let x = vec3<f32>(pos[3u * p], pos[3u * p + 1u], pos[3u * p + 2u]);
  var z : array<f32, 7>;
  for (var c = 0u; c < 7u; c++) { z[c] = lat[7u * p + c]; }

  let gx = x * invh;
  let base = vec3<i32>(floor(gx - 0.5));
  let fx = gx - vec3<f32>(base);
  let wx = quadWeights(fx.x);
  let wy = quadWeights(fx.y);
  let wz = quadWeights(fx.z);
  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var i = 0; i < 3; i++) {
        let wgt = wx[i] * wy[j] * wz[k] * pMass;
        let n = nodeIndexI(base + vec3<i32>(i, j, k));
        atomicAdd(&pmass[n], encodeFixed(wgt, MASS_SCALE));
        for (var c = 0u; c < 7u; c++) {
          atomicAdd(&plat[7u * n + c], encodeFixed(wgt * z[c], LAT_SCALE));
        }
      }
    }
  }
}

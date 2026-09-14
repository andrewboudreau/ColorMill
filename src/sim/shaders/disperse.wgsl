// Disperse (design §5): shear-driven relaxation of each particle's latent toward
// the node-averaged latent of its neighbourhood. gamma = ||sym(C)||_F.
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cbuf : array<f32>;
@group(0) @binding(3) var<storage, read_write> lat : array<f32>;
@group(0) @binding(4) var<storage, read> pmass : array<i32>;
@group(0) @binding(5) var<storage, read> plat : array<i32>;
@group(0) @binding(6) var<storage, read> flags : array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) != 0u) { return; }
  let k = P.part.w;
  let simDt = P.domain.w;   // sim seconds advanced this frame (dt * substeps), never wall clock
  if (k <= 0.0) { return; }

  // shear rate from the APIC velocity gradient
  let b = 9u * p;
  let c00 = cbuf[b]; let c01 = cbuf[b + 1u]; let c02 = cbuf[b + 2u];
  let c10 = cbuf[b + 3u]; let c11 = cbuf[b + 4u]; let c12 = cbuf[b + 5u];
  let c20 = cbuf[b + 6u]; let c21 = cbuf[b + 7u]; let c22 = cbuf[b + 8u];
  let s01 = 0.5 * (c01 + c10);
  let s02 = 0.5 * (c02 + c20);
  let s12 = 0.5 * (c12 + c21);
  let gamma = sqrt(c00 * c00 + c11 * c11 + c22 * c22 + 2.0 * (s01 * s01 + s02 * s02 + s12 * s12));
  let alpha = clamp(k * gamma * simDt, 0.0, 1.0);
  if (alpha <= 0.0) { return; }

  let invh = P.hdt.y;
  let x = pos[p].xyz;
  let gx = x * invh;
  let base = vec3<i32>(floor(gx - 0.5));
  let fx = gx - vec3<f32>(base);
  let wx = quadWeights(fx.x);
  let wy = quadWeights(fx.y);
  let wz = quadWeights(fx.z);
  var msum = 0.0;
  var zsum : array<f32, 7>;
  for (var c = 0u; c < 7u; c++) { zsum[c] = 0.0; }
  for (var kk = 0; kk < 3; kk++) {
    for (var j = 0; j < 3; j++) {
      for (var i = 0; i < 3; i++) {
        let wgt = wx[i] * wy[j] * wz[kk];
        let n = nodeIndexI(base + vec3<i32>(i, j, kk));
        msum += wgt * decodeFixed(pmass[n], MASS_SCALE);
        for (var c = 0u; c < 7u; c++) {
          zsum[c] += wgt * decodeFixed(plat[7u * n + c], LAT_SCALE);
        }
      }
    }
  }
  if (msum <= 1e-6) { return; }
  for (var c = 0u; c < 7u; c++) {
    let zg = zsum[c] / msum;
    let z = lat[7u * p + c];
    lat[7u * p + c] = z + alpha * (zg - z);
  }
}

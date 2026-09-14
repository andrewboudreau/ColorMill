// P2G (design §3.2): scatter mass and momentum (APIC + stress) to the grid.
// The affine matrix (-dt pVol 4/h^2 stressScale tau + pMass C) is premultiplied
// by G2P, so this kernel reads one 9-float matrix per particle instead of C and
// the stress separately.
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> abuf : array<f32>;   // P2G affine matrix, row-major (written by g2p)
@group(0) @binding(4) var<storage, read> flags : array<u32>;
@group(0) @binding(5) var<storage, read_write> gmass : array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> gmom : array<atomic<i32>>;

fn loadAffine(p : u32) -> mat3x3<f32> {
  let b = p * 9u;
  return mat3x3<f32>(
    vec3<f32>(abuf[b], abuf[b + 3u], abuf[b + 6u]),
    vec3<f32>(abuf[b + 1u], abuf[b + 4u], abuf[b + 7u]),
    vec3<f32>(abuf[b + 2u], abuf[b + 5u], abuf[b + 8u]));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) != 0u) { return; }   // kinematic: no grid coupling

  let h = P.hdt.x;
  let invh = P.hdt.y;
  let pMass = P.part.y;

  let x = pos[p].xyz;
  let v = vel[p].xyz;
  let affine = loadAffine(p);

  let gx = x * invh;
  let base = vec3<i32>(floor(gx - 0.5));
  let fx = gx - vec3<f32>(base);
  let wx = quadWeights(fx.x);
  let wy = quadWeights(fx.y);
  let wz = quadWeights(fx.z);
  let mv = pMass * v;

  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var i = 0; i < 3; i++) {
        let off = vec3<i32>(i, j, k);
        let dpos = (vec3<f32>(off) - fx) * h;
        let wgt = wx[i] * wy[j] * wz[k];
        let n = nodeIndexI(base + off);
        let mom = wgt * (mv + affine * dpos);
        atomicAdd(&gmass[n], encodeFixed(wgt * pMass, MASS_SCALE));
        atomicAdd(&gmom[3u * n], encodeFixed(mom.x, MOM_SCALE));
        atomicAdd(&gmom[3u * n + 1u], encodeFixed(mom.y, MOM_SCALE));
        atomicAdd(&gmom[3u * n + 2u], encodeFixed(mom.z, MOM_SCALE));
      }
    }
  }
}

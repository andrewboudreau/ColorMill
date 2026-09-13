// P2G (design §3.2): scatter mass and momentum (APIC + stress) to the grid.
@group(0) @binding(1) var<storage, read> pos : array<f32>;
@group(0) @binding(2) var<storage, read> vel : array<f32>;
@group(0) @binding(3) var<storage, read> cbuf : array<f32>;
@group(0) @binding(4) var<storage, read> sbuf : array<f32>;   // Cauchy stress of the current F (written by g2p)
@group(0) @binding(5) var<storage, read> flags : array<u32>;
@group(0) @binding(6) var<storage, read_write> gmass : array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> gmom : array<atomic<i32>>;

fn loadMatC(p : u32) -> mat3x3<f32> {
  let b = p * 9u;
  return mat3x3<f32>(
    vec3<f32>(cbuf[b], cbuf[b + 3u], cbuf[b + 6u]),
    vec3<f32>(cbuf[b + 1u], cbuf[b + 4u], cbuf[b + 7u]),
    vec3<f32>(cbuf[b + 2u], cbuf[b + 5u], cbuf[b + 8u]));
}
fn loadMatS(p : u32) -> mat3x3<f32> {
  let b = p * 9u;
  return mat3x3<f32>(
    vec3<f32>(sbuf[b], sbuf[b + 3u], sbuf[b + 6u]),
    vec3<f32>(sbuf[b + 1u], sbuf[b + 4u], sbuf[b + 7u]),
    vec3<f32>(sbuf[b + 2u], sbuf[b + 5u], sbuf[b + 8u]));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) != 0u) { return; }   // kinematic: no grid coupling

  let h = P.hdt.x;
  let invh = P.hdt.y;
  let dt = P.hdt.z;
  let pVol = P.part.x;
  let pMass = P.part.y;

  let x = vec3<f32>(pos[3u * p], pos[3u * p + 1u], pos[3u * p + 2u]);
  let v = vec3<f32>(vel[3u * p], vel[3u * p + 1u], vel[3u * p + 2u]);
  let C = loadMatC(p);
  let sigma = loadMatS(p);

  let affine = (-dt * pVol * 4.0 * invh * invh * P.part.z) * sigma + pMass * C;

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

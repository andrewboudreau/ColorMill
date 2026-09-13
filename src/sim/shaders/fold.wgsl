// Operator cut & fold (design §6, revised): a scripted kinematic move of the
// visible sheet on the front roller onto the bank, shifted by +0.25 L along x,
// that PRESERVES the sheet's shape: each selected particle is parameterised by
// (x, thickness t = d_front - R, arc angle theta around the front axis measured
// from the nip) and mapped onto a slab lying flat on the live bank top:
//   x1 = x + 0.25 L (clamped),  y1 = min(bankTop, yMax - 3h - clearance) + t + clearance,
//   z1 = nipZ + (theta R - arcMid) * zScale        (unrolled along z, centred on the nip)
// The bank top is the live one: `select` reduces max(y) over non-stray bank
// particles (|z - nipZ| < bankHalfDepth, y > axisY, raster pmass at the
// nearest node >= 2) into `info[0]` with atomicMax on the bit pattern of the
// positive float (monotonic), and the min/max arc of the selection into
// info[2]/info[1]; `move` and `finish` read them directly - no readback.
//   select: mark the sheet (x < 0.75 L, d_front < R + 3h, y < axisY or z > frontAxisZ,
//           i.e. never the nip channel) kinematic and store (p0, arc).
//   move:   each substep while active: p(s) = lerp(p0, p1, s) + up * sin(pi s) * lift,
//           v = dp/ds * ds/dt (P.fold = active, s, ds/dt, lift); y clamped to fold3.w.
//   finish: clear the flag, x = p1, v = scripted end velocity, C = 0, F kept,
//           P2G affine rebuilt from F.
@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cbuf : array<f32>;
@group(0) @binding(4) var<storage, read_write> fbuf : array<f32>;
@group(0) @binding(5) var<storage, read_write> abuf : array<f32>;
@group(0) @binding(6) var<storage, read_write> flags : array<u32>;
@group(0) @binding(7) var<storage, read_write> fold0 : array<vec4<f32>>;   // p0 xyz + arc length per particle
// [0] max bank y bits, [1] max arc bits, [2] min arc bits (init 0xffffffff), [3] selected count
@group(0) @binding(8) var<storage, read_write> info : array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> pmass : array<i32>;   // last raster (stray-particle filter)

const PMASS_MIN : f32 = 2.0;

fn liveBankTop() -> f32 {
  let y = bitcast<f32>(atomicLoad(&info[0]));
  return select(y, P.fold2.x, y <= 0.0);   // fallback: seeded bank top
}

fn foldTarget(p0 : vec3<f32>, arc : f32) -> vec3<f32> {
  let h = P.hdt.x;
  let L = P.fold2.y;
  let R = P.front.w;
  let x1 = clamp(p0.x + P.fold2.z, 1.5 * h, L - 1.5 * h);
  let t = max(rollerDist(P.front.x, P.front.y, p0) - R, 0.0);
  // slab base: the live bank top, lowered (pressed into the bank) when the full
  // sheet thickness (3h) would not fit under the ceiling - keep the thickness
  // rather than squashing the slab flat against fold3.w
  let base = min(liveBankTop(), P.fold3.w - 3.0 * h - P.fold3.y);
  let y1 = min(base + t + P.fold3.y, P.fold3.w);
  let arcMax = bitcast<f32>(atomicLoad(&info[1]));
  let arcMin = bitcast<f32>(atomicLoad(&info[2]));
  let arcMid = 0.5 * (arcMin + arcMax);
  let z1 = clamp(P.fold2.w + (arc - arcMid) * P.fold3.z, 1.5 * h, P.domain.z - 1.5 * h);
  return vec3<f32>(x1, y1, z1);
}

fn loadMatF(p : u32) -> mat3x3<f32> {
  let b = p * 9u;
  return mat3x3<f32>(
    vec3<f32>(fbuf[b], fbuf[b + 3u], fbuf[b + 6u]),
    vec3<f32>(fbuf[b + 1u], fbuf[b + 4u], fbuf[b + 7u]),
    vec3<f32>(fbuf[b + 2u], fbuf[b + 5u], fbuf[b + 8u]));
}

@compute @workgroup_size(128)
fn select_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let h = P.hdt.x;
  let x = pos[p].xyz;
  let R = P.front.w;
  let dy = x.y - P.front.x;
  let dz = x.z - P.front.y;
  let d = sqrt(dy * dy + dz * dz);

  // live bank top: non-stray material over the nip region
  if (abs(x.z - P.fold2.w) < P.fold3.x && x.y > P.front.x) {
    let c = clamp(vec3<i32>(round(x * P.hdt.y)), vec3<i32>(0), vec3<i32>(P.grid.xyz) - vec3<i32>(1));
    if (decodeFixed(pmass[nodeIndexI(c)], MASS_SCALE) >= PMASS_MIN) {
      atomicMax(&info[0], bitcast<u32>(x.y));
    }
  }

  // the visible sheet: front half of the front roll or below the axis, never the nip channel
  if (x.x < 0.75 * P.fold2.y && d < R + 3.0 * h && (x.y < P.front.x || x.z > P.front.y)) {
    // angle around the front axis from the nip, in the direction of rotation (nip -> bottom -> front -> top)
    var th = atan2(-dy, -dz);
    if (th < 0.0) { th += 2.0 * PI; }
    let arc = th * R;
    flags[p] = flags[p] | 1u;
    fold0[p] = vec4<f32>(x, arc);
    atomicMax(&info[1], bitcast<u32>(arc));
    atomicMin(&info[2], bitcast<u32>(arc));
    atomicAdd(&info[3], 1u);
  }
}

@compute @workgroup_size(128)
fn move_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) == 0u) { return; }
  let f0 = fold0[p];
  let p0 = f0.xyz;
  let p1 = foldTarget(p0, f0.w);
  let s = P.fold.y;
  let dsdt = P.fold.z;
  let lift = P.fold.w;
  let up = vec3<f32>(0.0, 1.0, 0.0);
  var x = mix(p0, p1, s) + up * (sin(PI * s) * lift);
  var v = ((p1 - p0) + up * (PI * cos(PI * s) * lift)) * dsdt;
  if (x.y > P.fold3.w) { x.y = P.fold3.w; v.y = 0.0; }
  let lo = vec3<f32>(1.5 * P.hdt.x);
  x = clamp(x, lo, P.domain.xyz - lo);
  pos[p] = vec4<f32>(x, 0.0);
  vel[p] = vec4<f32>(v, 0.0);
}

@compute @workgroup_size(128)
fn finish_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) == 0u) { return; }
  let f0 = fold0[p];
  let p0 = f0.xyz;
  let p1 = foldTarget(p0, f0.w);
  let lo = vec3<f32>(1.5 * P.hdt.x);
  let x = clamp(p1, lo, P.domain.xyz - lo);
  // scripted velocity at the end of the move (ds/dt -> 0, so ~0)
  let up = vec3<f32>(0.0, 1.0, 0.0);
  let v = ((p1 - p0) + up * (PI * cos(PI * P.fold.y) * P.fold.w)) * P.fold.z;
  pos[p] = vec4<f32>(x, 0.0);
  vel[p] = vec4<f32>(v, 0.0);
  // keep F (the sheet's deformation state), drop C, rebuild the P2G affine from F
  var F = loadMatF(p);
  if (isBadMat(F) || isBad(det3(F))) { F = identity3(); }
  let zero = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  let ar = matRows(p2gAffine(kirchhoffStressOf(F), zero));
  let fr = matRows(F);
  let b = 9u * p;
  for (var c = 0u; c < 9u; c++) {
    cbuf[b + c] = 0.0;
    fbuf[b + c] = fr[c];
    abuf[b + c] = ar[c];
  }
  flags[p] = flags[p] & ~1u;
}

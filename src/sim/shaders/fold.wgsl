// Operator move (design §6): cut the WHOLE sheet off the front roll, roll it up
// into a log, turn the log 90 degrees so it lies across the mill over the nip,
// and set it down on the bank for the rolls to draw in again. This is how an
// operator mixes along the roll axis: material that sat at one x now spans the
// log's cross-section, and the log is fed in end-on.
//
// Each selected particle is parameterised by (x, thickness t = d_front - R,
// arc length s around the front axis from the nip, in the direction of rotation).
// Rolling the sheet (thickness g = gap) from the cut end gives an area-preserving
// spiral: radius rho = sqrt(rc^2 + s_rel * g / pi), turns n = (rho - rc) / g,
// angle phi = 2 pi n; cross-section point (u, v) = (rho + t) (cos phi, sin phi).
// Turning the log 90 degrees maps the sheet's x to the log's axis along z:
//   x1 = L/2 + u,  y1 = liveBankTop + rLog + v,  z1 = nipZ + (x - L/2) * zScale
// The bank top is the live one: `select` reduces max(y) over non-stray bank
// particles into info[0] with atomicMax on the float bits (monotonic for y > 0),
// and the min/max arc of the selection into info[2]/info[1]; `move` and `finish`
// read them directly - no readback.
//   select: mark the sheet (d_front < R + 3h, y < axisY or z > frontAxisZ, i.e.
//           never the nip channel) kinematic and store (p0, arc).
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
  let g = max(P.fold2.z, 2.0 * h);            // sheet thickness = nip gap
  let t = clamp(rollerDist(P.front.x, P.front.y, p0) - R, 0.0, 0.95 * g);   // within one turn of the spiral
  let arcMax = bitcast<f32>(atomicLoad(&info[1]));
  let arcMin = bitcast<f32>(atomicLoad(&info[2]));
  let sRel = max(arc - arcMin, 0.0);
  let sLen = max(arcMax - arcMin, g);
  let rc = g;                                  // the small hole a rolled sheet leaves
  let rho = sqrt(rc * rc + sRel * g / PI);
  let rLog = sqrt(rc * rc + sLen * g / PI) + g;
  let phi = 2.0 * PI * (rho - rc) / g;
  let rr = rho + t;
  let u = rr * cos(phi);
  let v = rr * sin(phi);
  let base = min(liveBankTop(), P.fold3.w - 2.0 * rLog - P.fold3.y);
  let x1 = clamp(0.5 * L + u, 1.5 * h, L - 1.5 * h);
  let y1 = clamp(base + rLog + v + P.fold3.y, 1.5 * h, P.fold3.w);
  let z1 = clamp(P.fold2.w + (p0.x - 0.5 * L) * P.fold3.z, 1.5 * h, P.domain.z - 1.5 * h);
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
  if (d < R + 3.0 * h && (x.y < P.front.x || x.z > P.front.y)) {
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
  pos[p] = vec4<f32>(x, pos[p].w);
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
  pos[p] = vec4<f32>(x, pos[p].w);
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

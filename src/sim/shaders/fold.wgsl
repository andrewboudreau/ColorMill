// Operator move (design §6): cut the WHOLE sheet off the front roll, roll it up
// into a log, stand the log over the nip and FEED it in end-first, the way an
// operator does (the rolls consume it over a few seconds and its spiral
// cross-section, every colour interleaved, is squeezed out across the full roll
// width). That is how a two-roll mill mixes along the roll axis.
//
// Each selected particle is parameterised by (x, thickness t = d_front - R, arc
// length s around the front axis from the nip, in the direction of rotation).
// Rolling the sheet (thickness g = gap) from the cut end gives an area-preserving
// spiral: rho = sqrt(rc^2 + s_rel * g / pi), turns n = (rho - rc) / g,
// phi = 2 pi n; cross-section point (u, v) = (rho + t) (cos phi, sin phi).
// The sheet is folded in half across its width first (log = L/2 long, 2g thick),
// then the log stands tilted over the nip: axis a = (0, cos tilt, sin tilt) (up
// and toward the viewer), lower end just above the rolls at (L/2, *, nipZ); the
// sheet's x runs along the axis (the x = 0 and x = L ends go in first).
//   select: mark the sheet (d_front < R + 3h, y < axisY or z > frontAxisZ, i.e.
//           never the nip channel) kinematic and store (p0, arc); reduce the live
//           bank top and the arc range into `info` (atomics, no readback).
//   move:   t < T_roll: p(s) = lerp(p0, pLog, smoothstep(t / T_roll)).
//           t >= T_roll: the log translates along -a at the feed speed; a particle
//           that reaches the release plane (the roll tops) is released: flag
//           cleared, C = 0, F kept, v = feed velocity, P2G affine rebuilt.
//   finish: release whatever is still held.
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

fn logAxis() -> vec3<f32> {
  let tilt = P.fold3.z;
  return vec3<f32>(0.0, cos(tilt), sin(tilt));
}

// (position in the standing log at t = T_roll, radius of the log)
// (position in the standing log at t = T_roll). The sheet is first folded in
// half across its width (x = L/2 onto x = 0) as an operator does, so the log is
// half the roll length and twice the sheet thickness; then rolled up.
fn foldTarget(p0 : vec3<f32>, arc : f32) -> vec3<f32> {
  let h = P.hdt.x;
  let L = P.fold2.y;
  let R = P.front.w;
  let g = max(P.fold2.z, 2.0 * h);            // sheet thickness = nip gap
  let g2 = 2.0 * g;                            // folded sheet thickness
  let tRaw = clamp(rollerDist(P.front.x, P.front.y, p0) - R, 0.0, 0.95 * g);
  let secondHalf = p0.x >= 0.5 * L;
  let along = select(p0.x, L - p0.x, secondHalf);      // x = 0 and x = L ends go in first
  let t = tRaw + select(0.0, g, secondHalf);           // the folded-over half is the outer layer
  let arcMax = bitcast<f32>(atomicLoad(&info[1]));
  let arcMin = bitcast<f32>(atomicLoad(&info[2]));
  let sRel = max(arc - arcMin, 0.0);
  let sLen = max(arcMax - arcMin, g2);
  let rc = g2;                                 // the small hole a rolled sheet leaves
  let rho = sqrt(rc * rc + sRel * g2 / PI);
  let rLog = sqrt(rc * rc + sLen * g2 / PI) + g2;
  let phi = 2.0 * PI * (rho - rc) / g2;
  let rr = rho + t;
  let a = logAxis();
  let e1 = vec3<f32>(1.0, 0.0, 0.0);
  let e2 = normalize(cross(a, e1));
  // the log's lower end face rests on the live bank top (or clears the roll tops
  // when the bank is gone); its lowest point is rLog * sin(tilt) below the centre
  let tilt = P.fold3.z;
  let endDrop = rLog * sin(tilt);
  var baseY = max(liveBankTop() + endDrop + P.fold3.y, P.front.x + R + rLog + P.fold3.y);
  // never squash the log against the ceiling: if the bank (or a chunk still in the
  // air over it) is high, the log sinks into it instead
  let logLen = 0.5 * L;
  baseY = min(baseY, P.fold3.w - logLen * cos(tilt) - endDrop - h);
  let base = vec3<f32>(0.5 * L, baseY, P.fold2.w);
  var q = base + a * along + e1 * (rr * cos(phi)) + e2 * (rr * sin(phi));
  let lo = vec3<f32>(1.5 * h);
  q = clamp(q, lo, vec3<f32>(P.domain.x, P.fold3.w, P.domain.z) - lo);
  return q;
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

  // live bank top under the log's footprint (over the nip, around x = L/2):
  // non-stray material only
  if (abs(x.z - P.fold2.w) < P.fold3.x && abs(x.x - 0.5 * P.fold2.y) < 0.35 && x.y > P.front.x) {
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

fn release(p : u32, v : vec3<f32>) {
  vel[p] = vec4<f32>(v, 0.0);
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

@compute @workgroup_size(128)
fn move_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) == 0u) { return; }
  let f0 = fold0[p];
  let p0 = f0.xyz;
  let pLog = foldTarget(p0, f0.w);
  let t = P.fold.y;
  let tRoll = P.fold.z;
  let feed = P.fold.w;
  let a = logAxis();
  let lo = vec3<f32>(1.5 * P.hdt.x);
  let hi = vec3<f32>(P.domain.x, P.fold3.w, P.domain.z) - lo;
  if (t < tRoll) {
    // roll: fly from the sheet into the standing log
    let tau = clamp(t / tRoll, 0.0, 1.0);
    let s = tau * tau * (3.0 - 2.0 * tau);
    let dsdt = 6.0 * tau * (1.0 - tau) / tRoll;
    let x = clamp(mix(p0, pLog, s), lo, hi);
    let v = (pLog - p0) * dsdt;
    pos[p] = vec4<f32>(x, pos[p].w);
    vel[p] = vec4<f32>(v, 0.0);
    return;
  }
  // feed: the log descends along its axis; release what reaches the roll tops
  let x = clamp(pLog - a * (feed * (t - tRoll)), lo, hi);
  let v = -a * feed;
  pos[p] = vec4<f32>(x, pos[p].w);
  // release where the log meets the bank (or the roll tops when there is no bank)
  let releaseY = max(liveBankTop() + P.hdt.x, P.front.x + P.front.w + 3.0 * P.hdt.x);
  if (x.y <= releaseY) {
    release(p, v);
  } else {
    vel[p] = vec4<f32>(v, 0.0);
  }
}

@compute @workgroup_size(128)
fn finish_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) == 0u) { return; }
  release(p, -logAxis() * P.fold.w);
}

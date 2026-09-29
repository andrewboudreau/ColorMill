// Operator move (design §6): cut ALL the material off the mill (the sheet on the
// front roll, the bank in the nip, whatever rides the back roll), roll it up
// into a log, stand the log over the nip and put it back end-first, the way an
// operator does: dropped whole onto the nip (P.fold.w = 0, the default) or
// lowered in at P.fold.w units/s. The rolls consume it over a few seconds and
// its spiral cross-section, every colour interleaved, is squeezed out across
// the full roll width. That is how a two-roll mill mixes along the roll axis.
//
// Each particle is parameterised by (x, radial depth dr = d_front - R, arc
// s = theta * R around the front axis from the nip, in the direction of
// rotation). The material is rolled up from the bank end (s = 2 pi R), so the
// thick bank becomes the core and the sheet wraps around it. Its thickness
// varies along the arc (a gap-thick sheet, a thick bank, nothing at all in
// places), so the spiral is built from a per-bin mass histogram over the arc
// (NB bins, kept separately for the two halves of the width fold): the
// thickness of bin b is its material volume spread over the bin's footprint,
// t = n * Vp / (ell * Lhalf), where the bin's arc ell is stretched beyond ds
// wherever that would make the layer thicker than TMAX (the operator kneads
// the bank flat as it is rolled in; the log stays a spiral of thin layers);
// the spiral is then wound bin by bin: the winding angle advances by
// ell / (rho + T/2) and a bin's inner radius is the OUTER surface of the bin one
// full turn earlier at the same angle (rc on the first turn), so layers of
// varying thickness stack without overlapping. A particle lands at rho(s) + u,
// where u = t * F(dr) and F is the bin's depth CDF (NS sqrt-spaced depth slices), so
// each slice of material gets exactly its share of the layer. That is
// volume-preserving bin by bin whatever the shape of the material, so the log
// never packs material denser than it was.
// The sheet is folded in half across its width first (log = L/2 long, the
// x >= L/2 half stacked outside the x < L/2 half), then the log stands tilted
// over the nip: axis a = (0, cos tilt, sin tilt) (up and toward the viewer),
// lower end just above the rolls at (L/2, *, nipZ); the sheet's x runs along
// the axis (the x = 0 and x = L ends go in first).
//   select: flag every particle kinematic, store (p0, s_rel) and bin it.
//   tables: one workgroup turns the histogram into per-bin spiral tables
//           (radius, angle, thickness, depth) and the log's size and base.
//   move:   t < T_roll: p(s) = lerp(p0, pLog, smoothstep(t / T_roll)).
//           t >= T_roll, feed = 0: the whole log is let go where it stands (v = 0;
//           gravity and the rolls take it from there).
//           t >= T_roll, feed > 0: the log translates along -a at the feed speed; a
//           particle that reaches the release plane (just above the live pile the
//           nip is eating, reduced by g2p every substep, or the roll tops) is
//           released: flag cleared, C = 0, F = I, v = feed velocity, P2G affine rebuilt.
//   finish: release whatever is still held.
@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cbuf : array<f32>;
@group(0) @binding(4) var<storage, read_write> fbuf : array<f32>;
@group(0) @binding(5) var<storage, read_write> abuf : array<f32>;
@group(0) @binding(6) var<storage, read_write> flags : array<u32>;
@group(0) @binding(7) var<storage, read_write> fold0 : array<vec4<f32>>;   // p0 xyz + arc s_rel per particle
// [0] live pile top y bits (reduced by g2p while the move runs), [1] max x bits, [2] min x bits
// (init +inf bits), [3] selected count, [8 + k*NB + b] particle count per (half k, bin b),
// [8 + 2NB + (k*NB + b)*NS + j] particle count per (half, bin, depth slice j)
@group(0) @binding(8) var<storage, read_write> info : array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> pmass : array<i32>;   // last raster (unused here, kept for the bind group)
// per bin: rhoStart, phiStart, t0, t1, T, ell, rhoEnd, 0; header at NB*8: rLog, baseY, Lhalf, count;
// then the depth CDF per (half, bin): NS + 1 cumulative fractions (0 .. 1)
@group(0) @binding(10) var<storage, read_write> tables : array<f32>;

const NB : u32 = 128u;          // arc bins
const NS : u32 = 32u;           // depth slices per bin, square-root spaced (fine near the roll, coarse deep in the bank)
const DMAX : f32 = 0.5;         // depth covered by the slices (sim units); deeper material lands in the last slice

// Second operator move, "cut & fold" (P.fold.x = 2: cut from the x = 0 end, 3: from
// the x = L end), the one an operator makes most: a knife held against the sheet
// on the FRONT FACE of the front roll, drawn from the roll end to the middle while
// the sheet comes up past it. The cut starts at the roll end just below the crown
// and reaches the middle FLAP_CUT_ARC further down (a diagonal: the end was cut
// first and has moved up since); from there down to where the sheet comes up from
// under the roll, the middle line is the UNCUT SEAM. The freed flap is the whole
// cut half of the sheet on the front face below the cut. The operator takes it by
// the roll end and flips it over the seam onto the other half: a page turned on a
// spine that runs around the roll at x = L/2, so what was against the roll is on
// top and the doubled sheet rides up into the nip. Kinematic script:
//   select: the flap = sheet on the front face (radial depth < flapDepth(),
//           FLAP_TH_MIN..FLAP_TH_MAX down from the crown) on the cut side of
//           cutX(theta); binned by arc (NB bins, slot 1) and depth (NS slices)
//           and flagged. The sheet on the receiving half over the same arc is
//           binned the same way (slot 0), unflagged.
//   tables: per arc bin the thickness of each half's sheet (a high quantile of
//           its depth): the flap's, and the receiving sheet's it lands on.
//   move:   at its own angle theta a particle turns about the seam point (x = L/2,
//           the top of the flap there) in the plane of the roll axis and the local
//           radial direction, through pi: the whole flap swings out from the roll
//           and over, keeping its shape, and lands mirrored across the seam on top
//           of the receiving sheet, roll side up. The swing's radial reach is
//           flattened (FLOP_LIFT) so a half-width flap stays inside the domain in
//           front of the roll. Released at rest with F = I.
const FLOP_SECONDS : f32 = 1.0;
const FLOP_LIFT : f32 = 0.6;        // radial reach of the swing as a fraction of the lateral one (fits the domain in front of the roll)
const FLOP_TOP_Q : f32 = 0.97;      // sheet thickness = this quantile of its particles' depth
const FLAP_TH_MIN : f32 = 0.08;     // the cut starts just in front of the crown line (rad down from the crown)
const FLAP_CUT_ARC : f32 = 0.6;     // and reaches the middle this much further down; below that the middle is the uncut seam
const FLAP_TH_MAX : f32 = 2.2;      // the flap reaches down to where the sheet comes up from under the roll
const INFO_COUNT : u32 = 8u;
const INFO_DEPTH : u32 = 8u + 2u * NB;
const HDR : u32 = NB * 8u;
const CDF : u32 = NB * 8u + 8u;

fn arcTotal() -> f32 { return 2.0 * PI * P.front.w; }
/** Fractional depth slice of a radial depth dr: slice j spans DMAX * (j/NS)^2 .. DMAX * ((j+1)/NS)^2. */
fn depthSlice(dr : f32) -> f32 { return f32(NS) * sqrt(max(dr, 0.0) / DMAX); }
fn binWidth() -> f32 { return arcTotal() / f32(NB); }

fn isFlop() -> bool { return P.fold.x > 1.5; }
/** Which end the cut starts from (0: the x = 0 end, 1: the x = L end); that side's flap is lifted. */
fn flopSide() -> u32 { return select(0u, 1u, P.fold.x > 2.5); }
/** How deep off the roll the flap reaches: the sheet, with slack for a doubled one. */
fn flapDepth() -> f32 { return 3.0 * P.fold2.z + P.hdt.x; }
fn flapBinWidth() -> f32 { return (FLAP_TH_MAX - FLAP_TH_MIN) / f32(NB); }

/** Where the cut sits at angle theta down the front face, as a distance in from the cut end:
    the roll end at the top (cut first, moved up since), the middle FLAP_CUT_ARC lower; below
    that the whole half is free and the middle is the uncut seam. */
fn cutX(theta : f32) -> f32 {
  let f = clamp((theta - FLAP_TH_MIN) / FLAP_CUT_ARC, 0.0, 1.0);
  return 0.5 * P.fold2.y * f;
}
/** Thickness of the sheet at angle theta down the front face (linear between bins):
    slot 1 the flap (the cut half), slot 0 the receiving half. */
fn sheetThick(theta : f32, slot : u32) -> f32 {
  let fb = clamp((theta - FLAP_TH_MIN) / flapBinWidth() - 0.5, 0.0, f32(NB) - 1.0001);
  let b = u32(fb);
  return mix(tables[b * 8u + slot], tables[(b + 1u) * 8u + slot], fb - f32(b));
}

/** Top of what the nip is currently eating under the log (or the roll tops). */
fn pileTop() -> f32 {
  let y = bitcast<f32>(atomicLoad(&info[0]));
  return max(y, P.front.x + P.front.w);
}

fn logAxis() -> vec3<f32> {
  let tilt = P.fold3.z;
  return vec3<f32>(0.0, cos(tilt), sin(tilt));
}

/** Sheet thickness the nip makes (at least two cells so the log's layers stay resolvable). */
fn sheetThickness() -> f32 { return max(P.fold2.z, 2.0 * P.hdt.x); }
/** Core radius of the log: the small hole a rolled sheet leaves (the bank is kneaded around it). */
fn coreRadius() -> f32 { return 0.5 * sheetThickness(); }
/** Thickest layer one half of the fold may add per bin; thicker material is spread along the arc. */
fn layerMax() -> f32 { return 1.5 * sheetThickness(); }

/** Position in the standing log at t = T_roll for a particle at p0 with arc s_rel (0 = bank end). */
fn foldTarget(p0 : vec3<f32>, sRel : f32) -> vec3<f32> {
  let h = P.hdt.x;
  let L = P.fold2.y;
  let R = P.front.w;
  let ds = binWidth();
  let rc = coreRadius();
  let fb = clamp(sRel / ds, 0.0, f32(NB) - 1e-4);
  let b = u32(fb);
  let f = fb - f32(b);
  let o = b * 8u;
  let rho0 = tables[o];
  let phi0 = tables[o + 1u];
  let t0 = tables[o + 2u];
  let t1 = tables[o + 3u];
  let T = tables[o + 4u];
  let ell = tables[o + 5u];
  let rho = mix(rho0, tables[o + 6u], f);
  let phi = phi0 + f * ell / max(rho + 0.5 * T, rc);
  let dr = max(rollerDist(P.front.x, P.front.y, p0) - R, 0.0);
  let secondHalf = p0.x >= 0.5 * L;
  let k = select(0u, 1u, secondHalf);
  let along = select(p0.x, L - p0.x, secondHalf);      // x = 0 and x = L ends go in first
  // depth within the layer from the bin's depth CDF (linear inside a slice)
  let fj = clamp(depthSlice(dr), 0.0, f32(NS) - 1e-4);
  let j = u32(fj);
  let c = CDF + (k * NB + b) * (NS + 1u);
  let frac = mix(tables[c + j], tables[c + j + 1u], fj - f32(j));
  var u = frac * t0;
  if (secondHalf) { u = t0 + frac * t1; }
  // radial position: uniform in r^2 across the layer, so a thick layer near the
  // core is not denser on its inside than on its outside
  let rOut = rho + T;
  let rr = sqrt(mix(rho * rho, rOut * rOut, clamp(u / max(T, 1e-6), 0.0, 1.0)));
  let a = logAxis();
  let e1 = vec3<f32>(1.0, 0.0, 0.0);
  let e2 = normalize(cross(a, e1));
  let base = vec3<f32>(0.5 * L, tables[HDR + 1u], P.fold2.w);
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
  let x = pos[p].xyz;
  if (isFlop()) {
    let L = P.fold2.y;
    let k = flopSide();
    let R = P.front.w;
    // the sheet on the front face over the flap's arc
    let dy = x.y - P.front.x;
    let dz = x.z - P.front.y;
    let dr = sqrt(dy * dy + dz * dz) - R;
    let th = atan2(dz, dy);   // 0 at the crown, pi/2 at the axis level in front
    if (dr < 0.0 || dr > flapDepth() || th < FLAP_TH_MIN || th > FLAP_TH_MAX) { return; }
    let u = select(L - x.x, x.x, k == 0u);   // distance in from the cut end
    let b = min(u32((th - FLAP_TH_MIN) / flapBinWidth()), NB - 1u);
    let j = min(u32(dr / flapDepth() * f32(NS)), NS - 1u);
    if (u < cutX(th)) {
      // the flap: flagged, slot 1
      atomicAdd(&info[INFO_COUNT + NB + b], 1u);
      atomicAdd(&info[INFO_DEPTH + (NB + b) * NS + j], 1u);
      flags[p] = flags[p] | 1u;
      fold0[p] = vec4<f32>(x, th);
      atomicAdd(&info[3], 1u);
    } else if (u >= 0.5 * L) {
      // the receiving half it lands on: slot 0, unflagged
      atomicAdd(&info[INFO_COUNT + b], 1u);
      atomicAdd(&info[INFO_DEPTH + b * NS + j], 1u);
    }
    return;
  }
  let R = P.front.w;
  let dy = x.y - P.front.x;
  let dz = x.z - P.front.y;
  let d = sqrt(dy * dy + dz * dz);
  // angle around the front axis from the nip, in the direction of rotation (nip -> bottom -> front -> top -> bank)
  var th = atan2(-dy, -dz);
  if (th < 0.0) { th += 2.0 * PI; }
  let sRel = clamp(arcTotal() - th * R, 0.0, arcTotal());   // 0 at the bank end: the bank is the core
  let ds = binWidth();
  let b = min(u32(sRel / ds), NB - 1u);
  let k = select(0u, 1u, x.x >= 0.5 * P.fold2.y);
  let dr = max(d - R, 0.0);
  let j = min(u32(depthSlice(dr)), NS - 1u);
  flags[p] = flags[p] | 1u;
  fold0[p] = vec4<f32>(x, sRel);
  atomicAdd(&info[INFO_COUNT + k * NB + b], 1u);
  atomicAdd(&info[INFO_DEPTH + (k * NB + b) * NS + j], 1u);
  atomicMax(&info[1], bitcast<u32>(x.x));
  atomicMin(&info[2], bitcast<u32>(x.x));
  atomicAdd(&info[3], 1u);
}

var<workgroup> wT : array<f32, NB>;
var<workgroup> wL : array<f32, NB>;
var<workgroup> wPhi : array<f32, NB>;
var<workgroup> wRho : array<f32, NB>;

/** One workgroup: histogram -> spiral tables (radius / angle prefix), log size, base height. */
@compute @workgroup_size(128)
fn tables_(@builtin(local_invocation_id) lid : vec3<u32>) {
  let b = lid.x;
  if (isFlop()) {
    // per arc bin and half (slot 1 the flap, slot 0 the receiving half): the sheet's thickness
    // off the roll, FLOP_TOP_Q of the way up its depth (a few strays above do not count), at
    // least a cell
    for (var k = 0u; k < 2u; k++) {
      var thick = P.hdt.x;
      let n = atomicLoad(&info[INFO_COUNT + k * NB + b]);
      if (n > 0u) {
        let want = FLOP_TOP_Q * f32(n);
        var acc = 0.0;
        var j = 0u;
        for (; j < NS; j++) {
          let nj = f32(atomicLoad(&info[INFO_DEPTH + (k * NB + b) * NS + j]));
          if (acc + nj >= want) {
            thick = (f32(j) + (want - acc) / max(nj, 1.0)) * flapDepth() / f32(NS);
            break;
          }
          acc += nj;
        }
        if (j == NS) { thick = flapDepth(); }
        thick = max(thick, P.hdt.x);
      }
      tables[b * 8u + k] = thick;
    }
    if (b == 0u) { tables[HDR + 3u] = f32(atomicLoad(&info[3])); }
    return;
  }
  let h = P.hdt.x;
  let L = P.fold2.y;
  let R = P.front.w;
  let ds = binWidth();
  let count = atomicLoad(&info[3]);
  var xLen = bitcast<f32>(atomicLoad(&info[1])) - bitcast<f32>(atomicLoad(&info[2]));
  if (count == 0u || !(xLen > 0.0)) { xLen = L; }
  let Lhalf = max(0.5 * xLen, 4.0 * h);
  let Vp = P.part.x;
  let n0 = f32(atomicLoad(&info[INFO_COUNT + b]));
  let n1 = f32(atomicLoad(&info[INFO_COUNT + NB + b]));
  // stretch the bin along the arc until neither half is thicker than layerMax()
  let ell = max(ds, max(n0, n1) * Vp / (layerMax() * Lhalf));
  let t0 = n0 * Vp / (ell * Lhalf);
  let t1 = n1 * Vp / (ell * Lhalf);
  let o = b * 8u;
  tables[o + 2u] = t0;
  tables[o + 3u] = t1;
  tables[o + 4u] = t0 + t1;
  tables[o + 5u] = ell;
  tables[o + 6u] = 0.0;
  tables[o + 7u] = 0.0;
  // depth CDF of each half of this bin: cumulative fraction at the start of each slice, then 1
  for (var k = 0u; k < 2u; k++) {
    let n = select(n0, n1, k == 1u);
    let c = CDF + (k * NB + b) * (NS + 1u);
    var acc = 0u;
    for (var j = 0u; j < NS; j++) {
      tables[c + j] = select(0.0, f32(acc) / n, n > 0.0);
      acc += atomicLoad(&info[INFO_DEPTH + (k * NB + b) * NS + j]);
    }
    tables[c + NS] = 1.0;
  }
  wT[b] = t0 + t1;
  wL[b] = ell;
  workgroupBarrier();
  if (b == 0u) {
    let rc = coreRadius();
    var phi = 0.0;
    var rLog = rc;
    var j = 0u;   // bin one turn back (phi - 2 pi), advanced monotonically
    for (var i = 0u; i < NB; i++) {
      let T = wT[i];
      let ell = wL[i];
      var rho0 = rc;
      if (phi >= 2.0 * PI) {
        // inner radius = outer surface of the layer one turn earlier at this angle
        let back = phi - 2.0 * PI;
        for (; j + 1u < i && wPhi[j + 1u] <= back; j++) {}
        let phiA = wPhi[j];
        let phiB = select(phi, wPhi[j + 1u], j + 1u < i);
        let f = clamp((back - phiA) / max(phiB - phiA, 1e-6), 0.0, 1.0);
        let rhoA = wRho[j];
        let rhoB = select(wRho[j] + wT[j] * wL[j] / max(2.0 * PI * (wRho[j] + 0.5 * wT[j]), 1e-6), wRho[j + 1u], j + 1u < i);
        rho0 = mix(rhoA, rhoB, f) + wT[j];
      }
      wPhi[i] = phi;
      wRho[i] = rho0;
      tables[i * 8u] = rho0;
      tables[i * 8u + 1u] = phi;
      phi += ell / max(rho0 + 0.5 * T, rc);
      rLog = max(rLog, rho0 + T);
    }
    for (var i = 0u; i < NB; i++) {
      // radius at the end of the bin: where the next bin starts (or one more step along the spiral)
      let ell = wL[i];
      let T = wT[i];
      let stepOut = T * ell / max(2.0 * PI * (wRho[i] + 0.5 * T), 1e-6);
      tables[i * 8u + 6u] = select(wRho[i] + stepOut, wRho[i + 1u], i + 1u < NB);
    }
    // the log's lower end face rests just above what is left on the mill (nothing,
    // normally: everything was taken) or the roll tops; its lowest point is
    // rLog * sin(tilt) below the centre. Never squash it against the ceiling.
    let tilt = P.fold3.z;
    let endDrop = rLog * sin(tilt);
    var baseY = pileTop() + endDrop + P.fold3.y;
    let logLen = 0.5 * L;
    baseY = min(baseY, P.fold3.w - logLen * cos(tilt) - endDrop - h);
    tables[HDR] = rLog;
    tables[HDR + 1u] = baseY;
    tables[HDR + 2u] = Lhalf;
    tables[HDR + 3u] = f32(count);
  }
}

fn release(p : u32, v : vec3<f32>) {
  vel[p] = vec4<f32>(v, 0.0);
  fold0[p].w = P.fold.y;   // release time: g2p leaves it out of the pile-top estimate while it settles
  // the log was placed kinematically, so the particle's old F says nothing about
  // its new neighbourhood: release it unstressed (with a nearly incompressible
  // putty a stale J would fire it out of the pile)
  let F = identity3();
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
  let t = P.fold.y;
  if (isFlop()) {
    // flip the flap over the uncut seam at the middle onto the other half
    let L = P.fold2.y;
    let R = P.front.w;
    let h = P.hdt.x;
    let th = f0.w;
    let dy0 = p0.y - P.front.x;
    let dz0 = p0.z - P.front.y;
    let dr = max(sqrt(dy0 * dy0 + dz0 * dz0) - R, 0.0);
    let left = flopSide() == 0u;
    let rhat = vec2<f32>(cos(th), sin(th));                  // radial direction (y, z) at the particle's angle
    let sx = select(-1.0, 1.0, left);                        // +x points at the receiving half when the cut is at x = 0
    let w = select(p0.x - 0.5 * L, 0.5 * L - p0.x, left);    // lateral distance in from the seam
    let thick = sheetThick(th, 1u);
    let recv = sheetThick(th, 0u);
    // pivot: the top of the flap at the seam; the particle sits d below it
    let rhoP = R + thick;
    let d = rhoP - (R + dr);
    let tau = clamp(t / FLOP_SECONDS, 0.0, 1.0);
    let u = tau * tau * (3.0 - 2.0 * tau);
    let dudt = 6.0 * tau * (1.0 - tau) / FLOP_SECONDS;
    let phi = PI * u;
    // the turn about the pivot in the (lateral, radial) plane: lateral -w -> +w (mirrored across
    // the seam), radial -d -> +d (roll side up), the radial reach of the swing flattened by
    // FLOP_LIFT; the receiving sheet may be thicker or thinner than the flap, so the radial
    // offset eases onto it over the turn
    let lat = -w * cos(phi) + d * sin(phi);
    let rad = FLOP_LIFT * w * sin(phi) - d * cos(phi);
    let dlat = (w * sin(phi) + d * cos(phi)) * PI;
    let drad = (FLOP_LIFT * w * cos(phi) + d * sin(phi)) * PI;
    let settle = recv + 0.5 * h - thick;
    let rho = rhoP + rad + settle * u;
    let lo = vec3<f32>(1.5 * h);
    let hi = vec3<f32>(P.domain.x, P.fold3.w, P.domain.z) - lo;
    let x = clamp(vec3<f32>(0.5 * L + sx * lat, P.front.x + rho * rhat.x, P.front.y + rho * rhat.y), lo, hi);
    let vrad = (drad + settle) * dudt;
    let v = vec3<f32>(sx * dlat * dudt, vrad * rhat.x, vrad * rhat.y);
    pos[p] = vec4<f32>(x, pos[p].w);
    if (t >= FLOP_SECONDS) {
      release(p, vec3<f32>(0.0));
    } else {
      vel[p] = vec4<f32>(v, 0.0);
    }
    return;
  }
  let pLog = foldTarget(p0, f0.w);
  let tRoll = P.fold.z;
  let feed = P.fold.w;
  let a = logAxis();
  let lo = vec3<f32>(1.5 * P.hdt.x);
  let hi = vec3<f32>(P.domain.x, P.fold3.w, P.domain.z) - lo;
  if (t < tRoll) {
    // roll: fly from the mill into the standing log
    let tau = clamp(t / tRoll, 0.0, 1.0);
    let s = tau * tau * (3.0 - 2.0 * tau);
    let dsdt = 6.0 * tau * (1.0 - tau) / tRoll;
    let x = clamp(mix(p0, pLog, s), lo, hi);
    let v = (pLog - p0) * dsdt;
    pos[p] = vec4<f32>(x, pos[p].w);
    vel[p] = vec4<f32>(v, 0.0);
    return;
  }
  if (feed <= 0.0) {
    // drop: the log is set down whole; it stands on the nip and the rolls pull it in
    pos[p] = vec4<f32>(clamp(pLog, lo, hi), pos[p].w);
    release(p, vec3<f32>(0.0));
    return;
  }
  // lower in: the log descends along its axis; release what reaches the pile the nip
  // is eating (or the roll tops), so nothing is let go in mid-air
  let x = clamp(pLog - a * (feed * (t - tRoll)), lo, hi);
  let v = -a * feed;
  pos[p] = vec4<f32>(x, pos[p].w);
  let releaseY = max(pileTop() + P.hdt.x, P.front.x + P.front.w + 3.0 * P.hdt.x);
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
  if (isFlop()) { release(p, vec3<f32>(0.0)); return; }
  release(p, -logAxis() * P.fold.w);
}

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
// Each half of the width (x < L/2 and x >= L/2) is wound into its own spiral, the
// full-width roll is then folded in half (the x >= L/2 half over onto the top of
// the other, so the log is L/2 long and two barrels side by side), and the pair
// stands tilted over the nip: axis a = (0, cos tilt, sin tilt) (up and toward
// the viewer), lower end just above the rolls at (L/2, *, nipZ); the sheet's x
// runs along the axis (the x = 0 and x = L ends go in first).
//   select: flag every particle kinematic, store (p0, s_rel) and bin it.
//   tables: one workgroup turns the histogram into per-bin spiral tables
//           (radius, angle, thickness, depth) and the log's size and base.
//   move:   t < T_wind ("peel and wind"): the operator rolls the material off the mill.
//           A full-width coil sits on the crown of the front roll (WIND_CONTACT), axis along
//           the roll, each half resting on the crown at its own radius; the bank and what is
//           beyond the crown gather into its core at once; the rest of the sheet rides the
//           roll toward the crown at the wind speed (P.fold4.x, the roll's own surface speed
//           or faster) and, as its arc reaches the crown, hops onto the coil at its tabled
//           radius and winding angle. The coil spins as it winds (rolling without slipping
//           on the sheet) and rises as it grows.
//           then T_double (P.fold4.z): the roll is folded in half: the x >= L/2 half swings
//           up and over about the roll's middle onto the top of the other half.
//           then T_lift (P.fold4.w): the doubled roll swings up rigidly from the crown to
//           the standing tilted pose over the nip, the two barrels side by side.
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
// per bin (TS floats): for each half k at 8k: rhoStart, phiStart, rhoEnd, rMax (the half's
// spiral's outer radius once wound up to and including this bin); shared at 4: t0, t1, ell;
// (the flag fold keeps its per-bin thicknesses at 0 / 1); header at NB*TS: rLog (the larger
// half), baseY, Lhalf, count; then the depth CDF per (half, bin): NS + 1 cumulative fractions
@group(0) @binding(10) var<storage, read_write> tables : array<f32>;

const NB : u32 = 128u;          // arc bins
const NS : u32 = 32u;           // depth slices per bin, square-root spaced (fine near the roll, coarse deep in the bank)
const DMAX : f32 = 0.5;         // depth covered by the slices (sim units); deeper material lands in the last slice
// Peel and wind (mirrors WIND_* in mpm.ts): the coil winds on the crown of the front roll
// (angle around the front axis from the nip, in the direction of rotation: 3 pi / 2 is the
// crown); material beyond the crown (the bank) is the core and gathers at once; a hop from
// the roll onto the coil takes WIND_HOP seconds.
const WIND_CONTACT : f32 = 4.71238898;   // 1.5 pi
const WIND_HOP : f32 = 0.3;
// Doubling the roll: the barrels' axes end up this fraction of the sum of their radii apart
// (each radius is its half's largest, reached at one angle only, so a little less than the sum
// presses the barrels together as an operator would), and the swinging half's arc is flattened
// by DOUBLE_FLAT so its far end clears the ceiling.
const DOUBLE_GAP : f32 = 0.8;
const DOUBLE_FLAT : f32 = 0.75;

// Second operator move, "cut & fold" (P.fold.x = 2 / 3: the end the cut starts from, x = 0 /
// x = L), the one an operator makes most, done the way a flag is folded. The sheet is cut
// across at the crown of the front roll and everything past the cut round to the nip exit
// (the sheet only: radial depth off the roll below sheetDepth(), not the bank) is the flag,
// 1.5 pi R long. The operator pulls it off over the crown into a flat TONGUE that leaves the
// crown up and toward the viewer at FLAG_ANGLE, halving it lengthwise as it comes off (the half
// away from the cut end folds over onto the other, roll side up, in a FLAG_HOP hop), so the
// strip is W = L / 2 wide, two sheets thick, and about two squares long. Then the flag fold:
// a triangle rolls along the strip, flipping over one edge then the other, taking in a fresh
// triangle of strip each time:
//   fold 0 (at P.fold4.y, once square 0 is off the roll): the corner at the free end, u + v < W,
//     over the square's diagonal u + v = W onto the triangle below it;
//   fold 1 (at P.fold4.z, once the whole strip is off): the doubled triangle over the square's
//     bottom edge v = W onto the next square (its seam-side triangle, u >= v - W);
//   fold 2 (at P.fold4.w): that bundle over the next square's diagonal u = v - W onto the
//     roll-end triangle beside it.
// Each flip is a page turn on the hinge, in the strip's own flat coordinates (u in from the
// cut end, v along the strip from the free end, n out of the strip plane): in-plane offset
// -w -> +w across the hinge, height mirrored through the bundle's top (roll side up), the
// swing's reach out of the plane flattened by FLAG_LIFT, P.fold.w seconds each. After the last
// flip (P.fold.z) the bundle, a flat triangle W across and 2 * 4 sheets thick, is carried over
// onto the other half of the roll and set down on the nip (FLAG_SET seconds), then released.
// Kinematic script (one selection, one move, P.fold.z + FLAG_SET long):
//   select: the sheet past the cut, flagged, fold0 = (p0, arc from the bank end).
//   move:   riding the roll to the crown at the wind speed (P.fold4.x, as cut & roll), then the
//           tongue frame with the flag-fold coordinates, then the carry to the nip.
//   finish: release at rest with F = I.
const FLAG_ANGLE : f32 = 1.0471976;    // 60 degrees: the tongue leaves the crown up and toward the viewer
const FLAG_LIFT : f32 = 0.5;           // the flipping triangle's reach out of the strip plane, as a fraction of its in-plane one
const FLAG_HOP : f32 = 0.25;           // seconds for a sheet element to settle into the tongue as it leaves the crown
const FLAG_SET : f32 = 0.5;            // seconds to carry the bundle onto the nip (mirrors FLAG_SET_SECONDS in mpm.ts)
const FLAG_SPREAD : f32 = 0.6;         // the set-down bundle's v spread about the nip, as a fraction
const INFO_COUNT : u32 = 8u;
const INFO_DEPTH : u32 = 8u + 2u * NB;
const TS : u32 = 16u;           // floats per bin in tables
const HDR : u32 = NB * TS;
const CDF : u32 = NB * TS + 8u;

fn arcTotal() -> f32 { return 2.0 * PI * P.front.w; }
/** Fractional depth slice of a radial depth dr: slice j spans DMAX * (j/NS)^2 .. DMAX * ((j+1)/NS)^2. */
fn depthSlice(dr : f32) -> f32 { return f32(NS) * sqrt(max(dr, 0.0) / DMAX); }
fn binWidth() -> f32 { return arcTotal() / f32(NB); }

fn isFlop() -> bool { return P.fold.x > 1.5; }
/** Which end the cut starts from (0: the x = 0 end, 1: the x = L end): modes 2 / 3. */
fn flopSide() -> u32 { return u32(round(P.fold.x)) & 1u; }
/** How deep off the roll the sheet reaches (what the flag fold takes; the bank is deeper). */
fn sheetDepth() -> f32 { return 2.5 * P.fold2.z + P.hdt.x; }
/** One sheet's thickness allowance in the bundle, and a ply of the halved strip (two sheets). */
fn sheetT() -> f32 { return P.fold2.z + 0.5 * P.hdt.x; }
fn plyT() -> f32 { return 2.0 * sheetT(); }
/** Width of the strip: half the roll. */
fn stripW() -> f32 { return 0.5 * P.fold2.y; }

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

/** Bin and fraction along the arc sRel (0 at the bank end). */
fn binAt(sRel : f32) -> vec2<f32> {
  let fb = clamp(sRel / binWidth(), 0.0, f32(NB) - 1e-4);
  return vec2<f32>(floor(fb), fb - floor(fb));
}

/** Winding angle of half k's spiral at arc sRel, from the tables. */
fn windPhi(sRel : f32, k : u32) -> f32 {
  let bf = binAt(sRel);
  let o = u32(bf.x) * TS;
  let hk = o + 8u * k;
  let rho = mix(tables[hk], tables[hk + 2u], bf.y);
  return tables[hk + 1u] + bf.y * tables[o + 6u] / max(rho + 0.5 * tables[o + 4u + k], coreRadius());
}

/** Outer radius of half k's coil once wound up to arc sRel: the running maximum the tables keep
    per bin (so the coil never shrinks when a thin bin follows a thick one), blended across the
    bin so it grows smoothly. */
fn windRadius(sRel : f32, k : u32) -> f32 {
  let bf = binAt(sRel);
  let b = u32(bf.x);
  let prev = select(coreRadius(), tables[(b - 1u) * TS + 8u * k + 3u], b > 0u);
  return max(mix(prev, tables[b * TS + 8u * k + 3u], bf.y), coreRadius());
}

/** Arc at which the winding starts: the crown; material beyond it (the bank) is the core. */
fn contactArc() -> f32 { return (2.0 * PI - WIND_CONTACT) * P.front.w; }

/** Unit vector from the front axis at angle th (as in select_: 0 the nip, pi / 2 the bottom). */
fn rhat(th : f32) -> vec3<f32> { return vec3<f32>(0.0, -sin(th), -cos(th)); }

/** Which half of the width a particle came from. */
fn halfOf(p0 : vec3<f32>) -> u32 { return select(0u, 1u, p0.x >= 0.5 * P.fold2.y); }

/** A particle's place in its half's spiral (design §6): radius and winding angle, for a particle
    at p0 with arc sRel (0 = bank end). */
fn coilCoords(p0 : vec3<f32>, sRel : f32) -> vec2<f32> {
  let R = P.front.w;
  let k = halfOf(p0);
  let bf = binAt(sRel);
  let b = u32(bf.x);
  let o = b * TS;
  let hk = o + 8u * k;
  let T = tables[o + 4u + k];
  let rho = mix(tables[hk], tables[hk + 2u], bf.y);
  let dr = max(rollerDist(P.front.x, P.front.y, p0) - R, 0.0);
  // depth within the layer from the bin's depth CDF (linear inside a slice)
  let fj = clamp(depthSlice(dr), 0.0, f32(NS) - 1e-4);
  let j = u32(fj);
  let c = CDF + (k * NB + b) * (NS + 1u);
  let frac = mix(tables[c + j], tables[c + j + 1u], fj - f32(j));
  // radial position: uniform in r^2 across the layer, so a thick layer near the
  // core is not denser on its inside than on its outside
  let rOut = rho + T;
  let rr = sqrt(mix(rho * rho, rOut * rOut, clamp(frac, 0.0, 1.0)));
  return vec2<f32>(rr, windPhi(sRel, k));
}

/** The log's frame turned by theta from lying along the roll (theta = 0: axis x, cross-section
    in y, z) to standing (theta = pi / 2: axis logAxis()). A rotation about the horizontal
    n = (0, -sin tilt, cos tilt), so the axis swings up in the plane of x and the standing axis. */
fn liftFrame(theta : f32) -> mat3x3<f32> {
  let tilt = P.fold3.z;
  let sT = sin(tilt);
  let cT = cos(tilt);
  let c = cos(theta);
  let sn = sin(theta);
  let ax = vec3<f32>(c, cT * sn, sT * sn);
  let g1 = vec3<f32>(-cT * sn, c + sT * sT * (1.0 - c), -sT * cT * (1.0 - c));
  let g2 = vec3<f32>(-sT * sn, -sT * cT * (1.0 - c), c + cT * cT * (1.0 - c));
  return mat3x3<f32>(ax, g1, g2);
}

/** Height of the crown of the front roll (where the coil rests) and its z. */
fn crownY() -> f32 { return P.front.x + P.front.w; }

/** Where a held particle is at time t of the move (unclamped): riding the roll, hopping onto
    the coil, wound on the spinning coil, folded over with its half, or swinging up with the
    doubled roll into the standing log (t >= T_roll gives the standing log, as the drop and the
    lowering-in need). */
fn windPos(p0 : vec3<f32>, sRel : f32, t : f32) -> vec3<f32> {
  let R = P.front.w;
  let L = P.fold2.y;
  let vW = max(P.fold4.x, 1e-3);
  let tWind = P.fold4.y;
  let doubled = P.fold4.z > 0.0;              // else the roll stays long: no fold in half
  let tDouble = select(0.0, max(P.fold4.z, 1e-3), doubled);
  let tLift = max(P.fold4.w, 1e-3);
  let sCon = contactArc();
  let k = halfOf(p0);
  let cc = coilCoords(p0, sRel);
  let rr = cc.x;
  let phiP = cc.y;
  // the winding at time t: the arc that has reached the crown, and the coil that holds it
  let tw = min(t, tWind - WIND_HOP);
  let sCur = min(sCon + vW * tw, arcTotal());
  let psi = PI + windPhi(sCur, k) - phiP;      // the sheet joins at the coil's bottom; the coil spins as it winds
  let rk = windRadius(sCur, k);                 // this half's coil rests on the crown at its own radius
  let onCoil = vec3<f32>(p0.x, crownY() + rk + rr * cos(psi), P.front.y + rr * sin(psi));
  if (t >= tWind) {
    // the finished roll: both halves' final radii
    let r0 = windRadius(arcTotal(), 0u);
    let r1 = windRadius(arcTotal(), 1u);
    // double: the x >= L/2 half swings up and over about the roll's middle (a half turn about
    // the z line through x = L/2, at the height that lands its axis `spacing` above the other
    // half's) onto the top of the other half, its arc flattened by DOUBLE_FLAT
    let spacing = DOUBLE_GAP * (r0 + r1);
    var q = onCoil;
    if (doubled && k == 1u) {
      let tau = clamp((t - tWind) / tDouble, 0.0, 1.0);
      let th = PI * tau * tau * (3.0 - 2.0 * tau);
      let hinge = vec2<f32>(0.5 * L, crownY() + 0.5 * (r0 + r1 + spacing));
      let d = q.xy - hinge;
      q = vec3<f32>(hinge + vec2<f32>(d.x * cos(th) - d.y * sin(th), DOUBLE_FLAT * d.x * sin(th) + d.y * cos(th)), q.z);
    }
    if (t < tWind + tDouble) { return q; }
    // lift: the doubled roll swings up from the crown to the standing pose over the nip; its
    // axis is the line midway between the two barrels' axes, so the pair stands centred
    let tau = clamp((t - tWind - tDouble) / tLift, 0.0, 1.0);
    let s = tau * tau * (3.0 - 2.0 * tau);
    // the roll's axis: midway between the two barrels when doubled, the mean of the two halves'
    // centre lines (each resting on the crown at its own radius) when kept long
    let baseCoil = vec3<f32>(0.0, select(crownY() + 0.5 * (r0 + r1), crownY() + r0 + 0.5 * spacing, doubled), P.front.y);
    let baseStand = vec3<f32>(0.5 * L, tables[HDR + 1u], P.fold2.w);
    return mix(baseCoil, baseStand, s) + liftFrame(0.5 * PI * s) * (q - baseCoil);
  }
  // departure: the bank (arc < sCon) goes at once; the sheet when the roll brings it to the crown
  let tDep = max(sRel - sCon, 0.0) / vW;
  if (t >= tDep + WIND_HOP) { return onCoil; }
  // ride: the sheet on the roll moves with it (arc-wise at the wind speed) until it departs;
  // anything deep off the roll (strays on the floor, the back roll) waits where it is
  let dr = max(rollerDist(P.front.x, P.front.y, p0) - R, 0.0);
  let th0 = 2.0 * PI - sRel / R;
  var ride = p0;
  if (dr < 3.0 * P.fold2.z + P.hdt.x) {
    let th = th0 + vW * min(t, tDep) / R;
    ride = vec3<f32>(p0.x, P.front.x, P.front.y) + (R + dr) * rhat(th);
  }
  if (t < tDep) { return ride; }
  let tau = clamp((t - tDep) / WIND_HOP, 0.0, 1.0);
  return mix(ride, onCoil, tau * tau * (3.0 - 2.0 * tau));
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
    // the sheet past the cut at the crown, round to the nip exit: the flag
    let R = P.front.w;
    let dy = x.y - P.front.x;
    let dz = x.z - P.front.y;
    let dr = sqrt(dy * dy + dz * dz) - R;
    if (dr < 0.0 || dr > sheetDepth()) { return; }
    var th = atan2(-dy, -dz);
    if (th < 0.0) { th += 2.0 * PI; }
    let sRel = clamp(arcTotal() - th * R, 0.0, arcTotal());
    if (sRel < contactArc()) { return; }   // beyond the crown: on its way into the nip, not cut
    flags[p] = flags[p] | 1u;
    fold0[p] = vec4<f32>(x, sRel);
    atomicAdd(&info[3], 1u);
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
    if (b == 0u) { tables[HDR + 3u] = f32(atomicLoad(&info[3])); }   // the flag fold needs no tables
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
  let o = b * TS;
  tables[o + 4u] = t0;
  tables[o + 5u] = t1;
  tables[o + 6u] = ell;
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
  workgroupBarrier();
  if (b == 0u) {
    // each half of the width is its own spiral (the roll is wound full width, then doubled)
    let rc = coreRadius();
    var rMax = vec2<f32>(rc, rc);
    for (var k = 0u; k < 2u; k++) {
      for (var i = 0u; i < NB; i++) {
        wT[i] = tables[i * TS + 4u + k];
        wL[i] = tables[i * TS + 6u];
      }
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
        let hk = i * TS + 8u * k;
        tables[hk] = rho0;
        tables[hk + 1u] = phi;
        phi += ell / max(rho0 + 0.5 * T, rc);
        rLog = max(rLog, rho0 + T);
        tables[hk + 3u] = rLog;
      }
      for (var i = 0u; i < NB; i++) {
        // radius at the end of the bin: where the next bin starts (or one more step along the spiral)
        let ell = wL[i];
        let T = wT[i];
        let stepOut = T * ell / max(2.0 * PI * (wRho[i] + 0.5 * T), 1e-6);
        tables[i * TS + 8u * k + 2u] = select(wRho[i] + stepOut, wRho[i + 1u], i + 1u < NB);
      }
      rMax[k] = rLog;
    }
    // the doubled roll stands as two barrels side by side, its axis midway between theirs; its
    // lower end rests just above what is left on the mill (nothing, normally: everything was
    // taken) or the roll tops: the lower barrel's lowest point is its radius times sin(tilt)
    // below its own centre, which sits half the barrels' spacing below the axis (times sin^2).
    // Never squash it against the ceiling.
    let tilt = P.fold3.z;
    let doubled = P.fold4.z > 0.0;
    let rLog = max(rMax.x, rMax.y);
    let endDrop = rLog * sin(tilt) + select(0.0, 0.5 * DOUBLE_GAP * (rMax.x + rMax.y) * sin(tilt) * sin(tilt), doubled);
    var baseY = pileTop() + endDrop + P.fold3.y;
    let logLen = select(L, 0.5 * L, doubled);   // the long roll stands at its full length
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

/** The flag-fold coordinates (u in from the cut end, v along the strip from the free end, n out
    of the strip plane) of a sheet element that started at (u0, v0, n0), at time t of the move:
    each flip that has started reflects the bundle region it takes across its hinge, a page turn
    while it runs. */
fn flagCoords(u0 : f32, v0 : f32, n0 : f32, t : f32) -> vec3<f32> {
  let W = stripW();
  let ply = plyT();
  let tFlip = max(P.fold.w, 1e-3);
  var u = u0;
  var v = v0;
  var n = n0;
  for (var k = 0u; k < 3u; k++) {
    let tk = select(select(P.fold4.y, P.fold4.z, k == 1u), P.fold4.w, k == 2u);
    if (t < tk) { break; }
    var inR = false;
    var hp = vec2<f32>(0.0);
    var nrm = vec2<f32>(0.0, 1.0);
    if (k == 0u) {
      inR = u + v < W;                                     // the corner at the free end
      hp = vec2<f32>(W, 0.0);
      nrm = vec2<f32>(0.7071068, 0.7071068);               // over the diagonal u + v = W
    } else if (k == 1u) {
      inR = u + v >= W && v <= W;                          // the doubled triangle of square 0
      hp = vec2<f32>(0.0, W);
      nrm = vec2<f32>(0.0, 1.0);                           // over the bottom edge v = W
    } else {
      inR = v >= W && v <= 2.0 * W && u >= v - W;          // the bundle on square 1's seam side
      hp = vec2<f32>(0.0, W);
      nrm = vec2<f32>(-0.7071068, 0.7071068);              // over the diagonal u = v - W
    }
    if (!inR) { continue; }
    let H = f32(k + 1u) * ply;                             // the bundle's height before this flip
    let q = vec2<f32>(u, v) - hp;
    let foot = vec2<f32>(u, v) - nrm * dot(q, nrm);
    let w = max(-dot(q, nrm), 0.0);                        // in from the hinge, on the flap side
    let d = H - n;                                         // below the bundle's top
    let tau = clamp((t - tk) / tFlip, 0.0, 1.0);
    let sm = tau * tau * (3.0 - 2.0 * tau);
    let phi = PI * sm;
    let lat = -w * cos(phi) + d * sin(phi);
    let rad = FLAG_LIFT * w * sin(phi) - d * cos(phi);
    let uv = foot + nrm * lat;
    u = uv.x;
    v = uv.y;
    n = H + rad + (ply - H) * sm;                          // lands on one fresh ply, roll side up
  }
  return vec3<f32>(u, v, n);
}

/** A point of the tongue frame in the world: u across the strip (at the cut end's half of the
    roll), w along the tongue from the crown, n out of its plane (up and back). */
fn tonguePoint(u : f32, w : f32, n : f32) -> vec3<f32> {
  let left = flopSide() == 0u;
  let cA = cos(FLAG_ANGLE);
  let sA = sin(FLAG_ANGLE);
  return vec3<f32>(select(P.fold2.y - u, u, left), P.front.x + P.front.w + sA * w + cA * n, P.front.y + cA * w - sA * n);
}

/** Where a held sheet element is at time t of the flag fold (unclamped): riding the roll to the
    crown, settling into the tongue, folded with the bundle, or carried onto the nip. */
fn flagPos(p0 : vec3<f32>, sRel : f32, t : f32) -> vec3<f32> {
  let R = P.front.w;
  let L = P.fold2.y;
  let W = stripW();
  let vW = max(P.fold4.x, 1e-3);
  let tSet = P.fold.z;
  let sCon = contactArc();
  let Ls = arcTotal() - sCon;
  // the strip's own coordinates: u in from the cut end (both halves, the far one folded over),
  // v along the strip from the free end (the cut), n the element's height in the halved strip
  let u = min(p0.x, L - p0.x);
  let folded = (p0.x >= 0.5 * L) == (flopSide() == 0u);   // the half away from the cut end
  let v = sRel - sCon;
  let dy = p0.y - P.front.x;
  let dz = p0.z - P.front.y;
  let dr = clamp(sqrt(dy * dy + dz * dz) - R, 0.0, sheetT());
  let n0 = select(dr, 2.0 * sheetT() - dr, folded);
  // the ride: the sheet moves with the roll toward the crown until its arc has come off
  let tOff = v / vW;
  let th = 2.0 * PI - sRel / R + vW * min(t, tOff) / R;
  let ride = vec3<f32>(p0.x, P.front.x, P.front.y) + (R + dr) * rhat(th);
  if (t < tOff) { return ride; }
  // in the tongue, folded as the flips come; the strip stops coming off once it is all off
  let sOff = min(vW * t, Ls);
  let c = flagCoords(u, v, n0, min(t, tSet));
  let inTongue = tonguePoint(c.x, sOff - c.y, c.z);
  if (t < tSet) {
    // settle in: a hop from the crown, which for the folded half crosses the roll
    let tau = clamp((t - tOff) / FLAG_HOP, 0.0, 1.0);
    return mix(ride, inTongue, tau * tau * (3.0 - 2.0 * tau));
  }
  // set-down: the bundle is carried onto the nip on the other half, lowest ply just above the pile
  let yTop = max(pileTop(), P.front.x + R) + 1.5 * P.hdt.x;
  let xHere = select(L - c.x, c.x, flopSide() == 0u);
  let dest = vec3<f32>(L - xHere, yTop + c.z, P.fold2.w + (c.y - 1.5 * W) * FLAG_SPREAD);
  let src = tonguePoint(c.x, Ls - c.y, c.z);
  let tau = clamp((t - tSet) / FLAG_SET, 0.0, 1.0);
  return mix(src, dest, tau * tau * (3.0 - 2.0 * tau));
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
    // the flag fold (flagPos); the velocity is the move's over the last substep
    let lo = vec3<f32>(1.5 * P.hdt.x);
    let hi = vec3<f32>(P.domain.x, P.fold3.w, P.domain.z) - lo;
    let dt = P.hdt.z;
    let tPrev = max(t - dt, 0.0);
    let x = clamp(flagPos(p0, f0.w, t), lo, hi);
    let xPrev = clamp(flagPos(p0, f0.w, tPrev), lo, hi);
    let v = select(vec3<f32>(0.0), (x - xPrev) / (t - tPrev), t > tPrev);
    pos[p] = vec4<f32>(x, pos[p].w);
    vel[p] = vec4<f32>(v, 0.0);
    return;
  }
  let tRoll = P.fold.z;
  let feed = P.fold.w;
  let a = logAxis();
  let lo = vec3<f32>(1.5 * P.hdt.x);
  let hi = vec3<f32>(P.domain.x, P.fold3.w, P.domain.z) - lo;
  if (t < tRoll) {
    // peel and wind, then lift (windPos); the velocity is the move's over the last substep
    let dt = P.hdt.z;
    let tPrev = max(t - dt, 0.0);
    let x = clamp(windPos(p0, f0.w, t), lo, hi);
    let xPrev = clamp(windPos(p0, f0.w, tPrev), lo, hi);
    let v = select(vec3<f32>(0.0), (x - xPrev) / (t - tPrev), t > tPrev);
    pos[p] = vec4<f32>(x, pos[p].w);
    vel[p] = vec4<f32>(v, 0.0);
    return;
  }
  let pLog = windPos(p0, f0.w, tRoll);
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

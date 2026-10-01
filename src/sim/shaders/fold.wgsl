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

// Second operator move, "cut & fold" (P.fold.x = 2 / 3 / 4 / 5 / 6 / 7: the fold kind
// (P.fold.x - 2) / 2 and the end the cut starts from, x = 0 / x = L, in the low bit),
// the one an operator makes most, done the way a flag is folded: cut the sheet ACROSS
// on the front face of the front roll, from the roll end to the middle just below the
// crown, so the half-width strip below the cut is free at the top and still joined to
// the other half along the UNCUT SEAM at the middle. Then, for each of the squares of
// strip the roll brings up under the cut (BUNDLE_SQUARES of them, about one turn):
//   fold (kind 0, modes 2 / 3): the free top corner at the roll end is folded over
//     the diagonal of the strip's top square (from the seam at the cut down to the
//     roll end one strip-width lower), so the corner lands at the seam at the bottom
//     of the square: a triangle flopped onto the triangle below it.
//   lift (kind 1, modes 4 / 5, selected afresh, so it takes the doubled triangle and
//     whatever the roll has brought up under it): that doubled triangle is lifted off
//     the roll into the operator's hands out in front of the roll, laid flat on top of
//     what is already there (square P.fold4.x of the bundle), and PARKED: held there
//     (flag bit BUNDLE_FLAG, fold0 = its place) through the folds that follow.
//   set-down (kind 2, modes 6 / 7, after the last square): the whole bundle, a flat
//     triangle of 2 * BUNDLE_SQUARES plies, is carried over onto the other half of
//     the roll and set down on the nip, where the rolls pull it in.
// The fold is a page turn on a hinge drawn in the sheet's own coordinates (u in from
// the cut end, s down from the cut; the sheet is developable, so the hinge is
// straight there even though it is a helix on the roll): a particle turns about
// its foot on the hinge in the plane of the hinge's in-sheet normal and the local
// radial direction, through pi, keeping its depth below the flap's top as height
// above the landing, and its swing's radial reach is flattened (FLOP_LIFT) so the
// widest flap stays inside the domain. Kinematic script per step (P.fold.z = its length):
//   select: the flap (this step's triangle, radial depth < flapDepth(), inside the
//           strip's square, never a parked particle) is binned by arc (NB bins, slot 1)
//           and depth (NS slices) and flagged; the sheet it lands on (the lower
//           triangle) is binned the same way (slot 0), unflagged.
//   tables: per arc bin the thickness of each (a high quantile of its depth).
//   move:   the page turn / the flight into the hands / the carry to the nip; parked
//           particles are held where they are.
//   finish: release at rest with F = I (the fold), park (the lift), or release the
//           whole bundle (the set-down).
const FLOP_LIFT : f32 = 0.6;        // radial reach of the swing as a fraction of the lateral one (fits the domain in front of the roll)
const FLOP_TOP_Q : f32 = 0.97;      // sheet thickness = this quantile of its particles' depth
const FLAP_TH_MIN : f32 = 0.08;     // the cut runs across just in front of the crown line (rad down from the crown)
// The bundle in the operator's hands (mirrors mpm.ts): out in front of the front roll, this far
// above the crown level and beyond the face (the face is 0.59 from the front of the domain, so the
// triangle's s is spread over BUNDLE_SPREAD of its length toward the viewer about a centre
// BUNDLE_Z out), each square's doubled triangle BUNDLE_LAYER sheet thicknesses up.
const BUNDLE_FLAG : u32 = 2u;
const BUNDLE_Y : f32 = 0.3;
const BUNDLE_Z : f32 = 0.27;
const BUNDLE_SPREAD : f32 = 0.6;
const BUNDLE_LAYER : f32 = 2.2;
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
/** Which end the cut starts from (0: the x = 0 end, 1: the x = L end): modes 2 / 4 and 3 / 5. */
fn flopSide() -> u32 { return u32(round(P.fold.x)) & 1u; }
/** Which step of cut & fold: 0 the fold over the diagonal, 1 the lift into the hands, 2 the set-down. */
fn flopKind() -> u32 { return (u32(round(P.fold.x)) - 2u) >> 1u; }
/** How deep off the roll the flap reaches: the sheet, with slack for a doubled one. */
fn flapDepth() -> f32 { return 3.0 * P.fold2.z + P.hdt.x; }
/** Width of the strip: half the roll; the top square of the strip is this long down the face too. */
fn stripW() -> f32 { return 0.5 * P.fold2.y; }
/** Angle down from the crown of the bottom of the strip's top square. */
fn flapThMax() -> f32 { return FLAP_TH_MIN + stripW() / P.front.w; }
fn flapBinWidth() -> f32 { return (flapThMax() - FLAP_TH_MIN) / f32(NB); }
/** Thickness of the sheet at angle theta down the front face (linear between bins):
    slot 1 the flap, slot 0 what it lands on. */
fn sheetThick(theta : f32, slot : u32) -> f32 {
  let fb = clamp((theta - FLAP_TH_MIN) / flapBinWidth() - 0.5, 0.0, f32(NB) - 1.0001);
  let b = u32(fb);
  return mix(tables[b * TS + slot], tables[(b + 1u) * TS + slot], fb - f32(b));
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
  if (dr < flapDepth()) {
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
    if ((flags[p] & 1u) != 0u || flopKind() == 2u) { return; }   // parked in the bundle; the set-down takes the bundle as is
    let L = P.fold2.y;
    let k = flopSide();
    let R = P.front.w;
    let W = stripW();
    // the sheet on the front face inside the strip's top square (and the other half beside it)
    let dy = x.y - P.front.x;
    let dz = x.z - P.front.y;
    let dr = sqrt(dy * dy + dz * dz) - R;
    let th = atan2(dz, dy);   // 0 at the crown, pi/2 at the axis level in front
    if (dr < 0.0 || dr > flapDepth() || th < FLAP_TH_MIN || th > flapThMax()) { return; }
    let u = select(L - x.x, x.x, k == 0u);   // in from the cut end
    let sArc = R * (th - FLAP_TH_MIN);       // down from the cut
    let b = min(u32((th - FLAP_TH_MIN) / flapBinWidth()), NB - 1u);
    let j = min(u32(dr / flapDepth() * f32(NS)), NS - 1u);
    var flap = false;
    var landing = false;
    if (flopKind() == 1u) {
      // the doubled lower triangle of the square is lifted into the hands
      flap = u < W && u + sArc >= W;
    } else {
      // the free corner triangle goes over the diagonal onto the lower triangle
      flap = u + sArc < W;
      landing = u < W && u + sArc >= W;
    }
    if (flap) {
      atomicAdd(&info[INFO_COUNT + NB + b], 1u);
      atomicAdd(&info[INFO_DEPTH + (NB + b) * NS + j], 1u);
      flags[p] = flags[p] | 1u;
      fold0[p] = vec4<f32>(x, th);
      atomicAdd(&info[3], 1u);
    } else if (landing) {
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
      tables[b * TS + k] = thick;
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
  flags[p] = flags[p] & ~(1u | BUNDLE_FLAG);
}

/** Where a lifted particle sits in the bundle in the operator's hands: at its own x, its depth off
    the roll as height within its square's layer (roll side down), its arc down the cut spread
    toward the viewer; layers stack up square by square (P.fold4.x). */
fn bundlePos(p0 : vec3<f32>) -> vec3<f32> {
  let R = P.front.w;
  let gap = P.fold2.z;
  let dy = p0.y - P.front.x;
  let dz = p0.z - P.front.y;
  let dr = clamp(sqrt(dy * dy + dz * dz) - R, 0.0, 3.0 * gap);
  let th = atan2(dz, dy);
  let sArc = clamp(R * (th - FLAP_TH_MIN), 0.0, stripW());
  let y = P.front.x + R + BUNDLE_Y + P.fold4.x * BUNDLE_LAYER * gap + dr;
  let z = P.front.y + R + BUNDLE_Z + (sArc - 0.5 * stripW()) * BUNDLE_SPREAD;
  return vec3<f32>(p0.x, y, z);
}

/** Where a parked particle (at pb in the hands) is set down: on the other half of the roll, the
    bundle's lowest ply just above the pile the nip is eating (or the roll tops), centred on the nip. */
fn setDownPos(pb : vec3<f32>) -> vec3<f32> {
  let R = P.front.w;
  let yB = P.front.x + R + BUNDLE_Y;
  let zB = P.front.y + R + BUNDLE_Z;
  let yTop = max(pileTop(), P.front.x + R) + 1.5 * P.hdt.x;
  return vec3<f32>(P.fold2.y - pb.x, yTop + (pb.y - yB), P.fold2.w + (pb.z - zB));
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
    let L = P.fold2.y;
    let R = P.front.w;
    let h = P.hdt.x;
    let W = stripW();
    let T = max(P.fold.z, 1e-3);
    let lo = vec3<f32>(1.5 * h);
    let hi = vec3<f32>(P.domain.x, P.fold3.w, P.domain.z) - lo;
    let kind = flopKind();
    let parked = (flags[p] & BUNDLE_FLAG) != 0u;
    if (kind == 2u) {
      // set-down: the bundle is carried from the hands onto the nip on the other half
      let tau = clamp(t / T, 0.0, 1.0);
      let sm = tau * tau * (3.0 - 2.0 * tau);
      let dsdt = 6.0 * tau * (1.0 - tau) / T;
      let dest = setDownPos(p0);
      let x = clamp(mix(p0, dest, sm), lo, hi);
      pos[p] = vec4<f32>(x, pos[p].w);
      if (t >= T) { release(p, vec3<f32>(0.0)); } else { vel[p] = vec4<f32>((dest - p0) * dsdt, 0.0); }
      return;
    }
    if (parked) {
      // in the hands: held where it is while the next square is folded and lifted
      pos[p] = vec4<f32>(p0, pos[p].w);
      vel[p] = vec4<f32>(0.0);
      return;
    }
    if (kind == 1u) {
      // lift: the doubled triangle flies off the roll into the hands, onto the bundle
      let tau = clamp(t / T, 0.0, 1.0);
      let sm = tau * tau * (3.0 - 2.0 * tau);
      let dsdt = 6.0 * tau * (1.0 - tau) / T;
      let dest = bundlePos(p0);
      let x = clamp(mix(p0, dest, sm), lo, hi);
      pos[p] = vec4<f32>(x, pos[p].w);
      vel[p] = vec4<f32>((dest - p0) * dsdt, 0.0);
      return;
    }
    // the fold: a page turn on a hinge drawn in the sheet's own coordinates
    let th = f0.w;
    let dy0 = p0.y - P.front.x;
    let dz0 = p0.z - P.front.y;
    let dr = max(sqrt(dy0 * dy0 + dz0 * dz0) - R, 0.0);
    let left = flopSide() == 0u;
    let u0 = select(L - p0.x, p0.x, left);
    let s0 = R * (th - FLAP_TH_MIN);
    // hinge: a point and the in-sheet normal pointing at the landing side: the diagonal u + s = W
    let hp = vec2<f32>(W, 0.0);
    let nrm = vec2<f32>(0.7071068, 0.7071068);
    let q0 = vec2<f32>(u0, s0) - hp;
    let foot = vec2<f32>(u0, s0) - nrm * dot(q0, nrm);
    let w = max(-dot(q0, nrm), 0.0);             // distance in from the hinge, on the flap side
    let thick = sheetThick(th, 1u);
    // the landing point in sheet coordinates, and the thickness of what is there
    let land = foot + nrm * w;
    let thLand = clamp(FLAP_TH_MIN + land.y / R, FLAP_TH_MIN, flapThMax());
    let recv = sheetThick(thLand, 0u);
    // pivot: the top of the flap; the particle sits d below it
    let rhoP = R + thick;
    let d = rhoP - (R + dr);
    let tau = clamp(t / T, 0.0, 1.0);
    let u = tau * tau * (3.0 - 2.0 * tau);
    let dudt = 6.0 * tau * (1.0 - tau) / T;
    let phi = PI * u;
    // in-sheet offset along the normal -w -> +w (mirrored across the hinge), radial -d -> +d
    // (roll side up), the radial reach flattened by FLOP_LIFT; the landing sheet may be
    // thicker or thinner than the flap, so the radial offset eases onto its top over the turn
    let lat = -w * cos(phi) + d * sin(phi);
    let rad = FLOP_LIFT * w * sin(phi) - d * cos(phi);
    let dlat = (w * sin(phi) + d * cos(phi)) * PI;
    let drad = (FLOP_LIFT * w * cos(phi) + d * sin(phi)) * PI;
    let settle = recv + 0.5 * h - thick;
    let rho = rhoP + rad + settle * u;
    let us = foot + nrm * lat;                   // (u, s) now
    let thNow = FLAP_TH_MIN + us.y / R;
    let rhat = vec2<f32>(cos(thNow), sin(thNow));
    let xNow = select(L - us.x, us.x, left);
    let x = clamp(vec3<f32>(xNow, P.front.x + rho * rhat.x, P.front.y + rho * rhat.y), lo, hi);
    // velocity: the in-sheet motion (u along x, s around the roll) plus the radial one
    let dus = nrm * (dlat * dudt);
    let vth = dus.y / R;                         // d(theta)/dt
    let tang = vec2<f32>(-sin(thNow), cos(thNow));
    let vrad = (drad + settle) * dudt;
    let vyz = rhat * vrad + tang * (rho * vth);
    let v = vec3<f32>(select(-dus.x, dus.x, left), vyz.x, vyz.y);
    pos[p] = vec4<f32>(x, pos[p].w);
    if (t >= T) {
      release(p, vec3<f32>(0.0));
    } else {
      vel[p] = vec4<f32>(v, 0.0);
    }
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
  if (isFlop()) {
    let kind = flopKind();
    if (kind == 1u) {
      // park the lifted triangle in the hands: held there until the set-down
      fold0[p] = vec4<f32>(pos[p].xyz, 0.0);
      flags[p] = flags[p] | BUNDLE_FLAG;
      return;
    }
    if (kind == 0u && (flags[p] & BUNDLE_FLAG) != 0u) { return; }   // the bundle stays in the hands
    release(p, vec3<f32>(0.0));
    return;
  }
  release(p, -logAxis() * P.fold.w);
}

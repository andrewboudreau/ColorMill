// Operator move (design §6): cut ALL the material off the mill (the sheet on the
// front roll, the bank in the nip, whatever rides the back roll), roll it up
// into a log, stand the log over the nip and FEED it in end-first, the way an
// operator does (the rolls consume it over a few seconds and its spiral
// cross-section, every colour interleaved, is squeezed out across the full roll
// width). That is how a two-roll mill mixes along the roll axis.
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
//           t >= T_roll: the log translates along -a at the feed speed; a particle
//           that reaches the release plane (just above the live pile the nip is
//           eating, reduced by g2p every substep, or the roll tops) is released:
//           flag cleared, C = 0, F = I, v = feed velocity, P2G affine rebuilt.
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
// on the FRONT FACE of the front roll, drawn from the roll end toward the middle
// while the sheet comes up past it, so the cut runs diagonally on the sheet (the
// end was cut first and has moved up) and the freed flap is a triangle: a point
// at the top near the roll end, wide at the bottom where the sheet leaves the
// nip. The operator takes the bottom corner and folds the flap over the cut
// onto the sheet beside it, toward the middle: what was against the roll is now
// on top, and the doubled sheet rides up into the nip. Kinematic script:
//   select: the flap = sheet on the front face (radial depth < flapDepth(),
//           FLAP_TH_MIN..FLAP_TH_MAX down from the crown) on the cut side of the
//           diagonal cutX(theta); binned by arc (NB bins, slot 1) and depth
//           (NS slices) and flagged. The rest of that half's top-of-mill material
//           is binned by z (NB bins, slot 0) and height (NS slices), for the part
//           of the fold that reaches back over the crown.
//   tables: per arc bin the flap's thickness (a high quantile of its depth); per
//           z bin the top of the material behind the crown (a high quantile, or
//           the mill surface).
//   move:   in the sheet's own coordinates (u along the roll from the cut end, s
//           down the front face from the crown) the flap is mirrored across the
//           cut line, so it lands on the sheet beside the cut, outside it by its
//           own thickness with the roll side up; the part of the mirror image
//           that falls behind the crown lands on the bank top there. It gets
//           there as a page turning on the cut: straight toward its mirror
//           image, lifted up in proportion to its distance from the hinge, so
//           halfway it stands on the cut line and the corner is at the top.
//           Released at rest with F = I.
const FLOP_SECONDS : f32 = 0.8;
const FLOP_CAP_SIN : f32 = 0.766;   // sin 50 deg: how far down the back crown the receiving region reaches
const FLOP_YSPAN : f32 = 1.0;       // height above the axes covered by the receiving y slices
const FLOP_TOP_Q : f32 = 0.97;      // column top / flap thickness = this quantile of its particles
const FLAP_TH_MIN : f32 = 0.08;     // flap starts just in front of the crown line (rad down from the crown)
const FLAP_TH_MAX : f32 = 1.92;     // and reaches a little below the axis level, where the sheet comes up from the nip
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
fn flopBinWidth() -> f32 { return P.domain.z / f32(NB); }
fn flopSliceHeight() -> f32 { return FLOP_YSPAN / f32(NS); }
/** How deep off the roll the flap reaches: the sheet, with slack for a doubled one. */
fn flapDepth() -> f32 { return 3.0 * P.fold2.z + P.hdt.x; }
fn flapBinWidth() -> f32 { return (FLAP_TH_MAX - FLAP_TH_MIN) / f32(NB); }

/** Height of the mill's top surface at depth z: the roll crowns, or the channel floor between them. */
fn millTop(z : f32) -> f32 {
  let R = P.front.w;
  var top = P.front.x + 0.06;
  let dzf = z - P.front.y;
  if (abs(dzf) < R) { top = max(top, P.front.x + sqrt(R * R - dzf * dzf)); }
  let dzb = z - P.back.y;
  if (abs(dzb) < R) { top = max(top, P.back.x + sqrt(R * R - dzb * dzb)); }
  return top;
}

/** Material behind the crown the fold can reach: on top of the mill, between the crowns' 50-degree lines. */
fn inFlopRegion(x : vec3<f32>) -> bool {
  let reach = FLOP_CAP_SIN * P.front.w;
  return x.y > P.front.x + 0.06 && x.z >= P.back.y - reach && x.z <= P.front.y + reach
      && x.y > millTop(x.z) - 0.5 * P.hdt.x;
}

/** Where the cut sits at angle theta down the front face: the roll end at the top (cut first,
    moved up since), the middle at the bottom (cut last, just up from the nip). */
fn cutX(theta : f32) -> f32 {
  let f = clamp((theta - FLAP_TH_MIN) / (FLAP_TH_MAX - FLAP_TH_MIN), 0.0, 1.0);
  return 0.5 * P.fold2.y * f;
}

/** Top of the material on top of the mill at depth z (linear between bins). */
fn flopTop(z : f32) -> f32 {
  let fb = clamp(z / flopBinWidth() - 0.5, 0.0, f32(NB) - 1.0001);
  let b = u32(fb);
  return mix(tables[b * 8u], tables[(b + 1u) * 8u], fb - f32(b));
}
/** Thickness of the flap at angle theta down the front face (linear between bins). */
fn flapThick(theta : f32) -> f32 {
  let fb = clamp((theta - FLAP_TH_MIN) / flapBinWidth() - 0.5, 0.0, f32(NB) - 1.0001);
  let b = u32(fb);
  return mix(tables[b * 8u + 1u], tables[(b + 1u) * 8u + 1u], fb - f32(b));
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
    // only the cut side's half takes part
    if (!select(x.x >= 0.5 * L, x.x < 0.5 * L, k == 0u)) { return; }
    // the flap: the sheet on the front face inside the triangle
    let dy = x.y - P.front.x;
    let dz = x.z - P.front.y;
    let dr = sqrt(dy * dy + dz * dz) - R;
    let th = atan2(dz, dy);   // 0 at the crown, pi/2 at the axis level in front
    let xc = cutX(th);
    if (dr >= 0.0 && dr <= flapDepth() && th >= FLAP_TH_MIN && th <= FLAP_TH_MAX && select(x.x > L - xc, x.x < xc, k == 0u)) {
      let b = min(u32((th - FLAP_TH_MIN) / flapBinWidth()), NB - 1u);
      let j = min(u32(dr / flapDepth() * f32(NS)), NS - 1u);
      atomicAdd(&info[INFO_COUNT + NB + b], 1u);
      atomicAdd(&info[INFO_DEPTH + (NB + b) * NS + j], 1u);
      flags[p] = flags[p] | 1u;
      fold0[p] = vec4<f32>(x, th);
      atomicAdd(&info[3], 1u);
      return;
    }
    // everything else on top of the mill on this half, by z column and height
    if (!inFlopRegion(x)) { return; }
    let b = min(u32(x.z / flopBinWidth()), NB - 1u);
    let j = min(u32(max(x.y - P.front.x, 0.0) / flopSliceHeight()), NS - 1u);
    atomicAdd(&info[INFO_COUNT + b], 1u);
    atomicAdd(&info[INFO_DEPTH + b * NS + j], 1u);
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
    // slot 0, per z bin: the top of the material behind the crown, FLOP_TOP_Q of the way up its column
    // (a few strays above do not count), or the mill surface where it is empty
    let z = (f32(b) + 0.5) * flopBinWidth();
    var top = millTop(z);
    let n0 = atomicLoad(&info[INFO_COUNT + b]);
    if (n0 > 0u) {
      let want = FLOP_TOP_Q * f32(n0);
      var acc = 0.0;
      var j = 0u;
      for (; j < NS; j++) {
        let nj = f32(atomicLoad(&info[INFO_DEPTH + b * NS + j]));
        if (acc + nj >= want) {
          top = P.front.x + (f32(j) + (want - acc) / max(nj, 1.0)) * flopSliceHeight();
          break;
        }
        acc += nj;
      }
      if (j == NS) { top = P.front.x + FLOP_YSPAN; }
      top = max(top, millTop(z));
    }
    tables[b * 8u] = top;
    // slot 1, per arc bin: the flap's thickness off the roll, the same quantile of its depth
    var thick = P.hdt.x;
    let n1 = atomicLoad(&info[INFO_COUNT + NB + b]);
    if (n1 > 0u) {
      let want = FLOP_TOP_Q * f32(n1);
      var acc = 0.0;
      var j = 0u;
      for (; j < NS; j++) {
        let nj = f32(atomicLoad(&info[INFO_DEPTH + (NB + b) * NS + j]));
        if (acc + nj >= want) {
          thick = (f32(j) + (want - acc) / max(nj, 1.0)) * flapDepth() / f32(NS);
          break;
        }
        acc += nj;
      }
      if (j == NS) { thick = flapDepth(); }
      thick = max(thick, P.hdt.x);
    }
    tables[b * 8u + 1u] = thick;
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
    // fold the flap over the cut onto the sheet beside it
    let L = P.fold2.y;
    let R = P.front.w;
    let h = P.hdt.x;
    let th = f0.w;
    let dy0 = p0.y - P.front.x;
    let dz0 = p0.z - P.front.y;
    let dr = max(sqrt(dy0 * dy0 + dz0 * dz0) - R, 0.0);
    let left = flopSide() == 0u;
    // the sheet's own coordinates: u along the roll from the cut end, s down from the crown;
    // the cut runs from (0, 0) to (L/2, S)
    let S = R * (FLAP_TH_MAX - FLAP_TH_MIN);
    let q = vec2<f32>(select(L - p0.x, p0.x, left), R * (th - FLAP_TH_MIN));
    let d = normalize(vec2<f32>(0.5 * L, S));
    let q1 = 2.0 * dot(q, d) * d - q;             // mirror image across the cut
    let x1 = clamp(select(L - q1.x, q1.x, left), 1.5 * h, L - 1.5 * h);
    let th1 = FLAP_TH_MIN + q1.y / R;
    let thick = flapThick(th);
    var p1 : vec3<f32>;
    if (th1 > 0.0) {
      // on the front face beside the cut: outside the sheet there, roll side up
      let rho = R + flapThick(clamp(th1, FLAP_TH_MIN, FLAP_TH_MAX)) + 0.5 * h + max(thick - dr, 0.0);
      p1 = vec3<f32>(x1, P.front.x + rho * cos(th1), P.front.y + rho * sin(th1));
    } else {
      // the part of the image that reaches back over the crown lands on the bank top there
      let z1 = P.front.y + R * th1;
      p1 = vec3<f32>(x1, flopTop(z1) + 0.5 * h + max(thick - dr, 0.0), z1);
    }
    // the turn: the flap swings up as a page on its hinge, the cut line in space (from
    // the top corner at the roll end to the middle at the bottom), the whole flap
    // through the same angle so it keeps its shape, out from the roll and over onto
    // its mirror image; it settles onto the landing spot over the last stretch (the
    // hinge is a chord inside the roll, so the rigid image sits a little under the
    // surface and the swing is held outside the roll)
    let g = P.fold2.z;
    let xEnd = select(L, 0.0, left);
    let a3 = vec3<f32>(xEnd, P.front.x + (R + g) * cos(FLAP_TH_MIN), P.front.y + (R + g) * sin(FLAP_TH_MIN));
    let c3 = vec3<f32>(0.5 * L, P.front.x + (R + g) * cos(FLAP_TH_MAX), P.front.y + (R + g) * sin(FLAP_TH_MAX));
    let e = normalize(c3 - a3);
    // sense of the turn: the corner comes off the roll outward
    let qb = vec3<f32>(xEnd, c3.y, c3.z) - a3;
    let qbp = qb - e * dot(e, qb);
    let sgn = sign(dot(cross(e, qbp), vec3<f32>(0.0, cos(FLAP_TH_MAX), sin(FLAP_TH_MAX))));
    let q3 = p0 - a3;
    let qpar = e * dot(e, q3);
    let qperp = q3 - qpar;
    let tau = clamp(t / FLOP_SECONDS, 0.0, 1.0);
    let u = tau * tau * (3.0 - 2.0 * tau);
    let dudt = 6.0 * tau * (1.0 - tau) / FLOP_SECONDS;
    let phi = sgn * PI * u;
    var rigid = a3 + qpar + qperp * cos(phi) + cross(e, qperp) * sin(phi);
    var drigid = (cross(e, qperp) * cos(phi) - qperp * sin(phi)) * (sgn * PI);
    let rel = rigid.yz - vec2<f32>(P.front.x, P.front.y);
    let dRoll = length(rel);
    let rMin = R + dr + 0.5 * h;
    if (dRoll < rMin) { rigid = vec3<f32>(rigid.x, vec2<f32>(P.front.x, P.front.y) + rel * (rMin / max(dRoll, 1e-4))); }
    let sw = clamp((u - 0.7) / 0.3, 0.0, 1.0);
    let w = sw * sw * (3.0 - 2.0 * sw);
    let dwdu = 6.0 * sw * (1.0 - sw) / 0.3;
    let lo = vec3<f32>(1.5 * h);
    let hi = vec3<f32>(P.domain.x, P.fold3.w, P.domain.z) - lo;
    let x = clamp(mix(rigid, p1, w), lo, hi);
    let v = ((1.0 - w) * drigid + (p1 - rigid) * dwdu) * dudt;
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
  // feed: the log descends along its axis; release what reaches the pile the nip
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

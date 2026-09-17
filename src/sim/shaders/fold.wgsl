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

// Second operator move, "cut & fold" (P.fold.x = 2: the x < L/2 half is lifted, 3: the
// x >= L/2 half): the operator cuts the sheet across at the middle of the roll,
// peels what is on top of the mill on one side (the bank between the rolls and
// the sheet over the front roll's crown, out to FLOP_CAP_SIN of the radius) and
// flops it over onto the other half like turning a page, hinged on the cut. In
// the flop the bins run along z (NB over the domain depth) and the slices up y
// (NS over FLOP_YSPAN above the axes); the tables hold, per bin, the top of
// each half's material (a high percentile of its column, or the mill surface
// where the column is empty). A particle at (x, y) turns about the hinge line
// x = L/2, y = (topLift + topRecv)/2 of its column through FLOP_SECONDS, so it
// lands mirrored in x with the flap's former top resting on the receiving
// half's top and its former underside on the outside. Nothing else moves; the
// flap is released at rest and the rolls take it from there.
const FLOP_SECONDS : f32 = 0.8;
const FLOP_CAP_SIN : f32 = 0.766;   // sin 50 deg: how far down the crown the flap reaches
const FLOP_YSPAN : f32 = 1.0;       // height above the axes covered by the y slices
const FLOP_TOP_Q : f32 = 0.97;      // column top = this quantile of its particles
const INFO_COUNT : u32 = 8u;
const INFO_DEPTH : u32 = 8u + 2u * NB;
const HDR : u32 = NB * 8u;
const CDF : u32 = NB * 8u + 8u;

fn arcTotal() -> f32 { return 2.0 * PI * P.front.w; }
/** Fractional depth slice of a radial depth dr: slice j spans DMAX * (j/NS)^2 .. DMAX * ((j+1)/NS)^2. */
fn depthSlice(dr : f32) -> f32 { return f32(NS) * sqrt(max(dr, 0.0) / DMAX); }
fn binWidth() -> f32 { return arcTotal() / f32(NB); }

fn isFlop() -> bool { return P.fold.x > 1.5; }
/** Which half of the width the flop lifts (0: x < L/2, 1: x >= L/2). */
fn flopSide() -> u32 { return select(0u, 1u, P.fold.x > 2.5); }
fn flopBinWidth() -> f32 { return P.domain.z / f32(NB); }
fn flopSliceHeight() -> f32 { return FLOP_YSPAN / f32(NS); }

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

/** Material the flop can take: on top of the mill, between the crowns' 50-degree lines. */
fn inFlopRegion(x : vec3<f32>) -> bool {
  let reach = FLOP_CAP_SIN * P.front.w;
  return x.y > P.front.x + 0.06 && x.z >= P.back.y - reach && x.z <= P.front.y + reach
      && x.y > millTop(x.z) - 0.5 * P.hdt.x;
}

/** Top of half k's material at depth z (linear between bins). */
fn flopTop(k : u32, z : f32) -> f32 {
  let fb = clamp(z / flopBinWidth() - 0.5, 0.0, f32(NB) - 1.0001);
  let b = u32(fb);
  return mix(tables[b * 8u + k], tables[(b + 1u) * 8u + k], fb - f32(b));
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
    // bin what is on top of the mill by (half, z bin, y slice); flag the lifted half
    if (!inFlopRegion(x)) { return; }
    let k = select(0u, 1u, x.x >= 0.5 * P.fold2.y);
    let b = min(u32(x.z / flopBinWidth()), NB - 1u);
    let j = min(u32(max(x.y - P.front.x, 0.0) / flopSliceHeight()), NS - 1u);
    atomicAdd(&info[INFO_COUNT + k * NB + b], 1u);
    atomicAdd(&info[INFO_DEPTH + (k * NB + b) * NS + j], 1u);
    if (k == flopSide()) {
      flags[p] = flags[p] | 1u;
      fold0[p] = vec4<f32>(x, 0.0);
      atomicAdd(&info[3], 1u);
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
    // per (half, z bin): the top of the column, FLOP_TOP_Q of the way up its particles
    // (a few strays above do not count), or the mill surface where it is empty
    let z = (f32(b) + 0.5) * flopBinWidth();
    for (var k = 0u; k < 2u; k++) {
      let n = atomicLoad(&info[INFO_COUNT + k * NB + b]);
      var top = millTop(z);
      if (n > 0u) {
        let want = FLOP_TOP_Q * f32(n);
        var acc = 0.0;
        var j = 0u;
        for (; j < NS; j++) {
          let nj = f32(atomicLoad(&info[INFO_DEPTH + (k * NB + b) * NS + j]));
          if (acc + nj >= want) {
            top = P.front.x + (f32(j) + (want - acc) / max(nj, 1.0)) * flopSliceHeight();
            break;
          }
          acc += nj;
        }
        if (j == NS) { top = P.front.x + FLOP_YSPAN; }
        top = max(top, millTop(z));
      }
      tables[b * 8u + k] = top;
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
    // turn the page: rotate (x, y) about the hinge line at the cut, in the plane of
    // the particle's own z, from lying on its half to lying mirrored on the other
    let L = P.fold2.y;
    let kLift = flopSide();
    let topLift = flopTop(kLift, p0.z);
    let topRecv = flopTop(1u - kLift, p0.z) + 0.5 * P.hdt.x;   // half a cell of clearance
    let yHinge = 0.5 * (topLift + topRecv);
    let dirX = select(-1.0, 1.0, kLift == 0u);   // the lifted half is at x = L/2 + dirX * u, u <= 0
    let u = -abs(p0.x - 0.5 * L);
    let w = p0.y - yHinge;
    let tau = clamp(t / FLOP_SECONDS, 0.0, 1.0);
    let sm = tau * tau * (3.0 - 2.0 * tau);
    let phi = PI * sm;
    let dphi = PI * 6.0 * tau * (1.0 - tau) / FLOP_SECONDS;
    let u1 = u * cos(phi) + w * sin(phi);
    let w1 = -u * sin(phi) + w * cos(phi);
    let lo = vec3<f32>(1.5 * P.hdt.x);
    let hi = vec3<f32>(P.domain.x, P.fold3.w, P.domain.z) - lo;
    let x = clamp(vec3<f32>(0.5 * L + dirX * u1, yHinge + w1, p0.z), lo, hi);
    let v = vec3<f32>(dirX * w1 * dphi, -u1 * dphi, 0.0);
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

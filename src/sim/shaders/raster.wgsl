// Pigment raster (design §3.5): scatter particle mass, pigment load and load-weighted
// latent to the node accumulators pack reads. Kinematic particles are
// included (they carry pigment and must be visible). The clear base has load 0 and
// scatters mass only: its latent slot has no weight anywhere.
// On the solver grid only the mass is scattered (the density the renderer, the grid update and the
// fold read); that grid's load and latent accumulators are the dispersion exchange's pools
// (disperse.wgsl). The pigment the picture shows is the colour raster's.
// Uses the tight 8-node trilinear stencil (not the solver's 27-node quadratic
// one): pigment streaks in a milled sheet are one or two cells thick, and the
// wide kernel averaged them into the surrounding white before they were drawn.
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lat : array<f32>;
@group(0) @binding(3) var<storage, read_write> pmass : array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> plat : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> pload : array<atomic<i32>>;
// Pigment coverage (colour raster only): the mass of pigment-carrying particles per node, so the
// renderer can tell a cell that is one-eighth masterbatch (a lamina edge) from one that is evenly
// tinted, which the load ratio alone cannot. The solver-grid kernel binds a small dummy here and
// never writes it (nothing reads coverage on the solver grid).
@group(0) @binding(6) var<storage, read_write> pcov : array<atomic<i32>>;
// the material's box (box.wgsl): the colour raster's lattice (BOX_FINE)
@group(0) @binding(7) var<storage, read> box : array<u32>;
// the lamina state per particle (lamina.wgsl; lamina_common.wgsl is concatenated before this file):
// normal xyz and q = ln(s / s0), then ln s0 and the exchange's reference; only the colour raster reads them
@group(0) @binding(8) var<storage, read> lam : array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> mixs : array<vec2<f32>>;

// A particle counts as covered (pigment) above this load. Dispersion relaxes every particle of a
// milled few-percent batch toward load ~1, and that must still count as covered so a homogeneous
// mix reads solid; only near-clear base (0, or the faint trace a chunk sheds into it) is uncovered.
const COVERAGE_MIN_LOAD : f32 = 0.3;

/// Scatter particle p, as if it sat at x, with every term times `scale` and its pigment (load and
/// latent, not its mass or coverage) times `ink` besides, onto the node lattice
/// of spacing 1/invh whose node `origin` is index 0 of a grid of node counts dims (the accumulators
/// bound to pmass / plat / pload / pcov are that grid's); withCov adds the coverage term (the
/// colour raster only).
fn scatterAt(p : u32, x : vec3<f32>, scale : f32, ink : f32, invh : f32, origin : vec3<i32>, dims : vec3<i32>, withCov : bool) {
  let pMass = P.part.y;
  // no floor: a clear particle (load 0) adds mass only. (A floor of 1e-3 gave the base's white latent
  // real weight: it whitened thin pigment and, through disperse, tinted clear material from nothing.)
  let load = max(pos[p].w, 0.0) * ink;
  let inked = load > 0.0;
  let covered = withCov && pos[p].w > COVERAGE_MIN_LOAD;
  var z : array<f32, 7>;
  for (var c = 0u; c < 7u; c++) { z[c] = lat[7u * p + c]; }

  let gx = x * invh - 0.5;                  // node-centred trilinear
  let cell = vec3<i32>(floor(gx));
  let f = gx - vec3<f32>(cell);
  let base = cell - origin;
  let nmax = dims - vec3<i32>(1);
  for (var k = 0; k < 2; k++) {
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        let o = vec3<i32>(i, j, k);
        let w = select(vec3<f32>(1.0) - f, f, o == vec3<i32>(1));
        let wgt = w.x * w.y * w.z * pMass * scale;
        let c = clamp(base + o, vec3<i32>(0), nmax);
        let n = (u32(c.z) * u32(dims.y) + u32(c.y)) * u32(dims.x) + u32(c.x);
        atomicAdd(&pmass[n], encodeFixed(wgt, MASS_SCALE));
        if (covered) { atomicAdd(&pcov[n], encodeFixed(wgt, MASS_SCALE)); }
        if (inked) {
          atomicAdd(&pload[n], encodeFixed(wgt * load, MASS_SCALE));
          for (var cc = 0u; cc < 7u; cc++) {
            atomicAdd(&plat[7u * n + cc], encodeFixed(wgt * load * z[cc], LAT_SCALE));
          }
        }
      }
    }
  }
}

/// The solver-grid raster: mass only, with the same stencil as scatterAt.
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let gx = pos[p].xyz * P.hdt.y - 0.5;
  let cell = vec3<i32>(floor(gx));
  let f = gx - vec3<f32>(cell);
  let nmax = vec3<i32>(P.grid.xyz) - vec3<i32>(1);
  for (var k = 0; k < 2; k++) {
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        let o = vec3<i32>(i, j, k);
        let w = select(vec3<f32>(1.0) - f, f, o == vec3<i32>(1));
        atomicAdd(&pmass[nodeIndexI(clamp(cell + o, vec3<i32>(0), nmax))], encodeFixed(w.x * w.y * w.z * P.part.y, MASS_SCALE));
      }
    }
  }
}

const STREAK_MIN_LOAD : f32 = 0.05;  // clear base is not streaked
const STREAK_TAPS : i32 = 6;         // most taps along a line
const STREAK_TAP_CELLS : f32 = 2.0;  // taps this many fine cells apart: a tap's trilinear footprint is two cells wide, so the line stays continuous
const STREAK_INK_RICH : f32 = 3.4;   // the ink of one particle in four ...
const STREAK_INK_LEAN : f32 = 0.2;   // ... and of the other three: 0.25 * 3.4 + 0.75 * 0.2 = 1

/// The colour raster (design §3.5): the same scatter onto the finer lattice the renderer reads
/// pigment from, so streaks thinner than a solver cell survive to the picture, plus coverage.
/// The lattice's cell and where its box sits are this frame's (box.wgsl frame_).
@compute @workgroup_size(128)
fn mainFine(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let origin = vec3<i32>(vec3<u32>(box[BOX_FINE], box[BOX_FINE + 1u], box[BOX_FINE + 2u]));
  let invh = P.fineH.y * 8.0 / f32(box[BOX_FINE + 3u]);
  let x = pos[p].xyz;

  // laminae (design §5): a pigment-carrying particle is drawn as a thread lying in its laminae, K taps
  // of weight 1/K; K = 1 is the particle's own position. A blob thinned from s0 to s has grown s0 / s
  // times as long in the plane of the laminae, so that (less the 1 it started at) is the thread's
  // length in particle spacings, up to the user's cap. The mill's flow is across the rolls' axis, so
  // the stretch within a lamina runs along x cross n; a lamina stacked along the axis (n along x)
  // has no direction to draw and stays a point.
  let cap = P.fineH.w;              // user's cap in particle spacings; 0 = off
  let spacing = 0.5 * P.hdt.x;      // particle spacing in world units
  var K = 1;
  var ink = 1.0;
  var span = vec3<f32>(0.0);
  if (cap > 0.0 && pos[p].w > STREAK_MIN_LOAD) {
    let st = lam[p];
    let dir = vec3<f32>(0.0, -st.z, st.y);    // x cross n
    let dl = length(dir);
    let len = min(exp(-clamp(st.w, LAMINA_Q_MIN, 0.0)) - 1.0, cap) * dl;
    if (len > 1e-3 && dl > 1e-3 && !isBad3(st.xyz) && !isBad(len)) {
      span = (dir / dl) * len * spacing;
      let cell = 1.0 / invh;        // this frame's fine cell
      K = clamp(i32(ceil(len * spacing / (STREAK_TAP_CELLS * cell))), 1, STREAK_TAPS);
      // unequal ink: a smeared chunk is laminae of unequal thickness, not an even film. One particle
      // in four draws its pigment STREAK_INK_RICH times over and the rest STREAK_INK_LEAN (a mean of
      // one), so lines lying side by side differ and the smear reads as threads where it is thin.
      // It comes in as the particle smears: unsmeared pigment (a fresh dollop, a stamp) is even.
      // And it goes out again as the laminae pass below what the picture can resolve: under the
      // particle spacing the threads are s / spacing of their contrast, and the colour is the blend.
      var hsh = p * 747796405u + 2891336453u;
      hsh = ((hsh >> ((hsh >> 28u) + 4u)) ^ hsh) * 277803737u;
      hsh = (hsh >> 22u) ^ hsh;
      let uneven = select(STREAK_INK_LEAN, STREAK_INK_RICH, (hsh & 3u) == 0u);
      let resolved = clamp(exp(st.w + mixs[p].x - lnSpacing()), 0.0, 1.0);
      ink = mix(1.0, uneven, clamp((len - 0.5) / 3.0, 0.0, 1.0) * resolved);
    }
  }
  let lo = vec3<f32>(spacing);
  let hi = P.domain.xyz - vec3<f32>(spacing);
  for (var i = 0; i < K; i++) {
    var xi = x;
    if (K > 1) { xi = clamp(x + span * ((f32(i) + 0.5) / f32(K) - 0.5), lo, hi); }
    scatterAt(p, xi, 1.0 / f32(K), ink, invh, origin, vec3<i32>(P.fine.xyz), true);
  }
}

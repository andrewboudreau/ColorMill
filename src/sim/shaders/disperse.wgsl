// Disperse (design §5): exchange of pigment between neighbouring particles once the flow has thinned
// its laminae below what a particle can carry, in passes through the solver grid's nodes so that what
// one particle gains another loses. (Concatenated after common.wgsl and lamina_common.wgsl.)
//
// When. A particle is a parcel one particle spacing (delta = h / 2) across. While its laminae are
// thicker than that, advection carries them and nothing is exchanged. Below it the parcel is striped,
// and a box delta wide that samples stripes s thick differs from the local mean in proportion to
// s / delta. So the share a particle trades is tied to how far its laminae thinned to a new minimum:
// with `mixed` the log thickness already accounted for and ref = min(mixed, ln delta),
//       beta = 1 - (s / exp(ref))^k   if ln s < ref, else 0;     mixed <- min(mixed, ln s)
// The product of (1 - beta) over the frames is (s_min / delta)^k whatever the frame length, the grid
// or the strain rate, and elastic chatter in s trades nothing (lamina.ts exchangeFraction is the
// mirror). k = P.part.w (MillParams.dispersion), 1 for the argument above.
//
// What is transported is the conserved pair itself: the pigment load w (pos.w) and the load-weighted
// latent w·z (seven channels), and with the pigment its lamina state (load-weighted q and ln s).
//
//   mark_ (scatter)  each particle with beta > 0 raises the 8 nodes round it to its beta (a max).
//   give_ (scatter)  each particle has a rate alpha: its own beta, or, for a particle with room for
//                    pigment, the marked beta round it (trilinear) times that room, 1 - w / PIGMENT_LOAD:
//                    clear material next to thinning laminae takes part, neat masterbatch whose own
//                    laminae are still thick does not. It puts alpha·w and alpha·w·z into the pools
//                    of the 8 nodes round it (trilinear weights N), and alpha·m into each node's
//                    capacity: how much of the pool the particle will take back.
//   take_ (gather)   each particle removes what it gave and takes from each node's pool the share
//                    its capacity is of the node's, with the same weights.
//
// Summed over particles, the shares of a node's pool are 1, so the total of w and of each w·z is
// unchanged: only round-off moves it. Three things make that exact and not approximate:
//   * one kernel both ways, one rate for what a particle gives and what it may take (so an even
//     field stays even whatever the rates are);
//   * take_ recomputes the fixed-point terms give_ added, from the same inputs: a particle
//     subtracts the integers it added (not alpha·w), and its share of a node is its integer
//     capacity term over the node's integer sum; a particle whose capacity term rounds to zero at
//     a node gives that node nothing, so no pool is left without a taker;
//   * a particle that does not exchange does neither half. Kinematic particles (a held pigment
//     chunk, material in the operator's hand or log) are skipped in every pass.
//
// The latent follows from the pair: z = (w·z) / w after the exchange. A clear particle (w = 0) has
// no colour of its own, so the first pigment it takes sets its latent, and its lamina state, outright.
// A particle keeps its own lamina normal: it follows the flow whatever the load.
//
// The pools are the solver-grid raster's load and latent accumulators (the raster itself only needs
// the mass there) plus the capacity, lamina and mark accumulators, cleared with them over the frame's
// box (clear.wgsl).
@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> lat : array<f32>;
@group(0) @binding(3) var<storage, read> flags : array<u32>;
@group(0) @binding(4) var<storage, read_write> lam : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> mixs : array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> xcap : array<atomic<i32>>;    // sum of N · alpha · m
@group(0) @binding(7) var<storage, read_write> xload : array<atomic<i32>>;   // sum of N · alpha · w
@group(0) @binding(8) var<storage, read_write> xlat : array<atomic<i32>>;    // sum of N · alpha · w · z, 7 per node
@group(0) @binding(9) var<storage, read_write> xq : array<atomic<i32>>;      // sum of N · alpha · w · (q, ln(s / delta)), 2 per node
@group(0) @binding(10) var<storage, read_write> xbeta : array<atomic<u32>>;  // max of beta over the particles round the node (float bits)

// Fixed point of the exchange terms, 2^21: a node's pool holds up to 1024 (mass · load · alpha;
// a packed node of neat masterbatch trading everything is 8 · 12), and a capacity term of a slowly
// thinning particle (alpha ~ 1e-3, weight 1/8) is still a few hundred counts.
const EX_SCALE : f32 = 2097152.0;
// ... and of the lamina terms, 2^17: they carry the load times up to 12 more.
const XQ_SCALE : f32 = 131072.0;

/// ln s of particle p over ln of the spacing, as the pools carry it.
fn relThickness(p : u32) -> f32 {
  return clamp(lam[p].w + mixs[p].x - lnSpacing(), -LAMINA_REL_MAX, LAMINA_REL_MAX);
}

/// The share of its pigment particle p trades because its own laminae thinned to a new minimum.
fn ownBeta(p : u32) -> f32 {
  if ((flags[p] & 1u) != 0u) { return 0.0; }
  let k = P.part.w;
  let w = pos[p].w;
  if (!(k > 0.0) || !(w > 0.0) || isBad(w)) { return 0.0; }
  let lnS = lam[p].w + mixs[p].x;
  let ref_ = min(mixs[p].y, lnSpacing());
  if (isBad(lnS) || isBad(ref_) || !(lnS < ref_)) { return 0.0; }
  return clamp(1.0 - exp(k * (lnS - ref_)), 0.0, 1.0);
}

/// The 8 nodes round a particle and their trilinear weights: the solver-grid raster's stencil
/// (raster.wgsl), so every node touched lies in the box the frame's clear covers.
struct Stencil {
  node : array<u32, 8>,
  weight : array<f32, 8>,
};

fn stencilAt(x : vec3<f32>) -> Stencil {
  let gx = x * P.hdt.y - 0.5;
  let cell = vec3<i32>(floor(gx));
  let f = gx - vec3<f32>(cell);
  let nmax = vec3<i32>(P.grid.xyz) - vec3<i32>(1);
  var s : Stencil;
  for (var i = 0u; i < 8u; i++) {
    let o = vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
    let w = select(vec3<f32>(1.0) - f, f, o == vec3<i32>(1));
    s.node[i] = nodeIndexI(clamp(cell + o, vec3<i32>(0), nmax));
    s.weight[i] = w.x * w.y * w.z;
  }
  return s;
}

/// The rate particle p exchanges at this frame; 0 = it takes no part. The same value in give_ and take_.
fn exchangeRate(p : u32, st : Stencil) -> f32 {
  if ((flags[p] & 1u) != 0u) { return 0.0; }
  if (!(P.part.w > 0.0) || isBad(pos[p].w)) { return 0.0; }
  var around = 0.0;
  for (var i = 0u; i < 8u; i++) { around += st.weight[i] * bitcast<f32>(atomicLoad(&xbeta[st.node[i]])); }
  let room = clamp(1.0 - max(pos[p].w, 0.0) / PIGMENT_LOAD, 0.0, 1.0);
  let alpha = max(ownBeta(p), around * room);
  if (isBad(alpha)) { return 0.0; }
  return clamp(alpha, 0.0, 1.0);
}

@compute @workgroup_size(128)
fn mark_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let beta = ownBeta(p);
  if (beta <= 0.0) { return; }
  let st = stencilAt(pos[p].xyz);
  // beta > 0, so its float bits order as it does
  for (var i = 0u; i < 8u; i++) { atomicMax(&xbeta[st.node[i]], bitcast<u32>(beta)); }
}

@compute @workgroup_size(128)
fn give_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let st = stencilAt(pos[p].xyz);
  let alpha = exchangeRate(p, st);
  if (alpha <= 0.0) { return; }
  let w = max(pos[p].w, 0.0);
  var z : array<f32, 7>;
  for (var c = 0u; c < 7u; c++) { z[c] = lat[7u * p + c]; }
  let q = clamp(lam[p].w, LAMINA_Q_MIN, LAMINA_Q_MAX);
  let rel = relThickness(p);
  for (var i = 0u; i < 8u; i++) {
    let a = st.weight[i] * alpha;
    let cap = encodeFixed(a * P.part.y, EX_SCALE);
    if (cap <= 0) { continue; }
    let n = st.node[i];
    atomicAdd(&xcap[n], cap);
    if (w > 0.0) {
      atomicAdd(&xload[n], encodeFixed(a * w, EX_SCALE));
      for (var c = 0u; c < 7u; c++) {
        atomicAdd(&xlat[7u * n + c], encodeFixed(a * w * z[c], EX_SCALE));
      }
      atomicAdd(&xq[2u * n], encodeFixed(a * w * q, XQ_SCALE));
      atomicAdd(&xq[2u * n + 1u], encodeFixed(a * w * rel, XQ_SCALE));
    }
  }
}

@compute @workgroup_size(128)
fn take_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let st = stencilAt(pos[p].xyz);
  let alpha = exchangeRate(p, st);
  if (alpha <= 0.0) { return; }
  let w = max(pos[p].w, 0.0);
  var z : array<f32, 7>;
  var dq : array<f32, 7>;   // change of w·z, in fixed-point counts
  for (var c = 0u; c < 7u; c++) { z[c] = lat[7u * p + c]; dq[c] = 0.0; }
  var dw = 0.0;             // change of w, in fixed-point counts
  var outW = 0.0;           // what it gave, in counts
  var inW = 0.0;            // what it takes, in counts
  var inQ = 0.0;            // ... and the lamina state that comes with it (XQ counts)
  var inR = 0.0;
  for (var i = 0u; i < 8u; i++) {
    // the very terms give_ added (the same expressions on the same inputs)
    let a = st.weight[i] * alpha;
    let cap = encodeFixed(a * P.part.y, EX_SCALE);
    if (cap <= 0) { continue; }
    let n = st.node[i];
    // this particle's share of the node's pool: its capacity term is part of the sum, so the sum is >= cap > 0
    let share = f32(cap) / f32(atomicLoad(&xcap[n]));
    let got = share * f32(atomicLoad(&xload[n]));
    dw += got;
    inW += got;
    inQ += share * f32(atomicLoad(&xq[2u * n]));
    inR += share * f32(atomicLoad(&xq[2u * n + 1u]));
    for (var c = 0u; c < 7u; c++) { dq[c] += share * f32(atomicLoad(&xlat[7u * n + c])); }
    if (w > 0.0) {
      let gave = f32(encodeFixed(a * w, EX_SCALE));
      dw -= gave;
      outW += gave;
      for (var c = 0u; c < 7u; c++) { dq[c] -= f32(encodeFixed(a * w * z[c], EX_SCALE)); }
    }
  }
  let wNew = max(w + dw / EX_SCALE, 0.0);
  if (isBad(wNew)) { return; }
  if (wNew > 1e-9) {
    for (var c = 0u; c < 7u; c++) {
      let v = (w * z[c] + dq[c] / EX_SCALE) / wNew;
      if (!isBad(v)) { lat[7u * p + c] = v; }
    }
  }
  // the lamina state of the blend: what the particle kept of its own and what it took, by load.
  // Its own pigment is now accounted for down to the thinnest it has been; what came in was
  // traded at its giver's thickness, so that is what it is accounted for down to.
  let kept = max(w - outW / EX_SCALE, 0.0);
  let gained = max(inW / EX_SCALE, 0.0);
  let total = kept + gained;
  let lnD = lnSpacing();
  let st0 = lam[p];
  let mx = mixs[p];
  let lnS = st0.w + mx.x;
  let mixedOwn = min(mx.y, lnS);
  if (total > 1e-9) {
    let q = clamp((kept * clamp(st0.w, LAMINA_Q_MIN, LAMINA_Q_MAX) + inQ / XQ_SCALE) / total, LAMINA_Q_MIN, LAMINA_Q_MAX);
    let lnSNew = lnD + (kept * relThickness(p) + inR / XQ_SCALE) / total;
    let mixedNew = lnD + (kept * clamp(mixedOwn - lnD, -LAMINA_REL_MAX, LAMINA_REL_MAX) + inR / XQ_SCALE) / total;
    if (!isBad(q) && !isBad(lnSNew) && !isBad(mixedNew)) {
      lam[p] = vec4<f32>(st0.xyz, q);
      mixs[p] = vec2<f32>(lnSNew - q, min(mixedNew, lnSNew));
    }
  } else if (!isBad(mixedOwn)) {
    mixs[p] = vec2<f32>(mx.x, mixedOwn);
  }
  pos[p].w = wNew;
}

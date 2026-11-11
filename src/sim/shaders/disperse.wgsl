// Disperse (design §5): shear-driven exchange of pigment between neighbouring particles, in two
// passes through the solver grid's nodes so that what one particle gains another loses.
//
// What is transported is the conserved pair itself: the pigment load w (pos.w) and the load-weighted
// latent w·z (seven channels). Every particle has an exchange rate alpha = clamp(k · gamma · dt, 0, 1)
// from its own shear (gamma = ||sym(C)||_F), the share of its pigment it trades this frame.
//
//   give_ (scatter)  each particle puts alpha·w and alpha·w·z into the pools of the 8 nodes round it
//                    (trilinear weights N), and alpha·m into each node's capacity: how much of the
//                    pool the particle will take back.
//   take_ (gather)   each particle removes what it gave and takes from each node's pool the share
//                    its capacity is of the node's, with the same weights.
//
// Summed over particles, the shares of a node's pool are 1, so the total of w and of each w·z is
// unchanged: only round-off moves it. Three things make that exact and not approximate:
//   * one kernel both ways (the old gather read with 27 quadratic weights what the raster had
//     scattered with 8 trilinear ones, and each particle relaxed toward the neighbourhood mean at
//     its own rate, so nothing balanced one particle's gain against another's loss);
//   * take_ recomputes the fixed-point terms give_ added, from the same inputs: a particle
//     subtracts the integers it added (not alpha·w), and its share of a node is its integer
//     capacity term over the node's integer sum; a particle whose capacity term rounds to zero at
//     a node gives that node nothing, so no pool is left without a taker;
//   * a particle that does not exchange does neither half. Kinematic particles (a held pigment
//     chunk, material in the operator's hand or log) are skipped in both passes: a held chunk used
//     to be rastered but never debited, an unlimited source for whatever flowed past it.
//
// The latent follows from the pair: z = (w·z) / w after the exchange. A clear particle (w = 0) has
// no colour of its own, so the first pigment it takes sets its latent outright; one that ends with
// no pigment keeps its old latent at zero weight.
//
// The pools are the solver-grid raster's load and latent accumulators (the raster itself only needs
// the mass there) plus a capacity accumulator, cleared with them over the frame's box (clear.wgsl).
@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cbuf : array<f32>;
@group(0) @binding(3) var<storage, read_write> lat : array<f32>;
@group(0) @binding(4) var<storage, read> flags : array<u32>;
@group(0) @binding(5) var<storage, read_write> xcap : array<atomic<i32>>;    // sum of N · alpha · m
@group(0) @binding(6) var<storage, read_write> xload : array<atomic<i32>>;   // sum of N · alpha · w
@group(0) @binding(7) var<storage, read_write> xlat : array<atomic<i32>>;    // sum of N · alpha · w · z, 7 per node

// Fixed point of the exchange terms, 2^21: a node's pool holds up to 1024 (mass · load · alpha;
// a packed node of neat masterbatch trading everything is 8 · 12), and a capacity term of a slowly
// sheared particle (alpha ~ 1e-3, weight 1/8) is still a few hundred counts.
const EX_SCALE : f32 = 2097152.0;

/// The share of its pigment particle p trades this frame; 0 = it takes no part.
fn exchangeRate(p : u32) -> f32 {
  if ((flags[p] & 1u) != 0u) { return 0.0; }
  let k = P.part.w;
  if (k <= 0.0) { return 0.0; }
  // shear rate from the APIC velocity gradient
  let b = 9u * p;
  let c00 = cbuf[b]; let c01 = cbuf[b + 1u]; let c02 = cbuf[b + 2u];
  let c10 = cbuf[b + 3u]; let c11 = cbuf[b + 4u]; let c12 = cbuf[b + 5u];
  let c20 = cbuf[b + 6u]; let c21 = cbuf[b + 7u]; let c22 = cbuf[b + 8u];
  let s01 = 0.5 * (c01 + c10);
  let s02 = 0.5 * (c02 + c20);
  let s12 = 0.5 * (c12 + c21);
  let gamma = sqrt(c00 * c00 + c11 * c11 + c22 * c22 + 2.0 * (s01 * s01 + s02 * s02 + s12 * s12));
  // P.domain.w: sim seconds advanced this frame (dt * substeps), never wall clock
  let alpha = k * gamma * P.domain.w;
  if (isBad(alpha) || isBad(pos[p].w)) { return 0.0; }
  return clamp(alpha, 0.0, 1.0);
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

@compute @workgroup_size(128)
fn give_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let alpha = exchangeRate(p);
  if (alpha <= 0.0) { return; }
  let w = max(pos[p].w, 0.0);
  var z : array<f32, 7>;
  for (var c = 0u; c < 7u; c++) { z[c] = lat[7u * p + c]; }
  let st = stencilAt(pos[p].xyz);
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
    }
  }
}

@compute @workgroup_size(128)
fn take_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let alpha = exchangeRate(p);
  if (alpha <= 0.0) { return; }
  let w = max(pos[p].w, 0.0);
  var z : array<f32, 7>;
  var dq : array<f32, 7>;   // change of w·z, in fixed-point counts
  for (var c = 0u; c < 7u; c++) { z[c] = lat[7u * p + c]; dq[c] = 0.0; }
  var dw = 0.0;             // change of w, in fixed-point counts
  let st = stencilAt(pos[p].xyz);
  for (var i = 0u; i < 8u; i++) {
    // the very terms give_ added (the same expressions on the same inputs)
    let a = st.weight[i] * alpha;
    let cap = encodeFixed(a * P.part.y, EX_SCALE);
    if (cap <= 0) { continue; }
    let n = st.node[i];
    // this particle's share of the node's pool: its capacity term is part of the sum, so the sum is >= cap > 0
    let share = f32(cap) / f32(atomicLoad(&xcap[n]));
    dw += share * f32(atomicLoad(&xload[n]));
    for (var c = 0u; c < 7u; c++) { dq[c] += share * f32(atomicLoad(&xlat[7u * n + c])); }
    if (w > 0.0) {
      dw -= f32(encodeFixed(a * w, EX_SCALE));
      for (var c = 0u; c < 7u; c++) { dq[c] -= f32(encodeFixed(a * w * z[c], EX_SCALE)); }
    }
  }
  let wNew = max(w + dw / EX_SCALE, 0.0);
  if (isBad(wNew)) { return; }
  if (wNew > 1e-9) {
    for (var c = 0u; c < 7u; c++) {
      let q = (w * z[c] + dq[c] / EX_SCALE) / wNew;
      if (!isBad(q)) { lat[7u * p + c] = q; }
    }
  }
  pos[p].w = wNew;
}

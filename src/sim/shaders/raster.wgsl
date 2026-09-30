// Pigment raster (design §3.5): scatter particle mass and mass-weighted latent
// to the node accumulators used by disperse and pack. Kinematic particles are
// included (they carry pigment and must be visible).
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

// A particle counts as covered (pigment) above this load. Dispersion relaxes every particle of a
// milled few-percent batch toward load ~1, and that must still count as covered so a homogeneous
// mix reads solid; only near-clear base (0, or the faint trace a chunk sheds into it) is uncovered.
const COVERAGE_MIN_LOAD : f32 = 0.3;

/// Scatter particle p onto the node grid of spacing 1/invh and node counts dims (the
/// accumulators bound to pmass / plat / pload / pcov are that grid's); withCov adds the
/// coverage term (the colour raster only).
fn scatter(p : u32, invh : f32, dims : vec3<i32>, withCov : bool) {
  let pMass = P.part.y;
  let x = pos[p].xyz;
  let load = max(pos[p].w, 1e-3);
  let covered = withCov && pos[p].w > COVERAGE_MIN_LOAD;
  var z : array<f32, 7>;
  for (var c = 0u; c < 7u; c++) { z[c] = lat[7u * p + c]; }

  let gx = x * invh - 0.5;                  // node-centred trilinear
  let base = vec3<i32>(floor(gx));
  let f = gx - vec3<f32>(base);
  let nmax = dims - vec3<i32>(1);
  for (var k = 0; k < 2; k++) {
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        let o = vec3<i32>(i, j, k);
        let w = select(vec3<f32>(1.0) - f, f, o == vec3<i32>(1));
        let wgt = w.x * w.y * w.z * pMass;
        let c = clamp(base + o, vec3<i32>(0), nmax);
        let n = (u32(c.z) * u32(dims.y) + u32(c.y)) * u32(dims.x) + u32(c.x);
        atomicAdd(&pmass[n], encodeFixed(wgt, MASS_SCALE));
        atomicAdd(&pload[n], encodeFixed(wgt * load, MASS_SCALE));
        if (covered) { atomicAdd(&pcov[n], encodeFixed(wgt, MASS_SCALE)); }
        for (var cc = 0u; cc < 7u; cc++) {
          atomicAdd(&plat[7u * n + cc], encodeFixed(wgt * load * z[cc], LAT_SCALE));
        }
      }
    }
  }
}

/// The solver-grid raster (read by disperse and pack).
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  scatter(p, P.hdt.y, vec3<i32>(P.grid.xyz), false);
}

/// The colour raster (design §3.5): the same scatter onto the finer grid the renderer reads
/// pigment from, so streaks thinner than a solver cell survive to the picture, plus coverage.
@compute @workgroup_size(128)
fn mainFine(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  scatter(p, P.fineH.y, vec3<i32>(P.fine.xyz), true);
}

// The material's box (design §3.6): the grid is about 2% occupied, so the grid passes cover only
// the nodes the material can touch instead of the whole domain.
//
// G2P reduces the particles' cell range every substep (atomic min / max into BOX_ACC, kinematic
// particles included). `step_` (one invocation, after G2P) turns that into the solver's box for
// the next substep: the cells' 27-node stencils plus a cell of slack for what the operator's
// script moves before the next P2G, as an origin and size the grid update offsets its nodes by
// and the workgroup counts its indirect dispatch reads. The bounds are exact for the positions
// P2G will scatter from, so nothing is assumed about how fast the material moves.
//
// `frame_` (one invocation, before the frame's raster) does the same for the raster's clear and
// pack: the cells' 8-node stencils, united with the previous frame's own box, so nodes the
// material has left are zeroed once and the accumulators and the density texture are empty
// everywhere outside the current box.
//
// The colour raster (design §3.5) has a box of its own, of fixed node counts (a budget, not the
// domain): `frame_` lays it over the material on a world-anchored lattice whose cell is the
// finest one (the particle spacing) times FINE_K / 8, the smallest FINE_K whose range fits the
// node counts. So the colour is at particle resolution whenever the material fits the budget
// (milling), and coarsens by eighths when it does not (a log standing over the nip). The lattice
// is anchored at the world origin, so texels do not swim as the box moves; a change of FINE_K
// re-rasters everything, and it has a little hysteresis (the finer cell must fit with room to
// spare before the raster goes back to it). The renderer reads where texel 0 is and the cell
// from `fineOut`, a uniform to it.
//
// When particles are placed from the CPU (seeding, reset, a pigment chunk) the host writes the
// whole grid into every box (mpm.ts resetBox): the full grid is always a valid box.
@group(0) @binding(1) var<storage, read_write> box : array<atomic<u32>>;
// indirect dispatch arguments: [0..2] the solver's grid update, [4..6] the raster's clear and pack,
// [8..10] the colour raster's clear and pack
@group(0) @binding(2) var<storage, read_write> args : array<u32>;
// for the renderer: [0] = world position of the colour raster's texel 0 (xyz) and its cell (w),
// [1] = (the pack normalisation at that cell, FINE_K, 0, 0)
@group(0) @binding(3) var<storage, read_write> fineOut : array<vec4<f32>>;

struct FineRange {
  origin : vec3<i32>,   // lattice index of texel 0
  size : vec3<i32>,     // texels in use
};

/// The colour raster's range over the world box lo..hi at cell hF: a particle at x rasters to
/// lattice nodes floor(x / hF - 0.5) and the next, so two nodes before lo and two after hi leave an
/// empty texel at each end (what a clamped lookup outside the range reads), within the domain.
fn fineRange(lo : vec3<f32>, hi : vec3<f32>, hF : f32) -> FineRange {
  var r : FineRange;
  r.origin = max(vec3<i32>(floor(lo / hF)) - vec3<i32>(2), vec3<i32>(0));
  let last = min(vec3<i32>(floor(hi / hF)) + vec3<i32>(2), vec3<i32>(ceil(P.domain.xyz / hF)));
  r.size = last - r.origin + vec3<i32>(1);
  return r;
}

/// The smallest FINE_K whose range over lo..hi fits the colour raster's node counts.
fn fineFit(lo : vec3<f32>, hi : vec3<f32>) -> i32 {
  let tex = vec3<i32>(P.fine.xyz);
  for (var k = FINE_K_MIN; k < FINE_K_MAX; k++) {
    let r = fineRange(lo, hi, P.fineH.x * f32(k) / 8.0);
    if (all(r.size <= tex)) { return k; }
  }
  return FINE_K_MAX;
}

fn loadCells(at : u32) -> vec3<i32> {
  return vec3<i32>(i32(atomicLoad(&box[at])), i32(atomicLoad(&box[at + 1u])), i32(atomicLoad(&box[at + 2u])));
}

fn storeCells(at : u32, v : vec3<i32>) {
  atomicStore(&box[at], u32(v.x));
  atomicStore(&box[at + 1u], u32(v.y));
  atomicStore(&box[at + 2u], u32(v.z));
}

/// Store the node range lo..hi (inclusive) as origin and size at `at`, and its 4x4x4 workgroup
/// counts in the indirect arguments at `arg`.
fn storeBox(at : u32, arg : u32, lo : vec3<i32>, hi : vec3<i32>) {
  let size = hi - lo + vec3<i32>(1);
  storeCells(at, lo);
  storeCells(at + 3u, size);
  let groups = (size + vec3<i32>(3)) / 4;
  args[arg] = u32(groups.x);
  args[arg + 1u] = u32(groups.y);
  args[arg + 2u] = u32(groups.z);
}

@compute @workgroup_size(1)
fn step_() {
  let top = vec3<i32>(P.grid.xyz) - vec3<i32>(1);
  let mnU = vec3<u32>(atomicLoad(&box[BOX_ACC]), atomicLoad(&box[BOX_ACC + 1u]), atomicLoad(&box[BOX_ACC + 2u]));
  let mxU = vec3<u32>(atomicLoad(&box[BOX_ACC + 3u]), atomicLoad(&box[BOX_ACC + 4u]), atomicLoad(&box[BOX_ACC + 5u]));
  var mn = vec3<i32>(0);
  var mx = top;
  // nothing reduced (no particle ran G2P): keep the whole grid
  if (all(mnU <= mxU)) {
    mn = vec3<i32>(mnU);
    mx = min(vec3<i32>(mxU), top);
  }
  storeCells(BOX_LAST, mn);
  storeCells(BOX_LAST + 3u, mx);
  // a particle in cell c scatters to nodes c-1 .. c+2
  storeBox(BOX_GRID, 0u, max(mn - vec3<i32>(1 + BOX_SLACK), vec3<i32>(0)), min(mx + vec3<i32>(2 + BOX_SLACK), top));
  for (var c = 0u; c < 3u; c++) {
    atomicStore(&box[BOX_ACC + c], 0xffffffffu);
    atomicStore(&box[BOX_ACC + 3u + c], 0u);
  }
}

@compute @workgroup_size(1)
fn frame_() {
  let top = vec3<i32>(P.grid.xyz) - vec3<i32>(1);
  let mn = loadCells(BOX_LAST);
  let mx = loadCells(BOX_LAST + 3u);
  // a particle in cell c rasters to nodes c-1 .. c+1
  let lo = max(mn - vec3<i32>(1 + BOX_SLACK), vec3<i32>(0));
  let hi = min(mx + vec3<i32>(1 + BOX_SLACK), top);
  let plo = loadCells(BOX_PREV);
  let phi = plo + loadCells(BOX_PREV + 3u) - vec3<i32>(1);
  storeBox(BOX_FRAME, 4u, min(lo, plo), min(max(hi, phi), top));
  storeCells(BOX_PREV, lo);
  storeCells(BOX_PREV + 3u, hi - lo + vec3<i32>(1));

  // the colour raster: the world box of the particles' cells, on the finest lattice that fits
  let h = P.hdt.x;
  // widened by half the longest colour streak line (raster.wgsl) so no tap lands on the box's empty edge
  let margin = vec3<f32>(0.5 * P.fineH.w * 0.5 * h);
  let wlo = max(vec3<f32>(mn) * h - margin, vec3<f32>(0.0));
  let whi = min(vec3<f32>(mx + vec3<i32>(1)) * h + margin, P.domain.xyz);
  let pad = vec3<f32>(2.0 * h);
  let kPrev = i32(atomicLoad(&box[BOX_FINE + 3u]));
  let kFit = fineFit(wlo, whi);
  let kRoomy = fineFit(wlo - pad, whi + pad);
  var k = kFit;
  if (kPrev >= kRoomy) { k = kRoomy; } else if (kPrev > kFit) { k = kPrev; }
  let hF = P.fineH.x * f32(k) / 8.0;
  let tex = vec3<i32>(P.fine.xyz);
  let r = fineRange(wlo, whi, hF);
  let size = min(r.size, tex);
  let prev = loadCells(BOX_FINE_PREV);
  storeCells(BOX_FINE, r.origin);
  atomicStore(&box[BOX_FINE + 3u], u32(k));
  let cover = min(max(size, prev), tex);
  storeCells(BOX_FINE_RANGE, cover);
  storeCells(BOX_FINE_PREV, size);
  let groups = (cover + vec3<i32>(3)) / 4;
  args[8] = u32(groups.x);
  args[9] = u32(groups.y);
  args[10] = u32(groups.z);
  let s = 8.0 / f32(k);
  fineOut[0] = vec4<f32>(vec3<f32>(r.origin) * hF, hF);
  fineOut[1] = vec4<f32>(P.fineH.z * s * s * s, f32(k), 0.0, 0.0);
}

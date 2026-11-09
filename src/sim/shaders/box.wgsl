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
// When particles are placed from the CPU (seeding, reset, a pigment chunk) the host writes the
// whole grid into every box (mpm.ts resetBox): the full grid is always a valid box.
@group(0) @binding(1) var<storage, read_write> box : array<atomic<u32>>;
// indirect dispatch arguments: [0..2] the solver's grid update, [4..6] the raster's clear and pack
@group(0) @binding(2) var<storage, read_write> args : array<u32>;

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
}

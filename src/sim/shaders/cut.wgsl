// Drawn cut: a TEAR in the sheet along a polyline (src/sim/cut.ts, GpuMpm.cutAlong).
//
// The solver keeps one velocity field per node; a tear needs the grid to see the
// two sides of the seam as two bodies. Particles in a band around the drawn line
// get a SIDE label (flag bits 2-3: 1 left of the line, 2 right of it). Each side
// scatters its mass and momentum a second time into its own channel (p2g.wgsl
// `side`), and a side's particles gather the velocity of "everything except the
// other side" (grid.wgsl `side`: (all - other) / (mass all - mass other), with
// the same boundary conditions), so no momentum and no stress crosses the seam,
// while unlabelled material (beyond the band, past the ends of the cut, the bank)
// still couples to both sides as before. The band is wide enough that no
// unlabelled particle's stencil touches both sides near the seam. Away from a cut
// nothing changes: unlabelled particles gather the ordinary field.
//
// The knife has a width: inside the kerf (half-width C.w.y, tapered to nothing at
// the ends of the cut so the tear has a tip) the material is pushed aside onto
// the lips, mirrored across the kerf edge and laid on top of the lip (lifted by
// C.w.z), so nothing is lost and the cut shows as an open gap with raised edges.
// Moved particles restart unstressed (F = I, C = 0), as the operator moves do.
//
// A tear does not last: putty knits back together when it is pressed. heal_
// clears the side label of every particle passing through the nip, and the solver
// clears the whole cut (clear_) CUT_SECONDS after it was drawn.
//
// Sheet coordinates: x along the roll, s = R * atan2(dz, dy) around the front axis
// from the crown (positive down the front face). Only material within C.w.w of
// the front roll's surface (the sheet, not the bank behind it) is cut.

const CUT_MAX_POINTS : u32 = 64u;
const SIDE_SHIFT : u32 = 2u;
const SIDE_MASK : u32 = 12u;

struct CutParams {
  // polyline point count, 0, 0, 0
  info : vec4<u32>,
  // half-width of the labelled band, kerf half-width, lip lift, deepest material cut (off the roll surface)
  w : vec4<f32>,
  // total polyline length, 0, 0, 0
  w2 : vec4<f32>,
  // x, s, arc length along the polyline at this point, 0
  pts : array<vec4<f32>, CUT_MAX_POINTS>,
};

@group(0) @binding(1) var<uniform> C : CutParams;
@group(0) @binding(2) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cbuf : array<f32>;
@group(0) @binding(4) var<storage, read_write> fbuf : array<f32>;
@group(0) @binding(5) var<storage, read_write> abuf : array<f32>;
@group(0) @binding(6) var<storage, read_write> flags : array<u32>;

fn restartUnstressed(p : u32) {
  let b = 9u * p;
  for (var c = 0u; c < 9u; c++) {
    cbuf[b + c] = 0.0;
    abuf[b + c] = 0.0;   // p2gAffine(tau(I) = 0, C = 0)
    fbuf[b + c] = select(0.0, 1.0, (c % 4u) == 0u);
  }
}

/// Mark the two sides of the cut and open the kerf.
@compute @workgroup_size(128)
fn cut_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  var f = flags[p];
  if ((f & 1u) != 0u) { return; }   // held by an operator move
  f = f & ~SIDE_MASK;              // a new cut replaces the last one
  let h = P.hdt.x;
  let R = P.front.w;
  let x = pos[p].xyz;
  let dy = x.y - P.front.x;
  let dz = x.z - P.front.y;
  let r = sqrt(dy * dy + dz * dz);
  let dr = r - R;
  if (dr < -0.5 * h || dr > C.w.w) { flags[p] = f; return; }
  let th = atan2(dz, dy);
  let q = vec2<f32>(x.x, R * th);

  // nearest segment whose perpendicular foot lies inside it (beyond the ends there is no tear)
  var best = 1e9;
  var bestD = 0.0;
  var bestN = vec2<f32>(0.0);
  var bestAlong = 0.0;
  let n = min(C.info.x, CUT_MAX_POINTS);
  for (var i = 0u; i + 1u < n; i++) {
    let a = C.pts[i].xy;
    let e = C.pts[i + 1u].xy - a;
    let len2 = dot(e, e);
    if (len2 < 1e-12) { continue; }
    let t = dot(q - a, e) / len2;
    if (t < 0.0 || t > 1.0) { continue; }
    let len = sqrt(len2);
    let nrm = vec2<f32>(-e.y, e.x) / len;   // left of the direction of travel
    let d = dot(q - a, nrm);
    if (abs(d) < best) {
      best = abs(d);
      bestD = d;
      bestN = nrm;
      bestAlong = C.pts[i].z + t * len;
    }
  }
  if (best >= C.w.x) { flags[p] = f; return; }
  let left = bestD >= 0.0;
  f = f | (select(2u, 1u, left) << SIDE_SHIFT);

  // the kerf: tapered to nothing over two kerf widths at each end of the cut
  let fromEnd = min(bestAlong, C.w2.x - bestAlong);
  let k = C.w.y * smoothstep(0.0, 2.0 * C.w.y, fromEnd);
  if (best < k) {
    // mirror across the kerf edge (|d| in [0, k) -> (k, 2k]) and lay it on top of the lip
    let sg = select(-1.0, 1.0, left);
    let shift = sg * (2.0 * k - 2.0 * best);
    let nth = th + bestN.y * shift / R;
    let nr = r + C.w.z;
    var nx = vec3<f32>(x.x + bestN.x * shift, P.front.x + nr * cos(nth), P.front.y + nr * sin(nth));
    let lo = vec3<f32>(1.5 * h);
    nx = clamp(nx, lo, P.domain.xyz - lo);
    pos[p] = vec4<f32>(nx, pos[p].w);
    restartUnstressed(p);
  }
  flags[p] = f;
}

/// The nip presses the cut closed: material passing through it is one body again.
@compute @workgroup_size(128)
fn heal_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let f = flags[p];
  if ((f & SIDE_MASK) == 0u) { return; }
  let x = pos[p].xyz;
  if (abs(x.y - P.front.x) < 0.08 && x.z > P.back.y && x.z < P.front.y) {
    flags[p] = f & ~SIDE_MASK;
  }
}

/// The cut is over (or an operator move takes the material): forget every side label.
@compute @workgroup_size(128)
fn clear_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  flags[p] = flags[p] & ~SIDE_MASK;
}

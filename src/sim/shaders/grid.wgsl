// Grid update (design §3.3): momentum -> velocity, gravity, boundaries.
// The grid clears itself: this kernel is the only reader of the mass and momentum P2G scattered,
// so it zeroes them as it reads them (only where something was scattered, so empty nodes are not
// written) and P2G finds them empty next substep; there is no separate clear pass. It writes every
// node's velocity (zero where there is no mass), and the node's mass in the fourth component for
// G2P's pile test.
@group(0) @binding(1) var<storage, read_write> gmass : array<i32>;
@group(0) @binding(2) var<storage, read_write> gmom : array<i32>;
@group(0) @binding(3) var<storage, read_write> gvel : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> pmass : array<i32>;   // last frame's render raster (density = pmass / 8)
// the material's box (box.wgsl): the dispatch covers BOX_GRID's nodes only; outside it nothing was
// scattered, the accumulators are empty and no particle reads the velocity
@group(0) @binding(5) var<storage, read> box : array<u32>;

// The peel zone of the front roll's tack band while the hand holds the cut edge (cut & fold): this
// far back from the crown along the front face, in radians.
const PEEL_ARC : f32 = 0.7;

// The nip's shear (design §3.3, "Friction ratio"): the band's nodes in the channel carry the back roll's extra
// surface speed in proportion to their place across the gap, down to NIP_EXIT below the roll axes' height in
// full and fading out by the channel's lower end.
const NIP_HALF : f32 = 0.06;   // the channel's half height (the same 0.06 as below)
const NIP_EXIT : f32 = 0.03;

// The side guides (design §1, "Side guides"): two plates across the wedge between the rolls, from the channel's
// lower end up to their top and GUIDE_REACH past the roll axes in front and behind (the plate the renderer draws).
// Material found inside a plate's body (the sheet the roll brings round from outside the plates, or the bank when
// the simulator is reset to a wider batch than the plates now allow) is moved inward at GUIDE_PUSH.
const GUIDE_REACH : f32 = 0.06;
const GUIDE_PUSH : f32 = 0.3;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) lid : vec3<u32>) {
  let gid = lid + vec3<u32>(box[BOX_GRID], box[BOX_GRID + 1u], box[BOX_GRID + 2u]);
  if (gid.x >= P.grid.x || gid.y >= P.grid.y || gid.z >= P.grid.z) { return; }
  let n = nodeIndex(gid.x, gid.y, gid.z);
  let mi = gmass[n];
  let qi = vec3<i32>(gmom[3u * n], gmom[3u * n + 1u], gmom[3u * n + 2u]);
  if (mi != 0 || any(qi != vec3<i32>(0))) {
    gmass[n] = 0;
    gmom[3u * n] = 0;
    gmom[3u * n + 1u] = 0;
    gmom[3u * n + 2u] = 0;
  }
  let m = decodeFixed(mi, MASS_SCALE);
  if (m <= 0.0) {
    gvel[n] = vec4<f32>(0.0);
    return;
  }

  let h = P.hdt.x;
  let dt = P.hdt.z;
  let g = P.hdt.w;
  let mom = vec3<f32>(decodeFixed(qi.x, MOM_SCALE), decodeFixed(qi.y, MOM_SCALE), decodeFixed(qi.z, MOM_SCALE));
  var v = mom / m;
  v.y -= dt * g;

  let p = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z)) * h;
  let R = P.front.w;

  // 1. front roller. The sheet is carried around by a sticky band (full no-slip,
  //    gap + h thick) from the nip downward and up the front face. In the wedge
  //    above the nip, where the bank rests on the roll, the roll is simply a no-slip wall: the nodes inside its
  //    surface move with it and nothing else is prescribed, so the surface
  //    layer (as far as the particles' stencils reach, about a cell) is dragged
  //    in, the bank presses material onto the roll and packs the layer the nip
  //    takes to rest density, and when the channel cannot pass what arrives the
  //    material's own pressure holds the rest back and the bank rolls instead
  //    of being force-fed. (A hard tangential constraint through the whole band
  //    with the approach blocked froze the returning sheet at half density,
  //    lacy on every preset; with the approach free it packed the whole batch
  //    onto the roll at twice rest density, then starved.)
  //    The channel itself (the gap around the nip line) is not a conveyor that
  //    accepts any density: where a channel node is packed well past rest
  //    density the band lets go of it in proportion, so the material's own
  //    pressure pushes the excess back up into the bank instead of the sheet
  //    coming out at 1.5-1.8x rest density and the wrap running short.
  //    While the operator's hand holds the cut edge (cut & fold, fold.wgsl), the band just
  //    before the crown (PEEL_ARC back from it, on the front face) is a peel zone: the roll
  //    still carries what presses on it, but lets go of what is pulled away, so the sheet the
  //    hand holds can come off the roll instead of the hand tearing out of the sheet (the band
  //    clamps every node the sheet sits on, so the hand could pull only through the outermost
  //    layer of nodes and the putty gave way there).
  //    Friction ratio: in the channel the band is not a plug at the front roll's speed. Its velocity
  //    there is the front roll's rotation plus the back roll's extra surface speed times s, the
  //    node's place across the local gap (0 on the front roll's surface, 1 on the back roll's): a
  //    Couette profile, so the material against the back roll goes through at the back roll's
  //    speed, the shear rate across the gap is (U_back - U_front) / gap and the channel conveys at
  //    the mean of the two speeds. s is not clamped at the surfaces: the nodes inside the rolls
  //    carry the line on (ghost nodes, as far as P.chunk.w gaps), so what a particle in the gap
  //    gathers, velocity and C, is the linear profile itself and not one flattened at the walls
  //    (clamped, C held 0.6 to 0.67 of the nominal shear rate; carried on, 0.9). The profile is
  //    imposed, not solved: a gap of 1.3 cells has one layer of nodes in it. With equal roll
  //    speeds the term is zero and the band is the plug it was.
  let df = rollerDist(P.front.x, P.front.y, p);
  let db = rollerDist(P.back.x, P.back.y, p);
  let inWedge = p.y > P.front.x + 0.06 && p.z < P.front.y;
  if (df < R + P.bands.x && !(inWedge && df > R)) {
    var vr = rollerVel(P.front.x, P.front.y, P.front.z, p);
    let dyN = p.y - P.front.x;
    if (abs(dyN) < NIP_HALF && p.z < P.front.y && p.z > P.back.y) {
      let s = clamp((df - R) / max(df + db - 2.0 * R, 0.25 * h), -P.chunk.w, 1.0 + P.chunk.w);
      let across = s * smoothstep(-NIP_HALF, -NIP_EXIT, dyN);
      vr += (across * P.chunk.z * R / max(df, 0.5 * R)) * vr;
    }
    let inChannel = abs(p.y - P.front.x) < 0.06 && p.z < P.front.y && df > R;
    let peel = P.fold.x > 1.5 && df > R && p.z >= P.front.y && p.y - P.front.x > df * cos(PEEL_ARC);
    if (peel) {
      let nrm = rollerNormal(P.front.x, P.front.y, p);
      if (dot(v - vr, nrm) < 0.0) { v = vr; }
    } else {
      // packing is read off the render raster (a gap-wide channel at rest density
      // rasters to ~0.65 at `low`; the grid mass here includes the roll's own
      // stencil overlap and does not tell)
      let dens = decodeFixed(pmass[n], MASS_SCALE) / 8.0;
      let over = select(0.0, saturate((dens - 0.8) / 0.4), inChannel);
      v = mix(vr, v, over);
    }
  }

  // 2. back roller. In the wedge above the nip it is the same tacky no-slip wall
  //    as the front roll (nodes inside its surface move with it), so the bank is
  //    drawn in from both sides and rolls instead of leaving a dead pocket of
  //    material resting against the back roll. Everywhere else it is a
  //    separating contact with Coulomb friction, so the sheet peels off it cleanly
  //    below the nip and follows the (stickier) front roll.
  let inWedgeB = p.y > P.back.x + 0.06 && p.z > P.back.y;
  if (inWedgeB && db <= R) {
    v = rollerVel(P.back.x, P.back.y, P.back.z, p);
  } else if (db < R + P.bands.y) {
    let nrm = rollerNormal(P.back.x, P.back.y, p);
    let vr = rollerVel(P.back.x, P.back.y, P.back.z, p);
    let vrel = v - vr;
    let vn = dot(vrel, nrm);
    if (vn < 0.0) {
      var vt = vrel - vn * nrm;
      let vtl = length(vt);
      if (vtl > 1e-9) {
        vt *= max(0.0, 1.0 - P.back.w * (-vn) / vtl);
      }
      v = vr + vt;
    }
  }

  // 3. side guides, domain walls (separating), ceiling, floor with stick-slip
  let margin = P.bands.z;
  let dom = P.domain.xyz;
  // The guides are separating walls like the domain's: a node within the wall margin of a plate's face may not
  // move outward. A plate that is sliding in is a wall that moves: those nodes go inward at least at its speed,
  // so it ploughs the bank in front of it, and the putty gives way upward and into the nip as it would for a
  // real plate; nothing is trapped behind it or set down somewhere else. With the plates at the ends of the
  // rolls this is the domain wall's own condition (below) on the same nodes, and no node is inside a plate.
  // Outside the plates' extent nothing is confined: the sheet round the front of the roll has no force along
  // the roll on it and keeps its width.
  let off = p.x - 0.5 * dom.x;
  if (abs(off) > P.agglo2.y - margin && p.y > P.front.x - NIP_HALF && p.y < P.agglo2.w
      && p.z > P.back.y - GUIDE_REACH && p.z < P.front.y + GUIDE_REACH) {
    let inward = select(1.0, -1.0, off > 0.0);
    let least = select(P.agglo2.z, GUIDE_PUSH, abs(off) > P.agglo2.y);
    if (v.x * inward < least) { v.x = least * inward; }
  }
  if (p.x < margin && v.x < 0.0) { v.x = 0.0; }
  if (p.x > dom.x - margin && v.x > 0.0) { v.x = 0.0; }
  if (p.z < margin && v.z < 0.0) { v.z = 0.0; }
  if (p.z > dom.z - margin && v.z > 0.0) { v.z = 0.0; }
  if (p.y > dom.y - margin && v.y > 0.0) { v.y = 0.0; }
  if (p.y < margin && v.y < 0.0) {
    v.y = 0.0;
    v.x *= max(0.0, 1.0 - P.bands.w);
    v.z *= max(0.0, 1.0 - P.bands.w);
  }

  gvel[n] = vec4<f32>(v, m);
}

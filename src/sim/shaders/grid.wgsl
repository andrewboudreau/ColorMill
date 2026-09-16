// Grid update (design §3.3): momentum -> velocity, gravity, boundaries.
@group(0) @binding(1) var<storage, read> gmass : array<i32>;
@group(0) @binding(2) var<storage, read> gmom : array<i32>;
@group(0) @binding(3) var<storage, read_write> gvel : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> pmass : array<i32>;   // last frame's render raster (density = pmass / 8)

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= P.grid.x || gid.y >= P.grid.y || gid.z >= P.grid.z) { return; }
  let n = nodeIndex(gid.x, gid.y, gid.z);
  let m = decodeFixed(gmass[n], MASS_SCALE);
  if (m <= 0.0) { return; }   // gvel was zeroed by clearGrid

  let h = P.hdt.x;
  let dt = P.hdt.z;
  let g = P.hdt.w;
  let mom = vec3<f32>(decodeFixed(gmom[3u * n], MOM_SCALE), decodeFixed(gmom[3u * n + 1u], MOM_SCALE), decodeFixed(gmom[3u * n + 2u], MOM_SCALE));
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
  let df = rollerDist(P.front.x, P.front.y, p);
  let inWedge = p.y > P.front.x + 0.06 && p.z < P.front.y;
  if (df < R + P.bands.x && !(inWedge && df > R)) {
    let vr = rollerVel(P.front.x, P.front.y, P.front.z, p);
    let inChannel = abs(p.y - P.front.x) < 0.06 && p.z < P.front.y && df > R;
    // packing is read off the render raster (a gap-wide channel at rest density
    // rasters to ~0.65 at `low`; the grid mass here includes the roll's own
    // stencil overlap and does not tell)
    let dens = decodeFixed(pmass[n], MASS_SCALE) / 8.0;
    let over = select(0.0, saturate((dens - 0.8) / 0.4), inChannel);
    v = mix(vr, v, over);
  }

  // 2. back roller. In the wedge above the nip it is the same tacky no-slip wall
  //    as the front roll (nodes inside its surface move with it), so the bank is
  //    drawn in from both sides and rolls instead of leaving a dead pocket of
  //    material resting against the back roll. Everywhere else it is a
  //    separating contact with Coulomb friction, so the sheet peels off it cleanly
  //    below the nip and follows the (stickier) front roll.
  let db = rollerDist(P.back.x, P.back.y, p);
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

  // 3. domain walls (separating), ceiling, floor with stick-slip
  let margin = P.bands.z;
  let dom = P.domain.xyz;
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

  gvel[n] = vec4<f32>(v, 0.0);
}

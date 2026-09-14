// Grid update (design §3.3): momentum -> velocity, gravity, boundaries.
@group(0) @binding(1) var<storage, read> gmass : array<i32>;
@group(0) @binding(2) var<storage, read> gmom : array<i32>;
@group(0) @binding(3) var<storage, read_write> gvel : array<vec4<f32>>;

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

  // 1. front roller: sticky / tack band (full no-slip, sheet is carried around)
  let df = rollerDist(P.front.x, P.front.y, p);
  if (df < R + P.bands.x) {
    v = rollerVel(P.front.x, P.front.y, P.front.z, p);
  }

  // 2. back roller: separating with Coulomb friction
  let db = rollerDist(P.back.x, P.back.y, p);
  if (db < R + P.bands.y) {
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

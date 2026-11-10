// Colour streaks (design §5): every particle carries a unit direction d (xyz) and a length L in
// particle spacings (w). The colour raster draws a pigment-carrying particle as a line of that
// length along d. Only the picture reads it; the physics never does. Once per frame, with the sim
// time the frame advanced.
//
// The direction is a material line element turned by the velocity gradient, ds/dt = C s (second
// order), so it follows the flow. The length is NOT the line element's length: it grows with the
// shear strain the particle has been through (a smear in shear lengthens in proportion to the
// strain) and never shrinks. Two things rule the line element's own length out. A real sheet folds
// like a jelly roll and keeps its laminae, but the sim's under-resolved flow un-stretches material
// when the sheet re-enters the bank, so a reversible length forgets every smear (measured: a median
// of 1.3 spacings on the sheet, and falling). And the stretch along the line is small here: material
// beside a roll enters the nip already at the roll's speed, so the nip is shear, not extension, and
// it is several times wider for its rolls than a real one (a chunk is squeezed about 5x where a real
// nip gives 20x or more), hence STREAK_GAIN.
@group(0) @binding(1) var<storage, read> cbuf : array<f32>;
@group(0) @binding(2) var<storage, read> flags : array<u32>;
@group(0) @binding(3) var<storage, read_write> streak : array<vec4<f32>>;

const STREAK_SEED : f32 = 0.5;      // a fresh particle's length
const STREAK_MAX : f32 = 16.0;      // stored cap; the raster applies the user's cap
const STREAK_GAIN : f32 = 6.0;      // spacings of length per unit of shear strain
const STREAK_RATE_MIN : f32 = 2.0;  // 1/s: slower shear is the solver's elastic chatter and the bank at rest, not flow (the nip runs at 10 to 25)

fn hashU32(x : u32) -> u32 {
  var v = x;
  v = v ^ (v >> 16u);
  v = v * 0x7feb352du;
  v = v ^ (v >> 15u);
  v = v * 0x846ca68bu;
  v = v ^ (v >> 16u);
  return v;
}

/// A unit vector from the particle's index (any direction will do; never zero length).
fn seedDir(p : u32) -> vec3<f32> {
  let a = hashU32(p * 2u + 1u);
  let b = hashU32(a ^ 0x9e3779b9u);
  let z = 2.0 * (f32(a >> 8u) / 16777216.0) - 1.0;
  let phi = 2.0 * PI * (f32(b >> 8u) / 16777216.0);
  let r = sqrt(max(1.0 - z * z, 0.0));
  return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) != 0u) { return; }
  let b = 9u * p;
  // WGSL matrices are built from columns; C[r][c] = cbuf[9p + 3r + c], so C * s is the math product
  let C = mat3x3<f32>(
    vec3<f32>(cbuf[b], cbuf[b + 3u], cbuf[b + 6u]),
    vec3<f32>(cbuf[b + 1u], cbuf[b + 4u], cbuf[b + 7u]),
    vec3<f32>(cbuf[b + 2u], cbuf[b + 5u], cbuf[b + 8u]));
  let st = streak[p];
  var d = st.xyz;
  var L = st.w;
  let badL = isBad(L) || L <= 0.0;
  if (dot(d, d) < 0.25 || isBad3(d) || badL) {
    d = seedDir(p);
    if (badL) { L = STREAK_SEED; } else { L = max(L, STREAK_SEED); }
  }
  let dt = P.domain.w;   // sim seconds advanced this frame
  let Cd = C * d;
  // shear rate: the norm of the symmetric part of C (as the dispersion kernel takes it)
  let D = 0.5 * (C + transpose(C));
  let gamma = sqrt(dot(D[0], D[0]) + dot(D[1], D[1]) + dot(D[2], D[2]));
  let grow = STREAK_GAIN * clamp(gamma - STREAK_RATE_MIN, 0.0, 100.0) * dt;
  if (!isBad(grow)) { L = min(L + grow, STREAK_MAX); }
  let dn = d + dt * Cd + 0.5 * dt * dt * (C * Cd);
  let n2 = dot(dn, dn);
  if (!isBad3(dn) && n2 > 1e-12) { d = dn * inverseSqrt(n2); }
  if (isBad3(d) || isBad(L)) { d = seedDir(p); L = STREAK_SEED; }
  streak[p] = vec4<f32>(d, L);
}

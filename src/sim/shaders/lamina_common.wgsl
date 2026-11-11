// Lamellar mixing model (design §5), shared by the kernels that read or write the lamina state.
// Concatenated after common.wgsl for those kernels by src/sim/mpm.ts; src/sim/lamina.ts is the CPU mirror.
//
// Per particle:
//   lam : vec4   n xyz, the unit normal of the pigment laminae in the particle, and
//                q = ln(s / s0), the log of their thickness s over the thickness s0 they arrived with
//   mixs : vec4  ln s0 (world units); `mixed`: ln of the thickness the particle's pigment is already
//                blended down to (the exchange's reference, disperse.wgsl); the particle's UNDISPERSED
//                load, the part of its pigment load (pos.w) still in agglomerates (0 .. pos.w; the
//                undispersed fraction is this over the load; lamina.wgsl has the kinetics); spare
// A clear particle carries n and q too (they follow the flow whatever the load), but they mean
// nothing until it takes pigment: then it takes the pigment's q and ln s0 with it.
const LAMINA_Q_MIN : f32 = -12.0;   // q is kept in this range: beyond 1e-5 of the blob's size nothing reads it
const LAMINA_Q_MAX : f32 = 2.0;
// ln s over ln of the particle spacing is carried through the exchange's fixed-point pools within this range
const LAMINA_REL_MAX : f32 = 8.0;

/// ln of the particle spacing (two particles per cell per axis): the size of the parcel a particle stands for.
fn lnSpacing() -> f32 { return log(0.5 * P.hdt.x); }

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

/// The lamina state of a particle that takes a share of fresh pigment: `kept` is the load it keeps,
/// `gained` the load it gains, laid down as laminae exp(lnS0) thick with normal nNew. Blended by load, in
/// the log, like everything else about the pigment; the normal is the larger share's.
struct LaminaBlend {
  lam : vec4<f32>,
  mixs : vec2<f32>,
};
/// The undispersed load of a particle that keeps the share `keep` of its own pigment and gains `gained` of
/// fresh masterbatch: what it kept stays as dispersed as it was, and fresh pigment is all agglomerates.
fn undispersedTakeFresh(undispersed : f32, keep : f32, gained : f32, wNew : f32) -> f32 {
  let u = keep * max(undispersed, 0.0) + gained;
  if (isBad(u)) { return 0.0; }
  return clamp(u, 0.0, max(wNew, 0.0));
}
fn laminaTakeFresh(lam : vec4<f32>, mixs : vec2<f32>, kept : f32, gained : f32, nNew : vec3<f32>, lnS0 : f32) -> LaminaBlend {
  var out : LaminaBlend;
  out.lam = lam;
  out.mixs = mixs;
  let total = kept + gained;
  if (!(gained > 0.0) || !(total > 1e-9)) { return out; }
  let f = gained / total;
  let lnS = mix(lam.w + mixs.x, lnS0, f);        // ln s of the blend
  let q = clamp(mix(lam.w, 0.0, f), LAMINA_Q_MIN, LAMINA_Q_MAX);   // fresh pigment is unthinned
  var n = lam.xyz;
  if (f >= 0.5 && dot(nNew, nNew) > 0.25 && !isBad3(nNew)) { n = normalize(nNew); }
  out.lam = vec4<f32>(n, q);
  // fresh pigment is not blended at all: its reference is its own thickness
  out.mixs = vec2<f32>(lnS - q, mix(min(mixs.y, lam.w + mixs.x), lnS0, f));
  if (isBad(out.mixs.x) || isBad(out.mixs.y) || isBad(q)) { out.lam = vec4<f32>(n, 0.0); out.mixs = vec2<f32>(lnS0, lnS0); }
  return out;
}

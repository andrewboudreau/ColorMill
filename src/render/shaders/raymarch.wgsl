// ColorMill v2 — fullscreen ray-march of the mill (design §8).
// mixbox.wgsl (latentToRgb) is prepended by the renderer.
//
// One triangle covers the screen. Per pixel: build a camera ray, intersect the
// analytic scene (two capped roller cylinders, the bench and the shop floor), then march the
// density volume between the domain-box entry and the nearest analytic hit.
// The surface is not found on the solver's density as packed (volA) but on a smoothed copy of
// it (csSmooth, once per frame): splatted particles are noisy at the cell scale, and a glossy
// skin shows every degree of that noise in its normal.
// Blocks of the coarse max-density mip (built by csMip each frame) that hold no
// density above iso are skipped in one step. The first sample with density >= iso
// is refined by bisection; the normal is the gradient of the smoothed density; colour
// is the Mixbox latent decoded at the hit, with a thin-sheet transmittance term.
// Light: three soft boxes in an analytic studio environment that every glossy surface
// mirrors (dielectric putty skin, anisotropic steel rolls); the clear base refracts the
// analytic scene behind it; a hue-preserving tone curve and a dither finish the pixel.

struct Uniforms {
  camOrigin: vec4f,   // xyz = eye, w = tan(fovY/2)
  camRight: vec4f,    // xyz = screen right, w = aspect
  camUp: vec4f,       // xyz = screen up, w = time (s)
  camForward: vec4f,  // xyz = view dir, w = h (cell size)
  dims: vec4f,        // node counts (nx, ny, nz), w = iso threshold
  domain: vec4f,      // domain size xyz, w = roller length
  rollerBack: vec4f,  // axisY, axisZ, radius, angle (rad)
  rollerFront: vec4f,
  keyDir: vec4f,      // xyz = unit direction toward the light, w = intensity
  keyColor: vec4f,
  fillDir: vec4f,
  fillColor: vec4f,
  rimDir: vec4f,
  rimColor: vec4f,
  misc: vec4f,        // resolution.xy, exposure, flags (bit0 = volumes bound, bit1 = target is sRGB-encoded already, bit2 = surface colour mode)
  mip: vec4f,         // coarse max-density mip dims (cx, cy, cz), w = block edge in texels
  guides: vec4f,      // end-guide plates: bank top y, bank half depth (z), plate thickness, enabled
  fine: vec4f,        // colour raster (design §3.5): node counts of its textures (nxF, nyF, nzF), w = its finest cell
  quality: vec4f,     // x = the rolls mirror the putty (a short density march per roll pixel), y = debug view (1 = normal, 2 = normal confidence), z = reach of the normal's filter (texels), w spare
};

// where the colour raster's box is this frame (design §3.6; written on the GPU by the solver)
struct FineBox {
  origin: vec4f,      // xyz = world position of texel 0, w = the cell size hF
  info: vec4f,        // pack normalisation, cell in eighths of the finest, 0, 0
};

@group(0) @binding(0) var<uniform> U: Uniforms;
// The density volume. For the fragment stage and csMip this is the SMOOTHED volume (csSmoothWide's
// output: x = the density the surface is found on, y = a wider filter of it, with the rolls
// filled in, for the normal); for csSmooth it is the solver's density as packed, for csSmoothWide
// csSmooth's output.
@group(0) @binding(1) var volA: texture_3d<f32>;
@group(0) @binding(3) var volSampler: sampler;
@group(0) @binding(11) var smoothOut: texture_storage_3d<rgba16float, write>;
// coarse mip: max density over each block of mip.w^3 texels (plus a one-texel halo, so a
// trilinear sample anywhere inside the block is bounded by it); rebuilt by csMip every frame
@group(0) @binding(4) var<storage, read> mipMax: array<f32>;
@group(0) @binding(5) var<storage, read_write> mipOut: array<f32>;
// the colour raster (design §3.5), on its own finer grid: pigment load, latent and mass, so a
// streak thinner than a solver cell still reaches the picture; the surface itself is still
// found on volA's density
// The latent is stored premultiplied by the load (pack.wgsl), so the sampler's trilinear filter mixes
// pigment amounts; divide by the sum of the four concentration channels after sampling.
@group(0) @binding(7) var finA: texture_3d<f32>;   // (load, load·lat0, load·lat1, load·lat2), normalised like finC's mass
@group(0) @binding(8) var finB: texture_3d<f32>;   // (load·lat3, load·lat4, load·lat5, load·lat6)
@group(0) @binding(10) var<uniform> FB: FineBox;
@group(0) @binding(9) var finC: texture_3d<f32>;   // (mass, coverage, undispersed load); finA.x / finC.x = load per unit mass,
                                                   // finC.y / finC.x = share of the mass carrying pigment,
                                                   // finC.z / finA.x = share of the pigment still in agglomerates

// Pigment is an opaque colourant in a clear medium: the share of light a sample captures
// rises with its pigment load per unit mass (0 = clear base, PIGMENT_LOAD = pure masterbatch);
// a few percent of masterbatch already colours the sheet solidly, as it does on a real mill.
const PIGMENT_OPACITY: f32 = 6.0;
// Dispersive mixing (design §5, Agglomerates): pigment still in agglomerates tints at this share of its
// strength (the colour develops as the nip breaks them down; mirrors AGGLOMERATE_TINT in agglomerate.ts) ...
const AGGLO_WEAK: f32 = 0.12;
// ... and shows as specks instead: cells of AGGLO_SPECK_CELL colour-raster cells drawn as solid masstone,
// with a chance of AGGLO_SPECK_CHANCE times the undispersed share of the pigment (less where there is
// only a trace of it: AGGLO_SPECK_LOAD is the undispersed load per unit mass the chance reaches 63% of that at)
const AGGLO_SPECK_CELL: f32 = 0.5;
const AGGLO_SPECK_CHANCE: f32 = 0.3;
const AGGLO_SPECK_LOAD: f32 = 0.25;
// a speck's colour is the pigment's masstone: darker and more saturated than its tint (albedo ^ this)
const AGGLO_SPECK_TONE: f32 = 1.5;
// Least premultiplied latent sum (the filtered load of a colour texel) that is decoded: below it the
// half-float channels are a few subnormal steps and the ratio is noise.
const LATENT_MIN: f32 = 1e-5;
// Surface colour mode (design §8): the colour of the top-most material only, sampled this many
// colour-raster cells inside the hit (just at the outermost particles), with four sideways
// samples half a cell away in a cross so a single texel does not speckle.
const SURFACE_DEPTH: f32 = 0.6;
const SURFACE_SPREAD: f32 = 0.5;
// Surface mode draws pigment by coverage (the share of the cell's mass carried by pigment
// particles, design §3.5), not by the cell's mean load: a cell one-eighth masterbatch is the edge
// of a lamina, and averaging it into a translucent tint turned the marbling a real mill makes into
// a wash. Coverage below COV_LO draws clear, above COV_HI at the covered material's full strength,
// with a tight ramp between so lamina edges stay sharp; a per-fine-cell dither of COV_DITHER on
// the coverage breaks that ramp into cell-sized flecks, which reads as marbling rather than a
// smooth gradient. (This also replaces the old load floor: the faint trace a chunk sheds into the
// base is below the raster's coverage threshold, so it is uncovered and draws clear.)
const COV_LO: f32 = 0.2;
const COV_HI: f32 = 0.5;
const COV_DITHER: f32 = 0.08;

// Look of the materials (everything that happens to light once albedo, pigment opacity and the
// surface are known).
// body colour of the unpigmented base where it scatters: a faintly cool milk
const CLEAR_BODY: vec3f = vec3f(0.60, 0.65, 0.67);
// the clear base turns hazy over this many solver cells of thickness (1/e of the light passes)
const CLEAR_HAZE_CELLS: f32 = 3.2;
// roughness of the putty's skin (a satin sheen: at 0.11 it read as wet and too shiny) and of the rolls around their circumference
const PUTTY_ROUGH: f32 = 0.30;
// density gradient at the surface, per cell (rest density = 1): below WEAK the normal is noise and the
// skin is drawn rough, above FIRM it is a face and gets the full gloss
const NORMAL_WIDE: f32 = 3.0;   // weight of the wide field's gradient against the surface density's
const NORMAL_WEAK: f32 = 0.03;
const NORMAL_FIRM: f32 = 0.10;
const ROLL_ROUGH: f32 = 0.32;

// pixel being shaded, for screen-space dithers
var<private> gPix: vec2f;

const PI: f32 = 3.14159265358979;
const INF: f32 = 1e30;
const MAX_STEPS: i32 = 420;

// ---------------------------------------------------------------- surface density (compute)

// The density the surface is found on: the packed density under a [1 2 1]^3 / 64 binomial filter
// (eight trilinear taps on the texel's corners, each the mean of the eight texels round it).
// Why: the solver has eight particles a cell and the raster is a trilinear splat, so a full cell
// reads 0.85 +- 0.15 and the iso-surface and its gradient carry that as cell-sized lumps.
// The filter also spreads a sheet 1.3 cells thick over one more cell and lowers its peak by a
// third, which is why the threshold (Renderer.iso) is lower than it would be on the packed
// density. Pigment is not touched: colour comes from the colour raster.
@compute @workgroup_size(4, 4, 4)
fn csSmooth(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec3u(U.dims.xyz);
  if (any(gid >= dims)) { return; }
  let c = vec3f(gid) + vec3f(0.5);
  var s = 0.0;
  for (var i = 0; i < 8; i++) {
    let o = vec3f(select(-0.5, 0.5, (i & 1) != 0), select(-0.5, 0.5, (i & 2) != 0), select(-0.5, 0.5, (i & 4) != 0));
    s += textureSampleLevel(volA, volSampler, (c + o) / vec3f(dims), 0.0).x;
  }
  // y: the same with the rolls counted as material (1 inside, a ramp one cell wide across the
  // face), which csSmoothWide filters for the normal
  let q = vec3f(gid) * U.camForward.w;
  let sd = min(length(q.yz - U.rollerBack.xy) - U.rollerBack.z, length(q.yz - U.rollerFront.xy) - U.rollerFront.z);
  let fill = saturate(0.5 - sd / U.camForward.w) * step(q.x, U.domain.w + 0.5 * U.camForward.w);
  textureStore(smoothOut, vec3i(gid), vec4f(s / 8.0, max(s / 8.0, fill), 0.0, 0.0));
}

// Second pass, from csSmooth's output to the volume the picture reads: x = the surface density
// (csSmooth's, copied), y = a wider filter for the normal: csSmooth's y (the density with the
// rolls filled in) through eight taps U.quality.z texels out along the diagonals (at 1 that is
// [1 0 1]^3 / 8, and with the first pass [1 2 2 2 1]^3 / 512).
// Why a second field: what is left in the surface density after one pass is ripple a few cells
// long, which moves the surface by a tenth of a cell (nothing to see) but tilts it by degrees
// (a soft box's mirror image with a torn edge). The normal can take a wider filter than the
// surface, because it only has to say which way a face points, not where it is.
// Why the rolls are filled in: the sheet is a shell 1.3 cells thick, and a filter wider than that
// sees both of its faces and leaves no gradient at either (measured: a tenth of the gradient,
// normals pointing anywhere). With the roll under it counted as material the sheet is the face of
// a solid, which is also what it looks like: a film lying on a roll is parallel to the roll. Bare
// roll is never shaded with this normal (the march finds putty on x only); where the putty thins
// out to nothing the normal turns into the roll's, a rounded edge.
@compute @workgroup_size(4, 4, 4)
fn csSmoothWide(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec3u(U.dims.xyz);
  if (any(gid >= dims)) { return; }
  let c = vec3f(gid) + vec3f(0.5);
  let r = U.quality.z;
  var s = 0.0;
  for (var i = 0; i < 8; i++) {
    let o = vec3f(select(-r, r, (i & 1) != 0), select(-r, r, (i & 2) != 0), select(-r, r, (i & 4) != 0));
    s += textureSampleLevel(volA, volSampler, (c + o) / vec3f(dims), 0.0).y;
  }
  textureStore(smoothOut, vec3i(gid), vec4f(textureLoad(volA, vec3i(gid), 0).x, s / 8.0, 0.0, 0.0));
}

// ---------------------------------------------------------------- density mip (compute)

@compute @workgroup_size(4, 4, 4)
fn csMip(@builtin(global_invocation_id) gid: vec3u) {
  let cdims = vec3u(U.mip.xyz);
  if (any(gid >= cdims)) { return; }
  let dims = vec3i(U.dims.xyz);
  let B = i32(U.mip.w);
  let lo = max(vec3i(gid) * B - vec3i(1), vec3i(0));
  let hi = min(vec3i(gid) * B + vec3i(B), dims - vec3i(1));
  var m = 0.0;
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        m = max(m, textureLoad(volA, vec3i(x, y, z), 0).x);
      }
    }
  }
  mipOut[(gid.z * cdims.y + gid.y) * cdims.x + gid.x] = m;
}

// ---------------------------------------------------------------- vertex

struct VsOut {
  @builtin(position) pos: vec4f,
};

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VsOut {
  // fullscreen triangle
  var p = vec2f(-1.0, -1.0);
  if (vi == 1u) { p = vec2f(3.0, -1.0); }
  if (vi == 2u) { p = vec2f(-1.0, 3.0); }
  var o: VsOut;
  o.pos = vec4f(p, 0.0, 1.0);
  return o;
}

// ---------------------------------------------------------------- utilities

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn srgbToLinear(c: vec3f) -> vec3f {
  let lo = c / 12.92;
  let hi = pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
  return select(hi, lo, c <= vec3f(0.04045));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - vec3f(0.055);
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yzx + vec3f(33.33));
  return fract((q.x + q.y) * q.z);
}

fn hash31(p: vec3f) -> f32 {
  var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yxz + vec3f(33.33));
  return fract((q.x + q.y) * q.z);
}

fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}


// interleaved gradient noise (Jimenez): a screen-space dither with a blue-ish spectrum
fn ign(px: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715))));
}

// ---------------------------------------------------------------- studio environment
//
// The mill stands on a workshop bench in a dark studio lit by three soft boxes, placed along the
// light rig's key / fill / rim directions so the diffuse light and the reflections agree. The
// environment is analytic (no texture): glossy surfaces reflect the boxes as rectangles whose
// edges soften with roughness, which is where the wet look of the putty and the metal of the
// rolls come from.

// centre of the scene the soft boxes aim at
const SCENE_C: vec3f = vec3f(0.75, 0.55, 0.75);
// bench top (a slab whose upper face is the tray plane y = 0) and the shop floor under it
const BENCH_LO: vec3f = vec3f(-1.0, -0.07, -0.85);
const BENCH_HI: vec3f = vec3f(2.5, 0.0, 2.7);
const SHOP_Y: f32 = -0.95;
// radiance of the boxes per unit of the light's intensity (irradiance / solid angle, with some
// artistic gain: the boxes read as lit panels in the reflections)
const KEY_L: f32 = 9.0;
const FILL_L: f32 = 3.2;
const RIM_L: f32 = 4.0;

// what a camera ray sees when it leaves the scene: charcoal walls with a soft pool of light low
// on the cyclorama, darker toward the ceiling and the floor
fn backdrop(d: vec3f) -> vec3f {
  let wall = vec3f(0.040, 0.043, 0.050);
  let g = (d.y - 0.04) / 0.30;
  let glow = vec3f(0.15, 0.155, 0.17) * exp(-g * g);
  let c = wall + glow;
  return mix(c, vec3f(0.028, 0.027, 0.027), smoothstep(0.0, -0.22, d.y));
}

// Coverage of the reflection ray (p, r) by a soft box: a rectangle of half-size `half` centred
// `dist` from the scene centre along `dir`, facing the scene. `a` (GGX alpha) blurs its edges
// in proportion to the distance travelled, as a rough surface would.
fn softBox(p: vec3f, r: vec3f, dir: vec3f, dist: f32, half: vec2f, a: f32) -> f32 {
  let dn = dot(r, dir);
  if (dn < 1e-3) { return 0.0; }
  let c = SCENE_C + dir * dist;
  let t = dot(c - p, dir) / dn;
  if (t <= 0.0) { return 0.0; }
  let q = p + r * t - c;
  let ax = normalize(cross(vec3f(0.0, 1.0, 0.0), dir));
  let ay = cross(dir, ax);
  let u = vec2f(dot(q, ax), dot(q, ay));
  let w = (0.008 + 1.1 * a) * t;
  let e = half - abs(u);
  let m = smoothstep(-w, w, e.x) * smoothstep(-w, w, e.y);
  // a soft box is hotter in the middle than at its edges
  let hot = 1.0 - 0.3 * saturate(dot(u / half, u / half) * 0.5);
  return m * hot;
}

// Radiance arriving at p from direction r, pre-blurred for a surface of roughness `rough`.
fn envRadiance(p: vec3f, r: vec3f, rough: f32) -> vec3f {
  let a = rough * rough;
  var c = backdrop(r);
  // light bounced off the ceiling: nothing reflects pure black
  c += vec3f(0.085, 0.09, 0.10) * smoothstep(0.15, 1.0, r.y);
  // the lit bench top below
  c = mix(c, vec3f(0.13, 0.115, 0.10), smoothstep(0.02 + a, -0.12 - a, r.y));
  c += U.keyColor.rgb * (U.keyDir.w * KEY_L * softBox(p, r, U.keyDir.xyz, 3.0, vec2f(0.80, 0.52), a));
  c += U.fillColor.rgb * (U.fillDir.w * FILL_L * softBox(p, r, U.fillDir.xyz, 2.8, vec2f(0.42, 1.25), a));
  c += U.rimColor.rgb * (U.rimDir.w * RIM_L * softBox(p, r, U.rimDir.xyz, 3.0, vec2f(1.5, 0.32), a));
  return c;
}

// Diffuse light from the studio other than the three boxes (ceiling above, bench bounce below).
fn ambient(n: vec3f) -> vec3f {
  return mix(vec3f(0.075, 0.068, 0.062), vec3f(0.20, 0.21, 0.235), n.y * 0.5 + 0.5);
}

// Schlick Fresnel, with the grazing term damped on rough surfaces (Lagarde)
fn fresnel(cosTheta: f32, f0: vec3f, rough: f32) -> vec3f {
  let m = pow(1.0 - saturate(cosTheta), 5.0);
  return f0 + (max(vec3f(1.0 - rough), f0) - f0) * m;
}

// How much of the reflected environment an ambient occlusion `ao` leaves visible (Lagarde)
fn specOcclusion(nv: f32, ao: f32, rough: f32) -> f32 {
  return saturate(pow(nv + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao);
}

// ---------------------------------------------------------------- analytic geometry

struct Hit {
  t: f32,
  n: vec3f,
  kind: i32, // 0 = miss, 1 = cylinder body, 2 = end cap
};

fn missHit() -> Hit {
  var h: Hit;
  h.t = INF;
  h.n = vec3f(0.0, 1.0, 0.0);
  h.kind = 0;
  return h;
}

// Capped cylinder along +x from x = 0 to x = length, axis at (y, z) = (pose.x, pose.y), radius pose.z.
fn hitRoller(ro: vec3f, rd: vec3f, pose: vec4f) -> Hit {
  var best = missHit();
  let ay = pose.x;
  let az = pose.y;
  let R = pose.z;
  let L = U.domain.w;
  let oyz = vec2f(ro.y - ay, ro.z - az);
  let dyz = vec2f(rd.y, rd.z);
  let a = dot(dyz, dyz);
  let b = dot(oyz, dyz);
  let c = dot(oyz, oyz) - R * R;
  if (a > 1e-8) {
    let disc = b * b - a * c;
    if (disc > 0.0) {
      let sq = sqrt(disc);
      let t0 = (-b - sq) / a;
      if (t0 > 0.0) {
        let x = ro.x + rd.x * t0;
        if (x >= 0.0 && x <= L) {
          best.t = t0;
          let p = ro + rd * t0;
          best.n = normalize(vec3f(0.0, p.y - ay, p.z - az));
          best.kind = 1;
        }
      }
    }
  }
  // end caps
  if (abs(rd.x) > 1e-6) {
    for (var i = 0; i < 2; i++) {
      let xc = select(0.0, L, i == 1);
      let t = (xc - ro.x) / rd.x;
      if (t > 0.0 && t < best.t) {
        let p = ro + rd * t;
        let r2 = (p.y - ay) * (p.y - ay) + (p.z - az) * (p.z - az);
        if (r2 <= R * R) {
          best.t = t;
          best.n = vec3f(select(-1.0, 1.0, i == 1), 0.0, 0.0);
          best.kind = 2;
        }
      }
    }
  }
  return best;
}

// slab test against the domain box [0, domain]; returns (tNear, tFar), tNear > tFar on miss
fn hitBox(ro: vec3f, rd: vec3f) -> vec2f {
  let inv = 1.0 / rd;
  let t0 = (vec3f(0.0) - ro) * inv;
  let t1 = (U.domain.xyz - ro) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  let tf = min(min(tmax.x, tmax.y), tmax.z);
  return vec2f(tn, tf);
}

// Soft occlusion of a shadow ray (from p toward l) by a roller: penumbra from the
// distance between the ray and the axis line versus the radius.
fn rollerShadow(p: vec3f, l: vec3f, pose: vec4f) -> f32 {
  let ay = pose.x;
  let az = pose.y;
  let R = pose.z;
  let oyz = vec2f(p.y - ay, p.z - az);
  let dyz = vec2f(l.y, l.z);
  let a = dot(dyz, dyz);
  if (a < 1e-6) { return 1.0; }
  // closest approach along the ray
  let tc = -dot(oyz, dyz) / a;
  if (tc <= 0.0) { return 1.0; }
  let q = oyz + dyz * tc;
  let d = length(q);
  let x = p.x + l.x * tc;
  let inX = smoothstep(-0.08, 0.02, x) * (1.0 - smoothstep(U.domain.w - 0.02, U.domain.w + 0.08, x));
  let w = 0.02 + 0.12 * tc; // penumbra widens with distance from the occluder
  return 1.0 - inX * (1.0 - smoothstep(R - w, R + w, d));
}

// ---------------------------------------------------------------- volume

fn volUvw(p: vec3f) -> vec3f {
  return (p / U.camForward.w + vec3f(0.5)) / U.dims.xyz;
}

// Texture coordinate of world point p on the colour raster (texel i sits at world origin + i*hF).
// Outside the box the lookup clamps to an edge texel, which the raster leaves empty.
fn fineUvw(p: vec3f) -> vec3f {
  return ((p - FB.origin.xyz) / FB.origin.w + vec3f(0.5)) / U.fine.xyz;
}

fn density(p: vec3f) -> f32 {
  return textureSampleLevel(volA, volSampler, volUvw(p), 0.0).x;
}

// x = the surface density, y = the wide field the normal is mostly taken from (csSmoothWide)
fn densityBoth(p: vec3f) -> vec2f {
  return textureSampleLevel(volA, volSampler, volUvw(p), 0.0).xy;
}

// Surface normal, by central differences one cell apart: the gradient of the wide field
// (csSmoothWide), NORMAL_WIDE times over, plus that of the surface density itself. The wide
// one is the smooth one and carries the normal wherever it has a gradient; on a film with air on
// both sides (the flap of a cut, a strand) it has next to none, and the sum falls back to the
// surface density's own. The stencil has to be narrower than the thinnest thing drawn: this used
// to be a gradient over +-1.5 cells along every axis, to average the noise away, which on the
// sheet (1.3 cells) reached through into empty space and measured its lumps instead of its face.
struct SurfaceNormal {
  n: vec3f,
  // 0 = a firm gradient (a face), 1 = hardly any (a crumb, a film about to tear): the normal
  // there is whatever the noise says
  rough: f32,
};

fn densityNormal(p: vec3f) -> SurfaceNormal {
  let h = U.camForward.w;
  let dx = densityBoth(p + vec3f(h, 0.0, 0.0)) - densityBoth(p - vec3f(h, 0.0, 0.0));
  let dy = densityBoth(p + vec3f(0.0, h, 0.0)) - densityBoth(p - vec3f(0.0, h, 0.0));
  let dz = densityBoth(p + vec3f(0.0, 0.0, h)) - densityBoth(p - vec3f(0.0, 0.0, h));
  // per cell, as a mean of the two fields' gradients
  let grad = (vec3f(dx.x, dy.x, dz.x) + NORMAL_WIDE * vec3f(dx.y, dy.y, dz.y)) * (0.5 / (1.0 + NORMAL_WIDE));
  let len = length(grad);
  var out: SurfaceNormal;
  out.n = vec3f(0.0, 1.0, 0.0);
  out.rough = 1.0;
  if (len < 1e-6) { return out; }
  out.n = -grad / len;
  out.rough = 1.0 - smoothstep(NORMAL_WEAK, NORMAL_FIRM, len);
  return out;
}

// Coarse block containing world point p (texel i sits at world i*h); clamped to the mip.
fn mipCell(p: vec3f) -> vec3i {
  let tex = p / U.camForward.w + vec3f(0.5);
  let c = vec3i(floor(tex / U.mip.w));
  return clamp(c, vec3i(0), vec3i(U.mip.xyz) - vec3i(1));
}

fn mipMaxAt(c: vec3i) -> f32 {
  let cd = vec3i(U.mip.xyz);
  return mipMax[(c.z * cd.y + c.y) * cd.x + c.x];
}

// Distance along the ray (from p) to the exit face of block c, plus a nudge into the next block.
fn blockExit(p: vec3f, rd: vec3f, c: vec3i) -> f32 {
  let h = U.camForward.w;
  let lo = (vec3f(c) * U.mip.w - vec3f(0.5)) * h;
  let hi = lo + vec3f(U.mip.w * h);
  let bound = select(lo, hi, rd > vec3f(0.0));
  let inv = 1.0 / select(rd, vec3f(1e-9), abs(rd) < vec3f(1e-9));
  let tx = (bound - p) * inv;
  return max(min(min(tx.x, tx.y), tx.z), 0.0) + 0.05 * h;
}

// March from tStart to tEnd; returns the hit distance or -1. Blocks whose max density is
// below the iso value are skipped in one step (the coarse mip is conservative: it includes
// a one-texel halo, so the trilinear footprint of any sample inside the block is covered).
fn marchVolume(ro: vec3f, rd: vec3f, tStart: f32, tEnd: f32) -> f32 {
  let h = U.camForward.w;
  let iso = U.dims.w;
  let fine = 0.5 * h;
  let coarse = 0.9 * h;
  var tPrev = tStart;
  var t = tStart;
  for (var i = 0; i < MAX_STEPS; i++) {
    if (t > tEnd) { break; }
    let p = ro + rd * t;
    let c = mipCell(p);
    if (mipMaxAt(c) < iso) {
      // empty block: jump to its far face; the density there is below iso by construction
      t += blockExit(p, rd, c);
      tPrev = t;
      continue;
    }
    let d = density(p);
    if (d >= iso) {
      if (t <= tStart) { return t; }
      // 3 bisection refinements between the last outside sample and this one
      var lo = tPrev;
      var hi = t;
      for (var k = 0; k < 3; k++) {
        let mid = 0.5 * (lo + hi);
        let dm = density(ro + rd * mid);
        if (dm >= iso) { hi = mid; } else { lo = mid; }
      }
      // linear interpolation inside the final bracket
      let dlo = density(ro + rd * lo);
      let dhi = density(ro + rd * hi);
      let f = saturate((iso - dlo) / max(dhi - dlo, 1e-5));
      return mix(lo, hi, f);
    }
    tPrev = t;
    // zero density inside a non-empty block: a bigger step (the sheet after the nip can be
    // as thin as ~1.6 cells, so it stays below one cell)
    t += select(fine, coarse, d < 0.015);
  }
  return -1.0;
}

// Transmittance toward the light through the density volume: a short march that jumps
// over empty mip blocks, so it reaches far enough for the bank to shadow the rollers.
// jitter (0..1, a screen-space dither) offsets the samples by up to one step, which turns the
// march's banding into noise too fine to see.
fn volumeShadow(p: vec3f, l: vec3f, jitter: f32) -> f32 {
  let h = U.camForward.w;
  let step = max(1.5 * h, 0.024);
  var od = 0.0;
  var t = 1.6 * h + (jitter - 0.5) * step;
  for (var i = 0; i < 14; i++) {
    let q = p + l * t;
    if (any(q < vec3f(0.0)) || any(q > U.domain.xyz)) { break; }
    let c = mipCell(q);
    if (mipMaxAt(c) < 0.05) {
      t += blockExit(q, l, c);
      continue;
    }
    od += density(q) * step;
    t += step;
  }
  return exp(-od * 22.0);
}

// Occlusion of the hemisphere around n by a roller (sphere-occlusion style estimate on the
// cylinder cross-section): darkens the crease where the bank meets the rolls.
fn rollerOcclusion(p: vec3f, n: vec3f, pose: vec4f) -> f32 {
  let d = vec2f(p.y - pose.x, p.z - pose.y);
  let dist = max(length(d), 1e-4);
  if (dist < pose.z) { return 0.0; }
  let toAxis = vec3f(0.0, -d.x, -d.y) / dist;
  let solid = (pose.z * pose.z) / (dist * dist);
  let facing = saturate(dot(n, toAxis) * 0.8 + 0.45);
  let inX = smoothstep(-0.05, 0.02, p.x) * (1.0 - smoothstep(U.domain.w - 0.02, U.domain.w + 0.05, p.x));
  return saturate(solid * facing) * inX;
}

// Cheap AO: density 2, 4 and 6.5 texels along the normal (creases and the nip get dark),
// times the analytic occlusion by the two rolls.
fn volumeAo(p: vec3f, n: vec3f) -> f32 {
  let h = U.camForward.w;
  let d2 = density(p + n * 2.0 * h);
  let d4 = density(p + n * 4.0 * h);
  let d6 = density(p + n * 6.5 * h);
  let vol = saturate(1.0 - 0.55 * d2 - 0.35 * d4 - 0.2 * d6);
  let rollers = (1.0 - 0.7 * rollerOcclusion(p, n, U.rollerBack)) * (1.0 - 0.7 * rollerOcclusion(p, n, U.rollerFront));
  return vol * rollers;
}

// Thickness of the material behind the hit, measured along -n (world units, capped): a
// soft integral of the density so that cell-scale noise does not make it jump per pixel.
fn sheetThickness(p: vec3f, n: vec3f) -> f32 {
  let h = U.camForward.w;
  let iso = U.dims.w;
  let step = 0.7 * h;
  var thick = 0.5 * step;
  for (var k = 1; k <= 7; k++) {
    let d = density(p - n * (f32(k) * step));
    thick += smoothstep(iso * 0.3, iso * 1.1, d) * step;
  }
  return thick;
}

// ---------------------------------------------------------------- shading

// Wrapped-diffuse sum of the three lights (key is shadowed by `keyShadow`) plus the studio's
// ambient light.
fn lightDiffuse(n: vec3f, keyShadow: f32, wrap: f32) -> vec3f {
  let kd = saturate((dot(n, U.keyDir.xyz) + wrap) / (1.0 + wrap));
  let fd = saturate((dot(n, U.fillDir.xyz) + wrap) / (1.0 + wrap));
  let rd = saturate(dot(n, U.rimDir.xyz));
  return U.keyColor.rgb * U.keyDir.w * kd * keyShadow
       + U.fillColor.rgb * U.fillDir.w * fd
       + U.rimColor.rgb * U.rimDir.w * rd * rd
       + ambient(n);
}

// ---- bench

// Bench top at p: rgb = albedo, a = 1 on the steel tray plate under the mill, 0 on the wood.
fn benchSurface(p: vec3f) -> vec4f {
  let L = U.domain.w;
  // planks running along x, each with its own tone and a grain stretched along its length
  let pw = 0.26;
  let id = floor(p.z / pw);
  let fz = fract(p.z / pw);
  let tone = 0.80 + 0.40 * hash21(vec2f(id, 7.0));
  let grain = vnoise(vec2f(p.x * 5.0 + id * 13.7, p.z * 70.0)) * 0.6 + vnoise(vec2f(p.x * 22.0 + id * 5.1, p.z * 240.0)) * 0.4;
  let seam = smoothstep(0.0, 0.035, fz) * smoothstep(1.0, 0.965, fz);
  let wood = vec3f(0.130, 0.090, 0.060) * tone * (0.72 + 0.56 * grain) * (0.35 + 0.65 * seam);
  // brushed steel tray plate under the mill
  let ex = abs(p.x - L * 0.5) - (L * 0.5 + 0.30);
  let ez = abs(p.z - U.rollerFront.y + 0.12) - 0.90;
  let plate = smoothstep(0.012, 0.0, max(ex, ez));
  let rimLine = 1.0 - 0.6 * smoothstep(0.03, 0.0, abs(max(ex, ez) + 0.012));
  let brushed = vnoise(vec2f(p.x * 9.0, p.z * 160.0)) * 0.6 + vnoise(vec2f(p.x * 30.0, p.z * 420.0)) * 0.4;
  let steel = vec3f(0.30, 0.305, 0.32) * (0.82 + 0.36 * brushed) * rimLine;
  return vec4f(mix(wood, steel, plate), plate);
}

// x = key-light visibility on the bench (soft roll shadows), y = contact darkening under the rolls
fn benchShadow(p: vec3f) -> vec2f {
  let key = U.keyDir.xyz;
  let L = U.domain.w;
  var shadow = rollerShadow(p, key, U.rollerBack) * rollerShadow(p, key, U.rollerFront);
  let inX = smoothstep(-0.15, 0.05, p.x) * (1.0 - smoothstep(L - 0.05, L + 0.15, p.x));
  let dzB = (p.z - U.rollerBack.y) / (U.rollerBack.z * 1.35);
  let dzF = (p.z - U.rollerFront.y) / (U.rollerFront.z * 1.35);
  let contact = 1.0 - 0.78 * inX * max(exp(-dzB * dzB * 2.2), exp(-dzF * dzF * 2.2));
  return vec2f(shadow * contact, contact);
}

// the bench as it shows in a blurred reflection
fn benchGlimpse(q: vec3f) -> vec3f {
  let s = benchSurface(q);
  let sh = benchShadow(q);
  return s.rgb * lightDiffuse(vec3f(0.0, 1.0, 0.0), sh.x, 0.0) * sh.y * mix(1.0, 1.6, s.a);
}

// a roll as it shows in a reflection (rdIn = direction of the ray that reached it)
fn rollGlimpse(p: vec3f, n: vec3f, rdIn: vec3f) -> vec3f {
  let r = reflect(rdIn, n);
  return vec3f(0.56, 0.57, 0.60) * envRadiance(p, r, 0.35) + ambient(n) * 0.03;
}

// What a reflection ray leaving p along r sees: the rolls, the bench, otherwise the studio.
fn reflectScene(p: vec3f, r: vec3f, rough: f32) -> vec3f {
  let hb = hitRoller(p, r, U.rollerBack);
  let hf = hitRoller(p, r, U.rollerFront);
  if (min(hb.t, hf.t) < INF) {
    if (hf.t < hb.t) { return rollGlimpse(p + r * hf.t, hf.n, r); }
    return rollGlimpse(p + r * hb.t, hb.n, r);
  }
  if (r.y < -1e-3) {
    let q = p + r * (-p.y / r.y);
    if (all(q.xz > BENCH_LO.xz) && all(q.xz < BENCH_HI.xz)) {
      // fade into the blurred environment toward the horizon, where the footprint is long
      return mix(envRadiance(p, r, rough), benchGlimpse(q), smoothstep(0.0, 0.25, -r.y));
    }
  }
  return envRadiance(p, r, rough);
}

// ---- putty

// Body colour of the putty at q for reflections: one tap of the colour raster, decoded like the
// primary hit (latentToRgb, PIGMENT_OPACITY). If the colour model changes, change it here too.
fn puttyTint(q: vec3f) -> vec3f {
  let fq = fineUvw(q);
  let fa = textureSampleLevel(finA, volSampler, fq, 0.0);
  let fb = textureSampleLevel(finB, volSampler, fq, 0.0);
  let fc = textureSampleLevel(finC, volSampler, fq, 0.0);
  // the load that tints: pigment still in agglomerates counts for little (AGGLO_WEAK)
  let pl = (fa.x - (1.0 - AGGLO_WEAK) * clamp(fc.z, 0.0, fa.x)) / max(fc.x, 1e-4);
  let s = fa.y + fa.z + fa.w + fb.x;   // the filtered load: un-premultiplies the latent
  var albedo = vec3f(1.0);
  if (s > LATENT_MIN && pl > 1e-4) { albedo = srgbToLinear(latentToRgb(vec4f(fa.yzw, fb.x) / s, fb.yzw / s)); }
  return mix(CLEAR_BODY, albedo, 1.0 - exp(-PIGMENT_OPACITY * pl));
}

// The putty as a roll mirrors it: a short march of the density along the reflected ray (8
// growing steps, about 18 cells), then one colour tap. rgb = colour, a = how much of the
// reflection it covers.
fn reflectPutty(p: vec3f, r: vec3f) -> vec4f {
  let h = U.camForward.w;
  let iso = U.dims.w;
  var step = 1.2 * h;
  var t = 1.6 * h;
  // coverage builds up smoothly with the density met along the ray (a brushed roll blurs what
  // it mirrors, so there is no edge to find), fading with distance
  var cover = 0.0;
  var tTint = -1.0;
  for (var i = 0; i < 8; i++) {
    let q = p + r * t;
    if (any(q < vec3f(0.0)) || any(q > U.domain.xyz)) { break; }
    let d = density(q);
    let fade = 1.0 - smoothstep(0.5, 1.0, f32(i) / 8.0);
    cover = max(cover, smoothstep(0.5 * iso, 1.6 * iso, d) * fade);
    if (d >= iso) { tTint = t; break; }
    if (d >= 0.5 * iso && tTint < 0.0) { tTint = t + step; }
    t += step;
    step *= 1.15;
  }
  if (tTint < 0.0) { return vec4f(0.0); }
  let tint = puttyTint(p + r * (tTint + 0.6 * h));
  let light = lightDiffuse(normalize(vec3f(0.0, 0.5, 0.0) - r), 1.0, 0.5);
  return vec4f(tint * light * 0.8, cover * 0.5);
}

struct PuttyShade {
  diffuse: vec3f, // body colour: what the material scatters back
  spec: vec3f,    // the gloss on its skin, which is there whether or not the body is clear
};

// thickness: material thickness behind the hit (sheetThickness), in world units
// rough: how unsure the normal is (densityNormal): roughens the skin there, so a crumb or a
// tearing film does not sparkle
fn shadePutty(p: vec3f, n: vec3f, v: vec3f, albedo: vec3f, ao: f32, thickness: f32, rough: f32) -> PuttyShade {
  let key = U.keyDir.xyz;
  let h = U.camForward.w;
  var shadow = rollerShadow(p, key, U.rollerBack) * rollerShadow(p, key, U.rollerFront);
  // the march toward the key only matters where the key reaches (wrap 0.25)
  if (dot(n, key) > -0.25) { shadow *= mix(0.3, 1.0, volumeShadow(p, key, ign(gPix))); }
  let nv = saturate(dot(n, v));

  // diffuse with a little wrap: uncured silicone scatters light in its top layer
  var col = albedo * lightDiffuse(n, shadow, 0.25) * ao;

  // thin-sheet transmittance: light that enters the sheet, scatters and leaves again is
  // filtered twice by the pigment (albedo^2), attenuated by the thickness, and shows most
  // where the sheet is seen at a grazing angle
  let trans = exp(-thickness / (2.2 * h));
  let tint = albedo * albedo;
  let backKey = saturate((dot(-n, key) + 0.7) / 1.7);
  let backFill = saturate((dot(-n, U.fillDir.xyz) + 0.7) / 1.7);
  let backRim = saturate(dot(-n, U.rimDir.xyz) + 0.2);
  let through = U.keyColor.rgb * U.keyDir.w * backKey * mix(0.5, 1.0, shadow)
              + U.fillColor.rgb * U.fillDir.w * backFill
              + U.rimColor.rgb * U.rimDir.w * backRim * 0.8
              + vec3f(0.25);
  let graze = 0.35 + 0.65 * pow(1.0 - nv, 2.0);
  col += tint * through * trans * graze * 0.55 * ao;

  // gloss: a dielectric skin (F0 = 0.04) mirroring the studio, the rolls and the bench
  let sr = mix(PUTTY_ROUGH, 0.5, rough);
  let f = fresnel(nv, vec3f(0.04), sr).x;
  let r = reflect(-v, n);
  var out: PuttyShade;
  out.spec = reflectScene(p + n * (0.5 * h), r, sr) * (f * specOcclusion(nv, ao, sr));
  out.diffuse = col * (1.0 - f);
  return out;
}

// ---- rolls

// lite: the roll is seen through clear putty or in a place where its own shadow and reflection
// marches are not worth their samples
fn shadeRollerBody(p: vec3f, n: vec3f, v: vec3f, pose: vec4f, other: vec4f, lite: bool) -> vec3f {
  let R = pose.z;
  let angle = pose.w;
  // material angle around the axis (rotates with the roll)
  let theta = atan2(p.y - pose.x, p.z - pose.y) + angle;
  let circ = theta * R; // arc length coordinate

  // brushed streaks: long along the circumference (lathe/grinding marks), fine along x
  // (each octave fades to its mean once a pixel covers more than half its period, or it aliases
  // into combs along every highlight edge)
  let foot = length(p - U.camOrigin.xyz) * 2.0 * U.camOrigin.w / U.misc.y;
  let b1 = mix(0.5, vnoise(vec2f(p.x * 150.0, circ * 6.0)), saturate(1.5 - foot * 300.0));
  let b2 = mix(0.5, vnoise(vec2f(p.x * 420.0, circ * 14.0)), saturate(1.5 - foot * 840.0));
  let brush = b1 * 0.5 + b2 * 0.5;
  // rotation cues: isotropic speckle, broad wear/tarnish patches and faint spiral marks that
  // all travel with the surface
  let speck = vnoise(vec2f(p.x * 42.0, circ * 42.0));
  let spiral = smoothstep(0.86, 1.0, sin(theta * 22.0 + p.x * 9.0 + speck * 1.2)) * 0.07;
  let wear = vnoise(vec2f(p.x * 2.2, circ * 3.2)) * 0.22 + vnoise(vec2f(p.x * 5.0, circ * 7.5)) * 0.10;
  // polished steel: the pattern modulates how much it reflects
  let f0 = vec3f(0.62, 0.63, 0.66) * (0.66 + 0.14 * brush + 0.10 * speck + 0.75 * wear) + vec3f(spiral);

  let key = U.keyDir.xyz;
  var shadow = 1.0;
  var ao = 1.0 - 0.75 * rollerOcclusion(p, n, other);
  if (!lite) {
    // the bank sitting on the roll shadows and darkens the roll next to it
    shadow = mix(0.3, 1.0, volumeShadow(p, key, ign(gPix)));
    ao *= volumeAo(p, n);
  }
  let nv = saturate(dot(n, v));

  // metal: almost no diffuse, the look is the mirrored studio. The grinding marks run around
  // the roll, so its micro-normals fan out along the axis: the reflection is sharp around the
  // roll and smeared along it (five taps of the analytic environment, no texture samples).
  var col = f0 * lightDiffuse(n, shadow, 0.1) * 0.09 * ao;
  let wob = (brush - 0.5) * 0.05;
  var refl = vec3f(0.0);
  for (var k = 0; k < 5; k++) {
    let s = (f32(k) - 2.0) * 0.5;
    let nk = normalize(n + vec3f(s * 0.10 + wob, 0.0, 0.0));
    refl += envRadiance(p, reflect(-v, nk), ROLL_ROUGH) * (1.0 - 0.3 * abs(s));
  }
  refl /= 4.1;
  let r = reflect(-v, n);
  // nearer things in the mirror: the bench, the other roll, the putty
  if (r.y < -1e-3) {
    let q = p + r * (-p.y / r.y);
    if (all(q.xz > BENCH_LO.xz) && all(q.xz < BENCH_HI.xz)) { refl = mix(refl, benchGlimpse(q), smoothstep(0.0, 0.2, -r.y)); }
  }
  let ho = hitRoller(p, r, other);
  if (ho.t < INF) { refl = rollGlimpse(p + r * ho.t, ho.n, r); }
  if (!lite && U.quality.x > 0.5) {
    let pr = reflectPutty(p, r);
    refl = mix(refl, pr.rgb, pr.a);
  }
  let f = fresnel(nv, f0, 0.2);
  col += refl * f * (specOcclusion(nv, ao, 0.25) * mix(0.75, 1.0, shadow));
  return col;
}

fn shadeRollerCap(p: vec3f, n: vec3f, v: vec3f, pose: vec4f) -> vec3f {
  let dy = p.y - pose.x;
  let dz = p.z - pose.y;
  let rr = length(vec2f(dy, dz));
  let ringN = vnoise(vec2f(rr * 240.0, 0.5));
  let rings = 0.86 + 0.14 * ringN;
  let hub = 1.0 - 0.35 * smoothstep(0.09, 0.07, rr); // a darker axle stub in the middle
  // six bolt heads on a ring that turn with the roll (a rotation cue on the end faces)
  let phi = atan2(dy, dz) + pose.w;
  let sector = fract(phi * 6.0 / (2.0 * PI));
  let boltR = pose.z * 0.55;
  let bolt = vec2f((sector - 0.5) * (2.0 * PI * boltR / 6.0), rr - boltR);
  let boltMask = smoothstep(0.024, 0.016, length(bolt));
  let boltShade = 1.0 - 0.45 * boltMask + 0.25 * boltMask * smoothstep(0.012, 0.0, length(bolt));
  let f0 = vec3f(0.50, 0.51, 0.54) * rings * hub * boltShade;
  let nv = saturate(dot(n, v));
  var col = f0 * lightDiffuse(n, 1.0, 0.2) * 0.16;
  // a faced (lathe-turned) end: the tool marks are circles, so the reflection smears radially
  let radial = vec3f(0.0, dy, dz) / max(rr, 1e-4);
  var refl = vec3f(0.0);
  for (var k = 0; k < 3; k++) {
    let s = f32(k) - 1.0;
    let nk = normalize(n + radial * (s * 0.11 + (ringN - 0.5) * 0.05));
    refl += envRadiance(p, reflect(-v, nk), 0.26);
  }
  col += refl / 3.0 * fresnel(nv, f0, 0.26);
  // and each light draws the bow-tie glint of a turned face (Kajiya-Kay on the circular marks)
  let tang = cross(n, radial);
  let tk = dot(tang, normalize(U.keyDir.xyz + v));
  let tf = dot(tang, normalize(U.fillDir.xyz + v));
  let glint = U.keyColor.rgb * (U.keyDir.w * pow(max(1.0 - tk * tk, 0.0), 45.0) * saturate(dot(n, U.keyDir.xyz) * 3.0))
            + U.fillColor.rgb * (U.fillDir.w * pow(max(1.0 - tf * tf, 0.0), 45.0) * saturate(dot(n, U.fillDir.xyz) * 3.0));
  col += glint * f0 * (0.55 * smoothstep(0.03, 0.12, rr));
  return col;
}

// ---- bench and shop floor

fn shadeBench(p: vec3f, n: vec3f, rd: vec3f) -> vec3f {
  let v = -rd;
  if (n.y < 0.5) {
    // the edge of the slab
    let grain = vnoise(vec2f((p.x + p.z) * 6.0, p.y * 160.0));
    return vec3f(0.11, 0.072, 0.045) * (0.7 + 0.5 * grain) * lightDiffuse(n, 1.0, 0.0);
  }
  let s = benchSurface(p);
  let sh = benchShadow(p);
  let plate = s.a;
  let rough = mix(0.30, 0.34, plate);
  let f0 = mix(vec3f(0.04), s.rgb * 0.9, plate);
  // varnished wood scatters; the steel plate mostly mirrors
  var col = s.rgb * lightDiffuse(n, sh.x, 0.0) * sh.y * mix(1.0, 0.35, plate);
  let r = reflect(rd, n);
  let hb = hitRoller(p + n * 1e-4, r, U.rollerBack);
  let hf = hitRoller(p + n * 1e-4, r, U.rollerFront);
  var refl: vec3f;
  if (hf.t < min(hb.t, INF)) { refl = rollGlimpse(p + r * hf.t, hf.n, r); }
  else if (hb.t < INF) { refl = rollGlimpse(p + r * hb.t, hb.n, r); }
  else { refl = envRadiance(p, r, rough); }
  let f = fresnel(saturate(dot(n, v)), f0, rough);
  return col * (vec3f(1.0) - f) + refl * f * mix(sh.y, 1.0, 0.35);
}

fn shadeShopFloor(p: vec3f) -> vec3f {
  let n = vec3f(0.0, 1.0, 0.0);
  let mottle = vnoise(p.xz * 1.7) * 0.6 + vnoise(p.xz * 9.0) * 0.4;
  // the bench keeps the light off the floor under it
  let c = (BENCH_LO.xz + BENCH_HI.xz) * 0.5;
  let e = (BENCH_HI.xz - BENCH_LO.xz) * 0.5;
  let d = length(max(abs(p.xz - c) - e, vec2f(0.0)));
  let under = mix(0.25, 1.0, smoothstep(-0.1, 0.9, d));
  return vec3f(0.060, 0.059, 0.058) * (0.75 + 0.5 * mottle) * lightDiffuse(n, 1.0, 0.0) * under;
}

// ---------------------------------------------------------------- analytic scene

struct PlateHit {
  t: f32,
  n: vec3f,
};

// Entry point of a ray into the axis-aligned box [lo, hi]; t = INF on a miss.
fn hitAabb(ro: vec3f, rd: vec3f, lo: vec3f, hi: vec3f) -> PlateHit {
  var out: PlateHit;
  out.t = INF;
  out.n = vec3f(1.0, 0.0, 0.0);
  let inv = 1.0 / select(rd, vec3f(1e-9), abs(rd) < vec3f(1e-9));
  let t0 = (lo - ro) * inv;
  let t1 = (hi - ro) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tn = max(max(tmin.x, tmin.y), tmin.z);
  let tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tn > tf || tf < 0.0) { return out; }
  if (tn < 0.0) { return out; } // inside the box: ignore
  out.t = tn;
  if (tn == tmin.x) { out.n = vec3f(-sign(rd.x), 0.0, 0.0); }
  else if (tn == tmin.y) { out.n = vec3f(0.0, -sign(rd.y), 0.0); }
  else { out.n = vec3f(0.0, 0.0, -sign(rd.z)); }
  return out;
}

struct SceneHit {
  t: f32,
  n: vec3f,
  kind: i32,   // 0 = backdrop, 1 = roll body, 2 = roll end, 3 = bench, 4 = shop floor
  pose: vec4f, // the roll that was hit, and the other one
  other: vec4f,
};

// Nearest analytic surface along the ray: rolls, bench slab, shop floor.
fn traceScene(ro: vec3f, rd: vec3f) -> SceneHit {
  var s: SceneHit;
  let hb = hitRoller(ro, rd, U.rollerBack);
  let hf = hitRoller(ro, rd, U.rollerFront);
  var roller = hb;
  s.pose = U.rollerBack;
  s.other = U.rollerFront;
  if (hf.t < hb.t) { roller = hf; s.pose = U.rollerFront; s.other = U.rollerBack; }
  s.t = roller.t;
  s.n = roller.n;
  s.kind = roller.kind;
  let bh = hitAabb(ro, rd, BENCH_LO, BENCH_HI);
  if (bh.t < s.t) { s.t = bh.t; s.n = bh.n; s.kind = 3; }
  if (rd.y < -1e-6) {
    let tf = (SHOP_Y - ro.y) / rd.y;
    if (tf > 0.0 && tf < s.t) { s.t = tf; s.n = vec3f(0.0, 1.0, 0.0); s.kind = 4; }
  }
  return s;
}

fn shadeScene(ro: vec3f, rd: vec3f, s: SceneHit, lite: bool) -> vec3f {
  let p = ro + rd * s.t;
  if (s.kind == 1) { return shadeRollerBody(p, s.n, -rd, s.pose, s.other, lite); }
  if (s.kind == 2) { return shadeRollerCap(p, s.n, -rd, s.pose); }
  if (s.kind == 3) { return shadeBench(p, s.n, rd); }
  if (s.kind == 4) { return shadeShopFloor(p); }
  return backdrop(rd);
}

// ---------------------------------------------------------------- end-guide plates

// The translucent end guide at x = x0 (thickness U.guides.z toward the outside), spanning the
// bank in y and z; composited over the scene colour `behind` when it is in front of tHit.
fn compositeGuide(col: vec3f, ro: vec3f, rd: vec3f, tHit: f32, x0: f32, outward: f32) -> vec3f {
  let tk = U.guides.z;
  let lo = vec3f(select(x0 - tk, x0, outward > 0.0), U.rollerBack.x + U.rollerBack.z * 0.55, U.rollerBack.y - 0.06);
  let hi = vec3f(select(x0, x0 + tk, outward > 0.0), U.guides.x, U.rollerFront.y + 0.06);
  let ph = hitAabb(ro, rd, lo, hi);
  if (ph.t >= tHit) { return col; }
  let n = ph.n;
  let v = -rd;
  let nv = saturate(dot(n, v));
  let f = fresnel(nv, vec3f(0.04), 0.05).x;
  // clear acrylic: mostly what is behind, a little haze, the studio mirrored in its faces
  let haze = vec3f(0.62, 0.64, 0.66) * lightDiffuse(n, 1.0, 0.5) * 0.10;
  let refl = envRadiance(ro + rd * ph.t, reflect(rd, n), 0.05) * f;
  // thin edges (y/z faces) read as solid, bright rims: light piped through the sheet
  let edge = 1.0 - abs(n.x);
  let alpha = mix(0.10 + 0.5 * f, 0.8, edge);
  return mix(col * vec3f(0.93, 0.96, 0.95), haze + vec3f(0.30, 0.34, 0.33) * edge, alpha) + refl;
}

// ---------------------------------------------------------------- tone

// Display transform. Pigment colours are the point of the picture, so the curve works on the
// largest channel and scales the colour with it (hue and saturation of the mid-tones pass
// through unchanged, where a per-channel filmic curve such as ACES skews them toward the
// primaries); above the knee it rolls off smoothly to white, and what overshoots bleaches
// toward white as on film, so a highlight on yellow putty reads as a highlight.
fn tonemap(c: vec3f) -> vec3f {
  let x = max(c, vec3f(0.0));
  let peak = max(max(x.r, x.g), max(x.b, 1e-5));
  let knee = 0.62;
  var mapped = peak;
  if (peak > knee) { mapped = knee + (1.0 - knee) * (1.0 - exp(-(peak - knee) / (1.0 - knee))); }
  let scaled = x * (mapped / peak);
  let bleach = saturate((peak - 1.3) / 4.0);
  return clamp(mix(scaled, vec3f(mapped), bleach), vec3f(0.0), vec3f(1.0));
}

// ---------------------------------------------------------------- fragment

@fragment
fn fsMain(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  gPix = fragCoord.xy;
  let res = U.misc.xy;
  let ndc = vec2f(fragCoord.x / res.x * 2.0 - 1.0, 1.0 - fragCoord.y / res.y * 2.0);
  let tanHalf = U.camOrigin.w;
  let aspect = U.camRight.w;
  let ro = U.camOrigin.xyz;
  let rd = normalize(U.camForward.xyz + U.camRight.xyz * (ndc.x * tanHalf * aspect) + U.camUp.xyz * (ndc.y * tanHalf));
  let flags = u32(U.misc.w);
  let h = U.camForward.w;

  // analytic scene
  let scene = traceScene(ro, rd);

  // volume
  var tVol = -1.0;
  if ((flags & 1u) != 0u) {
    let box = hitBox(ro, rd);
    if (box.x < box.y && box.x < scene.t) {
      tVol = marchVolume(ro, rd, box.x, min(box.y, scene.t));
    }
  }

  var col: vec3f;
  var tHit = scene.t;
  let v = -rd;
  if (tVol > 0.0) {
    tHit = tVol;
    let p = ro + rd * tVol;
    let sn = densityNormal(p);
    let n = sn.n;
    // colour: integrate through the material behind the hit (uncured silicone is
    // translucent, and milled pigment often sits one cell under a clear skin),
    // weighting samples by density and by depth so the surface dominates. The
    // latent is the pigment-weighted Mixbox mix of the PIGMENTS along the ray
    // (the clear base carries none, so it never whitens a streak); the pigment
    // load per unit mass along the ray sets how opaque the colour is.
    // Latents are stored premultiplied by the load, so the trilinear sample is the
    // pigment-amount mix of the texels round it; dividing by the sum of its four
    // concentration channels (the filtered load) gives the latent back.
    var c = vec4f(0.0);
    var resid = vec3f(0.0);
    var csum = 0.0;
    var dsum = 0.0;
    var load = 0.0;
    var cov = 0.0;
    var und = 0.0;   // the part of the load still in agglomerates
    // surface mode: five samples just inside the hit, one at the hit and a cross of four half a
    // cell away along the surface, no depth; depth mode: five samples down the normal with a
    // depth decay
    let surfaceMode = (flags & 4u) != 0u;
    let hF = FB.origin.w;
    let tx = normalize(cross(n, select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9)));
    let ty = cross(n, tx);
    for (var k = 0; k < 5; k++) {
      var q: vec3f;
      var decay = 1.0;
      if (surfaceMode) {
        var side = vec3f(0.0);
        if (k == 1) { side = tx; } else if (k == 2) { side = -tx; } else if (k == 3) { side = ty; } else if (k == 4) { side = -ty; }
        q = p - n * (SURFACE_DEPTH * hF) + side * (SURFACE_SPREAD * hF);
      } else {
        q = p - n * ((0.5 + 0.8 * f32(k)) * h);
        decay = exp(-0.45 * f32(k));
      }
      // the sample's weight comes from the solver-grid density (the surface the march found);
      // its pigment comes from the colour raster, which resolves streaks the solver grid cannot
      let a = textureSampleLevel(volA, volSampler, volUvw(q), 0.0);
      let w = smoothstep(0.15, 0.6, a.x) * decay;
      if (w <= 1e-5) { continue; }
      let fq = fineUvw(q);
      let fa = textureSampleLevel(finA, volSampler, fq, 0.0);
      let fc = textureSampleLevel(finC, volSampler, fq, 0.0);
      let fm = fc.x;
      let pl = fa.x / max(fm, 1e-4);   // load per unit mass
      dsum += w;
      load += w * pl;
      und += w * clamp(fc.z, 0.0, fa.x) / max(fm, 1e-4);
      cov += w * clamp(fc.y / max(fm, 1e-4), 0.0, 1.0);   // covered share of the mass
      let fb = textureSampleLevel(finB, volSampler, fq, 0.0);
      let s = fa.y + fa.z + fa.w + fb.x;
      if (s <= LATENT_MIN || pl <= 1e-4) { continue; }
      let wp = w * pl;
      c += wp * vec4f(fa.yzw, fb.x) / s;
      resid += wp * fb.yzw / s;
      csum += wp;
    }
    if (dsum > 1e-5) { load = load / dsum; cov = cov / dsum; und = und / dsum; }
    // dispersive mixing: only dispersed pigment develops its colour; what is still in agglomerates tints
    // weakly and shows as specks of masstone (below)
    und = min(und, load);
    let raw = und / max(load, 1e-5);   // undispersed share of the pigment here
    load = load - (1.0 - AGGLO_WEAK) * und;
    var albedo = vec3f(1.0);
    if (csum > 1e-5) { c = c / csum; resid = resid / csum; albedo = srgbToLinear(latentToRgb(c, resid)); }
    let ao = volumeAo(p, n);
    let thickness = sheetThickness(p, n);
    // an opaque colourant in a clear medium: its share of the mass sets how much of
    // the light it captures; even a few percent of masterbatch reads solid
    var pigment = 1.0 - exp(-PIGMENT_OPACITY * load);
    if (surfaceMode) {
      // coverage decides whether this cell is pigment or clear; where it is pigment, the colour
      // has the covered material's own load (the clear share of the cell does not dilute it)
      let cell = floor((p - n * (SURFACE_DEPTH * hF)) / hF + vec3f(0.5));   // nearest fine node
      let covD = cov + COV_DITHER * (2.0 * hash31(cell) - 1.0);
      let loadCovered = load / max(cov, 0.05);
      pigment = smoothstep(COV_LO, COV_HI, covD) * (1.0 - exp(-PIGMENT_OPACITY * loadCovered));
    }
    if (csum > 1e-5 && und > 1e-4) {
      // an agglomerate at the surface: a speck the size of a fraction of a colour cell, opaque, in the
      // pigment's masstone. Hash of the world cell, like the coverage dither above.
      let sc = floor((p - n * (SURFACE_DEPTH * hF)) / (AGGLO_SPECK_CELL * hF));
      let chance = AGGLO_SPECK_CHANCE * raw * (1.0 - exp(-und / AGGLO_SPECK_LOAD));
      if (hash31(sc + vec3f(17.0, 59.0, 83.0)) < chance) {
        albedo = pow(albedo, vec3f(AGGLO_SPECK_TONE));
        pigment = max(pigment, 0.97);
      }
    }

    // ---- from here on the albedo, the pigment opacity and the surface are inputs ----
    // clear base: little body colour of its own (a milky scatter that grows with thickness);
    // the look comes from the gloss and from what shows through; pigment restores the albedo
    let shade = shadePutty(p, n, v, mix(CLEAR_BODY, albedo, pigment), ao, thickness, sn.rough);
    // Uncured silicone is clear, not white. The base scatters a share of the light that grows
    // with the thickness behind the hit (haze, Beer-Lambert) and passes the rest; pigment makes
    // it opaque. What passes is the analytic scene along the ray as the surface refracts it
    // (n = 1.41), tinted by the path through the material.
    let haze = 1.0 - exp(-thickness / (CLEAR_HAZE_CELLS * h));
    let opacity = pigment + (1.0 - pigment) * haze;
    var body = shade.diffuse;
    if (opacity < 0.99) {
      // a thin parallel sheet only offsets the ray; a thick body bends it like a lens
      // (less where the normal is cell-scale noise, or the roll behind looks like crumpled foil)
      let bend = (0.1 + 0.5 * saturate(thickness / (4.0 * h))) * (1.0 - 0.7 * sn.rough);
      let rb = normalize(mix(rd, refract(rd, n, 1.0 / 1.41), bend));
      let bs = traceScene(p, rb);
      let behind = shadeScene(p, rb, bs, true);
      let path = pow(vec3f(0.80, 0.88, 0.90), vec3f(thickness / (3.0 * h)));
      body = mix(behind * mix(path, albedo, pigment), shade.diffuse, opacity);
    }
    col = body + shade.spec;
    if (U.quality.y > 1.5) { col = vec3f(sn.rough, 1.0 - sn.rough, 0.0) * 0.5; }
    else if (U.quality.y > 0.5) { col = (n * 0.5 + vec3f(0.5)) * 0.6; }
  } else {
    col = shadeScene(ro, rd, scene, false);
  }

  // end-guide plates at both ends of the rolls (far one first)
  if (U.guides.w > 0.5) {
    let L = U.domain.w;
    let nearX = select(0.0, L, ro.x > L * 0.5);
    let farX = L - nearX;
    col = compositeGuide(col, ro, rd, tHit, farX, select(-1.0, 1.0, farX > 0.0));
    col = compositeGuide(col, ro, rd, tHit, nearX, select(-1.0, 1.0, nearX > 0.0));
  }

  // distance haze into the backdrop
  if (tHit < INF) {
    let fog = 1.0 - exp(-max(tHit - 3.0, 0.0) * 0.16);
    col = mix(col, backdrop(rd), fog);
  }

  // vignette, exposure, tone curve, encode, dither (half an 8-bit step of noise hides the
  // banding of the dark gradients)
  let vig = 1.0 - 0.22 * dot(ndc * vec2f(0.9, 1.0), ndc * vec2f(0.9, 1.0));
  col = tonemap(col * U.misc.z * vig);
  if ((flags & 2u) == 0u) { col = linearToSrgb(col); }
  col += vec3f((ign(fragCoord.xy) - 0.5) / 255.0);
  return vec4f(col, 1.0);
}

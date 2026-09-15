// ColorMill v2 — fullscreen ray-march of the mill (design §8).
// mixbox.wgsl (latentToRgb) is prepended by the renderer.
//
// One triangle covers the screen. Per pixel: build a camera ray, intersect the
// analytic scene (two capped roller cylinders, the tray plane), then march the
// density volume between the domain-box entry and the nearest analytic hit.
// Blocks of the coarse max-density mip (built by csMip each frame) that hold no
// density above iso are skipped in one step. The first sample with density >= iso
// is refined by bisection; the normal is a box-filtered density gradient; colour
// is the Mixbox latent decoded at the hit, with a thin-sheet transmittance term.

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
  misc: vec4f,        // resolution.xy, exposure, flags (bit0 = volumes bound, bit1 = target is sRGB-encoded already)
  mip: vec4f,         // coarse max-density mip dims (cx, cy, cz), w = block edge in texels
  guides: vec4f,      // end-guide plates: bank top y, bank half depth (z), plate thickness, enabled
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var volA: texture_3d<f32>;
@group(0) @binding(2) var volB: texture_3d<f32>;
@group(0) @binding(3) var volSampler: sampler;
// coarse mip: max density over each block of mip.w^3 texels (plus a one-texel halo, so a
// trilinear sample anywhere inside the block is bounded by it); rebuilt by csMip every frame
@group(0) @binding(4) var<storage, read> mipMax: array<f32>;
@group(0) @binding(5) var<storage, read_write> mipOut: array<f32>;

const PI: f32 = 3.14159265358979;
const INF: f32 = 1e30;
const MAX_STEPS: i32 = 420;

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

// dark studio backdrop: vertical gradient with a faint warm floor bounce
fn skyColor(d: vec3f) -> vec3f {
  let t = saturate(d.y * 0.5 + 0.5);
  let low = vec3f(0.030, 0.031, 0.036);
  let mid = vec3f(0.075, 0.080, 0.092);
  let high = vec3f(0.16, 0.17, 0.19);
  var c = mix(low, mid, smoothstep(0.0, 0.55, t));
  c = mix(c, high, smoothstep(0.55, 1.0, t));
  return c;
}

// broad soft-box reflection seen in glossy surfaces: a bright patch around the key light
fn envReflection(r: vec3f) -> vec3f {
  let key = U.keyDir.xyz;
  let box = pow(saturate(dot(r, key)), 5.0) * 0.55 + pow(saturate(dot(r, key)), 40.0) * 0.5;
  let fill = pow(saturate(dot(r, U.fillDir.xyz)), 3.0) * 0.12;
  return skyColor(r) * 1.4 + U.keyColor.rgb * box + U.fillColor.rgb * fill;
}

fn fresnelSchlick(cosTheta: f32, f0: f32) -> f32 {
  let m = pow(1.0 - saturate(cosTheta), 5.0);
  return f0 + (1.0 - f0) * m;
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

fn density(p: vec3f) -> f32 {
  return textureSampleLevel(volA, volSampler, volUvw(p), 0.0).x;
}

// Density gradient. At low resolution the MPM density is noisy at the cell scale, so the
// main term is the gradient of a 2x2x2 box-filtered density (the 8 cube-corner taps at
// +-e), which smooths along the surface as well as across it; a tight 6-tap central
// difference is blended in so the silhouette and thin sheets keep their shape.
struct SurfaceNormal {
  n: vec3f,
  // 0 = the tight and the box-filtered gradients agree (smooth surface), 1 = cell-scale noise
  rough: f32,
};

fn densityNormal(p: vec3f) -> SurfaceNormal {
  let h = U.camForward.w;
  let e = 1.5 * h;
  var g = vec3f(0.0);
  for (var i = 0; i < 8; i++) {
    let sgn = vec3f(select(-1.0, 1.0, (i & 1) != 0), select(-1.0, 1.0, (i & 2) != 0), select(-1.0, 1.0, (i & 4) != 0));
    g += sgn * density(p + sgn * e);
  }
  g /= 8.0 * e;
  let e2 = 0.8 * h;
  let g2 = vec3f(
    density(p + vec3f(e2, 0.0, 0.0)) - density(p - vec3f(e2, 0.0, 0.0)),
    density(p + vec3f(0.0, e2, 0.0)) - density(p - vec3f(0.0, e2, 0.0)),
    density(p + vec3f(0.0, 0.0, e2)) - density(p - vec3f(0.0, 0.0, e2))
  ) / (2.0 * e2);
  let grad = g * 0.75 + g2 * 0.25;
  let len = length(grad);
  var out: SurfaceNormal;
  out.n = vec3f(0.0, 1.0, 0.0);
  out.rough = 0.0;
  if (len < 1e-6) { return out; }
  out.n = -grad / len;
  let l2 = length(g2);
  if (l2 > 1e-6) { out.rough = saturate((1.0 - dot(out.n, -g2 / l2)) * 45.0); }
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
fn volumeShadow(p: vec3f, l: vec3f) -> f32 {
  let h = U.camForward.w;
  let step = max(1.5 * h, 0.024);
  var od = 0.0;
  var t = 1.6 * h;
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

struct Lighting {
  diffuse: vec3f,
  keyVis: f32,
};

// Wrapped-diffuse sum of the three lights (key is shadowed by `keyShadow`).
fn lightDiffuse(n: vec3f, keyShadow: f32, wrap: f32) -> vec3f {
  let kd = saturate((dot(n, U.keyDir.xyz) + wrap) / (1.0 + wrap));
  let fd = saturate((dot(n, U.fillDir.xyz) + wrap) / (1.0 + wrap));
  let rd = saturate(dot(n, U.rimDir.xyz));
  let hemi = mix(vec3f(0.06, 0.06, 0.065), vec3f(0.22, 0.23, 0.26), n.y * 0.5 + 0.5);
  return U.keyColor.rgb * U.keyDir.w * kd * keyShadow
       + U.fillColor.rgb * U.fillDir.w * fd
       + U.rimColor.rgb * U.rimDir.w * rd * rd
       + hemi;
}

fn blinnLobe(n: vec3f, v: vec3f, l: vec3f, shininess: f32) -> f32 {
  let hv = normalize(l + v);
  let nh = saturate(dot(n, hv));
  let nl = saturate(dot(n, l));
  return pow(nh, shininess) * (shininess + 8.0) / (8.0 * PI) * nl;
}

// thickness: material thickness behind the hit (sheetThickness), in world units
// rough: normal-noise estimate from densityNormal (widens the specular lobes, Toksvig-style,
// so cell-scale density noise does not sparkle)
fn shadePutty(p: vec3f, n: vec3f, v: vec3f, albedo: vec3f, ao: f32, thickness: f32, rough: f32) -> vec3f {
  let key = U.keyDir.xyz;
  let h = U.camForward.w;
  var shadow = rollerShadow(p, key, U.rollerBack) * rollerShadow(p, key, U.rollerFront);
  shadow *= mix(0.35, 1.0, volumeShadow(p, key));
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

  // glossy reflection: wide + narrow Blinn-Phong lobes, Fresnel-weighted, plus the environment
  let f = fresnelSchlick(nv, 0.03);
  let sw = mix(32.0, 10.0, rough);
  let sn = mix(320.0, 24.0, rough);
  let wide = blinnLobe(n, v, key, sw) * U.keyColor.rgb * U.keyDir.w * 0.30
           + blinnLobe(n, v, U.fillDir.xyz, sw * 0.75) * U.fillColor.rgb * U.fillDir.w * 0.30
           + blinnLobe(n, v, U.rimDir.xyz, sw * 0.75) * U.rimColor.rgb * U.rimDir.w * 0.45;
  let narrow = blinnLobe(n, v, key, sn) * U.keyColor.rgb * U.keyDir.w * 0.8
             + blinnLobe(n, v, U.fillDir.xyz, sn * 0.85) * U.fillColor.rgb * U.fillDir.w * 0.35
             + blinnLobe(n, v, U.rimDir.xyz, sn) * U.rimColor.rgb * U.rimDir.w * 0.4;
  let r = reflect(-v, n);
  let env = envReflection(r) * 0.55;
  let gloss = 0.9; // a wet skin, not a mirror
  col += ((wide + narrow) * shadow * 0.7 + env * ao) * (f * 9.0 + 0.035) * gloss;
  return col;
}

fn shadeRollerBody(p: vec3f, n: vec3f, v: vec3f, pose: vec4f) -> vec3f {
  let R = pose.z;
  let angle = pose.w;
  // material angle around the axis (rotates with the roll)
  let theta = atan2(p.y - pose.x, p.z - pose.y) + angle;
  let circ = theta * R; // arc length coordinate

  // brushed streaks: long along the circumference (lathe/grinding marks), fine along x
  let brush = vnoise(vec2f(p.x * 260.0, circ * 6.0)) * 0.5 + vnoise(vec2f(p.x * 620.0, circ * 14.0)) * 0.5;
  // rotation cues: isotropic speckle, broad wear/tarnish patches and faint spiral marks that
  // all travel with the surface
  let speck = vnoise(vec2f(p.x * 42.0, circ * 42.0));
  let spiral = smoothstep(0.86, 1.0, sin(theta * 22.0 + p.x * 9.0 + speck * 1.2)) * 0.08;
  let wear = vnoise(vec2f(p.x * 2.2, circ * 3.2)) * 0.22 + vnoise(vec2f(p.x * 5.0, circ * 7.5)) * 0.10;
  var albedo = vec3f(0.50, 0.51, 0.54) * (0.74 + 0.16 * brush + 0.12 * speck + wear) + vec3f(spiral);

  let key = U.keyDir.xyz;
  let shadow = mix(0.3, 1.0, volumeShadow(p, key));
  // the bank sitting on the roll darkens the roll next to it
  let ao = volumeAo(p, n);
  let nv = saturate(dot(n, v));

  // chrome: almost no diffuse; the look comes from the reflected environment (dark studio,
  // bright soft box) modulated by the wear pattern, plus anisotropic highlights stretched
  // along the axis (Kajiya-Kay with a circumferential tangent)
  let tangent = normalize(cross(vec3f(1.0, 0.0, 0.0), n));
  var col = albedo * lightDiffuse(n, shadow, 0.1) * 0.13 * ao;

  let hk = normalize(key + v);
  let th = dot(tangent, hk);
  let anisoK = pow(sqrt(max(1.0 - th * th, 0.0)), 140.0) * saturate(dot(n, key) * 2.0);
  let hf = normalize(U.fillDir.xyz + v);
  let tf = dot(tangent, hf);
  let anisoF = pow(sqrt(max(1.0 - tf * tf, 0.0)), 60.0) * saturate(dot(n, U.fillDir.xyz) * 2.0);
  let hr = normalize(U.rimDir.xyz + v);
  let tr = dot(tangent, hr);
  let anisoR = pow(sqrt(max(1.0 - tr * tr, 0.0)), 80.0) * saturate(dot(n, U.rimDir.xyz) * 2.0);
  let streak = 0.7 + 0.3 * brush;
  col += (U.keyColor.rgb * U.keyDir.w * anisoK * 0.45 * shadow + U.fillColor.rgb * U.fillDir.w * anisoF * 0.18 + U.rimColor.rgb * U.rimDir.w * anisoR * 0.3) * streak * (0.8 + 0.2 * ao);

  // sharper Blinn lobe from the key
  col += blinnLobe(n, v, key, 140.0) * U.keyColor.rgb * U.keyDir.w * 0.16 * shadow;

  let f = mix(0.6, 1.0, pow(1.0 - nv, 4.0));
  let r = reflect(-v, n);
  // blur the reflection a little along the circumference (brushed metal)
  let rb = normalize(r + tangent * (brush - 0.5) * 0.25);
  col += envReflection(rb) * f * albedo * 1.1 * (0.5 + 0.5 * ao) * shadow;
  return col;
}

fn shadeRollerCap(p: vec3f, n: vec3f, v: vec3f, pose: vec4f) -> vec3f {
  let dy = p.y - pose.x;
  let dz = p.z - pose.y;
  let rr = length(vec2f(dy, dz));
  let rings = 0.86 + 0.14 * vnoise(vec2f(rr * 240.0, 0.5));
  let hub = 1.0 - 0.35 * smoothstep(0.09, 0.07, rr); // a darker axle stub in the middle
  // six bolt heads on a ring that turn with the roll (a rotation cue on the end faces)
  let phi = atan2(dy, dz) + pose.w;
  let sector = fract(phi * 6.0 / (2.0 * PI));
  let boltR = pose.z * 0.55;
  let bolt = vec2f((sector - 0.5) * (2.0 * PI * boltR / 6.0), rr - boltR);
  let boltMask = smoothstep(0.024, 0.016, length(bolt));
  let boltShade = 1.0 - 0.45 * boltMask + 0.25 * boltMask * smoothstep(0.012, 0.0, length(bolt));
  let albedo = vec3f(0.40, 0.41, 0.44) * rings * hub * boltShade;
  let key = U.keyDir.xyz;
  var col = albedo * lightDiffuse(n, 1.0, 0.2) * 0.5;
  col += blinnLobe(n, v, key, 60.0) * U.keyColor.rgb * U.keyDir.w * 0.35;
  let nv = saturate(dot(n, v));
  let f = mix(0.4, 1.0, pow(1.0 - nv, 4.0));
  col += envReflection(reflect(-v, n)) * f * 0.35 * albedo * 1.5;
  return col;
}

fn shadeFloor(p: vec3f, rd: vec3f) -> vec3f {
  let n = vec3f(0.0, 1.0, 0.0);
  let v = -rd;
  let key = U.keyDir.xyz;
  let L = U.domain.w;
  // machined tray with a subtle grain
  let grain = vnoise(vec2f(p.x * 90.0, p.z * 90.0)) * 0.5 + vnoise(vec2f(p.x * 22.0, p.z * 22.0)) * 0.5;
  var albedo = vec3f(0.11, 0.112, 0.12) * (0.85 + 0.3 * grain);
  // a lighter tray plate under the mill
  let plate = smoothstep(0.02, 0.0, abs(p.x - L * 0.5) - (L * 0.5 + 0.35)) * smoothstep(0.02, 0.0, abs(p.z - U.rollerFront.y) - 0.95);
  albedo = mix(albedo, vec3f(0.17, 0.172, 0.18) * (0.85 + 0.3 * grain), plate);

  // soft key shadow from the rollers + contact darkening under them
  var shadow = rollerShadow(p, key, U.rollerBack) * rollerShadow(p, key, U.rollerFront);
  let inX = smoothstep(-0.15, 0.05, p.x) * (1.0 - smoothstep(L - 0.05, L + 0.15, p.x));
  let dzB = (p.z - U.rollerBack.y) / (U.rollerBack.z * 1.35);
  let dzF = (p.z - U.rollerFront.y) / (U.rollerFront.z * 1.35);
  let contact = 1.0 - 0.75 * inX * max(exp(-dzB * dzB * 2.2), exp(-dzF * dzF * 2.2));
  shadow *= contact;

  var col = albedo * lightDiffuse(n, shadow, 0.0) * contact;
  // glossy tray: reflect the rollers and the environment
  let r = reflect(rd, n);
  let hb = hitRoller(p + n * 1e-4, r, U.rollerBack);
  let hf = hitRoller(p + n * 1e-4, r, U.rollerFront);
  var refl = envReflection(r) * 0.5;
  if (min(hb.t, hf.t) < INF) {
    refl = vec3f(0.22, 0.225, 0.24) * lightDiffuse(vec3f(0.0, -1.0, 0.0), 1.0, 0.5) + vec3f(0.08);
  }
  let f = fresnelSchlick(saturate(dot(n, v)), 0.03);
  col += refl * (f * 6.0 + 0.04) * 0.5;
  col += blinnLobe(n, v, key, 24.0) * U.keyColor.rgb * U.keyDir.w * 0.08 * shadow;
  return col;
}

// ---------------------------------------------------------------- end-guide plates

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
  if (tn < 0.0) { return out; } // inside the plate: ignore
  out.t = tn;
  if (tn == tmin.x) { out.n = vec3f(-sign(rd.x), 0.0, 0.0); }
  else if (tn == tmin.y) { out.n = vec3f(0.0, -sign(rd.y), 0.0); }
  else { out.n = vec3f(0.0, 0.0, -sign(rd.z)); }
  return out;
}

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
  let f = fresnelSchlick(nv, 0.04);
  // milky acrylic: mostly what is behind, a little diffuse haze, a glossy reflection at the edges
  let haze = vec3f(0.62, 0.64, 0.66) * lightDiffuse(n, 1.0, 0.5) * 0.28;
  let refl = envReflection(reflect(rd, n)) * (f * 4.0 + 0.05) * 0.5;
  let spec = blinnLobe(n, v, U.keyDir.xyz, 90.0) * U.keyColor.rgb * U.keyDir.w * 0.2;
  // thin edges (y/z faces) read as solid rims
  let edge = 1.0 - abs(n.x);
  let alpha = mix(0.3 + 0.5 * f, 0.85, edge);
  return mix(col, haze + refl + spec, alpha) + spec * 0.3;
}

// ACES-fitted tone curve (Narkowicz), input linear, output linear 0..1
fn tonemap(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

// ---------------------------------------------------------------- fragment

@fragment
fn fsMain(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let res = U.misc.xy;
  let ndc = vec2f(fragCoord.x / res.x * 2.0 - 1.0, 1.0 - fragCoord.y / res.y * 2.0);
  let tanHalf = U.camOrigin.w;
  let aspect = U.camRight.w;
  let ro = U.camOrigin.xyz;
  let rd = normalize(U.camForward.xyz + U.camRight.xyz * (ndc.x * tanHalf * aspect) + U.camUp.xyz * (ndc.y * tanHalf));
  let flags = u32(U.misc.w);
  let h = U.camForward.w;

  // analytic scene
  let hb = hitRoller(ro, rd, U.rollerBack);
  let hf = hitRoller(ro, rd, U.rollerFront);
  var roller = hb;
  var rollerPose = U.rollerBack;
  if (hf.t < hb.t) { roller = hf; rollerPose = U.rollerFront; }
  var tFloor = INF;
  if (rd.y < -1e-6) { tFloor = -ro.y / rd.y; }
  let tScene = min(roller.t, tFloor);

  // volume
  var tVol = -1.0;
  if ((flags & 1u) != 0u) {
    let box = hitBox(ro, rd);
    if (box.x < box.y && box.x < tScene) {
      tVol = marchVolume(ro, rd, box.x, min(box.y, tScene));
    }
  }

  var col: vec3f;
  var tHit = INF;
  let v = -rd;
  if (tVol > 0.0) {
    tHit = tVol;
    let p = ro + rd * tVol;
    let sn = densityNormal(p);
    let n = sn.n;
    // colour: integrate the latent through the material behind the hit (uncured
    // silicone is translucent, and milled pigment often sits one cell under a white
    // skin), weighting samples by density and by depth so the surface dominates.
    // Latents are renormalised by their pigment-weight sum because trilinear
    // blending with empty cells scales them toward zero.
    var c = vec4f(0.0);
    var resid = vec3f(0.0);
    var wsum = 0.0;
    for (var k = 0; k < 5; k++) {
      let depth = (0.5 + 0.8 * f32(k)) * h;
      let q = p - n * depth;
      let a = textureSampleLevel(volA, volSampler, volUvw(q), 0.0);
      let b = textureSampleLevel(volB, volSampler, volUvw(q), 0.0);
      let s = a.y + a.z + a.w + b.x;
      if (s <= 1e-4) { continue; }
      let w = smoothstep(0.15, 0.6, a.x) * exp(-0.45 * f32(k));
      c += w * vec4f(a.yzw, b.x) / s;
      resid += w * b.yzw / s;
      wsum += w;
    }
    if (wsum > 1e-5) { c = c / wsum; resid = resid / wsum; } else { c = vec4f(0.0, 0.0, 0.0, 1.0); }
    let albedo = srgbToLinear(latentToRgb(c, resid));
    let ao = volumeAo(p, n);
    let thickness = sheetThickness(p, n);
    // clear base: little diffuse (a clear material has almost no body colour), the
    // look comes from specular and what shows through; pigment restores albedo
    let pigmentDiffuse = saturate((1.0 - c.w) * 1.6);
    let surface = shadePutty(p, n, v, mix(vec3f(0.58, 0.62, 0.64), albedo, pigmentDiffuse), ao, thickness, sn.rough);
    // Uncured silicone is clear, not white: the base (titanium-white latent,
    // white weight c.w ~ 1) is rendered translucent, going milky with thickness,
    // while pigment (white weight falls) makes it opaque. What shows through is
    // the analytic scene behind the hit along the same ray.
    let pigment = saturate((1.0 - c.w) * 1.6);
    var behind: vec3f;
    if (roller.t < tFloor) {
      let pb = ro + rd * roller.t;
      if (roller.kind == 1) { behind = shadeRollerBody(pb, roller.n, v, rollerPose); } else { behind = shadeRollerCap(pb, roller.n, v, rollerPose); }
    } else if (tFloor < INF) {
      behind = shadeFloor(ro + rd * tFloor, rd);
    } else {
      behind = skyColor(rd);
    }
    let clearness = (1.0 - pigment) * exp(-thickness / (14.0 * h));  // clear sheet / bank: see through, milky when thick
    let alpha = 1.0 - 0.8 * clearness;                                 // base alone: up to 80% shows through
    let tint = mix(vec3f(0.84, 0.89, 0.92), albedo, pigment);         // transmitted light picks up the pigment
    col = mix(behind * tint, surface, alpha);
  } else if (roller.t < tFloor) {
    tHit = roller.t;
    let p = ro + rd * roller.t;
    if (roller.kind == 1) {
      col = shadeRollerBody(p, roller.n, v, rollerPose);
    } else {
      col = shadeRollerCap(p, roller.n, v, rollerPose);
    }
  } else if (tFloor < INF) {
    tHit = tFloor;
    col = shadeFloor(ro + rd * tFloor, rd);
  } else {
    col = skyColor(rd);
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
    let fog = 1.0 - exp(-max(tHit - 2.0, 0.0) * 0.32);
    col = mix(col, skyColor(rd), fog);
  }

  // vignette, exposure, tone curve, encode
  let vig = 1.0 - 0.22 * dot(ndc * vec2f(0.9, 1.0), ndc * vec2f(0.9, 1.0));
  col = tonemap(col * U.misc.z * vig);
  if ((flags & 2u) == 0u) { col = linearToSrgb(col); }
  return vec4f(col, 1.0);
}

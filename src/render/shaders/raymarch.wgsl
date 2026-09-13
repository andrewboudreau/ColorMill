// ColorMill v2 — fullscreen ray-march of the mill (design §8).
// mixbox.wgsl (latentToRgb) is prepended by the renderer.
//
// One triangle covers the screen. Per pixel: build a camera ray, intersect the
// analytic scene (two capped roller cylinders, the tray plane), then march the
// density volume between the domain-box entry and the nearest analytic hit.
// The first sample with density >= iso is refined by bisection; the normal is
// the density gradient; colour is the Mixbox latent decoded at the hit.

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
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var volA: texture_3d<f32>;
@group(0) @binding(2) var volB: texture_3d<f32>;
@group(0) @binding(3) var volSampler: sampler;

const PI: f32 = 3.14159265358979;
const INF: f32 = 1e30;
const MAX_STEPS: i32 = 420;

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

fn densityNormal(p: vec3f) -> vec3f {
  let e = U.camForward.w;
  let ex = vec3f(e, 0.0, 0.0);
  let ey = vec3f(0.0, e, 0.0);
  let ez = vec3f(0.0, 0.0, e);
  let g = vec3f(
    density(p + ex) - density(p - ex),
    density(p + ey) - density(p - ey),
    density(p + ez) - density(p - ez)
  );
  let len = length(g);
  if (len < 1e-6) { return vec3f(0.0, 1.0, 0.0); }
  return -g / len;
}

// March from tStart to tEnd; returns the hit distance or -1.
fn marchVolume(ro: vec3f, rd: vec3f, tStart: f32, tEnd: f32) -> f32 {
  let h = U.camForward.w;
  let iso = U.dims.w;
  let fine = 0.6 * h;
  let coarse = 1.3 * h;
  var tPrev = tStart;
  var dPrev = density(ro + rd * tPrev);
  if (dPrev >= iso) { return tPrev; }
  var t = tPrev;
  for (var i = 0; i < MAX_STEPS; i++) {
    // empty space: bigger steps (a sheet is >= 3 cells thick, so 1.3h cannot skip it)
    let step = select(fine, coarse, dPrev < 0.015);
    t = tPrev + step;
    if (t > tEnd) { break; }
    let d = density(ro + rd * t);
    if (d >= iso) {
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
    dPrev = d;
    tPrev = t;
  }
  return -1.0;
}

// Transmittance toward the light through the density volume (short march).
fn volumeShadow(p: vec3f, l: vec3f) -> f32 {
  let h = U.camForward.w;
  let step = max(1.6 * h, 0.028);
  var sum = 0.0;
  var t = step * 1.2;
  for (var i = 0; i < 8; i++) {
    let q = p + l * t;
    if (any(q < vec3f(0.0)) || any(q > U.domain.xyz)) { break; }
    sum += density(q);
    t += step;
  }
  return exp(-sum * step * 22.0);
}

// Cheap AO: density 2 and 4 texels along the normal (creases and the nip get dark).
fn volumeAo(p: vec3f, n: vec3f) -> f32 {
  let h = U.camForward.w;
  let d2 = density(p + n * 2.0 * h);
  let d4 = density(p + n * 4.0 * h);
  let d6 = density(p + n * 6.5 * h);
  return saturate(1.0 - 0.55 * d2 - 0.35 * d4 - 0.2 * d6);
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

fn shadePutty(p: vec3f, n: vec3f, v: vec3f, albedo: vec3f, ao: f32, thin: f32) -> vec3f {
  let key = U.keyDir.xyz;
  var shadow = rollerShadow(p, key, U.rollerBack) * rollerShadow(p, key, U.rollerFront);
  shadow *= mix(0.35, 1.0, volumeShadow(p, key));
  let nv = saturate(dot(n, v));

  // diffuse with a little wrap: uncured silicone scatters light in its top layer
  var col = albedo * lightDiffuse(n, shadow, 0.25) * ao;

  // sub-surface warmth where the material is thin (sheet edges, the sheet on the roll)
  let warm = albedo * vec3f(1.0, 0.72, 0.5) * thin * 0.22 * saturate(dot(n, key) + 0.4) * shadow;
  col += warm;

  // glossy reflection: wide + narrow Blinn-Phong lobes, Fresnel-weighted, plus the environment
  let f = fresnelSchlick(nv, 0.03);
  let wide = blinnLobe(n, v, key, 32.0) * U.keyColor.rgb * U.keyDir.w * 0.30
           + blinnLobe(n, v, U.fillDir.xyz, 24.0) * U.fillColor.rgb * U.fillDir.w * 0.30
           + blinnLobe(n, v, U.rimDir.xyz, 24.0) * U.rimColor.rgb * U.rimDir.w * 0.45;
  let narrow = blinnLobe(n, v, key, 520.0) * U.keyColor.rgb * U.keyDir.w * 0.9
             + blinnLobe(n, v, U.fillDir.xyz, 260.0) * U.fillColor.rgb * U.fillDir.w * 0.35
             + blinnLobe(n, v, U.rimDir.xyz, 300.0) * U.rimColor.rgb * U.rimDir.w * 0.4;
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
  // isotropic speckle and faint spiral marks so the rotation reads
  let speck = vnoise(vec2f(p.x * 42.0, circ * 42.0));
  let spiral = smoothstep(0.90, 1.0, sin(theta * 22.0 + p.x * 9.0 + speck * 2.0)) * 0.09;
  let wear = vnoise(vec2f(p.x * 3.0, circ * 2.4)) * 0.12;
  var albedo = vec3f(0.52, 0.53, 0.56) * (0.86 + 0.16 * brush + 0.08 * speck + wear) + vec3f(spiral);

  let key = U.keyDir.xyz;
  let shadow = mix(0.3, 1.0, volumeShadow(p, key));
  let nv = saturate(dot(n, v));

  // low diffuse, strong anisotropic highlight stretched along the axis (Kajiya-Kay with a
  // circumferential tangent), plus a Fresnel-weighted environment reflection
  let tangent = normalize(cross(vec3f(1.0, 0.0, 0.0), n));
  var col = albedo * lightDiffuse(n, shadow, 0.1) * 0.42;

  let hk = normalize(key + v);
  let th = dot(tangent, hk);
  let anisoK = pow(sqrt(max(1.0 - th * th, 0.0)), 90.0) * saturate(dot(n, key) * 2.0);
  let hf = normalize(U.fillDir.xyz + v);
  let tf = dot(tangent, hf);
  let anisoF = pow(sqrt(max(1.0 - tf * tf, 0.0)), 40.0) * saturate(dot(n, U.fillDir.xyz) * 2.0);
  let hr = normalize(U.rimDir.xyz + v);
  let tr = dot(tangent, hr);
  let anisoR = pow(sqrt(max(1.0 - tr * tr, 0.0)), 60.0) * saturate(dot(n, U.rimDir.xyz) * 2.0);
  let streak = 0.75 + 0.25 * brush;
  col += (U.keyColor.rgb * U.keyDir.w * anisoK * 0.9 * shadow + U.fillColor.rgb * U.fillDir.w * anisoF * 0.35 + U.rimColor.rgb * U.rimDir.w * anisoR * 0.5) * streak;

  // sharper Blinn lobe from the key
  col += blinnLobe(n, v, key, 140.0) * U.keyColor.rgb * U.keyDir.w * 0.35 * shadow;

  let f = mix(0.55, 1.0, pow(1.0 - nv, 4.0));
  let r = reflect(-v, n);
  // blur the reflection a little along the circumference (brushed metal)
  let rb = normalize(r + tangent * (brush - 0.5) * 0.25);
  col += envReflection(rb) * f * 0.55 * albedo * 1.7;
  return col;
}

fn shadeRollerCap(p: vec3f, n: vec3f, v: vec3f, pose: vec4f) -> vec3f {
  let rr = length(vec2f(p.y - pose.x, p.z - pose.y));
  let rings = 0.86 + 0.14 * vnoise(vec2f(rr * 240.0, 0.5));
  let hub = 1.0 - 0.35 * smoothstep(0.09, 0.07, rr); // a darker axle stub in the middle
  let albedo = vec3f(0.40, 0.41, 0.44) * rings * hub;
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
    let n = densityNormal(p);
    // sample the latent slightly inside the surface and renormalise the pigment weights
    // (trilinear blending with empty cells scales the latent toward zero)
    let q = p - n * 0.75 * h;
    let a = textureSampleLevel(volA, volSampler, volUvw(q), 0.0);
    let b = textureSampleLevel(volB, volSampler, volUvw(q), 0.0);
    var c = vec4f(a.yzw, b.x);
    var resid = b.yzw;
    let s = c.x + c.y + c.z + c.w;
    if (s > 1e-4) { c = c / s; resid = resid / s; }
    let albedo = srgbToLinear(latentToRgb(c, resid));
    let ao = volumeAo(p, n);
    let thin = 1.0 - density(p - n * 2.5 * h);
    col = shadePutty(p, n, v, albedo, ao, thin);
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

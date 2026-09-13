// Mixbox latent -> RGB (port of src/sim/mixbox.c: EvalPolynomial + residual + clamp).
// A latent is 7 floats: c0..c3 (Kubelka-Munk pigment weights, sum to 1) and an
// RGB residual. Mixing is a mass-weighted average of latents; this decodes.
// Mixbox is (c) Secret Weapons, CC BY-NC 4.0 (non-commercial use).

fn mixboxEvalPolynomial(c0: f32, c1: f32, c2: f32, c3: f32) -> vec3f {
  let c00 = c0 * c0;
  let c11 = c1 * c1;
  let c22 = c2 * c2;
  let c33 = c3 * c3;
  let c01 = c0 * c1;
  let c02 = c0 * c2;
  let c12 = c1 * c2;

  var rgb = vec3f(0.0);
  var w: f32;
  w = c0 * c00;  rgb += vec3f(0.07717053, 0.02826978, 0.24832992) * w;
  w = c1 * c11;  rgb += vec3f(0.95912302, 0.80256528, 0.03561839) * w;
  w = c2 * c22;  rgb += vec3f(0.74683774, 0.04868586, 0.00000000) * w;
  w = c3 * c33;  rgb += vec3f(0.99518138, 0.99978149, 0.99704802) * w;
  w = c00 * c1;  rgb += vec3f(0.04819146, 0.83363781, 0.32515377) * w;
  w = c01 * c1;  rgb += vec3f(-0.68146950, 1.46107803, 1.06980936) * w;
  w = c00 * c2;  rgb += vec3f(0.27058419, -0.15324870, 1.98735057) * w;
  w = c02 * c2;  rgb += vec3f(0.80478189, 0.67093710, 0.18424500) * w;
  w = c00 * c3;  rgb += vec3f(-0.35031003, 1.37855826, 3.68865000) * w;
  w = c0 * c33;  rgb += vec3f(1.05128046, 1.97815239, 2.82989073) * w;
  w = c11 * c2;  rgb += vec3f(3.21607125, 0.81270228, 1.03384539) * w;
  w = c1 * c22;  rgb += vec3f(2.78893374, 0.41565549, -0.04487295) * w;
  w = c11 * c3;  rgb += vec3f(3.02162577, 2.55374103, 0.32766114) * w;
  w = c1 * c33;  rgb += vec3f(2.95124691, 2.81201112, 1.17578442) * w;
  w = c22 * c3;  rgb += vec3f(2.82677043, 0.79933038, 1.81715262) * w;
  w = c2 * c33;  rgb += vec3f(2.99691099, 1.22593053, 1.80653661) * w;
  w = c01 * c2;  rgb += vec3f(1.87394106, 2.05027182, -0.29835996) * w;
  w = c01 * c3;  rgb += vec3f(2.56609566, 7.03428198, 0.62575374) * w;
  w = c02 * c3;  rgb += vec3f(4.08329484, -1.40408358, 2.14995522) * w;
  w = c12 * c3;  rgb += vec3f(6.00078678, 2.55552042, 1.90739502) * w;
  return rgb;
}

/// Decode a latent (c0..c3 in `c`, residual rgb in `res`) to sRGB in 0..1.
fn latentToRgb(c: vec4f, res: vec3f) -> vec3f {
  let rgb = mixboxEvalPolynomial(c.x, c.y, c.z, c.w) + res;
  return clamp(rgb, vec3f(0.0), vec3f(1.0));
}

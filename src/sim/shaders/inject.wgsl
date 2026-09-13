// Inject (design §5): blend a pigment latent into particles inside a sphere,
// z = mix(z, zPigment, strength * t): a solid core (t = 1 for |d| < 0.6 r)
// with a linear falloff to the edge, so a tap reads as a dollop of
// concentrated masterbatch rather than a faint tint.
// mode 1 = set every particle (clearPigment / reset base colour).
struct Inject {
  // center xyz, radius
  center : vec4<f32>,
  lat0 : vec4<f32>,
  // lat4, lat5, lat6, strength
  lat1 : vec4<f32>,
  // mode, pad
  mode : vec4<u32>,
};
@group(0) @binding(1) var<uniform> I : Inject;
@group(0) @binding(2) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> lat : array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  var zp : array<f32, 7>;
  zp[0] = I.lat0.x; zp[1] = I.lat0.y; zp[2] = I.lat0.z; zp[3] = I.lat0.w;
  zp[4] = I.lat1.x; zp[5] = I.lat1.y; zp[6] = I.lat1.z;
  var a = 1.0;
  if (I.mode.x == 0u) {
    let x = pos[p].xyz;
    let d = length(x - I.center.xyz);
    if (d >= I.center.w) { return; }
    let t = clamp((1.0 - d / I.center.w) / 0.4, 0.0, 1.0);
    a = clamp(I.lat1.w * t, 0.0, 1.0);
  }
  for (var c = 0u; c < 7u; c++) {
    let z = lat[7u * p + c];
    lat[7u * p + c] = z + a * (zp[c] - z);
  }
}

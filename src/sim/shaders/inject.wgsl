// Inject (design §5): blend a pigment latent into particles inside a sphere,
// z = mix(z, zPigment, strength * t): a solid core (t = 1 for |d| < 0.6 r)
// with a linear falloff to the edge, so a tap reads as a dollop of
// concentrated masterbatch rather than a faint tint.
// mode 1 = set every particle (clearPigment / reset base colour).
// mode 2 = "surface" injection: the sphere is centred just below the highest
// particle found by `probe` in the column |x - cx| < r, |z - cz| < r, so a tap
// always lands on the material wherever the bank has slumped or drained to.
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
@group(0) @binding(2) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> lat : array<f32>;
// probe[0] = float bits of the highest y in the column (0 = nothing found)
@group(0) @binding(4) var<storage, read_write> probe : array<atomic<u32>>;

@compute @workgroup_size(128)
fn probeColumn(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let x = pos[p].xyz;
  let r = I.center.w;
  if (abs(x.x - I.center.x) < r && abs(x.z - I.center.z) < r) {
    // y > 0 so the float bit pattern is monotonic and atomicMax works
    atomicMax(&probe[0], bitcast<u32>(max(x.y, 0.0)));
  }
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  var zp : array<f32, 7>;
  zp[0] = I.lat0.x; zp[1] = I.lat0.y; zp[2] = I.lat0.z; zp[3] = I.lat0.w;
  zp[4] = I.lat1.x; zp[5] = I.lat1.y; zp[6] = I.lat1.z;
  var a = 1.0;
  if (I.mode.x != 1u) {
    var c = I.center.xyz;
    if (I.mode.x == 2u) {
      let top = bitcast<f32>(atomicLoad(&probe[0]));
      if (top > 0.0) { c.y = top - 0.5 * I.center.w; }
    }
    let x = pos[p].xyz;
    let d = length(x - c);
    if (d >= I.center.w) { return; }
    let t = clamp((1.0 - d / I.center.w) / 0.4, 0.0, 1.0);
    a = clamp(I.lat1.w * t, 0.0, 1.0);
  }
  for (var c = 0u; c < 7u; c++) {
    let z = lat[7u * p + c];
    lat[7u * p + c] = z + a * (zp[c] - z);
  }
  // pigment load: the clear base carries none (set-all mode); tinted particles move toward a chunk's load
  if (I.mode.x == 1u) {
    pos[p].w = 0.0;
  } else {
    let w = pos[p].w;
    pos[p].w = w + a * (PIGMENT_LOAD - w);
  }
}

// Operator cut & fold (design §6): a scripted kinematic move of the visible
// sheet on the front roller onto the bank, shifted by +0.25 L along x.
//   select: mark particles (x < 0.75 L, d_front < R + 3h, z > zNip) kinematic and store p0.
//   move:   each substep while active: p(s) = lerp(p0, p1, s) + up * sin(pi s) * lift,
//           v = dp/ds * ds/dt (P.fold = active, s, ds/dt, lift).
//   finish: clear the flag, x = p1, v = 0, C = 0, F = I, stress = 0.
@group(0) @binding(1) var<storage, read_write> pos : array<f32>;
@group(0) @binding(2) var<storage, read_write> vel : array<f32>;
@group(0) @binding(3) var<storage, read_write> cbuf : array<f32>;
@group(0) @binding(4) var<storage, read_write> fbuf : array<f32>;
@group(0) @binding(5) var<storage, read_write> sbuf : array<f32>;
@group(0) @binding(6) var<storage, read_write> flags : array<u32>;
@group(0) @binding(7) var<storage, read_write> fold0 : array<f32>;   // p0 xyz + pad per particle

fn foldTarget(p0 : vec3<f32>) -> vec3<f32> {
  let h = P.hdt.x;
  let L = P.fold2.y;
  let x1 = clamp(p0.x + P.fold2.z, 1.5 * h, L - 1.5 * h);
  return vec3<f32>(x1, P.fold2.x, P.fold2.w);
}

@compute @workgroup_size(128)
fn select_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let h = P.hdt.x;
  let x = vec3<f32>(pos[3u * p], pos[3u * p + 1u], pos[3u * p + 2u]);
  let R = P.front.w;
  let d = rollerDist(P.front.x, P.front.y, x);
  if (x.x < 0.75 * P.fold2.y && d < R + 3.0 * h && x.z > P.fold2.w) {
    flags[p] = flags[p] | 1u;
    fold0[4u * p] = x.x; fold0[4u * p + 1u] = x.y; fold0[4u * p + 2u] = x.z; fold0[4u * p + 3u] = 0.0;
  }
}

@compute @workgroup_size(128)
fn move_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) == 0u) { return; }
  let p0 = vec3<f32>(fold0[4u * p], fold0[4u * p + 1u], fold0[4u * p + 2u]);
  let p1 = foldTarget(p0);
  let s = P.fold.y;
  let dsdt = P.fold.z;
  let lift = P.fold.w;
  let up = vec3<f32>(0.0, 1.0, 0.0);
  let x = mix(p0, p1, s) + up * (sin(PI * s) * lift);
  let v = ((p1 - p0) + up * (PI * cos(PI * s) * lift)) * dsdt;
  let lo = vec3<f32>(1.5 * P.hdt.x);
  let xc = clamp(x, lo, P.domain.xyz - lo);
  pos[3u * p] = xc.x; pos[3u * p + 1u] = xc.y; pos[3u * p + 2u] = xc.z;
  vel[3u * p] = v.x; vel[3u * p + 1u] = v.y; vel[3u * p + 2u] = v.z;
}

@compute @workgroup_size(128)
fn finish_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) == 0u) { return; }
  let p0 = vec3<f32>(fold0[4u * p], fold0[4u * p + 1u], fold0[4u * p + 2u]);
  let p1 = foldTarget(p0);
  let lo = vec3<f32>(1.5 * P.hdt.x);
  let xc = clamp(p1, lo, P.domain.xyz - lo);
  pos[3u * p] = xc.x; pos[3u * p + 1u] = xc.y; pos[3u * p + 2u] = xc.z;
  for (var c = 0u; c < 3u; c++) { vel[3u * p + c] = 0.0; }
  for (var c = 0u; c < 9u; c++) {
    cbuf[9u * p + c] = 0.0;
    sbuf[9u * p + c] = 0.0;
    fbuf[9u * p + c] = select(0.0, 1.0, (c % 4u) == 0u);
  }
  flags[p] = flags[p] & ~1u;
}

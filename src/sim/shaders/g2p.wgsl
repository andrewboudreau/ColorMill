// G2P (design §3.4): gather velocity + APIC C, update F, plastic return,
// stress for the next P2G, advection, clamps, hard roller push-out.
@group(0) @binding(1) var<storage, read_write> pos : array<f32>;
@group(0) @binding(2) var<storage, read_write> vel : array<f32>;
@group(0) @binding(3) var<storage, read_write> cbuf : array<f32>;
@group(0) @binding(4) var<storage, read_write> fbuf : array<f32>;
@group(0) @binding(5) var<storage, read_write> sbuf : array<f32>;
@group(0) @binding(6) var<storage, read> flags : array<u32>;
@group(0) @binding(7) var<storage, read> gvel : array<f32>;

fn loadMatF(p : u32) -> mat3x3<f32> {
  let b = p * 9u;
  return mat3x3<f32>(
    vec3<f32>(fbuf[b], fbuf[b + 3u], fbuf[b + 6u]),
    vec3<f32>(fbuf[b + 1u], fbuf[b + 4u], fbuf[b + 7u]),
    vec3<f32>(fbuf[b + 2u], fbuf[b + 5u], fbuf[b + 8u]));
}
fn storeMatC(p : u32, m : mat3x3<f32>) {
  let b = p * 9u;
  cbuf[b] = m[0][0]; cbuf[b + 1u] = m[1][0]; cbuf[b + 2u] = m[2][0];
  cbuf[b + 3u] = m[0][1]; cbuf[b + 4u] = m[1][1]; cbuf[b + 5u] = m[2][1];
  cbuf[b + 6u] = m[0][2]; cbuf[b + 7u] = m[1][2]; cbuf[b + 8u] = m[2][2];
}
fn storeMatF(p : u32, m : mat3x3<f32>) {
  let b = p * 9u;
  fbuf[b] = m[0][0]; fbuf[b + 1u] = m[1][0]; fbuf[b + 2u] = m[2][0];
  fbuf[b + 3u] = m[0][1]; fbuf[b + 4u] = m[1][1]; fbuf[b + 5u] = m[2][1];
  fbuf[b + 6u] = m[0][2]; fbuf[b + 7u] = m[1][2]; fbuf[b + 8u] = m[2][2];
}
fn storeMatS(p : u32, m : mat3x3<f32>) {
  let b = p * 9u;
  sbuf[b] = m[0][0]; sbuf[b + 1u] = m[1][0]; sbuf[b + 2u] = m[2][0];
  sbuf[b + 3u] = m[0][1]; sbuf[b + 4u] = m[1][1]; sbuf[b + 5u] = m[2][1];
  sbuf[b + 6u] = m[0][2]; sbuf[b + 7u] = m[1][2]; sbuf[b + 8u] = m[2][2];
}

fn isFinite3(v : vec3<f32>) -> bool {
  let a = abs(v);
  return all(a < vec3<f32>(1e30)) && all(v == v);
}

fn pushOut(axisY : f32, axisZ : f32, R : f32, x : vec3<f32>, h : f32) -> vec3<f32> {
  let d = rollerDist(axisY, axisZ, x);
  if (d < R) {
    let nrm = rollerNormal(axisY, axisZ, x);
    return vec3<f32>(x.x, axisY + nrm.y * (R + 0.25 * h), axisZ + nrm.z * (R + 0.25 * h));
  }
  return x;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) != 0u) { return; }   // kinematic: moved by the fold kernel

  let h = P.hdt.x;
  let invh = P.hdt.y;
  let dt = P.hdt.z;

  var x = vec3<f32>(pos[3u * p], pos[3u * p + 1u], pos[3u * p + 2u]);
  let gx = x * invh;
  let base = vec3<i32>(floor(gx - 0.5));
  let fx = gx - vec3<f32>(base);
  let wx = quadWeights(fx.x);
  let wy = quadWeights(fx.y);
  let wz = quadWeights(fx.z);

  var v = vec3<f32>(0.0);
  var B = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var i = 0; i < 3; i++) {
        let off = vec3<i32>(i, j, k);
        let dpos = (vec3<f32>(off) - fx) * h;
        let wgt = wx[i] * wy[j] * wz[k];
        let n = nodeIndexI(base + off);
        let gv = vec3<f32>(gvel[3u * n], gvel[3u * n + 1u], gvel[3u * n + 2u]);
        v += wgt * gv;
        // outer(gv, dpos): column c = gv * dpos[c]  (B[r][c] = gv_r * dpos_c)
        B += wgt * mat3x3<f32>(gv * dpos.x, gv * dpos.y, gv * dpos.z);
      }
    }
  }
  let C = (4.0 * invh * invh) * B;

  // deformation gradient update + plastic return (design §4)
  var F = (identity3() + dt * C) * loadMatF(p);
  var J = det3(F);
  if (!(J > 1e-6) || !(J < 1e6) || !isFinite3(F[0]) || !isFinite3(F[1]) || !isFinite3(F[2])) {
    F = identity3();
  }
  var d = svd3(F);
  d.S = clamp(d.S, vec3<f32>(1.0 - P.mat.z), vec3<f32>(1.0 + P.mat.w));
  let Sig = mat3x3<f32>(vec3<f32>(d.S.x, 0.0, 0.0), vec3<f32>(0.0, d.S.y, 0.0), vec3<f32>(0.0, 0.0, d.S.z));
  F = d.U * Sig * transpose(d.V);
  let Rot = d.U * transpose(d.V);
  J = d.S.x * d.S.y * d.S.z;
  let sigma = corotatedStress(F, Rot, J, P.mat.x, P.mat.y);

  // advection, clamp, hard push-out of the rollers
  if (!isFinite3(v)) { v = vec3<f32>(0.0); }
  x += dt * v;
  let lo = vec3<f32>(1.5 * h);
  let hi = P.domain.xyz - lo;
  x = clamp(x, lo, hi);
  let R = P.front.w;
  x = pushOut(P.front.x, P.front.y, R, x, h);
  x = pushOut(P.back.x, P.back.y, R, x, h);
  x = clamp(x, lo, hi);

  pos[3u * p] = x.x; pos[3u * p + 1u] = x.y; pos[3u * p + 2u] = x.z;
  vel[3u * p] = v.x; vel[3u * p + 1u] = v.y; vel[3u * p + 2u] = v.z;
  storeMatC(p, C);
  storeMatF(p, F);
  storeMatS(p, sigma);
}

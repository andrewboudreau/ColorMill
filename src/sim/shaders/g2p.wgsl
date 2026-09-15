// G2P (design §3.4): gather velocity + APIC C, update F, plastic return,
// the premultiplied P2G affine matrix for the next substep, advection,
// clamps, hard roller push-out.
@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cbuf : array<f32>;
@group(0) @binding(4) var<storage, read_write> fbuf : array<f32>;
@group(0) @binding(5) var<storage, read_write> abuf : array<f32>;   // P2G affine (read by p2g)
@group(0) @binding(6) var<storage, read> flags : array<u32>;
@group(0) @binding(7) var<storage, read> gvel : array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> gmass : array<i32>;
// cut & fold reductions: [0] = live pile top under the standing log (atomicMax on float bits)
@group(0) @binding(9) var<storage, read_write> foldInfo : array<atomic<u32>>;
@group(0) @binding(10) var<storage, read> fold0 : array<vec4<f32>>;   // .w = release time of a fed particle

const FOLD_SETTLE : f32 = 0.5;   // s: material just let go of the log does not count as pile yet

fn loadMatF(p : u32) -> mat3x3<f32> {
  let b = p * 9u;
  return mat3x3<f32>(
    vec3<f32>(fbuf[b], fbuf[b + 3u], fbuf[b + 6u]),
    vec3<f32>(fbuf[b + 1u], fbuf[b + 4u], fbuf[b + 7u]),
    vec3<f32>(fbuf[b + 2u], fbuf[b + 5u], fbuf[b + 8u]));
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

  let pw = pos[p].w;
  var x = pos[p].xyz;

  // while the operator move feeds the log: reduce the top of the (non-stray) material
  // under its footprint, so the log releases onto the pile the nip is eating
  if (P.fold.x > 0.5 && P.fold.y - fold0[p].w >= FOLD_SETTLE
      && abs(x.z - P.fold2.w) < P.fold3.x && abs(x.x - 0.5 * P.fold2.y) < 0.35 && x.y > P.front.x) {
    let cn = clamp(vec3<i32>(round(x * invh)), vec3<i32>(0), vec3<i32>(P.grid.xyz) - vec3<i32>(1));
    if (decodeFixed(gmass[nodeIndexI(cn)], MASS_SCALE) >= 2.0 * P.part.y) {
      atomicMax(&foldInfo[0], bitcast<u32>(x.y));
    }
  }
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
        let gv = gvel[nodeIndexI(base + off)].xyz;
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
  if (isBad(J) || J < 1e-6 || J > 1e6 || isBadMat(F)) {
    F = identity3();
  }
  var d = svd3(F);
  // Deviatoric plastic return: putty yields in shape but not in volume.
  // Split the stretches into a volumetric part J^(1/3) and a unit-volume
  // deviatoric part, clamp only the deviatoric stretches (design §4), and
  // put the volume back so the pressure term still resists compression
  // (otherwise the nip could pack material to several times rest density).
  let Jraw = clamp(d.S.x * d.S.y * d.S.z, 0.25, 4.0);
  let Jc = pow(Jraw, 1.0 / 3.0);
  var dev = clamp(d.S / Jc, vec3<f32>(1.0 - P.mat.z), vec3<f32>(1.0 + P.mat.w));
  dev = dev / pow(max(dev.x * dev.y * dev.z, 1e-6), 1.0 / 3.0);
  d.S = dev * Jc;
  let Sig = mat3x3<f32>(vec3<f32>(d.S.x, 0.0, 0.0), vec3<f32>(0.0, d.S.y, 0.0), vec3<f32>(0.0, 0.0, d.S.z));
  F = d.U * Sig * transpose(d.V);
  let Rot = d.U * transpose(d.V);
  J = d.S.x * d.S.y * d.S.z;
  let tau = kirchhoffStress(F, Rot, J, P.mat.x, P.mat.y);
  let affine = p2gAffine(tau, C);

  // advection, clamp, hard push-out of the rollers
  if (isBad3(v)) { v = vec3<f32>(0.0); }
  x += dt * v;
  let lo = vec3<f32>(1.5 * h);
  let hi = P.domain.xyz - lo;
  x = clamp(x, lo, hi);
  let R = P.front.w;
  x = pushOut(P.front.x, P.front.y, R, x, h);
  x = pushOut(P.back.x, P.back.y, R, x, h);
  x = clamp(x, lo, hi);

  pos[p] = vec4<f32>(x, pw);
  vel[p] = vec4<f32>(v, 0.0);
  let cr = matRows(C);
  let fr = matRows(F);
  let ar = matRows(affine);
  let b = 9u * p;
  for (var c = 0u; c < 9u; c++) {
    cbuf[b + c] = cr[c];
    fbuf[b + c] = fr[c];
    abuf[b + c] = ar[c];
  }
}

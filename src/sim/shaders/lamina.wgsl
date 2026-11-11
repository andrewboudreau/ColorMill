// Lamina kinematics (design §5): the pigment in a particle is a stack of laminae with unit normal n
// and thickness s, both carried by the flow. With L the velocity gradient (the APIC matrix C):
//
//   a material surface element's direction m = F^-T n0 obeys dm/dt = -L^T m, so for n = m / |m|
//       dn/dt = -(I - n n^T) L^T n
//   and two neighbouring material surfaces are s = n . dx apart with d(dx)/dt = L dx, so
//       d(ln s)/dt = n^T L n
//   (equivalently s / s0 = 1 / |F^-T n0|). Volume change is not mixing: L is replaced by its
//   deviatoric part L' = L - tr(L)/3 I.
//
// The step integrates m and takes both from it, m = exp(-dt L'^T) n to second order, q -= ln |m|,
// n = m / |m|: exact for simple shear, and a rotation by theta costs theta^4 / 8 in q.
//
// The solver's F cannot stand in for this: after the plastic return it holds only the elastic part of
// the deformation and the operator's moves reset it. So the state is integrated here from C, in its own
// buffer, and survives the moves. Kinematic particles (a held chunk, material in the operator's hand or
// log) keep their state: their C is stale. The scripted roll-up turns the normals itself when it lets
// go (fold.wgsl).
@group(0) @binding(1) var<storage, read> cbuf : array<f32>;
@group(0) @binding(2) var<storage, read> flags : array<u32>;
@group(0) @binding(3) var<storage, read_write> lam : array<vec4<f32>>;

fn stepLamina(p : u32, dt : f32) {
  if ((flags[p] & 1u) != 0u) { return; }
  let b = 9u * p;
  let tr = (cbuf[b] + cbuf[b + 4u] + cbuf[b + 8u]) / 3.0;
  // A = L'^T. WGSL matrices are built from columns and C[r][c] = cbuf[9p + 3r + c], so the columns of
  // A are the rows of C.
  let A = mat3x3<f32>(
    vec3<f32>(cbuf[b] - tr, cbuf[b + 1u], cbuf[b + 2u]),
    vec3<f32>(cbuf[b + 3u], cbuf[b + 4u] - tr, cbuf[b + 5u]),
    vec3<f32>(cbuf[b + 6u], cbuf[b + 7u], cbuf[b + 8u] - tr));
  let st = lam[p];
  var n = st.xyz;
  var q = st.w;
  if (dot(n, n) < 0.25 || isBad3(n) || isBad(q)) {
    n = seedDir(p);
    if (isBad(q)) { q = 0.0; }
  }
  let An = A * n;
  let m = n - dt * An + (0.5 * dt * dt) * (A * An);
  let m2 = dot(m, m);
  if (!isBad3(m) && m2 > 1e-12 && m2 < 1e12) {
    n = m * inverseSqrt(m2);
    q = clamp(q - 0.5 * log(m2), LAMINA_Q_MIN, LAMINA_Q_MAX);
  }
  lam[p] = vec4<f32>(n, q);
}

/// Once a substep, after G2P (P.hdt.z is the substep).
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  stepLamina(p, P.hdt.z);
}

/// Once a frame, with the sim time the frame advanced and the last substep's C (kept for comparison:
/// mpm.ts LAMINA_PER_SUBSTEP).
@compute @workgroup_size(128)
fn frame_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  stepLamina(p, P.domain.w);
}

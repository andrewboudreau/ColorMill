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
//
// Agglomerates (design §5, dispersive mixing). Fresh masterbatch is pigment in agglomerates, which tint
// weakly; shear stress over their cohesive strength erodes them into dispersed pigment, and only that
// develops the colour. Per particle the UNDISPERSED load u (mixs.z, 0 .. the load pos.w) decays where
// the flow stresses the putty enough:
//       tau   = tau_y + k (shear rate)^n          the material's flow curve (the stress the plastic
//                                                 return holds a flowing particle at, design §4)
//       shear rate = sqrt(2 D':D'), D' the symmetric part of L'
//       du/dt = -u (shear rate) [ k_e max(tau / tau_e - 1, 0) + k_r max(tau / tau_c - 1, 0) ]
// so nothing happens below the cohesive stress tau_e, and above it the loss per unit of shear strain
// grows with the excess stress (erosion); over the higher tau_c a second, faster term stands for rupture
// (tau_c = 0: none). The step is the exact decay over dt at the frame's rate, so it never overshoots
// and is the same whatever the frame is split into at a steady rate. agglomerate.ts is the CPU mirror.
// The shear rate is read through a scale: the presets narrow the gap with the cell (1.28 cells at each), which
// doubles the nip's strain rate from `low` to `high` for a reason that is numerics, not the mill, while the
// bank's hardly changes. The rate from C is multiplied by P.agglo2.x = sqrt(the preset's nominal gap / the
// reference gap), 1 at `low`: an empirical middle (design §5 has the measurements). The gap the user sets
// still counts: narrower than the preset's, more stress.
// P.agglo = (tau_e, k_e, tau_c, k_r).
// MAT_STRIDE vec4 per particle (common.wgsl); C is floats 18 to 26 of the particle's 28, row-major
@group(0) @binding(1) var<storage, read> mats : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> flags : array<u32>;
@group(0) @binding(3) var<storage, read_write> lam : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> mixs : array<vec4<f32>>;

/// The rate (1/s) a particle's undispersed load decays at under the shear rate `rawRate` (1/s) of its C.
fn agglomerateLossRate(rawRate : f32) -> f32 {
  let tauE = P.agglo.x;
  let rate = rawRate * P.agglo2.x;
  if (!(tauE > 0.0) || !(rate > 0.0) || isBad(rate)) { return 0.0; }
  let tau = P.mat.z + P.mat.w * pow(rate, 1.0 / P.mat2.x);
  var loss = P.agglo.y * max(tau / tauE - 1.0, 0.0);
  if (P.agglo.z > 0.0) { loss += P.agglo.w * max(tau / P.agglo.z - 1.0, 0.0); }
  loss *= rate;
  if (isBad(loss)) { return 0.0; }
  return max(loss, 0.0);
}

fn stepLamina(p : u32, dt : f32) {
  if ((flags[p] & 1u) != 0u) { return; }
  let b = MAT_STRIDE * p;
  let r0 = mats[b + 4u];   // f7, f8, c0, c1
  let r1 = mats[b + 5u];   // c2 .. c5
  let r2 = mats[b + 6u];   // c6, c7, c8, spare
  let tr = (r0.z + r1.z + r2.z) / 3.0;
  // A = L'^T. WGSL matrices are built from columns and C[r][c] is float 3r + c of C, so the columns of
  // A are the rows of C.
  let A = mat3x3<f32>(
    vec3<f32>(r0.z - tr, r0.w, r1.x),
    vec3<f32>(r1.y, r1.z - tr, r1.w),
    vec3<f32>(r2.x, r2.y, r2.z - tr));
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
  // agglomerates: D' = (A + A^T) / 2
  let u = mixs[p].z;
  if (u > 0.0) {
    let S = 0.5 * (A + transpose(A));
    let rate = sqrt(2.0 * (dot(S[0], S[0]) + dot(S[1], S[1]) + dot(S[2], S[2])));
    let loss = agglomerateLossRate(rate);
    if (loss > 0.0) { mixs[p].z = u * exp(-loss * dt); }
  }
}

/// Once a substep, after G2P (P.hdt.z is the substep). Not dispatched unless mpm.ts LAMINA_PER_SUBSTEP is set.
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  stepLamina(p, P.hdt.z);
}

/// Once a frame, with the sim time the frame advanced and the last substep's C: what runs (measured the
/// same as once a substep, mpm.ts LAMINA_PER_SUBSTEP).
@compute @workgroup_size(128)
fn frame_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  stepLamina(p, P.domain.w);
}

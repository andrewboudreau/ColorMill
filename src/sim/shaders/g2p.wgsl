// G2P (design §3.4): gather velocity + APIC C, update F, plastic return (design §4),
// the premultiplied P2G affine matrix for the next substep, advection,
// clamps, hard roller push-out.
@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;
// P2G affine (read by p2g), F and C, MAT_STRIDE vec4 per particle (common.wgsl)
@group(0) @binding(3) var<storage, read_write> mats : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> flags : array<u32>;
@group(0) @binding(5) var<storage, read> gvel : array<vec4<f32>>;   // xyz velocity, w the node's mass (grid.wgsl)
// cut & fold reductions: [0] = live pile top under the standing log (atomicMax on float bits)
@group(0) @binding(6) var<storage, read_write> foldInfo : array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> fold0 : array<vec4<f32>>;   // .w = release time of a fed particle
// the material's box (box.wgsl): BOX_ACC is the min / max cell over every particle this substep
@group(0) @binding(8) var<storage, read_write> box : array<atomic<u32>>;

// the bounds step_ settled on last substep, in a plain buffer of its own (min cell xyz, max cell xyz)
@group(0) @binding(9) var<storage, read> boxLast : array<u32>;

/// Report a particle at x to the box's reduction. Only a particle on or beyond a face of the last
/// substep's bounds touches the shared atomics; step_ takes a face nothing reported on as having
/// moved in by a cell (box.wgsl). Reading the bounds from the atomic words themselves had every
/// particle contend for the same six words every substep, and was four fifths of this kernel's time.
fn growBox(x : vec3<f32>) {
  let c = vec3<u32>(clamp(x * P.hdt.y, vec3<f32>(0.0), vec3<f32>(P.grid.xyz - vec3<u32>(1u))));
  if (c.x <= boxLast[0]) { atomicMin(&box[BOX_ACC], c.x); }
  if (c.y <= boxLast[1]) { atomicMin(&box[BOX_ACC + 1u], c.y); }
  if (c.z <= boxLast[2]) { atomicMin(&box[BOX_ACC + 2u], c.z); }
  if (c.x >= boxLast[3]) { atomicMax(&box[BOX_ACC + 3u], c.x); }
  if (c.y >= boxLast[4]) { atomicMax(&box[BOX_ACC + 4u], c.y); }
  if (c.z >= boxLast[5]) { atomicMax(&box[BOX_ACC + 5u], c.z); }
}

/// The rest mass (as a share of rest density) the rolls displace from a particle's kernel at x:
/// the gather of node mass reads low next to a roll, since nodes inside it hold no particles. For a
/// node at signed distance s cells from a flat wall the share of its kernel behind the wall is the
/// integral of the quadratic B-spline, and gathering that with the particle's own weights
/// convolves it with the spline once more: the distribution of a sum of six uniforms, standard
/// deviation sqrt(1/2) cell, which a smoothstep of half-width GHOST_REACH matches to 0.04
/// (tests/diagnostics.test.ts).
const GHOST_REACH : f32 = 1.45;
fn rollGhost(x : vec3<f32>) -> f32 {
  let sF = (rollerDist(P.front.x, P.front.y, x) - P.front.w) * P.hdt.y;
  let sB = (rollerDist(P.back.x, P.back.y, x) - P.front.w) * P.hdt.y;
  return (1.0 - smoothstep(-GHOST_REACH, GHOST_REACH, sF)) + (1.0 - smoothstep(-GHOST_REACH, GHOST_REACH, sB));
}

// The density tie lowers ln J at most this fast (1/s): through the nip (0.03 s) it follows the packing,
// but material the fed log of a cut & roll is laid into is loaded over tens of substeps, not in one.
const TIE_RATE : f32 = 10.0;

const FOLD_SETTLE : f32 = 0.5;   // s: material just let go of the log does not count as pile yet

fn loadMatF(p : u32) -> mat3x3<f32> {
  let b = p * MAT_STRIDE;
  // F is floats 9 to 17 of the particle's 28: r0 = (a8, f0, f1, f2), r1 = (f3 .. f6), r2 = (f7, f8, c0, c1)
  let r0 = mats[b + 2u];
  let r1 = mats[b + 3u];
  let r2 = mats[b + 4u];
  return mat3x3<f32>(
    vec3<f32>(r0.y, r1.x, r1.w),
    vec3<f32>(r0.z, r1.y, r2.x),
    vec3<f32>(r0.w, r1.z, r2.y));
}

fn pushOut(axisY : f32, axisZ : f32, R : f32, x : vec3<f32>, h : f32) -> vec3<f32> {
  let d = rollerDist(axisY, axisZ, x);
  if (d < R) {
    atomicAdd(&box[BOX_DIAG + 1u], 1u);
    let nrm = rollerNormal(axisY, axisZ, x);
    return vec3<f32>(x.x, axisY + nrm.y * (R + 0.25 * h), axisZ + nrm.z * (R + 0.25 * h));
  }
  return x;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  if ((flags[p] & 1u) != 0u) {   // kinematic: moved by the fold kernel
    growBox(pos[p].xyz);
    return;
  }

  let h = P.hdt.x;
  let invh = P.hdt.y;
  let dt = P.hdt.z;

  let pw = pos[p].w;
  var x = pos[p].xyz;

  // while the operator move feeds the log: reduce the top of the (non-stray) material
  // under its footprint, so the log releases onto the pile the nip is eating
  if (P.fold.x > 0.5 && P.fold.y - fold0[p].w >= FOLD_SETTLE
      && abs(x.z - P.fold2.w) < P.fold3.x && abs(x.x - 0.5 * P.fold2.y) < 0.6 && x.y > P.front.x) {
    let cn = clamp(vec3<i32>(round(x * invh)), vec3<i32>(0), vec3<i32>(P.grid.xyz) - vec3<i32>(1));
    if (gvel[nodeIndexI(cn)].w >= 2.0 * P.part.y) {
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
  var mass = 0.0;
  var B = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var i = 0; i < 3; i++) {
        let off = vec3<i32>(i, j, k);
        let dpos = (vec3<f32>(off) - fx) * h;
        let wgt = wx[i] * wy[j] * wz[k];
        let g = gvel[nodeIndexI(base + off)];
        let gv = g.xyz;
        v += wgt * gv;
        mass += wgt * g.w;
        // outer(gv, dpos): column c = gv * dpos[c]  (B[r][c] = gv_r * dpos_c)
        B += wgt * mat3x3<f32>(gv * dpos.x, gv * dpos.y, gv * dpos.z);
      }
    }
  }
  let C = (4.0 * invh * invh) * B;

  // deformation gradient update + plastic return (design §4)
  var F = (identity3() + dt * C) * loadMatF(p);
  let J = det3(F);
  // a particle whose F is not a number restarts unstressed (counted: BOX_DIAG)
  if (isBad(J) || isBadMat(F)) {
    F = identity3();
    atomicAdd(&box[BOX_DIAG], 1u);
  }
  let d = svd3(F);
  // Plastic return (design §4): putty yields in shape, not in volume. On the principal Hencky
  // strains, split off the volume (ln J) and bring the deviatoric part back radially toward the von
  // Mises yield surface. The return is viscoplastic: the stress beyond yield follows a
  // Herschel-Bulkley law, tau = tau_y + k (shear rate)^n, so the flowing putty carries shear like a
  // shear-thinning fluid and the drag of the rolls reaches into the bank instead of stopping in a
  // thin slip layer at the roll surface, while the elastic strain it carries stays tau / 2 mu.
  let eps = henckyStrain(d.S);
  var ev = (eps.x + eps.y + eps.z) / 3.0;
  var e = eps - vec3<f32>(ev);
  // Density tie (design §4): the volume the pressure is computed from may not exceed what the
  // density the grid measures at the particle allows. det F only integrates the velocity gradient
  // the particle saw, and next to the rolls (push-out, the band's prescribed velocities) the
  // particles pack without F hearing of it; then the nip passes more than the gap holds.
  if (P.mat2.y > 0.0) {
    let rho = mass / (8.0 * P.part.y) + rollGhost(x);
    ev = max(min(ev, log((1.0 + P.mat2.y) / max(rho, 1e-3)) / 3.0), ev - TIE_RATE * dt / 3.0);
  }
  // The elastic volume saturates (design §4, Stability): a particle the hand, the fed log or the
  // rolls' push-out squeeze further than the putty can carry keeps its shape history and a bounded
  // pressure, K ln(1 / volumeMin). (Resetting F to I at J = 1/2, as before, dropped the whole
  // pressure in one substep and kicked the neighbours; no bound at all lets a prescribed motion
  // store any amount of energy, and the cut & fold then blows up 10% past the preset's time step.)
  ev = clamp(ev, P.mat2.z, P.mat2.w);
  let trial = 1.41421356 * P.mat.x * length(e);
  if (trial > P.mat.z) { e *= hbShear(trial) / trial; }
  let epsN = e + vec3<f32>(ev);
  let Sn = exp(epsN);
  F = mat3x3<f32>(d.U[0] * Sn.x, d.U[1] * Sn.y, d.U[2] * Sn.z) * transpose(d.V);
  let tau = henckyStress(d.U, epsN);
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
  growBox(x);

  pos[p] = vec4<f32>(x, pw);
  vel[p] = vec4<f32>(v, 0.0);
  let cr = matRows(C);
  let fr = matRows(F);
  let ar = matRows(affine);
  let b = MAT_STRIDE * p;
  mats[b] = vec4<f32>(ar[0], ar[1], ar[2], ar[3]);
  mats[b + 1u] = vec4<f32>(ar[4], ar[5], ar[6], ar[7]);
  mats[b + 2u] = vec4<f32>(ar[8], fr[0], fr[1], fr[2]);
  mats[b + 3u] = vec4<f32>(fr[3], fr[4], fr[5], fr[6]);
  mats[b + 4u] = vec4<f32>(fr[7], fr[8], cr[0], cr[1]);
  // C is read once a frame (lamina.wgsl), so only the frame's last substep stores the two vec4 that are
  // all C; between substeps the stored C is stale but for c0 and c1, which share a vec4 with F
  if (P.fine.w != 0u) {
    mats[b + 5u] = vec4<f32>(cr[2], cr[3], cr[4], cr[5]);
    mats[b + 6u] = vec4<f32>(cr[6], cr[7], cr[8], 0.0);
  }
}

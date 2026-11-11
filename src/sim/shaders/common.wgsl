// ColorMill v2 — shared WGSL prelude for the MLS-MPM kernels (design §1–§5).
// Concatenated in front of every kernel by src/sim/mpm.ts.
//
// Conventions:
//   * grid node (i,j,k) has index n = (k*NY + j)*NX + i, position (i,j,k)*h
//   * particle positions / velocities and the grid velocity are vec4 (xyz + pad)
//     so a load is one 16-byte transaction instead of three scalar loads.
//   * particle matrices (P2G affine, F, C) share one buffer, MAT_STRIDE vec4 per particle: seen as
//     floats, the affine matrix is at 28p, F at 28p + 9 and C at 28p + 18, each row-major
//     (float 3r + c = M[r][c], math notation, row r / column c), and float 27 is spare. A vec4
//     store is one transaction where nine scalar stores were nine, which was most of G2P's stores.
//     WGSL mat3x3 is column-major: m[c][r] is row r of column c.
//   * fixed-point grid accumulators: mass 2^20, momentum 2^16, latent 2^18.

struct Params {
  // nx, ny, nz (node counts), particleCount
  grid : vec4<u32>,
  // h, 1/h, dt, gravity
  hdt : vec4<f32>,
  // domain xyz, sim seconds advanced this frame (dt * substeps; NOT wall clock)
  domain : vec4<f32>,
  // back roller: axisY, axisZ, omegaX, Coulomb mu
  back : vec4<f32>,
  // front roller: axisY, axisZ, omegaX, radius
  front : vec4<f32>,
  // material (design §4): shear modulus mu, bulk modulus K, yield stress tau_y, Herschel-Bulkley consistency k
  mat : vec4<f32>,
  // pVol, pMass, stressScale, the exchange closure's exponent k (disperse.wgsl)
  part : vec4<f32>,
  // fold script: active (0/1), s, ds/dt, lift height
  fold : vec4<f32>,
  // fold targets: fallback bankTop y (used when no bank particle was found), roller length L, x shift, nip z
  fold2 : vec4<f32>,
  // tackBand, backBand, wallMargin, floorFriction
  bands : vec4<f32>,
  // fold slab: bank half depth (z window for the live bank-top estimate), slab clearance above the bank,
  // arc -> z scale (0.6), max y any folded particle may reach (domain.y - 3h)
  fold3 : vec4<f32>,
  // colour raster (design §3.5): the node counts of its box nxF, nyF, nzF; then 1 on the substeps whose G2P
  // stores C (the frame's last, g2p.wgsl), else 0
  fine : vec4<u32>,
  // colour raster at its finest cell: hF, 1/hF, normalisation (h/hF)^3 / 8 so a packed fine cell reads ~1,
  // the longest colour thread in particle spacings (0 = points)
  // (the cell in use is FINE_K / 8 of this, box.wgsl)
  fineH : vec4<f32>,
  // cut & roll, peel and wind (fold.wgsl): wind speed along the arc, T_wind, T_double, T_lift
  fold4 : vec4<f32>,
  // held pigment chunks (chunk.wgsl): feed speed (how fast a chunk sinks through its level), the node mass that
  // counts as material under it; then the nip's shear (grid.wgsl): the back roll's surface speed over the front
  // roll's less one, and how far past the roll surfaces the profile across the gap is carried on, in gaps
  chunk : vec4<f32>,
  // material, continued: 1 / n (n the Herschel-Bulkley index), the density tie's tolerance (0 = off),
  // the bounds of ln J / 3
  mat2 : vec4<f32>,
  // pigment agglomerates (lamina.wgsl, design §5): the cohesive stress erosion starts at, the erosion rate
  // constant (per unit of shear strain and of excess stress over the cohesive stress), the stress rupture
  // starts at (0 = no rupture) and its rate constant
  agglo : vec4<f32>,
  // ... and the scale its shear rates are read through (config agglomerateRateScale: the square root of the
  // preset's nominal gap over the reference gap, design §5); three spare
  agglo2 : vec4<f32>,
};

@group(0) @binding(0) var<uniform> P : Params;

const MASS_SCALE : f32 = 1048576.0;   // 2^20
const MOM_SCALE  : f32 = 65536.0;     // 2^16
const LAT_SCALE  : f32 = 262144.0;    // 2^18
const PI : f32 = 3.14159265358979;
// Pigment load of a masterbatch chunk (pos.w per particle; the clear base
// carries 0). Latents are averaged weighted by load, so the base has no say in
// a mix's colour and only the load per unit mass thins it.
const PIGMENT_LOAD : f32 = 12.0;

// The material's box buffer (box.wgsl, design §3.6), u32 words. Cells and nodes share indices
// (cell c spans nodes c .. c+1).
const BOX_ACC : u32 = 0u;     // min cell xyz, max cell xyz: G2P's reduction over the particles this substep
const BOX_GRID : u32 = 6u;    // origin xyz, size xyz: the nodes the solver's grid update covers
const BOX_LAST : u32 = 12u;   // min cell xyz, max cell xyz of the last finished reduction
const BOX_FRAME : u32 = 18u;  // origin xyz, size xyz: the nodes the frame's raster clear and pack cover
const BOX_PREV : u32 = 24u;   // origin xyz, size xyz of the previous frame's own raster box
const BOX_SLACK : i32 = 1;    // extra cells around every box
// the colour raster's box (finer lattice, design §3.6): its cell is FINE_K / 8 finest cells
const BOX_FINE : u32 = 32u;        // lattice index xyz of its texel 0, FINE_K
const BOX_FINE_RANGE : u32 = 36u;  // texels xyz its clear and pack cover (this frame's range and the last one's)
const BOX_FINE_PREV : u32 = 40u;   // texels xyz of the previous frame's own range
// diagnostics (physics review 4.5): G2P counts its F resets and its roller push-outs here; nothing in the solver reads them
const BOX_DIAG : u32 = 44u;
const FINE_K_MIN : i32 = 8;
const FINE_K_MAX : i32 = 64;

fn nodeIndex(i : u32, j : u32, k : u32) -> u32 {
  return (k * P.grid.y + j) * P.grid.x + i;
}

fn nodeIndexI(c : vec3<i32>) -> u32 {
  return nodeIndex(u32(c.x), u32(c.y), u32(c.z));
}

/// Node index on the colour raster's finer grid.
fn fineIndexI(c : vec3<i32>) -> u32 {
  return (u32(c.z) * P.fine.y + u32(c.y)) * P.fine.x + u32(c.x);
}

/// Quadratic B-spline weights for the three nodes base..base+2 (fx = x/h - base).
fn quadWeights(fx : f32) -> vec3<f32> {
  let a = 1.5 - fx;
  let b = fx - 1.0;
  let c = fx - 0.5;
  return vec3<f32>(0.5 * a * a, 0.75 - b * b, 0.5 * c * c);
}

fn encodeFixed(v : f32, scale : f32) -> i32 {
  return i32(round(v * scale));
}

fn decodeFixed(v : i32, scale : f32) -> f32 {
  return f32(v) / scale;
}

/// Particle index from a (possibly 2D) dispatch of 128-wide workgroups.
fn particleIndex(gid : vec3<u32>, nwg : vec3<u32>) -> u32 {
  return gid.x + gid.y * (nwg.x * 128u);
}

/// Surface velocity of a roller rotating about +x: omega x (p - axis).
fn rollerVel(axisY : f32, axisZ : f32, omegaX : f32, p : vec3<f32>) -> vec3<f32> {
  let dy = p.y - axisY;
  let dz = p.z - axisZ;
  return vec3<f32>(0.0, -omegaX * dz, omegaX * dy);
}

/// (distance to the axis, outward unit normal) for a roller.
fn rollerDist(axisY : f32, axisZ : f32, p : vec3<f32>) -> f32 {
  let dy = p.y - axisY;
  let dz = p.z - axisZ;
  return sqrt(dy * dy + dz * dz);
}

fn rollerNormal(axisY : f32, axisZ : f32, p : vec3<f32>) -> vec3<f32> {
  let dy = p.y - axisY;
  let dz = p.z - axisZ;
  let d = sqrt(dy * dy + dz * dz);
  if (d < 1e-8) { return vec3<f32>(0.0, 1.0, 0.0); }
  return vec3<f32>(0.0, dy / d, dz / d);
}

/// True for NaN or +-Inf. Tests the exponent bits directly so a fast-math
/// backend cannot fold the check away (as it may with `x == x` / `!(x > a)`).
fn isBad(x : f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u;
}

fn isBad3(v : vec3<f32>) -> bool {
  let e = vec3<u32>(0x7f800000u);
  return any((bitcast<vec3<u32>>(v) & e) == e);
}

fn isBadMat(m : mat3x3<f32>) -> bool {
  return isBad3(m[0]) || isBad3(m[1]) || isBad3(m[2]);
}

fn identity3() -> mat3x3<f32> {
  return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
}

fn det3(m : mat3x3<f32>) -> f32 {
  return dot(m[0], cross(m[1], m[2]));
}

// ---------------------------------------------------------------------------
// 3x3 SVD: Jacobi eigen-decomposition of F^T F (<= 8 sweeps) -> V, sigma^2,
// then U = F V sigma^-1 with guards, and a sign fix so det(U) det(V) > 0.
// ---------------------------------------------------------------------------
struct Svd3 {
  U : mat3x3<f32>,
  S : vec3<f32>,
  V : mat3x3<f32>,
};

/// One Jacobi rotation in the (p,q) plane of the symmetric matrix A, accumulating V
/// (Numerical Recipes convention: J_pp = J_qq = c, J_pq = s, J_qp = -s; A' = J^T A J, V' = V J).
fn jacobiRotate(A : ptr<function, mat3x3<f32>>, V : ptr<function, mat3x3<f32>>, p : i32, q : i32) {
  let apq = (*A)[q][p];
  if (abs(apq) < 1e-12) { return; }
  let app = (*A)[p][p];
  let aqq = (*A)[q][q];
  let theta = (aqq - app) / (2.0 * apq);
  let sg = select(-1.0, 1.0, theta >= 0.0);
  let t = sg / (abs(theta) + sqrt(theta * theta + 1.0));
  let c = inverseSqrt(t * t + 1.0);
  let s = t * c;
  let r = 3 - p - q;   // the third index
  let arp = (*A)[p][r];
  let arq = (*A)[q][r];
  (*A)[p][p] = app - t * apq;
  (*A)[q][q] = aqq + t * apq;
  (*A)[q][p] = 0.0;
  (*A)[p][q] = 0.0;
  let nrp = c * arp - s * arq;
  let nrq = s * arp + c * arq;
  (*A)[p][r] = nrp; (*A)[r][p] = nrp;
  (*A)[q][r] = nrq; (*A)[r][q] = nrq;
  // V' = V J: columns p and q of V are rotated
  let vp = (*V)[p];
  let vq = (*V)[q];
  (*V)[p] = c * vp - s * vq;
  (*V)[q] = s * vp + c * vq;
}

fn svd3(F : mat3x3<f32>) -> Svd3 {
  var A = transpose(F) * F;
  var V = identity3();
  // Relative convergence test: f32 round-off leaves off-diagonals of order
  // eps * |A| (eps ~ 6e-8, so off ~ 1e-14 * tr^2 at best); 1e-10 * tr^2 lets
  // an already-diagonal F (the common case, F ~ I) leave after 1-2 sweeps.
  let tr = A[0][0] + A[1][1] + A[2][2];
  let tol = 1e-10 * tr * tr;
  for (var sweep = 0; sweep < 8; sweep++) {
    let off = A[1][0] * A[1][0] + A[2][0] * A[2][0] + A[2][1] * A[2][1];
    if (off < tol) { break; }
    jacobiRotate(&A, &V, 0, 1);
    jacobiRotate(&A, &V, 0, 2);
    jacobiRotate(&A, &V, 1, 2);
  }
  var e = vec3<f32>(A[0][0], A[1][1], A[2][2]);
  // sort descending (swap columns of V together with eigenvalues)
  if (e.x < e.y) { let t = e.x; e.x = e.y; e.y = t; let c = V[0]; V[0] = V[1]; V[1] = c; }
  if (e.y < e.z) { let t = e.y; e.y = e.z; e.z = t; let c = V[1]; V[1] = V[2]; V[2] = c; }
  if (e.x < e.y) { let t = e.x; e.x = e.y; e.y = t; let c = V[0]; V[0] = V[1]; V[1] = c; }
  var S = sqrt(max(e, vec3<f32>(0.0)));
  var U = identity3();
  let u0 = F * V[0];
  let u1 = F * V[1];
  let u2 = F * V[2];
  if (S.x > 1e-6) { U[0] = u0 / S.x; } else { U[0] = vec3<f32>(1.0, 0.0, 0.0); }
  if (S.y > 1e-6) { U[1] = u1 / S.y; } else {
    var t = cross(U[0], vec3<f32>(0.0, 0.0, 1.0));
    if (dot(t, t) < 1e-6) { t = cross(U[0], vec3<f32>(0.0, 1.0, 0.0)); }
    U[1] = normalize(t);
  }
  if (S.z > 1e-6) { U[2] = u2 / S.z; } else { U[2] = normalize(cross(U[0], U[1])); }
  // Gram-Schmidt polish keeps U orthonormal when singular values are close.
  U[1] = normalize(U[1] - U[0] * dot(U[0], U[1]));
  U[2] = normalize(U[2] - U[0] * dot(U[0], U[2]) - U[1] * dot(U[1], U[2]));
  if (det3(U) * det3(V) < 0.0) {
    U[2] = -U[2];
    S.z = -S.z;
  }
  var out : Svd3;
  out.U = U;
  out.S = S;
  out.V = V;
  return out;
}

// ---------------------------------------------------------------------------
// The putty (design §4; src/sim/constitutive.ts is the CPU mirror, unit tested): Hencky elasticity,
// a von Mises radial return with a Herschel-Bulkley overstress, and a pressure K ln J.
// ---------------------------------------------------------------------------
const HB_NEWTON : i32 = 4;

/// The shear stress after one backward-Euler step of the Herschel-Bulkley overstress from a trial
/// shear stress: the root of  tau + mu dt ((tau - tau_y) / k)^(1/n) = trial  in [tau_y, trial]. The
/// left side is increasing and convex (n <= 1), so Newton from above cannot overshoot; it starts from
/// the smaller of the trial value and the root of the power term alone, within a factor of two of
/// the root.
fn hbShear(trial : f32) -> f32 {
  let ty = P.mat.z;
  if (!(trial > ty)) { return trial; }
  let xt = trial - ty;
  let k = max(P.mat.w, 1e-6);
  let p = P.mat2.x;
  let a = P.mat.x * P.hdt.z;
  var x = min(xt, k * pow(xt / a, 1.0 / p));
  for (var i = 0; i < HB_NEWTON; i++) {
    let w = a * pow(x / k, p);
    x = max(x - (x - xt + w) / (1.0 + p * w / max(x, 1e-20)), 0.0);
  }
  return ty + min(x, xt);
}

/// Principal Hencky strains of the singular values (floored: a log needs a positive stretch).
fn henckyStrain(S : vec3<f32>) -> vec3<f32> {
  return log(clamp(S, vec3<f32>(0.05), vec3<f32>(20.0)));
}

/// Kirchhoff stress of principal Hencky strains eps in the frame U: U diag(2 mu dev eps + K ln J) U^T.
/// (The P2G force term is V0 * tau, design §3.2.) Linear in the strain: the Cauchy pressure
/// -K ln J / J has no ceiling in compression, and the stiffness per unit mass does not change with J.
fn henckyStress(U : mat3x3<f32>, eps : vec3<f32>) -> mat3x3<f32> {
  let lnJ = eps.x + eps.y + eps.z;
  let t = (2.0 * P.mat.x) * (eps - vec3<f32>(lnJ / 3.0)) + vec3<f32>(P.mat.y * lnJ);
  return mat3x3<f32>(U[0] * t.x, U[1] * t.y, U[2] * t.z) * transpose(U);
}

/// The matrix P2G multiplies dpos by (design §3.2), premultiplied once in G2P:
/// affine = -dt * pVol * (4/h^2) * stressScale * tau + pMass * C.
fn p2gAffine(tau : mat3x3<f32>, C : mat3x3<f32>) -> mat3x3<f32> {
  let invh = P.hdt.y;
  let dt = P.hdt.z;
  return (-dt * P.part.x * 4.0 * invh * invh * P.part.z) * tau + P.part.y * C;
}

/// Kirchhoff stress of an already returned F (SVD inside; used by the fold release, where G2P
/// has not run for the particle in a while).
fn kirchhoffStressOf(F : mat3x3<f32>) -> mat3x3<f32> {
  let d = svd3(F);
  return henckyStress(d.U, henckyStrain(d.S));
}

/// Store a mat3x3 row-major into a flat f32 array slot (9 floats per particle).
/// vec4 per particle in the matrix buffer (affine, F, C; see the conventions at the top).
const MAT_STRIDE : u32 = 7u;

fn matRows(m : mat3x3<f32>) -> array<f32, 9> {
  return array<f32, 9>(m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]);
}

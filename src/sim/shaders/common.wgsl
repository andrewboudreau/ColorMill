// ColorMill v2 — shared WGSL prelude for the MLS-MPM kernels (design §1–§5).
// Concatenated in front of every kernel by src/sim/mpm.ts.
//
// Conventions:
//   * grid node (i,j,k) has index n = (k*NY + j)*NX + i, position (i,j,k)*h
//   * particle positions / velocities and the grid velocity are vec4 (xyz + pad)
//     so a load is one 16-byte transaction instead of three scalar loads.
//   * particle matrices (C, F, P2G affine) are stored row-major in flat f32 arrays:
//     buf[p*9 + r*3 + c] = M[r][c] (math notation, row r / column c).
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
  // material: mu, lambda, thetaC, thetaS
  mat : vec4<f32>,
  // pVol, pMass, stressScale, dispersion k
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
};

@group(0) @binding(0) var<uniform> P : Params;

const MASS_SCALE : f32 = 1048576.0;   // 2^20
const MOM_SCALE  : f32 = 65536.0;     // 2^16
const LAT_SCALE  : f32 = 262144.0;    // 2^18
const PI : f32 = 3.14159265358979;

fn nodeIndex(i : u32, j : u32, k : u32) -> u32 {
  return (k * P.grid.y + j) * P.grid.x + i;
}

fn nodeIndexI(c : vec3<i32>) -> u32 {
  return nodeIndex(u32(c.x), u32(c.y), u32(c.z));
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

/// Fixed corotated Kirchhoff stress tau = P F^T (design §4 with the MLS-MPM
/// volume: the P2G force term is V0 * P * F^T, not V0 * sigma):
/// P = 2mu (F - R) + lambda (J - 1) J F^-T, so tau = 2mu (F - R) F^T + lambda (J - 1) J I.
fn kirchhoffStress(F : mat3x3<f32>, R : mat3x3<f32>, J : f32, mu : f32, lambda : f32) -> mat3x3<f32> {
  let a = (2.0 * mu) * ((F - R) * transpose(F));
  let b = lambda * (J - 1.0) * J;
  return a + mat3x3<f32>(vec3<f32>(b, 0.0, 0.0), vec3<f32>(0.0, b, 0.0), vec3<f32>(0.0, 0.0, b));
}

/// The matrix P2G multiplies dpos by (design §3.2), premultiplied once in G2P:
/// affine = -dt * pVol * (4/h^2) * stressScale * tau + pMass * C.
fn p2gAffine(tau : mat3x3<f32>, C : mat3x3<f32>) -> mat3x3<f32> {
  let invh = P.hdt.y;
  let dt = P.hdt.z;
  return (-dt * P.part.x * 4.0 * invh * invh * P.part.z) * tau + P.part.y * C;
}

/// Kirchhoff stress of an already plastically-clamped F (SVD inside; used by the
/// fold release, where G2P has not run for the particle in a while).
fn kirchhoffStressOf(F : mat3x3<f32>) -> mat3x3<f32> {
  let d = svd3(F);
  let Rot = d.U * transpose(d.V);
  let J = d.S.x * d.S.y * d.S.z;
  return kirchhoffStress(F, Rot, J, P.mat.x, P.mat.y);
}

/// Store a mat3x3 row-major into a flat f32 array slot (9 floats per particle).
fn matRows(m : mat3x3<f32>) -> array<f32, 9> {
  return array<f32, 9>(m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]);
}

# ColorMill v2 — GPU two-roll mill simulation (design spec)

This is the engineering spec for ColorMill v2. It replaces the v1 CPU
proof-of-concept (a 40³ MLS-MPM grid whose material occupied a ~14×12×7-voxel
box, drawn as flat cubes) with a **WebGPU compute MLS-MPM solver** running
hundreds of thousands of material points on a fine grid, an **elastoplastic
(putty / uncured silicone) material model**, a **real two-roll mill geometry**
(bank, nip, sheet on the front roll, friction ratio, end guides), **shear-driven
pigment dispersion**, and a **ray-marched glossy surface renderer** with Mixbox
pigment mixing in latent space.

Everything below is normative. Modules are developed against
`src/config/mill.ts` and `src/sim/types.ts`, which are the shared contracts.

---

## 1. Coordinate system, units, geometry

Right-handed, **+y up**, **+z toward the viewer** (the front), **+x to the
viewer's right**. The rollers' axes are parallel to **x**.

Sim units are dimensionless. One unit ≈ 0.2 m if you want a physical feel
(roller radius 0.32 ≈ 64 mm; real lab mills are 150 mm radius with a 1–5 mm
gap — we exaggerate the gap so it is resolvable on the grid).

| Symbol | Value | Meaning |
| --- | --- | --- |
| `L` | 1.5 | Roller (and domain) length along x |
| `domain` | (1.5, 1.25, 1.5) | Domain size (x, y, z); origin at (0,0,0) |
| `R` | 0.32 | Roller radius |
| `yc` | 0.55 | Height of both roller axes |
| `zNip` | 0.75 | z of the nip centre (mid-plane between rollers) |
| `gap` | 0.02–0.10, default 0.05 | Nip opening (distance between surfaces) |
| back roller axis | (y=yc, z = zNip − (R + gap/2)) | further from viewer |
| front roller axis | (y=yc, z = zNip + (R + gap/2)) | nearer the viewer |
| `omega` | 0–6 rad/s, default 3.0 | Angular speed of the **front** roller |
| `frictionRatio` | 1.0–1.6, default 1.25 | back roller speed / front roller speed |

**Rotation sense.** Both roller surfaces move **downward through the nip**.
With angular velocity vectors along x: back roller `ω_b = (+omega·frictionRatio, 0, 0)`,
front roller `ω_f = (−omega, 0, 0)`. Surface velocity at a point `p` is
`v = ω × (p − axisPoint)`. (Check: top of the back roller moves +z, top of the
front roller moves −z, so both move toward the nip and then down.)

**End guides.** The bank is confined along x by the domain walls at x=0 and
x=L (they model the mill's guide cheeks). Walls are *separating* (no
penetration, free tangential slip).

**Tray.** The floor y=0 is a separating wall with Coulomb friction.

The **grid** is a uniform lattice with spacing `h = 1/quality.cellsPerUnit`.
Node counts are `N = round(domain / h) + 1` per axis; see
`MillConfig.gridDims()`. Presets:

| preset | cellsPerUnit | h | cells (x,y,z) | seeded particles (approx) |
| --- | --- | --- | --- | --- |
| low | 32 | 0.03125 | 48×40×48 | 60k |
| medium | 48 | 0.02083 | 72×60×72 | 204k |
| high (default) | 64 | 0.015625 | 96×80×96 | 484k |
| ultra | 80 | 0.0125 | 120×100×120 | 945k |

The nip gap at `high` is ~3.2 cells wide; the sheet is 2–4 cells thick. That
is the minimum for a resolvable sheet; do not lower `cellsPerUnit` below 32.

**Initial material (the bank).** A slab resting on the nip:
x ∈ [0.05, L−0.05], z ∈ [zNip−0.25, zNip+0.25], y from the roller top surface
(`ySurface(z)` = yc + sqrt(R² − (z − zRollerAxis)²) for whichever roller is
under that z, or the nip floor `yc` inside the gap) up to `yc + R + 0.22`.
Particles are seeded 2 per axis per cell (8 per cell) on a jittered lattice.
Particles inside a roller are skipped. This gives the counts in the table.

---

## 2. Material points

Per-particle state, all `f32`, stored as a structure-of-arrays or an
array-of-structs of 32 floats (128 bytes) — the implementer chooses, but the
readback helper `GpuMpmSim.readParticles()` must return the layout in
`src/sim/types.ts` (`ParticleSnapshot`).

| field | size | notes |
| --- | --- | --- |
| `x` | vec3 | position (sim units) |
| `v` | vec3 | velocity |
| `C` | mat3 | APIC affine velocity (row-major, `C[i][j] = ∂v_i/∂x_j`) |
| `F` | mat3 | elastic deformation gradient (row-major) |
| `latent` | 7 floats | Mixbox latent of the particle's pigment mix |
| `flags` | 1 float (bit field as int) | bit0 = kinematic (under operator control), unused otherwise |
| pad | to 32 floats |

All particles have equal mass `pMass = 1` and equal rest volume
`pVol = h³ / 8` for the grid transfers. The material constants of §4 are
expressed for a **unit density (ρ = 1)** material, so the P2G stress term is
scaled by `pMass / (pVol · ρ)` with ρ = 1 (`MATERIAL_DENSITY` in
`src/sim/mpm.ts`); this keeps `dt < ~0.4·h/√E` meaningful and the bank
self-supporting. The v1 solver used the same convention.

---

## 3. The substep (MLS-MPM, quadratic B-splines)

Fixed substep `dt = quality.dt` (see §7). `substepsPerFrame` substeps per
rendered frame. All kernels are WebGPU compute shaders in
`src/sim/shaders/`. Node index `n = (k·NY + j)·NX + i` for node (i,j,k)
where NX, NY, NZ are **node** counts.

### 3.1 Clear grid
Zero the node accumulators: `mass`, `mom.xyz` (i32 fixed point) and the
velocity output `vel.xyz` (f32).

Fixed-point encoding (WebGPU has no float atomics): `mass` uses scale 2^20,
momentum uses scale 2^16. Encode with `i32(round(v * scale))`, decode with
`f32(i) / scale`. With pMass=1 and ≤ ~64 particles influencing a node the
ranges are safe (mass ≤ 2^26, |mom| ≤ 2^30 for |v| ≤ 250).

### 3.2 P2G (particle → grid)
For each non-kinematic particle:

```
base = floor(x/h − 0.5);  fx = x/h − base
w[0] = 0.5(1.5−fx)², w[1] = 0.75−(fx−1)², w[2] = 0.5(fx−0.5)²   (per axis)
σ  = stress(F)                                (§4)
affine = −dt · pVol · (4/h²) · σ + pMass · C   (mat3)
for each of the 27 nodes (i,j,k) in base..base+2:
   dpos = (offset − fx) · h
   wgt  = wx[i]·wy[j]·wz[k]
   atomicAdd mass  += wgt · pMass
   atomicAdd mom   += wgt · (pMass · v + affine · dpos)
```

Kinematic particles (flags bit0) are skipped here and in G2P velocity
gather; they move on their own script (§6).

### 3.3 Grid update
For each node with `mass > 0`:

```
v = mom / mass
v.y −= dt · g                                 (g = config.gravity, default 2.0)
v = applyBoundaries(v, nodePos)
store vel = v
```

Boundaries, applied in this order:

1. **Front roller** (sticky / tack). Let `d` be the distance from the node to
   the front roller axis, `n` the outward unit normal, `vr = ω_f × (p − axis)`
   the surface velocity. If `d < R + tackBand` (tackBand = 1.5·h):
   `v = vr` (full no-slip incl. normal — the sheet is carried around).
2. **Back roller** (separating with Coulomb friction). If `d < R`:
   `vrel = v − vr`; `vn = dot(vrel, n)`; if `vn < 0`: `vt = vrel − vn·n`;
   `vt *= max(0, 1 − mu·(−vn)/|vt|)` with `mu = 0.4`; `v = vr + vt` (normal
   component removed). If `vn ≥ 0` leave v unchanged.
   (Same test uses d < R + h·0.5 so the surface is not inside the grid cell.)
3. **Domain walls**. For x, z faces and the ceiling: if inside a `2h` margin
   and moving outward, zero that component. Floor (y < 2h): if `v.y < 0`
   then `v.y = 0` and scale the tangential part by `max(0, 1 − 0.6)` once
   (simple stick-slip; the tray should not be a skating rink).

The mode `rollerMode = 'counter' | 'friction'` from v1 is subsumed by
`frictionRatio` (1.0 = pure counter-rotation, >1 = friction milling). Keep
only `frictionRatio`.

### 3.4 G2P (grid → particle) + advection + plasticity
For each non-kinematic particle:

```
gather v_new = Σ wgt · vel[n]
gather C_new = (4/h²) · Σ wgt · outer(vel[n], dpos)     (MLS-MPM)
v = v_new;  C = C_new
F = (I + dt · C) · F
F = plasticReturn(F)                            (§4)
x += dt · v
x = clamp(x, 1.5h, domain − 1.5h)
x = pushOutOfRollers(x)   // hard safety: if inside a roller, project to R + 0.25h along n
```

### 3.5 Frame-level kernels (once per rendered frame, after the substeps)
- **Pigment raster**: clear `pmass` + `plat[7]` node accumulators (i32, scale
  2^20 for mass, 2^18 for latent·mass); scatter `wgt·pMass` and
  `wgt·pMass·latent` with the same 27-node stencil.
- **Disperse** (shear-driven mixing, §5).
- **Pack**: write the render volume (§8): two `rgba16float` 3D textures of
  node dims: `volA = (density, lat0, lat1, lat2)`, `volB = (lat3, lat4, lat5, lat6)`
  where `density = pmass / 8` (so a fully packed cell ≈ 1) and `latN =
  plat[N] / pmass` (0 where pmass = 0).

---

## 4. Material model: elastoplastic putty (fixed corotated + yield clamp)

This is the model that makes it a mill and not a puddle: the bank holds its
shape under gravity, yields and flows under the nip's shear, and does not
spring back.

Constants (in `config.material`):

| name | default | meaning |
| --- | --- | --- |
| `E` | 60 | Young's modulus |
| `nu` | 0.35 | Poisson ratio |
| `thetaC` | 0.025 | plastic compression threshold |
| `thetaS` | 0.0075 | plastic stretch threshold |
| `mu`, `lambda` | derived | `mu = E/(2(1+nu))`, `lambda = E·nu/((1+nu)(1−2nu))` |

Fixed corotated (Stomakhin 2013) with the MLS-MPM stress form:

```
[U, Σ, V] = svd(F)                     (3×3, robust; see note)
Rot = U · Vᵀ
J = det(F) = Σ0·Σ1·Σ2
P = 2mu·(F − Rot) + lambda·(J − 1)·J·F⁻ᵀ
σ = (1/J) · P · Fᵀ                     (Cauchy stress, used in §3.2)
```

Plastic return (after the F update in G2P): clamp each singular value
`Σi = clamp(Σi, 1 − thetaC, 1 + thetaS)` and rebuild `F = U·diag(Σ)·Vᵀ`.
No hardening. (This is the snow model with hardening off — it reads as
plasticine/putty.)

SVD note: implement a 3×3 SVD as Jacobi eigen-decomposition of `FᵀF`
(≤ 8 sweeps) to get `V` and `Σ²`, then `U = F·V·Σ⁻¹` with a guard for tiny
singular values, and fix the sign so `det(U)·det(V) > 0`. Alternatively use
the polar decomposition for `Rot` and the eigendecomposition only for the
clamp. Precision: f32 is fine.

Stability: sound speed `c = sqrt((lambda + 2mu)/ρ)` with ρ = 8/h³ ⇒ in the
scaling above the P2G stress term is already multiplied by `pVol·4/h²`, and
the effective CFL condition is `dt < ~0.4·h/sqrt(E)`. For h = 1/64, E = 60:
dt < 8e-4. Presets use the values in §7. Keep the constants of §4 the same
across presets; only `dt` and `substepsPerFrame` change.

---

## 5. Pigment: Mixbox latents and shear-driven dispersion

Each particle carries a 7-float **Mixbox latent** `z` (Sochorová & Jamriška
2021). Mixing pigments is a mass-weighted average in latent space, and
`latentToRgb(z)` (the Mixbox polynomial; port of `src/sim/mixbox.c`
`EvalPolynomial` + residual) is evaluated only in the render shader.

- Fresh silicone base = Titanium White latent (`PIGMENTS.white`).
- Pigments are a palette of named real pigments with precomputed latents
  (generated once with the official `web/vendor/mixbox/mixbox.js`
  `rgbToLatent`, stored as constants in `src/color/pigments.ts`):
  Cadmium Red, Cadmium Yellow, Cobalt Blue, Phthalo Green, Ultramarine,
  Burnt Sienna, Ivory Black, Titanium White, plus "custom" from a colour
  picker (converted with mixbox.js at runtime).
- **Inject** kernel: `addPigment(center, radius, latent, strength)` blends
  particle latents inside the sphere: `z = mix(z, zPigment, strength·t)`,
  `t = 1 − |d|/radius`.
- **Disperse** kernel (once per frame): for each particle gather the
  node-averaged latent `zg` (from the raster of §3.5) with the 27-node
  stencil; compute the shear rate `γ = ‖(C + Cᵀ)/2‖_F` from the particle's
  current `C`; then `z += clamp(k · γ · frameDt, 0, 1) · (zg − z)` with
  `k = config.dispersion` (default 0.6, UI 0–2). Mixing therefore happens
  where the material is sheared — at the nip — and not in the resting bank.

---

## 6. Operator: cut & fold (implemented)

A real operator cuts the sheet on the front roll and folds it across the
mill to mix along x (there is no axial transport otherwise). Model it as a
scripted kinematic move over `T = 1.2 s`:

1. Select particles on the front roller sheet with `x < 0.75·L`,
   `d < R + 3h` from the front axis, and `y < axisY` or `z > frontAxisZ`
   (the visible sheet; never the nip channel).
2. Parameterise each by `(x, t = d − R, θ)` with θ the angle around the
   front axis from the nip, and map it onto a slab lying on the bank so the
   sheet keeps its shape: `x1 = x + 0.25·L` (clamped to the guides),
   `y1 = bankTop + t + h/2`, `z1 = zNip + 0.6·(θ·R − arcMid)`. `bankTop` is
   the *live* bank top: the select kernel atomically maxes the y of bank
   particles (|z − zNip| < bankHalfDepth, above the roller tops, in a node
   with raster mass ≥ 2), so the slab lands on the bank as it is, not as it
   was seeded. Move along a raised arc
   `p(s) = lerp(p0,p1,s) + up·sin(π s)·0.25`, `s = smoothstep(t/T)`, with `v`
   the analytical derivative and the lift clamped below the ceiling.
3. On release keep `F` (the sheet's strain history), set `C = 0` and `v` to
   the scripted end velocity, and rebuild the P2G affine term.

Exposed as `GpuMpmSim.cutAndFold()`. The UI has a "Cut & fold" button (F).

---

## 7. Time stepping and presets

`quality` presets (`src/config/mill.ts`):

| preset | cellsPerUnit | dt | substepsPerFrame | target device |
| --- | --- | --- | --- | --- |
| low | 32 | 1.6e-3 | 8 | integrated / mobile GPU |
| medium | 48 | 1.1e-3 | 12 | laptop GPU |
| high | 64 | 8e-4 | 16 | desktop GPU (default) |
| ultra | 80 | 6.4e-4 | 20 | discrete GPU |

Simulated time per rendered frame is `dt·substepsPerFrame` (≈13 ms at
`high`), so at 60 fps the mill runs at ~0.8× real time. Show the resulting
"sim speed" in the HUD; do not vary substeps with frame time (determinism).
Everything rate-dependent (dispersion included) advances by sim time, never
by wall-clock frame time.

The app also steps the preset **down** automatically when the average frame
time stays above 45 ms for 3 s, unless the preset was pinned by the user or
the URL.

The app auto-selects `medium` when `navigator.gpu` adapter info says the
architecture is not a discrete class (or `low` on mobile user agents) and
`high` otherwise; the user can override.

---

## 8. Rendering

Single fullscreen ray-march pass (`src/render/shaders/raymarch.wgsl`):

1. Camera ray per pixel (perspective, orbit camera in `src/render/camera.ts`).
2. Intersect the domain box; march through `volA.density` with step `0.6h`
   (trilinear, `float32-filterable` not required: use `rgba16float`).
   Surface = first sample with `density ≥ 0.45`; refine with 3 bisection
   steps. Normal = central-difference gradient of density (offset 1 texel).
3. Colour at the hit: sample `volA/volB`, decode the 7-float latent,
   `latentToRgb` → albedo (sRGB-linearised for lighting).
4. Shading: key light (warm, upper right front), fill (cool, left), rim from
   behind; Blinn-Phong specular with a wide + narrow lobe (silicone putty is
   glossy); a cheap ambient occlusion from density sampled 2 and 4 texels
   along the normal (darkens creases and the nip).
5. Rollers: analytic ray/cylinder intersection with end caps; brushed metal
   shading with a rotating stripe pattern (angle = `rollerAngle`) so rotation
   speed is visible. Depth-composite against the volume hit.
6. Tray/floor plane with a soft contact shadow under the front roller;
   background: dark vertical gradient.
7. Output sRGB. The canvas uses `navigator.gpu.getPreferredCanvasFormat()`.

Render at `min(devicePixelRatio, 2)`. Provide `Renderer.setVolumes(volA, volB)`
so the renderer can be tested with a procedural volume without the sim.

---

## 9. UI (touch-first)

- Bottom bar: pigment palette (large round swatches; tap = inject a blob of
  radius 0.09 at a random x along the bank top), a colour picker swatch,
  "Cut & fold", "Clear pigment", "Reset", Pause.
- Right drawer (collapsible; hidden by default on narrow screens): sliders
  for roller speed (rpm shown), friction ratio, nip gap, dispersion, gravity;
  quality select; camera auto-orbit toggle; stats (particles, grid, fps,
  ms/frame, sim speed).
- Drag in the viewport orbits; wheel/pinch zooms; double-tap resets the
  camera to the front view.
- Keyboard: Space pause, R reset, F cut&fold, 1–8 pigments, arrows speed/gap.

---

## 10. Files

```
index.html                      Vite entry (the v2 app)
src/main.ts                     glue and frame loop
src/gpu/device.ts               WebGPU init + capability report + fallback message
src/config/mill.ts              MillConfig, presets, geometry helpers (pure, tested)
src/sim/types.ts                shared interfaces (GpuMpmSim API, snapshots, stats)
src/sim/mpm.ts                  GpuMpmSim implementation
src/sim/shaders/*.wgsl          clear, p2g, grid, g2p, raster, disperse, pack, inject, fold
src/render/renderer.ts          Renderer (ray-march), src/render/camera.ts
src/render/shaders/raymarch.wgsl, mixbox.wgsl
src/color/pigments.ts           palette latents + latentToRgb (TS mirror, tested)
src/ui/*.ts, src/styles.css     panel, hud, palette
tests/*.test.ts                 vitest unit tests (pure TS)
tests/e2e/*.spec.ts             Playwright, headless Chromium + SwiftShader WebGPU
web/*.html                      docs pages, copied into dist by the build
src/sim/*.c, src/main.c         v1 CPU reference (kept; `make native`)
```

WGSL is imported with Vite `?raw` imports. `@webgpu/types` provides typings.

---

## 11. Verification bar

The v2 is done when, in headless Chromium (SwiftShader) at `low`:

- 200 frames run with no NaN/Inf in particle state, all particles inside
  the domain and outside both rollers, particle count unchanged.
- A sheet forms on the front roller within 3 s of sim time (≥ 15% of
  particles have `d_front < R + 4h` and `y < yc`), while the back roller
  carries less than a third of that.
- Injecting blue and yellow and running 6 s produces voxels whose decoded
  colour is green (G channel dominant) in the sheet — mixing happens.
- The rendered frame shows a smooth, lit surface (no visible cubes) —
  checked by eye from a saved screenshot.
- `npm run build`, `npm test`, `npm run test:e2e` all pass and the
  Pages workflow builds the Vite app.

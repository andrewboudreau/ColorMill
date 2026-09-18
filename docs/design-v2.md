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
| `domain` | (1.5, 2.25, 1.5) | Domain size (x, y, z); origin at (0,0,0); the headroom holds the operator's standing log and dropped pigment chunks |
| `R` | 0.32 | Roller radius |
| `yc` | 0.55 | Height of both roller axes |
| `zNip` | 0.75 | z of the nip centre (mid-plane between rollers) |
| `gap` | 0.02–0.10, default 0.04 | Nip opening (distance between surfaces) |
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
| low | 32 | 0.03125 | 48×72×48 | 39k at 1× batch (+50% pool) |
| medium | 48 | 0.02083 | 72×108×72 | 133k at 1× batch (+50% pool) |
| high (default) | 64 | 0.015625 | 96×144×96 | 315k at 1× batch (+50% pool) |
| ultra | 72 | 0.01389 | 108×162×108 | 448k at 1× batch (+50% pool) |

**Batch size.** `MillConfig.batch` is a volume multiplier on the seeded
material (0.5×–3× from the panel, 0.25×–4× via `?batch=`); the bank top is
solved from the volume (`bankTopY`), so 2× really is twice the putty. The
default batch is `GEOMETRY.bankVolume` = 0.15 units³ ≈ 1.2 L: the pocket
between the rolls plus a modest bank, of which a gap-thick sheet around the
front roll takes about 0.11 units³. Changing the batch rebuilds the sim. The HUD shows the
material on the mill as litres and kg (one sim unit ≈ 0.2 m, putty 1.1 kg/L)
and the particle count, so conservation is visible: the count only changes
when a pigment chunk is added or the sim is reset.

Each sim also reserves particle capacity for pigment chunks (50% of the
seeded bank): a tap adds a dab of new pigmented putty (radius 0.1, i.e. a
40 mm dollop, load `PIGMENT_LOAD` = 12 so it carries as much pigment as a
bigger chunk would) set down touching whatever is in that column over the
nip (the GPU column probe finds the surface: the bank, the nip floor or an
earlier chunk), instead of tinting existing particles. A dab this size is
what the nip can take quickly: a lump's intake time is its volume over the
nip's flux across its width (radius 0.14 measured ~1.5 s from tap to 80%
through at `low`, most of it the lump sitting wedged in the V while the
gap ate it; radius 0.1 is a third of the volume). When the pool is used
up, taps tint the surface (`addPigmentOnSurface`); Reset refills it.

The default nip gap (0.04) at `high` is ~2.6 cells wide; the sheet is 2–4 cells thick. That
is the minimum for a resolvable sheet; do not lower `cellsPerUnit` below 32.

**Initial material (the bank).** A slab resting on the nip:
x ∈ [0.05, L−0.05], z ∈ [zNip−0.25, zNip+0.25], y from the roller top surface
(`ySurface(z)` = yc + sqrt(R² − (z − zRollerAxis)²) for whichever roller is
under that z, or the nip floor `yc` inside the gap) up to `yc + R + 0.36`.
The bank must hold more material than a gap-thick sheet around the front
roll (2πR·gap per unit length) or the mill consumes it into a ring with no
bank in front of the nip (see docs/millref-notes.md §5b).
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
   the surface velocity. If `d < R + tackBand` (tackBand = gap + h, i.e. the
   thickness of the sheet the nip produces plus one cell of stencil slack):
   `v = vr` (full no-slip incl. normal — the sheet is carried around; in the
   channel itself, `|y − axisY| < 0.06`, a node whose render-raster density
   (last frame's `pmass / 8`; a gap-wide channel at rest rasters to ~0.65 at
   `low`) is past 0.8 is let go of in proportion,
   `v = mix(vr, v_free, (density − 0.8)/0.4)`, so the material's pressure can
   push excess back up rather than the band conveying it through) **except
   in the wedge above the nip** (`y > axisY + 0.06`, `z < frontAxisZ`, the
   region where the bank rests on the roll). There the roll is simply a
   no-slip wall: only the nodes inside its surface (`d ≤ R`) take `v = vr`,
   nothing else is prescribed, and g2p pushes particles out of the roll. The
   surface layer is dragged in as far as the particles' stencils reach (about
   a cell), the bank presses material onto the roll and packs the layer the
   nip is about to take, and when the channel cannot pass what arrives the
   material's own pressure (ν = 0.47) holds the rest back and the bank rolls
   instead of being force-fed.
   Measured (particles per 0.35 rad of sheet; 1650 at `low`, 5550 at `medium`
   is a gap-thick sheet at rest density): `low` 1.2 L 2000–2700 in every bin,
   `low` 2.4 L 2200–2900, `medium` 1.2 L 7700–8300, all steady from 1.3 s to
   3.9 s, nothing dropped, the sheet rendered without a hole. The channel
   itself is still velocity-prescribed, so it packs the sheet to 1.3–1.5× rest
   density (more with a heavier bank); that is the remaining dishonesty of the
   nip and it is bounded.
   Alternatives tried and rejected: Coulomb friction only (μ = 0.8) starves
   the nip once the bank is small, because the drag scales with the bank's
   weight (lacy, pulsing sheet); a hard tangential tack constraint through the
   whole band with the approach blocked freezes the returning sheet at the
   density it first formed with (700–1000 per bin at `low`, ~2800 at `medium`:
   half density, lacy on every preset once the seeded pocket is used up, and
   8800 particles dropped to the tray at `medium`); the same with the approach
   free packs the whole batch onto the roll at twice rest density in the first
   turn, then starves; a tack of bounded shear strength (1.0) pulls the batch
   onto the roll in pulses and leaves gaps; a deeper tack layer (2× the band)
   force-feeds and drops material off the underside; releasing the band's
   normal constraint where it is overpacked drops material off the underside
   too.
2. **Back roller**. In the wedge above the nip (`y > axisY + 0.06`,
   `z > backAxisZ`) it is the same tacky no-slip wall as the front roll:
   nodes inside its surface (`d ≤ R`) take `v = vr`, so the bank is drawn in
   from both sides. Everywhere else it is a separating contact with Coulomb
   friction, so the sheet peels off it cleanly below the nip and follows the
   front roll. If `d < R + backBand`: `vrel = v − vr`; `vn = dot(vrel, n)`;
   if `vn < 0`: `vt = vrel − vn·n`; `vt *= max(0, 1 − mu·(−vn)/|vt|)` with
   `mu = 0.4`; `v = vr + vt` (normal component removed). If `vn ≥ 0` leave v
   unchanged.
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
| `E` | 15 | Young's modulus (unit density; 60 makes a rigid slab that starves the nip) |
| `nu` | 0.47 | Poisson ratio (nearly incompressible, bulk modulus ≈ 80: at 0.35 the nip packed the putty to 1.5× rest density and carried 50% more than the gap allows; at 0.45 it still packed the sheet to 1.5× once the viscoplastic drag pulled the bank in, and the wrap ran short; at 0.49 (bulk modulus ≈ 500, dt at ~0.85 of the elastic CFL limit) the sheet was continuous at ~1.0–1.3× rest but the fed pile of a cut & roll blew up, speeds of 10–25; 0.47 is stable through a cut & roll) |
| `thetaC` | 0.025 | plastic compression threshold |
| `thetaS` | 0.03 | plastic stretch threshold (tensile cohesion; 0.0075 gives a lacy sheet after operator moves) |
| `mu`, `lambda` | derived | `mu = E/(2(1+nu))`, `lambda = E·nu/((1+nu)(1−2nu))` |

Fixed corotated (Stomakhin 2013) with the MLS-MPM stress form:

```
[U, Σ, V] = svd(F)                     (3×3, robust; see note)
Rot = U · Vᵀ
J = det(F) = Σ0·Σ1·Σ2
P = 2mu·(F − Rot) + lambda·(J − 1)·J·F⁻ᵀ
σ = (1/J) · P · Fᵀ                     (Cauchy stress, used in §3.2)
```

Plastic return (after the F update in G2P) is **deviatoric only**: split the
stretches into the volumetric part `J^(1/3)` and unit-volume deviatoric
stretches `Σi / J^(1/3)`, clamp only the deviatoric stretches to
`[1 − thetaC, 1 + thetaS]`, renormalise their product to 1 and put the volume
back, then rebuild `F = U·diag(Σ)·Vᵀ`. No hardening. Clamping the raw
singular values (the snow model) lets the material lose volume plastically,
so the nip packs it to several times rest density and swallows the bank
(docs/millref-notes.md §5a); putty yields in shape, not in volume, and the
pressure term must keep resisting compression.

The return is **viscoplastic** (Perzyna): the deviatoric stretch beyond the
yield range is not discarded at once but relaxes toward it,
`dev = devC + clamp(dev − devC, ±DEV_CAP) · exp(−dt / TAU_VP)` with
`TAU_VP` = 0.06 s and `DEV_CAP` = 1.0 (`g2p.wgsl`). For a short while the
putty carries shear like a viscous fluid (effective viscosity ≈ 2μ·TAU_VP),
so the drag of the roll surfaces reaches into the bank instead of stopping
in a thin slip layer at the roll: a perfectly plastic bank sat over the back
roll as a rigid lump moving at a fifth of roll speed while the rolls slid
under it (the dead pocket that footage of a real mill does not show); with
the relaxation its back half is drawn down into the nip at about half roll
speed and the mass shifts toward the front roll. Measured at `low`, 2.4 L,
z bands from the back roll to the front axis: mean speed of the back half
0.05–0.2 → 0.4–0.57, the lump over the back roll clearing within 4 s. The
extra drag would pack the sheet at the nip (1.5–1.8× rest at ν = 0.45, and
the wrap then ran short, leaving a bare stretch of roll circulating), which
is why ν went up: with a stiffer bulk modulus the excess is pushed back up
the wedge instead of compressed through the gap (0.47; 0.49 was better still
for the sheet but unstable under a cut & roll).

SVD note: implement a 3×3 SVD as Jacobi eigen-decomposition of `FᵀF`
(≤ 8 sweeps) to get `V` and `Σ²`, then `U = F·V·Σ⁻¹` with a guard for tiny
singular values, and fix the sign so `det(U)·det(V) > 0`. Alternatively use
the polar decomposition for `Rot` and the eigendecomposition only for the
clamp. Precision: f32 is fine.

Stability: sound speed `c = sqrt((lambda + 2mu)/ρ)` with ρ = 1; the measured
stable limit is `dt ≈ 0.8·h/sqrt(E)` (docs/millref-notes.md §2) and the
presets use `0.4·h/sqrt(60)`, i.e. a safety factor of 2 at E = 60 and 4 at
the default E = 15. Keep the constants of §4 the same across presets; only
`dt` and `substepsPerFrame` change.

---

## 5. Pigment: Mixbox latents and shear-driven dispersion

Each particle carries a 7-float **Mixbox latent** `z` (Sochorová & Jamriška
2021). Mixing pigments is a mass-weighted average in latent space, and
`latentToRgb(z)` (the Mixbox polynomial; port of `src/sim/mixbox.c`
`EvalPolynomial` + residual) is evaluated only in the render shader.

- Fresh silicone base is **clear**: it carries no pigment load and no colour
  of its own (its latent slot holds Titanium White but has zero weight
  everywhere it is used). Pigment is an opaque colourant in a transparent
  medium, so a node's colour is the Mixbox mix of the pigments present and
  the pigment load per unit mass sets the opacity; the base never whitens a
  streak. (Treating the base as white paint made every streak a pastel: a
  sheet 10% masterbatch by particles was 60% white in the mix, a grey dash
  where the reference footage shows a black streak.)
- Pigments are a palette of named real pigments with precomputed latents
  (generated once with the official `web/vendor/mixbox/mixbox.js`
  `rgbToLatent`, stored as constants in `src/color/pigments.ts`):
  Cadmium Red, Cadmium Yellow, Cobalt Blue, Phthalo Green, Ultramarine,
  Burnt Sienna, Ivory Black, Titanium White, plus "custom" from a colour
  picker (converted with mixbox.js at runtime).
- **Inject** kernel: `addPigment(center, radius, latent, strength)` blends
  particle latents inside the sphere: `z = mix(z, zPigment, strength·t)`,
  `t = 1 − |d|/radius`.
- **Pigment load.** Each particle carries a pigment load in `pos.w` (clear
  base 0, a masterbatch chunk `PIGMENT_LOAD` = 12). The raster accumulates
  per node the mass, the load and the load-weighted latent; `pack` writes
  `volA/volB` = density and the latent normalised by the node's load (the
  mix of the pigments only) and `volC` = load / 8 (same normalisation as the
  density, so `volC / density` is the load per unit mass). The raster uses
  the tight 8-node trilinear stencil so one-cell streaks are not averaged
  away before they are drawn, and the renderer integrates colour through
  the sheet's thickness.
- **Disperse** kernel (once per frame): for each particle gather the
  node-averaged latent `zg` (from the raster of §3.5) with the 27-node
  stencil; compute the shear rate `γ = ‖(C + Cᵀ)/2‖_F` from the particle's
  current `C`; then `z += clamp(k · γ · frameDt, 0, 1) · (zg − z)` with
  `k = config.dispersion` (default 0.02, UI 0–0.5; nearly off: colour mills as streaks that thin with folding, and at 0.1 a black chunk was 77% grey after 3 s). The load relaxes toward the
  neighbourhood's load per unit mass at the same rate; `zg` is the
  load-weighted (pigments-only) neighbourhood latent, and a particle that
  gains pigment blends `zg` in with the gained load, so a clear-base
  particle picking up pigment takes the neighbourhood's colour outright.
  Mixing therefore happens where the material is sheared — at the nip — and
  not in the resting bank.

---

## 6. Operator: cut, roll and feed (implemented)

A real operator cuts the sheet off the front roll, rolls it into a log, stands
the log over the nip and feeds it in end-first; the rolls consume it over a few
seconds and its spiral cross-section, every colour interleaved, is squeezed
out across the full roll width. That is the only source of mixing along the
roll axis. Modelled as a scripted kinematic move:

1. **Select** everything on the mill: the sheet on the front roll, the bank in
   the nip and whatever rides the back roll (an operator takes the lot). Each
   particle is parameterised by `(x, dr = d − R, s = θR)`: its depth off the
   front roll and its arc from the nip in the direction of rotation; it is
   binned by arc (`NB` = 128 bins over 2πR, per half of the width fold) and by
   depth (`NS` = 32 square-root-spaced slices over 0.5 units, fine near the
   roll so a gap-thick sheet is resolved, coarse deep in the bank). Counts go
   into a small atomic buffer; no readback.
2. **Tables** (one workgroup): the histogram becomes the log. Bin `b` holds
   volume `n·Vp`; spread over the bin's footprint `ell × Lhalf` that is a layer
   `t = n·Vp / (ell·Lhalf)` thick, where the bin's arc `ell` is stretched
   beyond `ds` wherever the layer would exceed `1.5 g` per half (the operator
   kneads the bank flat as it rolls it in). The spiral is then wound bin by
   bin from the bank end (the bank is the core, the sheet wraps around it):
   the angle advances by `ell / (ρ + T/2)` and a bin's inner radius is the
   outer surface of the bin one full turn earlier at the same angle (`rc = 2g`
   on the first turn), so layers of varying thickness stack without
   overlapping. Within a bin a particle lands at the radius that gives it its
   share of the layer's area (uniform in `r²`), ordered by the bin's depth
   CDF, and along the axis at `x` (or `L − x`: the width fold, the folded half
   as the outer sub-layer). Every bin is volume-preserving, so the log is never
   denser than the material was (validated: ≤ 1.6× rest with a trilinear
   raster, the same as the sheet it came from). The header stores the log's
   radius and base: the lower end face rests just above the roll tops (nothing
   is left on the mill), never squashed against the ceiling.
3. **Roll** (`FOLD_ROLL_SECONDS` = 1.2 s): each particle flies from where it
   was to its place in the standing log (smoothstep). The log stands tilted
   `FOLD_TILT` (0.42 rad) from vertical toward the viewer, axis
   `a = (0, cos, sin)`, over the nip at `x = L/2`.
4. **Feed** (`FOLD_FEED_SPEED` = 0.15 units/s along the axis): the held log
   descends along `−a`; a particle that reaches the release plane is released:
   flag cleared, `C = 0`, `F` kept, `v` = feed velocity, P2G affine rebuilt.
   The release plane is one cell above the **live pile** the nip is eating:
   g2p reduces, every substep, the top of the settled material under the
   log's footprint (node mass ≥ 2 particles; material released less than
   0.5 s ago does not count, otherwise the log would let go of itself in a
   cascade), falling back to the roll tops + 3h. So nothing is let go in
   mid-air and the log is never pushed into a pile the nip has not taken.
   The move ends when the log is used up (`FOLD_DURATION` ≈ 10.5 s: roll, then the
   half-length log plus its tilted end face at the feed speed); `finish`
   releases anything still held.

Nothing is placed inside existing material and no material is left behind. Exposed as
`GpuMpmSim.cutAndFold()`; the UI button is "Cut & roll" (F).

### 6b. Cut & fold (the flop)

The other move an operator makes, and the one most milling actually uses
(reference footage: silicone colour mixing on a lab mill): hold a knife
against the sheet on the **front face** of the front roll and draw it from
the roll end toward the middle while the sheet comes up past it. Because the
sheet moves during the stroke the cut is a diagonal on the sheet (the end was
cut first and has moved up since), and the freed flap is a **triangle**: a
point at the top near the roll end, wide at the bottom where the sheet leaves
the nip. The operator takes the bottom corner and folds the flap over the cut
onto the sheet beside it, toward the middle, so what was against the roll is
now on top and the doubled sheet rides up into the nip. Ends alternate. Same
kernels as the log, mode `P.fold.x` = 2 (cut from the x = 0 end) or 3 (from
the x = L end); `GpuMpmSim.cutAndFlop(side)`, UI button "Cut & fold" (C),
which alternates. No knife is drawn: the flap starts to fold when the button
is pressed.

1. **Select** the flap: the sheet on the front face (radial depth off the
   front roll below `flapDepth()` = 3 gap + h, angle `FLAP_TH_MIN`..
   `FLAP_TH_MAX` = 0.08..1.92 rad down from the crown, so from just in front
   of the crown line to a little below the axis level) on the cut side of the
   diagonal `cutX(θ)`, which runs from 0 at the top to L/2 at the bottom. It
   is binned by arc (`NB` bins) and depth (`NS` linear slices) and flagged.
   The rest of that half's top-of-mill material (above the axes and the mill
   surface, between the crowns' 50° lines) is binned by z (`NB` bins over the
   domain depth) and height (`NS` slices over one unit above the axes), for
   the part of the fold that reaches back over the crown.
2. **Tables**: per arc bin the flap's thickness (the 97th percentile of its
   depth, at least a cell); per z bin the top of the material behind the
   crown (the same percentile of its column, or the mill surface where empty).
3. **Fold** (`FLOP_SECONDS` = 0.8 s): in the sheet's own coordinates (u along
   the roll from the cut end, s down the front face from the crown) the flap
   is mirrored across the cut line from (0, 0) to (L/2, S). Its image lies on
   the sheet beside the cut, toward the middle, with a corner reaching just
   past the crown. A particle lands at its image, outside the sheet there by
   the flap's thickness less its own depth (roll side up), or, where the image
   falls behind the crown, on the bank top there at z = crown + R·θ'. It gets
   there as a page turning on the cut: straight toward its image, lifted up by
   `FLOP_LIFT` = 0.35 times its distance from the hinge (relative to the
   corner's), so halfway the flap stands on the cut line with the corner at
   the top. Nothing goes further forward than the front face; the corner rises
   about half a roll radius above it.
4. **Release** at rest, `F = I`, `C = 0`, as for the log. The roll carries the
   doubled sheet up into the nip; the bare patch on the cut side is covered
   again by the sheet coming up from the nip within half a turn.

Every fold doubles a wedge of the sheet toward the middle; alternating ends
walks material in from both sides, which is where a real mill's lateral
mixing comes from. The diagonal cut is more interface per move than a
straight one, and moving a wedge of sheet rather than half the mill keeps the
batch from sloshing from side to side. Cut & roll remains the move that
reaches from one end of the roll to the other.

---

## 7. Time stepping and presets

`quality` presets (`src/config/mill.ts`):

| preset | cellsPerUnit | dt | substepsPerFrame | target device |
| --- | --- | --- | --- | --- |
| low | 32 | 1.6e-3 | 8 | integrated / mobile GPU |
| medium | 48 | 1.1e-3 | 12 | laptop GPU |
| high | 64 | 8e-4 | 16 | desktop GPU (default) |
| ultra | 72 | 7.1e-4 | 18 | discrete GPU |

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
2. Intersect the domain box; march through `volA.density` with step
   `0.5h` (trilinear, `rgba16float`). A coarse max-density mip (one value
   per 4×4×4 block with a one-texel halo, rebuilt by a small compute pass
   before every render) lets empty blocks be skipped in one step. Surface =
   first sample with `density ≥ 0.45`; refine with 3 bisection steps.
   Normal = a 75/25 blend of the gradient of a 2×2×2 box-filtered density
   (±1.5 texels) and a tight central difference (±0.8 texel); the
   disagreement between the two widens the specular lobes so cell-scale
   noise reads as satin, not sparkle.
3. Colour at the hit: five samples along −n behind the hit, weighted by
   density and depth; the latent is the pigment-load-weighted mix of
   `volA/volB` (the pigments only), decoded with `latentToRgb` → albedo
   (sRGB-linearised for lighting); the mean pigment load per unit mass
   `load = volC / density` gives the opacity
   `pigment = 1 − exp(−PIGMENT_OPACITY · load)` (`PIGMENT_OPACITY` = 6: a
   sheet 10% masterbatch by particles, load 0.6, is 97% opaque; the faint
   fringe a chunk sheds into the base, load 0.2, is 70%). The base
   putty is **clear, not white**: where there is no pigment the surface is
   rendered translucent (the analytic scene behind the hit shows through a
   thin sheet, and the bank goes milky with thickness); pigment makes it
   opaque with the pigments' own colour, never a pastel of it.
4. Shading: key light (warm, upper right front), fill (cool, left), rim from
   behind; Blinn-Phong specular with a wide + narrow lobe (silicone putty is
   glossy); ambient occlusion from density sampled along the normal plus an
   analytic hemisphere occlusion by each roll (darkens the crease between
   bank and rolls); a thin-sheet transmittance term (thickness from a short
   march along −n, transmitted colour ≈ albedo²) so sheet edges glow.
5. Rollers: analytic ray/cylinder intersection with end caps; brushed metal
   shading with a rotating stripe pattern (angle = `rollerAngle`) so rotation
   speed is visible. Depth-composite against the volume hit.
6. Tray/floor plane with a soft contact shadow under the front roller;
   translucent end-guide plates at x = 0 and x = L; background: dark
   vertical gradient.
7. Output sRGB. The canvas uses `navigator.gpu.getPreferredCanvasFormat()`.

Render at `min(devicePixelRatio, 2)`. Provide `Renderer.setVolumes(volA, volB)`
so the renderer can be tested with a procedural volume without the sim.

---

## 9. UI (touch-first)

- Bottom bar: a drop-slot strip (`DROP_SLOTS` = 6 fixed x positions along
  the roll, evenly spaced between margins that keep a whole chunk clear of
  the end guides; the operator picks one, `?slot=` presets it), a chunk-size
  picker (`PIGMENT_CHUNK_SIZES`: small / medium / large, radius 0.06 / 0.1 /
  0.14; `?chunk=s|m|l`), the pigment palette (large round swatches; tap = a
  masterbatch chunk of the chosen radius set down on top of whatever is over
  the nip at the chosen slot, so repeated taps stack), a colour picker swatch,
  "Cut & roll",
  "Cut & fold" (alternating sides), "Clear pigment", "Reset", Pause.
- Right drawer (collapsible; hidden by default on narrow screens): sliders
  for roller speed (rpm shown), friction ratio, nip gap, dispersion, gravity;
  quality select; camera auto-orbit toggle; stats (particles, grid, fps,
  ms/frame, sim speed).
- Drag in the viewport orbits; wheel/pinch zooms; double-tap resets the
  camera to the front view.
- Keyboard: Space pause, R reset, F cut&roll, C cut&fold, 1–8 pigments, arrows speed/gap,
  `[` / `]` drop slot, `-` / `=` chunk size, P drawer.
- A separate page, mixer.html, samples pigment recipes outside the mill with
  the same latents and mixing rule (parts-weighted latent average), showing
  the Mixbox result beside the naive RGB average.

---

## 10. Files

```
index.html                      Vite entry (the v2 app)
src/main.ts                     glue and frame loop
mixer.html, src/mixer/main.ts, src/mixer/mixer.ts
                                the colour mixer page (shares src/color/pigments.ts)
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

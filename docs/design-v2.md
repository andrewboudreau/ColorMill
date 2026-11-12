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
| `domain` | (1.5, 2.25, 2.0) | Domain size (x, y, z); origin at (0,0,0); the headroom holds the operator's standing log, the hand carrying the cut edge and dropped pigment chunks |
| `R` | 0.32 | Roller radius |
| `yc` | 0.55 | Height of both roller axes |
| `zNip` | 0.75 | z of the nip centre (mid-plane between rollers) |
| `gap` | 0.02–0.10, default 0.04 | Nip opening (distance between surfaces) |
| back roller axis | (y=yc, z = zNip − (R + gap/2)) | further from viewer |
| front roller axis | (y=yc, z = zNip + (R + gap/2)) | nearer the viewer |
| `omega` | 0–6 rad/s, default 3.0 | Angular speed of the **front** roller |
| `frictionRatio` | 1.0–1.6, default 1.25 | back roller speed / front roller speed |
| `guideWidth` | 0.4–1.0, default 1.0 | open width between the side guides, as a fraction of `L` |

**Rotation sense.** Both roller surfaces move **downward through the nip**.
With angular velocity vectors along x: back roller `ω_b = (+omega·frictionRatio, 0, 0)`,
front roller `ω_f = (−omega, 0, 0)`. Surface velocity at a point `p` is
`v = ω × (p − axisPoint)`. (Check: top of the back roller moves +z, top of the
front roller moves −z, so both move toward the nip and then down.)

**Side guides.** The bank is confined along x by two plates (the mill's
guide cheeks), symmetric about the middle of the rolls and `guideWidth · L`
apart: at `x = L/2 ± guideWidth · L/2`. At the default 1 they stand at the
ends of the rolls, where they are the domain walls at x=0 and x=L. Walls
are *separating* (no penetration, free tangential slip). The operator slides
the plates together to work a narrower strip of the rolls
(`MillParams.guideWidth`, a drawer slider and `?guideWidth=`):

- A plate spans the wedge between the rolls: in y from the channel's lower
  end (`yc − 0.06`) to its top, in z from 0.06 behind the back roll's axis to
  0.06 in front of the front roll's. Inside that extent a grid node within
  the wall margin (`2h`) of a plate's face may not move outward (§3.3 step
  3). The sheet round the front and underside of the front roll is outside
  it and is not confined: nothing pushes it along the roll.
- A moved plate slides at `GUIDE_SPEED` = 0.15 (the simulator holds where
  the plates stand, `SimStats.guideOpen`, and moves it toward the setting
  every substep). While a plate closes, the nodes at its face move inward at
  least at its speed, so it ploughs the bank in front of it. Nodes inside a
  plate's body are moved inward at 0.3: that is for the edge of the sheet,
  which the roll brings round from outside the plates into the wedge for a
  turn or two after they have closed.
- Narrowing does not remove putty: the same batch in a narrower strip is a
  bigger bank. From the full roll to 0.6 on a running mill the share of the
  batch in the wedge between the rolls goes from about a third to over half
  (0.35 to 0.59 at `low`, 0.34 to 0.55 at `high`), and no particle is left
  outside the plates 10 s later (`tools/perf/guides.mjs`, `low` and `high`;
  at 0.4 the heap stands over the plates). The plates' top is
  `guideTopY(width)`: 0.15 above the seeded default batch at the ends, and
  higher the further in they stand.
- What is laid out along the roll fits between the plates: a new batch is
  seeded between them to the same height (`bankBounds`; the batch size is a
  charge for the whole roll length, so a strip 0.6 wide gets about 0.6 of
  the putty and the bank and sheet are the same in section), and the drop
  slots and a stamped picture are spaced between them. Reset rebuilds the
  simulator when the plates have moved since it was built.
- The operator's moves (§6) are unchanged: they work about the middle of the
  rolls and take the sheet's width from the particles.

**Tray.** The floor y=0 is a separating wall with Coulomb friction.

The **grid** is a uniform lattice with spacing `h = 1/quality.cellsPerUnit`.
Node counts are `N = round(domain / h) + 1` per axis; see
`MillConfig.gridDims()`. Presets:

| preset | cellsPerUnit | h | cells (x,y,z) | seeded particles (approx) |
| --- | --- | --- | --- | --- |
| low | 32 | 0.03125 | 48×72×64 | 39k at 1× batch (+50% pool) |
| medium | 48 | 0.02083 | 72×108×96 | 133k at 1× batch (+50% pool) |
| high (default) | 64 | 0.015625 | 96×144×128 | 315k at 1× batch (+50% pool) |
| ultra | 72 | 0.01389 | 108×162×144 | 448k at 1× batch (+50% pool) |

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
| `flags` | 1 float (bit field as int) | bit0 = kinematic (under operator control); bit1 = in the operator's hand (kinematic, but scattered to the grid in P2G); bit2 = a held pigment chunk's (kinematic, feeding in, §5) |
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
There is no clear pass. The node accumulators `mass` and `mom.xyz` (i32
fixed point) have one reader, the grid update (§3.3), which zeroes them as
it reads them (only where P2G scattered something, so empty nodes are not
written) and writes every node's velocity, zero where there is no mass. P2G
therefore always finds the accumulators empty. The node's mass rides in the
velocity's fourth component for G2P's pile test (§6). A separate clear pass
over the whole grid every substep was 8 to 10% of a frame on the software
GPU, and wrote 32 bytes per node per substep on a grid about 2% occupied.

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
gather; they move on their own script (§6). The exception is the hand
(bit1, §6b): P2G scatters its mass (`HAND_MASS` = 4 times a particle's)
and momentum, rigid, so it drags the putty with it; G2P still skips it.

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
   material's own pressure (§4) holds the rest back and the bank rolls
   instead of being force-fed.
   Measured (particles per 0.35 rad of sheet; 1650 at `low`, 5550 at `medium`
   is a gap-thick sheet at rest density): `low` 1.2 L 2000–2700 in every bin,
   `low` 2.4 L 2200–2900, `medium` 1.2 L 7700–8300, all steady from 1.3 s to
   3.9 s, nothing dropped, the sheet rendered without a hole. The channel
   itself is still velocity-prescribed, and next to the rolls the particles
   pack without their `F` recording it (those counts are 1.2–1.75× a gap-thick
   rest sheet, and at `high` it was 2.2×); the density tie of §4 is what
   bounds that now (1.08 / 1.26 / 1.35 at `low` / `medium` / `high`).
   The 0.06 is a length, not a number of gaps, on purpose: scaled to 1.5 gaps
   on every preset the rolls' grip on the bank is shorter at `medium` and
   `high` and a bank of the same proportions moves slower (back half 0.42 of
   roll speed against 0.69 at `high`).
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
   **Friction ratio.** With `v = vr` on every node of the band the material
   crossed the nip as a plug at the front roll's speed, whatever the back roll
   did: at ratio 1.25 the particles' shear strain rate there (`C_yz + C_zy`,
   which reads 0 for the band's rigid rotation) was 0.15–0.23 of
   `(U_back − U_front) / gap`, at 1.6 it was 0.05–0.08. So in the channel
   (`|y − axisY| < 0.06`, between the two axes) the band's velocity is

   ```
   s  = (d_front − R) / (d_front + d_back − 2R)      0 on the front roll, 1 on the back roll
   v  = vr · (1 + (ratio − 1) · s · fade · R / d_front)
   ```

   the front roll's rotation plus the back roll's extra surface speed in
   proportion to the node's place across the local gap: a Couette profile.
   `fade` is 1 down to 0.03 below the axes' height and falls to 0 at the
   channel's lower end, so the sheet leaves at the front roll's speed. `s` is
   not clamped at the roll surfaces but carried on into the rolls' nodes as
   far as `NIP_PROFILE_REACH` = 0.6 gaps (ghost nodes), so that what a
   particle in the gap gathers is the line itself and not one flattened at the
   walls. At ratio 1 the term is zero and the field is bit for bit the plug.
   **This is a model, not a resolved flow**: a gap of 1.3 cells has one layer
   of nodes in it, the profile is prescribed on them and the momentum the
   grid solved for is overwritten as everywhere else in the band (the back
   roll's contact is unchanged and gives no reaction). Only the drag part of
   the lubrication profile is there; the pressure-driven (Poiseuille) part,
   which in a real nip adds shear at the walls on the way in and takes it
   away on the way out, is not.
   Measured (`tools/perf/nip.mjs`, `low` / `medium` / `high`, shear in units of
   the nominal `(U_back − U_front) / gap`, flux over `U_front·gap·ρ0·L`):

   | ratio | shear before | shear now | flux before | flux now | sheet count before | now |
   | --- | --- | --- | --- | --- | --- | --- |
   | 1.0 | – | – | 1.12 / 1.12 / 1.19 | the same | 1.05 / 1.01 / 1.09 | the same |
   | 1.25 | 0.15 / 0.16 / 0.23 | 0.91 / 0.92 / 0.96 | 1.14 / 1.13 / 1.22 | 1.22 / 1.16 / 1.23 | 1.06 / 1.03 / 1.13 | 1.09 / 1.07 / 1.21 |
   | 1.6 | 0.08 / 0.05 / 0.07 | 0.91 / 0.95 / ~0.95 | 1.14 / 1.10 / 1.23 | 1.27 / 1.15 / 1.13 | 1.07 / 1.01 / 1.13 | 1.20 / 1.16 / 1.30 |

   No `F` resets, nothing in the tray, both operator moves clean on every
   preset at 1.25 and 1.6 (fastest particle 0.83–1.02 of twice the roll
   speed). The flux follows the mean of the two speeds at `low` and only
   part of the way at `medium` and `high` at 1.6 (0.88 of `gap × mean speed`):
   the channel's density release gives the rest back to the bank.
   Alternatives measured at 1.25: the same line clamped at the surfaces
   (no ghost nodes) 0.60–0.67 of the nominal shear; a step (front half of the
   gap at the front roll's speed, back half at the back roll's) 0.68–0.90,
   uneven between presets; a blend of the two rolls' rotations, which is not
   the plug at ratio 1, starved the nip at `low` (flux 0.89, bank's back half
   0.39 of roll speed); prescribing the nodes deep in the back roll too
   over-conveyed (count 1.2–1.39, the nip emptying in pulses); a reach of 1.2
   gaps measured the same shear and threw the fastest particle of a cut &
   fold at `high` to 1.22.
   **What it does to mixing, measured, is not what a mill does.** Steady
   milling (no moves) for 60 s, I(1/32) at 20 / 40 / 60 s: ratio 1.0, `low`
   0.155 / 0.088 / 0.068, `high` 0.227 / 0.192 / 0.176; ratio 1.6, `low`
   0.243 / 0.151 / 0.120, `high` 0.431 / 0.370 / 0.344 (with the plug, 1.6
   read 0.136 / 0.078 / 0.062 and 0.239 / 0.201 / 0.185: no different from
   1.0). The higher ratio mixes *slower*. Split by region, the lamina
   thinning `d ln s/dt = n·L'·n` is mostly the bank's (at ratio 1, 0.17 of
   0.21 /s at `low`, 0.20 of 0.29 at `high`); a striation reaches the nip
   lying along the sheet, where simple shear across the gap does not thin
   it (the nip's own share goes up 0–25%, and what the entry gains the exit
   gives back as the profile fades). Meanwhile the faster channel carries a
   thicker sheet, the bank is left with less of the batch (0.41 → 0.24 of
   the particles at `low`) and a small bank rides through as a lump rather
   than rolling (bank's contribution at 1.25 `low`: 0.013 /s). With the batch
   raised to give the bank its share back (1.12 at `low`, 0.56 at `high`)
   `high` at 1.6 is level with ratio 1 (0.248 / 0.198 / 0.174) and `low` is
   still behind (0.245 / 0.127 / 0.086). With the operator's moves the
   difference at the default ratio is small (0.139 / 0.018 / 0.008 at `low`,
   0.268 / 0.035 / 0.019 at `high`, against 0.115 / 0.013 / 0.007 and
   0.245 / 0.033 / 0.022). What a real nip adds, dispersive mixing from the
   shear stress, this model has no measure for. One more thing the ratio
   brings out: the gap in the sheet left by the start-up (the nip slice
   empty for a moment each revolution, there at ratio 1 until about 2 s)
   lasts to 8–10 s at `high` at 1.6.
2. **Back roller**. In the wedge above the nip (`y > axisY + 0.06`,
   `z > backAxisZ`) it is the same tacky no-slip wall as the front roll:
   nodes inside its surface (`d ≤ R`) take `v = vr`, so the bank is drawn in
   from both sides. Everywhere else it is a separating contact with Coulomb
   friction, so the sheet peels off it cleanly below the nip and follows the
   front roll. If `d < R + backBand`: `vrel = v − vr`; `vn = dot(vrel, n)`;
   if `vn < 0`: `vt = vrel − vn·n`; `vt *= max(0, 1 − mu·(−vn)/|vt|)` with
   `mu = 0.4`; `v = vr + vt` (normal component removed). If `vn ≥ 0` leave v
   unchanged.
3. **Side guides and domain walls**. Inside the guides' extent (§1), a node
   within `2h` of a plate's face: its x velocity toward the middle of the
   rolls is raised to the plate's closing speed (0 when the plate is at
   rest, which is the separating wall), or to 0.3 if the node is inside the
   plate's body. With the plates at the ends this is the domain wall's
   condition on the same nodes. For x, z faces and the ceiling: if inside a `2h` margin
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
gather mass  = Σ wgt · m[n]                             (for the density tie, §4)
gather C_new = (4/h²) · Σ wgt · outer(vel[n], dpos)     (MLS-MPM)
v = v_new;  C = C_new
F = (I + dt · C) · F
F = plasticReturn(F, mass, x)                   (§4)
x += dt · v
x = clamp(x, 1.5h, domain − 1.5h)
x = pushOutOfRollers(x)   // hard safety: if inside a roller, project to R + 0.25h along n
```

### 3.5 Frame-level kernels (once per rendered frame, after the substeps)
- **Pigment raster**: clear `pmass` + `plat[7]` node accumulators (i32, scale
  2^20 for mass, 2^18 for latent·mass); scatter `wgt·pMass` and
  `wgt·pMass·latent` with the same 27-node stencil.
- **Disperse** (shear-driven mixing, §5).
- **Pack**: write the coarse density volume (§8): one `rgba16float` 3D texture
  of node dims, `volA = (density, 0, 0, 0)`, where `density = pmass / 8`
  (so a fully packed cell ≈ 1). The solver-grid raster scatters mass only;
  its load and latent accumulators are the dispersion exchange's pools (§5),
  and only the fine raster's pigment is packed into render textures.
- **Colour raster**: the same scatter and pack a second time onto a finer
  lattice, in a box laid over the material each frame (§3.6). Its finest
  cell is the particle spacing, `colourCellsPerUnit` = twice the solver's
  cells per unit (64 / 96 / 128 / 144 at low / medium / high / ultra); its
  node counts are a budget (`colourNodes`, `colourGridDims`: the roll's
  whole length and as much height and depth as the budget buys, kept to the
  memory the raster used when it covered the domain and under the default
  128 MiB storage-binding limit for the 7-channel accumulator). It has its
  own accumulators and three `rgba16float`
  textures `finA = (load, lat0..2)`, `finB = (lat3..6)` (the latent
  premultiplied by the load, §5), `finC = (mass, coverage)`, load, latent,
  mass and coverage normalised by `(h/hF)³/8` so a packed
  fine cell reads ≈ 1, `finA.x / finC.x` is the load per unit mass and
  `finC.y / finC.x` is the covered mass fraction (0..1). **Coverage** is an
  eighth accumulator `pcov` (i32, scale 2^20) that sums `wgt·pMass` only for
  particles with `pos.w > COVERAGE_MIN_LOAD` (0.3): the load ratio alone
  cannot tell a cell one-eighth masterbatch (the edge of a lamina) from one
  evenly tinted, and the threshold sits below the load ≈ 1 that dispersion
  relaxes a milled few-percent batch to, so a homogeneous mix is fully
  covered while clear base and the faint trace a chunk sheds into it are
  not. The solver-grid raster shares the scatter but skips coverage (its
  binding is a 16-byte stand-in). The renderer takes
  pigment colour and load from these (§8) while the surface and the sample
  weights still come from `volA`: the sheet the nip makes is one or two
  solver cells thick, and a streak inside it was averaged into its cell
  before it was drawn; on the finer lattice the colour's detail limit is
  the particle spacing. Costs one more raster
  pass per frame (per particle, so independent of the box's size), a clear
  and a pack over the box's range in use, and
  the fine buffers and textures (64 bytes per fine node: 40 in the i32
  accumulators, coverage included, and 24 in the three textures; about
  116 MB at low, 225 MB at medium, 300 MB at high and ultra); a
  device whose storage-binding limit cannot hold the 7-channel accumulator
  falls back to the solver grid with a console warning. The dispersion
  exchange always runs on the solver grid.

### 3.6 The material's box

The domain is tall and deep for the operator's moves, but the material
fills little of it: about 2% of the solver's cells hold particles, and
while milling their bounding box is about 12% of the domain. The grid
passes therefore cover a box around the material, not every node
(`box.wgsl`; a 32-word buffer and indirect dispatch arguments).

- **Bounds.** G2P reduces the particles' cell range every substep: atomic
  min / max per axis, kinematic particles included, with a plain load first
  so nearly no particle writes.
- **The solver's box** (`step_`, one invocation after G2P): the cells'
  27-node stencils (`c−1 .. c+2`) plus a cell of slack for what the
  operator's script moves before the next P2G, as an origin the grid update
  adds to its node index and workgroup counts its
  `dispatchWorkgroupsIndirect` reads. The bounds are those of the very
  positions the next P2G scatters from, so nothing is assumed about how
  fast the material moves. Outside the box nothing was scattered: the
  accumulators are empty (§3.1) and no particle reads the velocity.
- **The raster's box** (`frame_`, one invocation before the frame's
  raster): the cells' 8-node stencils plus the slack, united with the
  previous frame's own box. The raster's clear and pack cover that union, so
  nodes the material has left are zeroed once, and the accumulators and the
  density texture are empty everywhere outside the current box (the
  dispersion gather, the grid update's packing test and the renderer rely on
  that).
- **The colour raster's box** (`frame_` again). The colour raster used to
  be a dense grid over the domain: 4.7 million nodes at high, about 1% of
  them holding material, cleared and repacked every frame, and capped at 88
  cells per unit by the binding limit while the particles are 128 per unit
  apart. It is now a box of fixed node counts (§3.5) laid over the
  particles' world bounds on a lattice anchored at the world origin, with
  cell `hF = finest · FINE_K / 8`. `frame_` takes the smallest `FINE_K ≥ 8`
  whose range (two lattice nodes of margin at each end, so a clamped lookup
  outside reads an empty texel) fits the node counts: at `FINE_K = 8` the
  colour is at particle resolution, which milling fits on every preset; a
  log standing over the nip can outgrow the box at the finer presets, and
  the cell then grows by eighths and shrinks back after, with a little
  hysteresis (the finer cell must fit with two solver cells to spare before
  the raster returns to it). Because the lattice is world-anchored, texels
  do not swim as the box follows the material. The raster reads the lattice
  from the box buffer; its clear and pack cover this frame's range and the
  last one's through an indirect dispatch; and `frame_` writes the world
  position of texel 0 and the cell into a 32-byte buffer the renderer binds
  as a uniform (`RenderVolumes.fineBox`), so the lookup needs no readback.
  At low the budget buys the whole domain at the particle spacing and the
  picture is unchanged, pixel for pixel; the gain there is the cropped
  clear and pack.
- **From the CPU.** Whenever particles are placed without a G2P (seeding,
  reset, a pigment chunk) the host writes the whole grid into every box
  (`resetBox`): the whole grid is always a valid box, and the next substep's
  G2P finds the real bounds.

Nothing about the physics or the density volume changes (a fingerprint run
hashing the particle state and the density volume through milling, a cut &
fold and a cut & roll matches bit for bit). On the software GPU the grid
update and the raster's clear and pack were about 15% of a frame, and the
colour raster's clear and pack 10 to 20%; the boxes take most of both (the
fingerprint run went from 455 to 293 ms a frame at low with the
self-clearing grid of §3.1). The box grows when the operator stands a log up over
the nip and shrinks again as the nip eats it.

---

## 4. Material model: elasto-viscoplastic putty (Hencky + Herschel–Bulkley)

This is the model that makes it a mill and not a puddle: the bank holds its
shape under gravity, yields and flows under the nip's shear, and does not
spring back. `src/sim/constitutive.ts` is a CPU mirror of the update in f64
(`tests/constitutive.test.ts`: the return is idempotent, never raises the
stress, keeps the volume, and steady simple shear reproduces the flow curve).

Constants (in `config.material`; URL overrides of the same names):

| name | default | meaning |
| --- | --- | --- |
| `E` | 22 | Young's modulus (unit density; at 15 a sheet pulled off the roll by its edge sagged as a film and pigment chunks smeared, at 22 it holds as a slab with the chunks still lumps and the bank still rolls; 60 makes a rigid slab that starves the nip) |
| `nu` | 0.47 | Poisson ratio. Bulk modulus `K = E/(3(1−2ν))` = 122, `λ` = 117, `μ` = 7.5. 0.45 (K = 73) leaves more margin under a cut & fold (clean at 1.3× the time step) and packs the nip more; 0.49 (K = 367) is past the time step's limit |
| `yieldStress` | 0.58 | `τ_y`, yield stress in shear (von Mises: `sqrt(2)·μ·|dev ε|`, the τ_xy of the equivalent simple shear). The old stretch clamp `[0.975, 1.05]` gave 0.3–0.6 depending on direction; 0.58 is its stretch side |
| `consistency` | 0.45 | `k` of the overstress |
| `flowIndex` | 0.35 | `n` of the overstress (below 1: shear-thinning) |
| `densityTolerance` | 0.1 | density tie: `J ≤ (1 + this)/ρ`; 0 turns it off |
| `volumeMin`, `volumeMax` | 0.7, 1.4 | where the elastic volume saturates |

**Elasticity** is Hencky (logarithmic strain), which makes the return below
exact and the pressure stiffen in compression:

```
[U, Σ, V] = svd(F)                     (3×3, robust; see note)
ε = log Σ                              (principal Hencky strains; Σ clamped to [0.05, 20])
ln J = ε0 + ε1 + ε2,   e = ε − ln J / 3
τ = U · diag(2μ·e + K·ln J) · Uᵀ       (Kirchhoff stress = J·σ; what §3.2 scatters)
```

The fixed corotated law this replaces has the pressure `λ(J − 1)J`, whose
stiffness `λ(2J − 1)` falls as the putty is compressed and is zero at J = 1/2:
the more the nip packed it the less it resisted. `K ln J` has the constant
stiffness K in ln J.

**Plastic return** (after the F update in G2P): putty yields in shape, not in
volume. The volume `ln J` is left alone and the deviatoric strain is brought
back radially toward the von Mises surface. The return is viscoplastic, with
a Herschel–Bulkley overstress: with `τ_tr = sqrt(2)·μ·|e|` the shear stress of
the trial state, solve

```
τ + μ·dt·((τ − τ_y)/k)^(1/n) = τ_tr        for τ in [τ_y, τ_tr]
e ← e · τ/τ_tr,     F ← U · exp(e + ln J / 3) · Vᵀ
```

by four Newton steps from above (`hbShear` in `common.wgsl`; the left side is
increasing and convex, so it cannot overshoot). In steady shear at rate γ̇ the
stress is the flow curve `τ = τ_y + k·γ̇ⁿ`: 0.58 at rest, 1.03 at 1/s, 1.6 at
10/s, 2.8 at 100/s. So the flowing putty carries shear
like a shear-thinning fluid and the drag of the rolls reaches into the bank
instead of stopping in a slip layer at the roll surface, while the elastic
strain it carries stays `τ/2μ` (the relaxation this replaces, a fixed 0.06 s
time on the stretch, let the stored strain grow with the rate to its cap).
Sweeps of `k` from 0.45 to 2 and `n` from 0.35 to 1 did not move the nip's
count or flux by more than a few percent; `k` = 4 breaks up at `high`. Clamping
the raw singular values (the snow model) lets the material lose volume
plastically, so the nip packs it to several times rest density and swallows
the bank (docs/millref-notes.md §5a).

**Volume: the density tie.** `det F` only integrates the velocity gradient a
particle saw. Next to the rolls (the hard push-out, the band's prescribed
velocities) particles pack without `F` recording it: with J ≈ 0.95 in the
sheet the grid measured 1.26 and 1.47 times rest density at `medium` and
`high`, the pressure never saw it, and the nip passed 1.5–2.2 gaps of rest
material where calendering theory allows 1.2–1.4. So G2P gathers the node
mass it already reads, adds the rest mass the rolls displace from the
particle's kernel (nodes inside a roll hold no particles: a smoothstep of the
signed distance to each roll over ±1.45 cells, within 0.04 of the node-level
integral, `tests/diagnostics.test.ts`), and caps the volume:

```
ρ = mass/(8·pMass) + rollGhost(x)
ln J ← max(min(ln J, ln((1 + densityTolerance)/ρ)), ln J − TIE_RATE·dt)
ln J ← clamp(ln J, ln volumeMin, ln volumeMax)
```

The tie only ever lowers J (a free surface under-reads ρ), and by at most
`TIE_RATE` = 10 /s: applied in one substep it once threw the fed log of a cut
& roll at `high`. The second line replaces the old reset of `F` to `I`
outside J ∈ [0.5, 2], which dropped a particle's whole pressure in one
substep and its shape history with it; only an `F` that is not a number
resets now. With no bound at all a prescribed motion (the hand, the fed log)
can store any amount of energy and a cut & fold blew up 10% past the preset's
time step.

Measured with `tools/perf/nip.mjs` (6 s of the default batch, then 5 windows;
`low` / `medium` / `high`; the count is the sheet's particles over those of a
gap-thick sheet at rest density, the flux is over `U·gap·ρ0·L`):

| | fixed corotated, clamp, reset | Hencky + HB | + volume bounds, tie |
| --- | --- | --- | --- |
| F resets /s | 27 / 483 / 7759 | 0 / 23 / 166 | 0 / 0 / 0 |
| sheet count | 1.09 / 1.59 / 2.24 | 1.09 / 1.47 / 1.75 | 1.08 / 1.26 / 1.35 |
| nip flux | 1.17 / 1.55 / 2.20 | 1.10 / 1.52 / 1.85 | 1.14 / 1.31 / 1.37 |
| sheet density (grid) | 0.90 / 1.26 / 1.47 | 0.90 / 1.18 / 1.24 | 0.87 / 0.93 / 0.93 |
| bank, back half / roll speed | 0.84 / 0.80 / 0.71 | 0.80 / 0.56 / 0.15 | 0.78 / 0.20 / 0.10 |
| bank particles | 9157 / 31117 / 78920 | 8887 / 36899 / 129391 | 9179 / 50975 / 167943 |

`low` (gap 1.28 cells, as at `high`) stays below the band at 1.08–1.14.

The sheet's thickness reads 0.84–1.03 gaps (90th percentile of the particles'
depth off the roll, which reads 0.9 for a lattice sheet one gap thick), so
the sheet is about one gap thick at 1.2 times rest density and the flux above
1 is in the density. (These columns were measured with the band a plug at the
front roll's speed; the friction ratio then added almost nothing to the flux,
1.14 against 1.12 at `low`. With the profile of §3.3 across the channel the
default ratio 1.25 reads flux 1.22 / 1.16 / 1.23, count 1.09 / 1.07 / 1.21,
sheet density 0.86 / 0.90 / 0.90, and `bank.mjs` at the default batches a
bank share of 0.25 / 0.27 / 0.25, back half 0.81 / 0.75 / 0.76 of roll
speed, half the chunks through in 3.8 / 3.3 / 3.2 s.) It does not swell after
the nip as a calendered sheet does, because the tack band prescribes its
normal velocity too (§3.3) and what leaves the nip stays as dense as it left.

**The bank and the batch.** The last two rows are not a property of the
model: a sheet that carries 1.3 gaps instead of 2.2 leaves the rest of the
same batch in the bank, and how a bank moves follows the share of the batch
it holds, in the old model as well. `tools/perf/bank.mjs` drops three chunks
on the bank and follows their particles through the nip:

| | bank's share of the batch | back half / roll speed | half the chunks through the nip |
| --- | --- | --- | --- |
| old model, default batch, `low` / `medium` / `high` | 0.30 / 0.26 / 0.26 | 0.69 / 0.74 / 0.73 | 3.2 / 3.5 / 3.6 s |
| old model, batch ×2 at `low`, ×2 at `high` | 0.45 / 0.35 | 0.29 / 0.31 | 5.2 / 6.3 s |
| this model, default batch | 0.31 / 0.46 / 0.59 | 0.59 / 0.13 / 0.10 | 3.9 s / not in 8 s / not in 8 s |
| this model, batch ×1 / ×0.65 / ×0.5 | 0.31 / 0.28 / 0.26 | 0.59 / 0.63 / 0.61 | 3.9 / 3.5 / 3.5 s |

Past a share of about a third the bank reaches over the back roll and its
back half sits there while the roll slides under it; the chunks on top are
not drawn in. Neither the yield stress (0.15 to 0.58), nor `k` and `n` (0.45
to 2, 0.35 to 1) nor the tie move that: at the default batch every one of
them leaves the back half at 0.09–0.20 of roll speed at `medium` and `high`.
At a bank of the old proportions `n` = 1 is a little quicker (0.54–0.76, half
the chunks through in 3.2 s) but starves `low` (count 0.95). So the default
batch is too large for the gap of the finer presets: the gap halves from
`low` to `high` and the batch does not. A batch of 0.65 at `medium` and 0.5
at `high` (or the gap of `low`, 0.04, on every preset: 0.50 / 0.54, 3.9 /
4.1 s, count 1.01 / 1.11) gives the old bank with the right flux; both moves
are clean there too.

SVD note: implement a 3×3 SVD as Jacobi eigen-decomposition of `FᵀF`
(≤ 8 sweeps) to get `V` and `Σ²`, then `U = F·V·Σ⁻¹` with a guard for tiny
singular values, and fix the sign so `det(U)·det(V) > 0`. Alternatively use
the polar decomposition for `Rot` and the eigendecomposition only for the
clamp. Precision: f32 is fine.

Stability: the sound speed is `c = sqrt((λ + 2μ)/ρ)` = 11.5 with ρ = 1
(`soundSpeed` in `mill.ts`), and with the Hencky law it does not change with
J. All three presets run at `c·dt/h` = 0.59 (`courantNumber`; 0.65–0.67 with
the material's own speed added). That is the margin, and it is not large: an
earlier version of this paragraph claimed a safety factor of 4 from
`0.8·h/sqrt(E)`, which leaves ν out; at ν = 0.47 `c` is 2.45·sqrt(E). Measured
by scaling `dt` (`nip.mjs --dt-scale`): steady milling holds to about
`c·dt/h` = 0.85; the operator moves are the tighter case, clean at 1.2× the
preset's step on all three presets (a cut & fold then peaks at 1.2–1.5 of
`2ωR`, at most one particle on the tray) and failing at 1.3× at `high`. Keep
the constants of §4 the same across presets; only `dt` and
`substepsPerFrame` change, and a stiffer `E` or `nu` needs `dt` scaled by
`1/c`.

---

## 5. Pigment: Mixbox latents and lamellar mixing

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
  `rgbToLatent` by `scripts/gen-pigments.mjs`, stored as constants in
  `src/color/pigments.ts`), in two families. **Silicone** (first in the
  palette, the 1–8 shortcuts): what a silicone colour house actually uses,
  which is iron oxides (PR101 red, PY42 yellow, PBr7 burnt sienna), titanium
  dioxide (PW6), carbon black (PBk7), phthalocyanine blue (PB15:3) and green
  (PG7), ultramarine (PB29), an azo red (PR170 naphthol) and yellow (PY3/74
  hansa), quinacridone magenta (PR122), a phthalo turquoise paste and a flesh
  paste. **Artist** (after a divider): the rest of the Mixbox oil-paint set,
  Cadmium Red / Yellow / Orange, Cobalt Blue / Violet, Permanent and Sap
  Green; cadmium and cobalt are avoided in silicone (RoHS / REACH / toy
  safety) so these are kept for range, not realism. Where a pigment is one
  of Mixbox's calibrated colours its sRGB is that pigment's; the naphthol
  red, the two iron oxides and the flesh paste have no published paste
  colour and their hex is estimated from artist swatches of the same
  colour-index pigment (the turquoise was sampled from the reference
  footage). No supplier publishes RGB values for its pastes, and there is no
  standard silicone palette: colour houses match Pantone / RAL targets from a
  small set of base pastes. Plus "custom" from a colour picker (converted
  with mixbox.js at runtime). Keys never change (`?drops=` links); old
  names stay as aliases.
- A paste chunk in the footage is a **masterbatch**, pigment already
  dispersed in silicone at a few percent by weight, so what the operator
  drops is a tint, not the pigment's masstone. The palette hexes are
  masstones (a phthalo blue chunk here is near-navy; a phthalo blue paste in
  the footage is a bright blue). Modelling the paste as pigment + white at a
  fixed loading is the natural next step and is not done yet.
- **A chunk feeds in, it is not swallowed** (`chunk.wgsl`). Let go whole on
  a running mill, a chunk sinks into the pool between the rolls and comes
  out as one patch a third to a half of the circumference long, mostly in
  the sheet's outer skin, which then circles for many revolutions without
  spreading (measured per chunk in twelve sectors around the roll at 2, 4,
  6 and 8 s): on screen a band flashes past every revolution and the rest
  of the roll stays white. On a real mill the chunk sits in the bank and
  every pass of the sheet takes a little of it. So a dropped chunk is held:
  its particles are kinematic (flags bit 0) and marked a chunk's (bit 2),
  and the lump sinks at the feed speed through a level, the top of the
  material it was set down on; a particle is let go as it crosses the
  level, with the lump's velocity. The feed is even and lasts
  `2r / speed`: `chunkFeed` seconds for a medium chunk (6 is about
  three revolutions; a drawer slider and a link key; the default, 0, lets
  chunks go whole). The same pigment is then laid around the whole circumference as
  a layer several times thinner, which is where the streak lines and the
  unequal ink show. The level is per chunk, in a small table (first
  particle, count, level, footprint; 32 chunks, beyond that a chunk is let
  go whole) the host fills when it adds the chunk. It only moves down:
  `settle_`, one invocation a frame, looks for material just under each
  chunk (node mass at the middle of its footprint and four points around)
  and lowers the level and the lump together where there is none, so a
  chunk follows the bank down as it drains. Letting a particle go when it
  touched material was tried first and fed most of a chunk in within a
  second: particles just let go raise the grid's mass under the lump and
  set off the layer above. A level fixed at the drop left chunks hanging
  in the air once the bank sank. An operator move (cut & fold, cut & roll)
  lets every held chunk go at once (`drop_`), as does switching the feed
  off; a chunk added during a move goes in whole. Held particles take no
  part in P2G or G2P, so material passes through the lump; the raster
  draws them like any others.
- **A softer paste was tried and does nothing.** Lowering a particle's
  shear modulus with its pigment load (to 30% of the base's at a
  masterbatch chunk's load, the bulk term left alone) was meant to let a
  chunk take the shear and smear between stiffer layers. Measured against
  the same run at full stiffness, with the chunk feed on: the spread around
  the roll, the share of the pigment in the sheet's outer skin and the
  surface area it colours were the same within noise at 4 and 8 s, and the
  renders match. The putty yields everywhere in the nip and the rolls
  dictate the motion, so a softer inclusion deforms like its neighbours;
  and with one velocity field per cell there is no slip between a chunk and
  the base inside a cell for softness to exploit. The change was not kept.
- **Inject** kernel: `addPigment(center, radius, latent, strength)` adds
  pigment inside the sphere, blended by load: with `a = strength·t`,
  `t = 1 − |d|/radius`, the particle keeps `(1 − a)·w` of its own pigment and
  gains `a·PIGMENT_LOAD` of the ink's, `w' = (1 − a)·w + a·PIGMENT_LOAD` and
  `z' = ((1 − a)·w·z + a·PIGMENT_LOAD·zPigment) / w'`. A clear particle
  (`w = 0`) therefore takes the ink's colour outright at any `a`, only
  thinner; `mix(z, zPigment, a)` blended in the clear base's unweighted
  white latent and gave every soft edge a pastel rim.
- **Stamp** kernel (`stamp.wgsl`): `stampImage(image, placement)` presses a
  picture onto the material seen from above. The image is a raster of up to
  256 px a side, 8 floats per pixel (a latent and an opacity), mapped onto a
  footprint `x0..x1 × z0..z1` on the bank (`fitStamp` in `src/stamp.ts` fits
  the picture's aspect inside the bank's footprint between the end margins;
  the top row goes to the back so it reads right from the front). `probeTop`
  records per solver column (i, k) the highest particle under the footprint
  (float bits, `atomicMax`), then `main` gives every particle within
  `STAMP_DEPTH` = 0.06 (about 1 cm; at least 1.5 cells) of its column's top the pixel it lies
  under, blended by load with `a = α·strength` exactly as inject does. The surface is found fresh each stamp, so a picture lands on
  the flat seeded bank (the use: a logo or a line of text laid on the putty
  before the mill pulls it in and streaks it) and on whatever the material
  has become. The app rasterises an image file (`Stamp picture`, or a file
  dropped on the viewport) or a line of text (`?stamp=Hello&ink=naphtholRed`)
  on a canvas, converts the pixels with mixbox.js once per distinct colour
  (text in a palette pigment uses its latent directly), and stamps them; the
  stamped text goes into the start link.
- **Laminae** (`lamina.wgsl`, `lamina_common.wgsl`, the thread raster in
  `raster.wgsl`; `src/sim/lamina.ts` is the CPU mirror, with the tests):
  colour lives on particles, but a real chunk becomes laminae far thinner
  than a particle. Every particle carries the unit normal `n` of its
  pigment's laminae and `q = ln(s/s0)`, the log of their thickness over the
  thickness they arrived with (one `vec4`), plus `ln s0` and the exchange's
  reference `mixed` (two floats of a second `vec4`, `mixs`, whose third is
  the undispersed load, see Agglomerates below). Both follow the flow exactly: a material
  surface element's direction is `m = F⁻ᵀ n0`, `dm/dt = −Lᵀ m`, so
  `dn/dt = −(I − n nᵀ) Lᵀ n` and `d(ln s)/dt = nᵀ L n`, with `L` the velocity
  gradient (the APIC `C`) less its trace (volume change is not mixing). The
  kernel integrates `m = (I − dt L'ᵀ + dt²/2 L'ᵀL'ᵀ) n`, `q −= ln|m|`,
  `n = m/|m|`: exact for simple shear (`s ∝ 1/√(1+γ²)`), exponential in
  extension, and a rigid rotation by `θ` a step costs `θ⁴/8` in `q`. The
  solver's `F` cannot stand in: after the plastic return it holds only the
  elastic part and the operator's moves reset it. Integrated once a frame
  from the last substep's `C` over the frame's time: once a substep gave the
  same thinning (mean `ln(s0/s)` of the pigment after 20 s of milling 1.88
  against 1.88 at low, 3.22 against 3.25 at high) for 8 to 16 times the
  dispatches, and a bank at rest does not thin at all either way
  (`LAMINA_PER_SUBSTEP` in `mpm.ts` switches). Kinematic particles keep
  their state. **Where a blob is made**: a chunk and an inject are one blob,
  `s0` = its diameter and `n` along its radius (the way the load falls
  off; `addPigment` takes an optional thickness); a stamp is a print layer,
  `s0` = its depth and `n` up; fresh pigment blended into a particle that
  already has some is blended by load in the log. Clear material carries an
  `n` and a `q` that mean nothing until it takes pigment. **The operator's
  moves**: the hand of cut & fold only translates its band, so the normals
  are as they were; the roll-up turns each normal with the script's own map
  when it lets go (sheet frame at the particle's place on the roll to the
  coil's frame, the half turn of the doubling, the lift), which is
  orthogonal, so `q` is untouched. **The picture**: the colour raster (fine
  lattice only) draws each pigment-carrying particle (load above 0.05) as a
  thread lying in its laminae, along `x̂ × n` (the mill's flow is across the
  axis), `s0/s − 1` particle spacings long up to the `streak` parameter
  (0 = points): `K` taps two fine cells apart, up to 6, each of weight
  `1/K`. **Unequal ink** as before (one particle in four draws its pigment
  3.4 times over, the rest 0.2 times, coming in as the thread lengthens),
  now fading with `s/Δ` once the laminae are thinner than the particle
  spacing: above the resolvable scale the smear reads as threads, below it
  the colour is the blend the exchange has made. The colour box
  (`box.wgsl`) is widened by half the longest line so no tap lands on its
  empty edge texels. The `streak` mill parameter (link key `streak`,
  default 12 spacings) is a picture setting. The slow feeding of a real mill, where a chunk sits in the bank for many
  revolutions and paints a stripe on every pass, is the chunk feed above
  (a bigger batch does not give it: it buries the pigment inside a thick
  sheet).
- **Pigment load.** Each particle carries a pigment load in `pos.w` (clear
  base 0, a masterbatch chunk `PIGMENT_LOAD` = 12). The raster accumulates
  per node the mass, the load and the load-weighted latent; a particle with no
  load adds to neither of the last two (there is no floor: a floor gave the
  clear base's white latent a weight in every thin edge). The fine raster's
  pack writes `finA/finB` = load and the latent **premultiplied by the load**
  (same normalisation as the load), and `finC` = mass, coverage and the
  undispersed load (Agglomerates below). Thus
  `finA.x / finC.x` is the load per unit mass, and the renderer divides the
  filtered latent by the sum of its four concentration channels (which sum
  to 1 per pigment, so the sum is the filtered load) to get the mix of the
  pigments only. Stored normalised instead, trilinear and mip filtering
  averaged each texel's colour with equal weight whatever its load, and an
  empty texel's zeros darkened and desaturated every edge. The raster uses
  the tight 8-node trilinear stencil so one-cell streaks are not averaged
  away before they are drawn, and the renderer integrates colour through
  the sheet's thickness.
- **Disperse** kernels (`disperse.wgsl`, once per frame): a conservative
  exchange of pigment between neighbouring particles through the solver
  grid's nodes, which happens only where the flow has thinned the pigment's
  laminae below what a particle can carry. A particle is a parcel one
  particle spacing `Δ = h/2` across. While its laminae are thicker than
  that, advection carries them and nothing is exchanged. Below it the parcel
  is striped, and a box `Δ` wide that samples stripes `s` thick differs
  from the local mean in proportion to `s/Δ`. So with `mixed` the log
  thickness already accounted for and `ref = min(mixed, ln Δ)`, a particle
  whose laminae reach a new minimum trades
  `β = 1 − (s / exp(ref))^k` of its pigment, and `mixed ← min(mixed, ln s)`.
  The product of `(1 − β)` over the frames is `(s_min/Δ)^k` whatever the
  frame length, the grid or the strain rate, so there is no rate constant to
  tune to the nip; elastic chatter in `s` trades nothing (only a new minimum
  does), and thickening undoes nothing. `k = config.dispersion` (default 1,
  the geometric value; UI 0–2; 0 = off) is the model's one number. This is a
  closure, not a derivation: the kinematics of `s` are exact, the step from
  "striped below the parcel size" to "this share is exchanged" is the
  argument above. (A per-frame `β = 1 − s_new/s_prev` gives the same product
  for a monotone `s` but counts every wobble of `s` as mixing.)
  Three passes. `mark_`: each particle with `β > 0` raises the 8 nodes round
  it to its `β` (an atomic max). `give_`: each particle's rate `α` is its
  own `β`, or, if it has room for pigment, the marked `β` round it
  (trilinear) times that room `1 − w/PIGMENT_LOAD`, so clear material beside
  thinning laminae takes part and neat masterbatch with thick laminae does
  not; it scatters `N·α·w` and `N·α·w·z` into the pools of the 8 nodes round
  it (trilinear `N`) and `N·α·m` into a per-node capacity. `take_` removes
  what the particle gave and takes from each node's pool the share its
  capacity term is of the node's. One rate for giving and taking, so an even
  field stays even; the shares of a pool sum to 1, so the batch's total load
  and each load-weighted latent channel change only by round-off (measured 1.2e-7 to 1.5e-7
  of the total over 60 s of milling with the operator's moves, growing
  steadily and always upward: a rounding bias, not yet tracked down).
  The latent follows as `z = (w·z)/w`, so a clear particle takes the colour
  of the pigment it receives outright. The lamina state travels with the
  pigment: two more pools carry `w·q` and `w·ln(s/Δ)`, and a particle's new
  `q`, `ln s` and `mixed` are the load-weighted blend of what it kept and
  what it took (what came in is accounted for down to its giver's
  thickness); it keeps its own normal. To keep the balance exact the passes
  compute the same fixed-point terms from the same inputs (2^21; a
  particle's share is its integer capacity term over the node's integer
  sum), and a particle that does not exchange does neither half: kinematic
  particles (a held chunk, material in the operator's hand) are skipped.
  The pools are the solver-grid raster's load and latent accumulators, which
  that raster does not fill (it scatters mass only), plus the capacity, the
  two lamina pools (and, in the same buffer, the undispersed load's) and the
  mark. Ten storage bindings, the software adapter's limit.
  Before this the rate was `α = k·γ·frameDt` from the particle's own shear
  rate: mixing wherever there was shear, at a rate that depended on the
  nip's strain rates and on `k`, whatever the laminae were doing.
  `tools/perf/mixcurve.mjs` prints the mixing-against-time table (intensity
  of segregation at boxes of 1/32, 1/16 and 1/8 of a sim unit, axial
  segregation, striation thickness) for a fixed recipe with or without the
  operator's moves.
- **Agglomerates: dispersive mixing** (`lamina.wgsl`, `agglomerate.ts` the
  CPU mirror with the tests, `AgglomerateConstants` in the config). The
  laminae and the exchange above are distributive mixing: they say where the
  pigment is. Fresh masterbatch is pigment in agglomerates, which tint
  weakly until stress breaks them down, and that is what the nip and the
  friction ratio are for. Each particle carries its **undispersed load** `u`
  (`mixs.z`, `0 ≤ u ≤ w`): the part of its load still in agglomerates. It is
  an amount, like the load, so it needs no new buffer and no new rule:
  a chunk, an inject and a stamp set `u` to the load they add; the exchange
  pools `N·α·u` beside `N·α·w` (a third slot per node in the lamina pool
  buffer, still ten bindings) and shares it out by the same integer shares,
  so agglomerates travel with the pigment that holds them and the exchange
  neither makes nor breaks any (e2e `pigment.spec.mjs` with the kinetics
  off: the batch's undispersed total holds to the load's own round-off);
  the fine raster accumulates it beside the coverage and `pack` writes it to
  `finC.z`.
  **Kinetics**, once a frame in the lamina kernel, from the same `C`:
  `γ̇ = √(2 D':D')`, `τ = τ_y + k γ̇ⁿ` (the flow curve of §4, not the stored
  elastic stress: measured, the two agree in the nip and the wedge to 0.05,
  and the flow curve needs no eigen-decomposition), and
  `du/dt = −u γ̇ [k_e (τ/τ_e − 1)₊ + k_r (τ/τ_c − 1)₊]`: nothing below the
  cohesive stress `τ_e`, above it a loss per unit of shear strain that grows
  with the excess stress (erosion), and an optional second term over a
  higher `τ_c` (rupture; off by default, `τ_c = 0`). First order in `u`, so
  the step is the exact decay `exp(−loss·dt)`. Kinematic particles are
  skipped. **What is sourced:** that agglomerates break by erosion and by
  rupture, that erosion is gradual, sets in at the lower stress and is first
  order, and that the applied shear stress over the cohesive strength
  governs it (Rwei, Manas-Zloczower & Feke, Polym. Eng. Sci. 1990 and 1991,
  read at the level of the abstracts). **What is not:** the rate expression
  (linear in the strain rate and in the excess stress) and every constant.
  **Where the threshold sits** (`tools/perf/stress.mjs`: flow-curve stress
  by region on the settled flow, 10th / 50th / 90th / 99th percentile; the
  "bank" is everything between the roll axes more than 0.06 over the nip):

  | `low` | ratio 1.0 | ratio 1.25 | ratio 1.6 |
  |---|---|---|---|
  | bank (25% of the particles) | 1.02 / 1.27 / 1.64 / 1.76 | 1.06 / 1.28 / 1.62 / 1.84 | 1.06 / 1.26 / 1.66 / 1.95 |
  | wedge over the nip (1.6%) | 1.15 / 1.48 / 1.71 / 1.76 | 1.39 / 1.57 / 1.80 / 1.87 | 1.65 / 1.71 / 1.92 / 2.01 |
  | nip (1.5%) | 0.96 / 1.12 / 1.28 / 1.40 | 1.38 / 1.41 / 1.48 / 1.55 | 1.68 / 1.70 / 1.71 / 1.74 |
  | sheet (70%) | 0.58 / 0.59 / 0.81 / 1.06 | 0.58 / 0.59 / 0.95 / 1.38 | 0.58 / 0.59 / 1.01 / 1.58 |

  The premise "the bank shears at about 1 /s, the nip far above" is not what
  the simulation does. The bulk of the bank is slow (median 3 /s), but the
  one or two cells of it that the rolls drag down into the wedge shear at 10
  to 30 /s, as fast as the nip, and there is ten times more of that layer
  than of nip: at any threshold the batch's erosion is mostly there, and the
  nip proper at the default ratio (1.41) is below the bank's 90th
  percentile. So no threshold separates "bank" from "nip". `τ_e = 1.75` is
  the bank's 99th percentile at equal roll speeds: the bulk of the bank and
  an even-speed nip do nothing, and what erodes is the dragged layer and the
  wedge, which the friction ratio loads (wedge median 1.48 → 1.57 → 1.71).
  `k_e = 1` puts a batch at the default ratio at 0.84 dispersed after a
  minute at `low`. **Presets.** The presets narrow the gap with the cell, so
  the nip's strain rate doubles from `low` to `high` while the bank's hardly
  changes. With the rates as they are `high` dispersed three times faster
  (0.92 after 20 s at ratio 1.0 with `τ_e` = 1.65); with every rate scaled by
  the gap ratio it dispersed at a third of `low`'s rate. The kinetics read
  the rate times `√(preset gap / 0.04)` (`agglomerateRateScale`), the
  measured middle and no more than that. The gap the user sets is not
  divided out.
  **Measured** (`mixcurve.mjs`, the batch's dispersed fraction `1 − Σu/Σw`
  after 20 / 40 / 60 s, three chunks, steady milling; with the operator's
  moves in brackets at 60 s):

  | ratio | `low` | `high` |
  |---|---|---|
  | 1.0 | 0.29 / 0.44 / 0.54 (0.55) | 0.54 / 0.79 / 0.90 (0.89) |
  | 1.25 | 0.69 / 0.78 / 0.84 (0.88) | 0.57 / 0.81 / 0.92 (0.94) |
  | 1.6 | 0.76 / 0.89 / 0.94 (0.96) | 0.60 / 0.72 / 0.77 (0.99) |

  At `low` the friction ratio does what it should. At `high` it barely
  matters between 1.0 and 1.25: there an even-speed bank is sheared against
  the rolls harder than at `low` (bank 90th percentile 14 /s against 10 /s at
  ratio 1.25), and that, not the nip, is what disperses. At `high` and ratio
  1.6 a fifth of the pigment falls off the front roll onto the tray within
  10 s and stays raw there (the median over the pigment is 0.97); that is
  the flow at that ratio, not the kinetics, and the operator's moves bring
  it back (0.99). Elsewhere the moves change little: they redistribute,
  they do not stress. Not preset-independent, then, and the constants are a
  game's, not a material's.
  **Appearance** (`raymarch.wgsl`). The load that tints is
  `w − (1 − 0.12) u`: agglomerates count for an eighth of their strength
  (`AGGLO_WEAK`), in the primary hit's opacity (both colour modes) and in
  the reflections' `puttyTint`. And they show: a cell half a colour-raster
  cell across is drawn as an opaque speck of the pigment's masstone
  (`albedo^1.5`) with a chance of 0.3 × the undispersed share of the pigment
  there, less where there is only a trace of it; a hash of the world cell,
  like the coverage dither. The latent is not touched, so nothing blurs: an
  under-milled batch is paler and speckled, a milled one clean. Neat
  masterbatch looks the same either way (its opacity saturates at an eighth
  of its load too). The specks are fixed in space, not carried by the
  material: on a turning roll they flicker rather than ride.
  `mixing()` reports the batch's dispersed fraction, its load-weighted
  10th / 50th / 90th percentile over the pigment, a histogram and the raw
  (< 0.1) and developed (> 0.9) shares; `inventory()` the undispersed total.

**Reference check.** Footage of a real two-roll mill (silicone, three bands
of about equal width across the front roll: red, yellow, teal) milled to a
terracotta. Sampling the frames gives the bands as `#d60720`, `#e8ef10`,
`#24a2a2` and the result as `#a4634b`; Mixbox mixes those three in equal
parts to `#a27242`, a few points yellower and within 0.1 per channel, while
the plain sRGB average is `#a18846`, a desaturated olive. The same
proportions with the silicone-family pigments (naphthol red, hansa yellow,
phthalo turquoise) give a warmer, yellower brown, because the palette
yellow is a stronger tint than the pale paste in the footage; the closest
recipe is 1½ naphthol red : ½ hansa yellow : 1 turquoise (`#a15c3f`, the
mixer's Terracotta starter, within 0.06 of the sample). A second clip (a violet, a white, a black and
an orange chunk on a clear slab, ending sky blue) does not add up under any
subtractive model and is not used.

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
2. **Tables** (one workgroup): the histogram becomes the log, one spiral per
   half of the width (each half is wound full width and the roll doubled
   afterwards, step 4). Bin `b` holds
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
   CDF, and along the axis at its own `x` (the `x ≥ L/2` half's `x` becomes
   `L − x` when the roll is doubled). Every bin is volume-preserving, so the log is never
   denser than the material was (validated: ≤ 1.6× rest with a trilinear
   raster, the same as the sheet it came from). The header stores the log's
   radius (the larger half's) and base: the lower end face rests just above
   the roll tops (nothing is left on the mill), never squashed against the
   ceiling.
3. **Peel and wind** (`windSeconds(omega)`): the operator rolls the material
   off the mill rather than the log appearing in place. A full-width coil
   sits on the crown of the front roll (`WIND_CONTACT` = 3π/2 around the
   front axis from the nip, axis along the roll); each half of the width
   winds its own spiral (the tables hold one per half, stride `TS` = 16
   floats per bin) and rests on the crown at its own radius. The bank and
   whatever is beyond the crown gather into the core at once (a
   `WIND_HOP_SECONDS` = 0.3 s hop). The rest of the
   sheet rides the roll toward the crown at the wind speed, the roll's own
   surface speed `omega·R` floored so the 1.5πR of arc from the nip exit to
   the crown takes at most `WIND_MAX_SECONDS` = 2.5 s (the operator pulls
   faster than a crawling roll delivers; strays off the roll wait where they
   are), and as its arc reaches the crown it hops onto the coil at its tabled
   radius and winding angle. The sheet joins at the coil's bottom, so the
   coil spins as it winds (rolling without slipping: the angle of a wound
   particle is π + Φ(arc wound so far) − φ(its own arc)) and rises on the
   crown as its outer radius grows. About 1.9 s at the default speed.
4. **Double** (`DOUBLE_SECONDS` = 0.5 s): the full-width roll is folded in
   half: the `x ≥ L/2` half swings up and over about the roll's middle (a
   half turn about the z line through `x = L/2`, its arc flattened by
   `DOUBLE_FLAT` = 0.75 to clear the ceiling) onto the top of the other
   half, the barrels' axes `DOUBLE_GAP` = 0.8 of the sum of their radii apart
   (each radius is its half's largest, so a little less presses them
   together), so the roll is `L/2` long and two barrels one on the other,
   every `x` position in its cross-section.
   With `rollStyle` = 'long' (`?roll=long`, the drawer's "Long roll"; `P.fold4.z`
   < 0) this step is skipped: the roll stays at its full length, its axis the
   mean of the two halves' centre lines, and stands up as one at the steeper
   `FOLD_TILT_LONG` = 0.85 rad so that 1.5 of roll clears the ceiling (top at
   about 2.1, far end 1.9 toward the viewer); lowering it in takes the full
   length at the feed speed.
5. **Lift** (`LIFT_SECONDS` = 0.6 s): the doubled roll swings up as a rigid
   body from the crown to the standing pose: a rotation through π/2 about the
   horizontal `n = (0, −sin tilt, cos tilt)` takes the roll's axis `x̂` (the
   line midway between the two barrels' axes) to the log axis `a` (in the
   plane of both), while its base slides from `x = 0` on the crown to the
   log's base over the nip at `x = L/2`, so the pair stands centred on the
   nip, the barrels side by side across the roll. The log stands tilted
   `FOLD_TILT` (0.42 rad) from vertical toward the viewer, axis
   `a = (0, cos, sin)`; its base height allows for the lower barrel. The
   pile-top window under it (g2p) is ±0.6 in `x`. `rollSeconds(omega)` =
   wind + double + lift (`FOLD_ROLL_SECONDS` is its value at the default
   speed, ≈ 3 s); the spec reads it from `operatorRollSeconds`. The held
   particles' velocities are the move's own over the last substep.
6. **Put it back** (`params.logFeed`, the "Log feed" slider, 0 to 0.6 units/s;
   captured when the move starts). **Drop** (`logFeed` = 0, the default): the
   substep after the roll ends, every held particle is released where it
   stands (`v = 0`, `C = 0`, `F = I`); the log stands on the nip and gravity
   and the rolls take it from there, so it goes in as one body. Lowering it
   in (below) released it a thin slice at a time, and the nip spread and tore
   each slice before the rest arrived. **Lower in** (`logFeed` > 0, the
   design's original move at `FOLD_FEED_SPEED` = 0.15): the held log
   descends along `−a` at that speed; a particle that reaches the release plane is released:
   flag cleared, `C = 0`, `F = I`, `v` = feed velocity, P2G affine rebuilt.
   The release plane is one cell above the **live pile** the nip is eating:
   g2p reduces, every substep, the top of the settled material under the
   log's footprint (node mass ≥ 2 particles; material released less than
   0.5 s ago does not count, otherwise the log would let go of itself in a
   cascade), falling back to the roll tops + 3h. So nothing is let go in
   mid-air and the log is never pushed into a pile the nip has not taken.
   The move ends when the log is used up (`foldDuration(feed)`: roll, then the
   half-length log plus its tilted end face at the feed speed, ≈ 10.5 s at
   0.15; roll + 0.02 s for the drop); `finish` releases anything still held.

Nothing is placed inside existing material and no material is left behind. Exposed as
`GpuMpmSim.cutAndFold()`; the UI button is "Cut & roll" (F).

### 6b. Cut & fold (the hand)

The other move an operator makes, and the one most milling actually uses
(reference footage: silicone colour mixing on a lab mill). It is the one
move here that the solver does rather than a script: the operator's hand
is scripted, the sheet is not. The sheet is cut **across** at the crown
of the front roll (`WIND_CONTACT`, as cut & roll's coil contact) and the
hand takes the cut edge: the band of sheet `HAND_CELLS` = 3 cells deep
along the arc just before the crown, the full width (the sheet only:
radial depth off the roll below `sheetDepth()` = 2.5 gap + h; the bank is
deeper and stays). Only that band is held.

A held particle in the hand carries flags bit 1 as well as bit 0. P2G
scatters its mass and momentum to the grid as it does any particle's,
`HAND_MASS` = 4 times heavier and rigid (no C, no stress), so where the
hand touches the grid it wins over the putty and over the roll's drag,
and the sheet it holds follows it instead of pulling it apart; G2P still
skips it (the hand sets where it goes). That is the whole coupling: the
band drags its neighbours through the grid nodes they share, those drag
theirs, and the sheet peels off the roll.

One pass is lift, pull, drop:

1. **The lift** (`HAND_LIFT_SECONDS` = 0.3): the hand rises `HAND_LIFT`
   = 0.2 off the crown (a smoothstep) while its speed back over the mill
   ramps up (another smoothstep) from 0 to the roll's own surface speed
   (`windSpeed(omega)`, as cut & roll). The roll keeps carrying the sheet
   past the cut away at that speed while the hand still holds, so the cut
   opens behind the hand, half a lift's worth of arc; the sheet before
   the cut, fed to the hand at the roll's speed, slackens by as much and
   takes it up as the hand gets going.
2. **The pull** (`handTimes(omega).pass` = lift + (`handPull(gap)` − half
   a lift's pull) / speed; `handPull` = `HAND_PULL` = 0.35 of `handBack`
   = 2R + gap = 0.68, so 0.24 back from the crown, short of the nip at
   0.34): the hand pulls the edge back over the top of the roll at the
   roll's speed, so sheet comes off the crown as fast as the roll delivers
   it and hangs from the hand to the crown.
3. **The drop**: just past the crown the band is let go with the hand's
   velocity, F = I. The flap flops back onto the front roll and the bank
   behind the crown, folded over itself, and the roll carries the fold
   into the nip. (Hauling the edge all the way to the back roll pulled too
   hard: a sheet's length stretched across the mill instead of a flap
   flopped over.)

The operator makes `params.foldPasses` such passes in a row (1 to 3, the
"Cut & fold passes" slider, default 2), `HAND_GAP_SECONDS` = 0.4 apart:
after a drop the hand is empty for the gap while the fold lands and the
roll brings fresh sheet up to the crown, then a fresh selection takes the
band there (a new cut) and the next pass begins. About 0.4 s a pass at
the default speed, so a default press is about 1.2 s
(`handTimes(omega).drops` are when each pass lets go, `.end` the last).
Earlier versions laid the edge down gently, folded it forward and back
again in laps with turns; the back-and-forth read as shaking, and the
third lap, dragging layers stuck to the bank and to each other, tore the
sheet into strands. Lift, pull, drop, repeated, is what the footage
shows.

Per pass one selection, one move script (`handOffset(t)`, a translation
of the band), one release. `P.fold` = (mode, t, lift seconds, 0),
`P.fold4` = (speed, distance back, 0, 0); the sim exposes
`operatorHandTimes` (speed, lift, pass, gap, passes, drops[], end) and
`operatorBusy` stays true through the gaps. Ends still alternate between presses in the UI, as an
operator alternates hands, though the move is the same from either.

Before the hand, the whole fold was scripted (a tongue peeled into a
bundle, then a flag fold of the halved strip in three page-turn flips on
the mill). Every particle was placed, so the sheet never sagged, hinges
were creases, the lengthwise fold was a cross-fade through the roll, and
the stack was dropped dead. Holding only the edge and leaving the rest to
the solver replaced three nested coordinate systems with one hand path.
`GpuMpmSim.cutAndFlop(side)` starts the press; UI button "Cut & fold" (C).
No knife is drawn: the pull starts when the button is pressed.

One press folds about a turn of the sheet into an eight-sheet triangle
half the roll wide and drops it into the middle of the mill;
alternating ends swaps which side ends up on top, and the rolls spread the
stack back out across the width, which is where a real mill's lateral
mixing comes from. Cut & roll remains the move that reaches from one end
of the roll to the other.

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

**The nip gap follows the preset** (`QualitySettings.gap`: 0.04 at low,
0.028 at medium, 0.02 at high and ultra; about 1.3 cells, the thinnest sheet
a grid carries, which the low preset's long-standing 0.04 already is). A
real mill's nip is far narrower for its rolls than 0.04 is for these (an
eighth of the roll's radius), and the gap sets how thin a chunk is spread.
Measured at medium with the chunk feed on, 0.028 against 0.04: the sheet
covered the roll face 81% against 46% after 2.6 s and 100% against 96%
three seconds later, and after 9 s the pigment coloured 36% of the roll's
surface against 16%; at 0.04 two of three chunks had hardly reached the
visible surface. The sheet came out 1.3 cells thick against 1.8. At high
with 0.02 the sheet covered the whole roll face within 1.9 s, 1.7 to 1.9
cells thick, at ordinary speeds (a short run: 3.3 s). The gap
is a live parameter: it starts at the preset's and follows the preset when
that changes, until it is set on purpose (the slider, the arrow keys or
`?gap=`, which pins it even at 0.04); the start link carries it only when
pinned, so a link's reader gets the gap of their own preset. `defaultConfig`
starts with the preset's gap too.

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
3. Colour at the hit, in one of two modes (`Renderer.mode`, `?render=`,
   the drawer's "Surface colour" toggle; depth is the default and surface
   the opt-in: the blended look is the tie-dye effect the mill is for, and
   surface's hard edges read as worse in use). **Surface**: three
   samples 0.6 colour-raster cells inside the hit, one at the hit and two
   half a cell away along the surface (against speckle), no depth, so the
   colour is the top-most material's: a pigment skin over clear base reads
   as pigment, clear over buried pigment reads as clear, which is the
   layering the particles carry through the nip. Surface mode draws pigment
   by **coverage** `cov = finC.y / finC.x` (§3.5), averaged over the same
   samples: `pigment = smoothstep(0.2, 0.5, cov + dither) · (1 − exp(−6 ·
   load / max(cov, 0.05)))`, so a cell under 20% covered draws clear, one
   over 50% draws at the load of its covered material (the clear share does
   not dilute it), and the tight ramp keeps a lamina's edge sharp where the
   mean load alone drew every cell a lamina touched near-opaque, a wash. The
   dither is a per-fine-cell hash of ±0.08 on `cov`, which breaks the ramp
   into cell-sized flecks that read as marbling rather than a gradient. This
   replaces the old load floor (loads under 0.06 drew clear): the trace a
   chunk sheds into the base is below the coverage threshold, so it is
   uncovered and invisible, while a milled few-percent batch (every particle
   at load ≈ 1) is fully covered and reads solid. **Depth** (the
   default): five samples along −n behind the hit, weighted by `volA`
   density and a depth decay, which gives the bank a milky body and blends a
   skin with what is under it, the soft tie-dye look. In both, the latent and the pigment load come
   from the colour raster `finA/finB/finC` (§3.5), the pigment-load-weighted
   mix of the pigments only, decoded with `latentToRgb` → albedo
   (sRGB-linearised for lighting); in depth mode the mean pigment load per
   unit mass `load = finA.x / finC.x` gives the opacity
   `pigment = 1 − exp(−PIGMENT_OPACITY · load)` (`PIGMENT_OPACITY` = 6: a
   sheet 10% masterbatch by particles, load 0.6, is 97% opaque; the faint
   fringe a chunk sheds into the base, load 0.2, is 70%). The base
   putty is **clear, not white**: where there is no pigment the surface is
   rendered translucent (the analytic scene behind the hit shows through a
   thin sheet, and the bank goes milky with thickness); pigment makes it
   opaque with the pigments' own colour, never a pastel of it.
4. Shading: key light (warm, upper right front), fill (cool, left), rim from
   behind, each a **soft box** in an analytic studio environment
   (`envRadiance`: three rectangles along the rig's directions whose edges
   blur with roughness, dark walls, ceiling and bench bounce; no texture).
   Diffuse is wrapped Lambert from the three lights plus the studio's
   ambient. The putty's gloss is a dielectric skin (F0 = 0.04, Schlick
   Fresnel, roughness 0.11 widened where the normal is cell-scale noise)
   mirroring that environment, the rolls and the bench (`reflectScene`,
   analytic), with specular occlusion from the AO. Ambient occlusion from
   density sampled along the normal plus an analytic hemisphere occlusion by
   each roll; a shadow march toward the key (dithered start, skipped where
   the key cannot reach); a thin-sheet transmittance term (thickness from a
   short march along −n, transmitted colour ≈ albedo²) so sheet edges glow.
   The clear base scatters a share `1 − exp(−thickness / 3.2 cells)` of the
   light as a faint milk and passes the rest: the analytic scene behind the
   hit along the **refracted** ray (n = 1.41), tinted by the path length.
   The gloss sits on top at full strength whether the body is clear or not.
5. Rollers: analytic ray/cylinder intersection with end caps; polished
   steel (F0 ≈ 0.6) whose reflection is sharp around the roll and smeared
   along the axis (five environment taps), with the rotating brush / wear /
   spiral pattern (angle = `rollerAngle`) modulating its reflectance so
   rotation speed is visible; the rolls mirror the bench, each other and,
   with `Renderer.reflections` (default on, `?reflections=0` in the
   harness), the putty next to them (8 density samples along the reflected
   ray + 3 colour samples). The end faces are turned steel with the bow-tie
   glint of circular tool marks. Depth-composite against the volume hit.
6. A wooden bench top with a brushed steel plate under the mill (soft roll
   shadows and contact darkening as before), a shop floor below it;
   clear side-guide plates where the simulator's plates stand (at x = 0 and
   x = L by default; slid in, they reach down between the rolls to the nip); background: dark studio walls
   with a pool of light at the horizon.
7. Display: exposure, a hue-preserving tone curve (a shoulder on the largest
   channel, so pigment mid-tones keep their hue and saturation; overshoot
   bleaches to white), sRGB encode, half an 8-bit step of interleaved
   gradient noise against banding. The canvas uses
   `navigator.gpu.getPreferredCanvasFormat()`.

Render at `min(devicePixelRatio, 2) * renderScale`, with a separate 0.5–1
resolution control that preserves the simulation. `Renderer.setVolumes(volumes)`
accepts procedural volumes for testing without the sim. Optional volume and
density versions allow unchanged frames and density-mip rebuilds to be skipped;
unversioned external volumes retain the always-refresh behavior. Fully opaque
putty skips shading the analytic scene behind it; translucent shading is unchanged.

---

## 9. UI (touch-first)

- Bottom bar: a drop-slot strip (`DROP_SLOTS` = 6 fixed x positions along
  the roll, evenly spaced between margins that keep a whole chunk clear of
  the side guides; the operator picks one, `?slot=` presets it), a chunk-size
  picker (`PIGMENT_CHUNK_SIZES`: small / medium / large, radius 0.06 / 0.1 /
  0.14; `?chunk=s|m|l`), the pigment palette (large round swatches; tap = a
  masterbatch chunk of the chosen radius set down on top of whatever is over
  the nip at the chosen slot, so repeated taps stack), a colour picker swatch,
  "Cut & roll",
  "Cut & fold" (alternating sides), "Stamp picture" (an image file pressed
  onto the bank, §5; an image dropped on the viewport does the same, and
  `?stamp=text&ink=pigment` stamps a line of text at the start), "Clear
  pigment", "Reset", Pause.
- Right drawer (collapsible; hidden by default on narrow screens): sliders
  for roller speed (rpm shown), friction ratio, nip gap, dispersion, gravity,
  back-roll friction, log feed (drop, or lower in at a speed; §6); an
  **experimental** toggle (`?experimental=1`) that lifts the sliders' caps
  (4× their top; URL values then accepted as written, floored at the
  minimum) with no stability promise past the normal range;
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
src/sim/shaders/*.wgsl          clear, p2g, grid, g2p, box, chunk, raster, disperse, lamina, lamina_common, pack, inject, stamp, fold
src/stamp.ts                    stamping a picture on the bank: raster size, footprint, pixels to latents (pure, tested)
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

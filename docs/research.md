# ColorMill Research Notes — Toward a 3D Voxel Mill

Working research log for moving ColorMill from a 2D Eulerian grid toward a
true 3D voxel simulation of a two-roll mill that folds, stretches, and
blends colored material.

> Status: brainstorming / reading list. No solver decision committed yet.
> See [Open question](#open-question-which-solver) at the bottom.

---

## Where we are today

The current C simulation (`src/sim/material_sim.{c,h}`) is a **2D Eulerian
grid**:

- Scalar fields per cell: `material`, `red`, `blue` (pigment masses).
- A velocity field: `vx`, `vy`.
- Roller-driven flow plus diffusion advects and blends pigment.
- `bandMask` / `mixMask` constrain where material lives and where the nip
  mixes.

This is, in spirit, a stripped-down *Stable Fluids* advection loop. The
jump we're considering is a **3D voxel grid** where material parcels carry
their own velocity vectors through the grid and colors merge based on those
vectors — which, decoded, is a **hybrid particle/grid solver** (PIC / FLIP
/ APIC / MPM).

---

## The mental model, decoded

What was described in chat → what it is in the literature:

| Described as | Actually is |
| --- | --- |
| "small 3D voxel grid" | An Eulerian simulation grid (cells store density + velocity) |
| "rollers apply force at the edges" | Moving / no-slip **boundary conditions** injecting momentum; the mill is two counter-rotating velocity boundaries with a **nip** between them |
| "the silicon goes through each grid with a force, the vector of each Si" | **Particles** (parcels of material), each carrying a velocity vector, transferred to/from the grid — a particle-in-cell scheme |
| "colors merged given the vector" | Pigment is **advected** (carried by the flow) and mixed where parcels share a cell |

So the whole thing is: *a hybrid particle/grid fluid solver, with advected
pigment, driven by moving roller boundaries.*

---

## Three separable problems

Keep these decoupled — they can be chosen and upgraded independently.

### 1. The motion solver (how material moves)

**Pure Eulerian grid** (smallest change from today):

- **Stam, *Stable Fluids* (SIGGRAPH 1999).** Semi-Lagrangian advection,
  unconditionally stable, ~150 lines. The minimal 3D path: add a `z` axis
  and `vz` to `MaterialSim`.

**Hybrid particle/grid** (the "Si with vectors" intuition, and the right
tool for folding/kneading — *see the deep dive below*):

- **Zhu & Bridson, *Animating Sand as a Fluid* (SIGGRAPH 2005)** — PIC/FLIP.
- **Jiang et al., *The Affine Particle-In-Cell Method* (SIGGRAPH 2015)** —
  APIC; fixes FLIP's noise and is the modern default transfer.
- **Stomakhin et al., *A Material Point Method for Snow Simulation*
  (SIGGRAPH 2013)** — MPM. Particles carry state and are rasterized to a
  voxel grid each step — almost exactly the "3D voxel integration grid"
  described. Gold standard for doughy/silicone material that folds and
  stretches without tearing.
- **Goktekin et al., *A Method for Animating Viscoelastic Fluids*
  (SIGGRAPH 2004)** — the springy/doughy silicone behavior itself.

### 2. The mixing math (why folding mixes)

A two-roll mill folding color is, mathematically, **taffy pulling / the
baker's map / chaotic advection**. The *folding* is what mixes, governed by
**topological entropy** — how fast nearby material gets stretched apart.

- ⭐ **Thiffeault, *The Mathematics of Taffy Pullers* (2018)** — the "soul"
  reference. Rollers folding material; pseudo-Anosov maps; why stretch
  factors come out as silver/golden ratios. **arXiv: 1608.00152**
  → <https://arxiv.org/abs/1608.00152> (PDF: <https://arxiv.org/pdf/1608.00152>).
  Published in *The Mathematical Intelligencer*
  (<https://link.springer.com/article/10.1007/s00283-018-9788-4>).
- **Aref, *Stirring by Chaotic Advection* (1984)** — the founding paper of
  chaotic mixing.
- **Boyland, Aref & Stremler, *Topological Fluid Mechanics of Stirring*
  (2000)** — topological entropy of rod/roller stirring protocols.

### 3. The color model (how pigment combines)

Linear RGB averaging (today's `mixColors`) is physically wrong for pigment —
red + green should darken toward mud, not gray. Use subtractive mixing:

- **Sochorová & Jamriška, *Practical Pigment Mixing (Mixbox)*
  (SIGGRAPH Asia 2021)** — easiest drop-in Kubelka–Munk mixing.
- **Baxter et al., *IMPaSTo: A Realistic, Interactive Model for Paint*
  (2004)** — Kubelka–Munk pigment mixing for paint.
- **Curtis et al., *Computer-Generated Watercolor* (SIGGRAPH 1997)** — fluid
  pigment transport on paper.
- **Chen et al., *Wetbrush: GPU-based 3D Painting Simulation*
  (SIGGRAPH Asia 2015)** — 3D bristle/paint with K–M mixing.

---

## Deep dive: MPM / APIC (the one to learn)

MPM = **Material Point Method**. APIC = **Affine Particle-In-Cell**, the
transfer scheme MPM normally uses. The short version:

> Material is represented as **particles** (the "material points") that carry
> all the physical state — mass, velocity, deformation, and here, pigment.
> A **background voxel grid** is scratch space rebuilt every frame. Each
> step: scatter particle state *to* the grid, solve forces/momentum on the
> grid, then gather the updated motion *back* to the particles and move them.
> The grid never stores permanent state — it's an "integration grid."

The single MPM step (the loop to internalize):

1. **P2G** (particle → grid): scatter mass and momentum to nearby grid nodes
   using interpolation weights (quadratic/cubic B-splines).
2. **Grid update**: compute forces (from each particle's deformation
   gradient + a constitutive/material model), integrate velocity, apply
   boundary conditions — *this is where the rollers push*.
3. **G2P** (grid → particle): gather the new velocity back to particles.
   FLIP/PIC/APIC differ only in *how* this gather is blended. APIC also
   carries a small affine velocity matrix per particle, which kills FLIP's
   noise and conserves angular momentum (so spinning/rolling looks right).
4. **Advance**: update each particle's deformation gradient and position.

Why it fits a mill better than pure Eulerian:

- Sharp color striations from folding stay crisp (color rides particles, not
  a diffusing grid field).
- Large deformation, folding, and self-contact "just work" — no mesh to
  tangle.
- The roller boundary is a clean grid-side velocity/collision condition.

### Suggested learning path

1. **Concept first — the 88-line taichi MPM (`mpm88`).** Smallest readable
   complete MPM. Read it, run it, watch P2G/G2P. (Taichi / Yuanming Hu's
   MLS-MPM demos.)
2. **The canonical course — Jiang, Schroeder, Stomakhin, Selle & Teran,
   *The Material Point Method for Simulating Continuum Materials*
   (SIGGRAPH 2016 course).** The from-scratch derivation everyone learns
   from. ACM: <https://dl.acm.org/doi/10.1145/2897826.2927348> · archive:
   <https://history.siggraph.org/learning/the-material-point-method-for-simulating-continuum-materials-by-schroeder-stomakhin-selle-and-teran/>
3. **Transfers — Jiang et al., APIC (SIGGRAPH 2015)** and the practical
   **MLS-MPM** (Hu et al., 2018) simplification. Understand why APIC > FLIP >
   PIC for rolling/shearing motion.
4. **Origin — Stomakhin et al., snow MPM (SIGGRAPH 2013).** Read once the
   loop makes sense; it motivates the plasticity/material model.

### Minimum viable MPM for ColorMill

- 3D grid, modest resolution (≈32³–48³ to start).
- Particles seeded inside the band; each carries `mass`, `velocity`,
  affine matrix `C`, deformation gradient `F`, and `pigment` (K-M coeffs).
- Two roller boundary conditions: prescribe grid-node velocity along each
  roller surface; enforce the gap/nip.
- Quadratic B-spline weights; APIC transfers.
- A simple weakly-compressible or fixed-corotated material for a
  silicone-ish feel.
- Render: splat particle pigment to voxels → composite (or raymarch).

---

## Color model: Mixbox WebGL integration

We're adopting **Mixbox** (Sochorová & Jamriška, *Practical Pigment Mixing*,
SIGGRAPH Asia 2021) as the color model — Kubelka–Munk pigment mixing, so
blue + yellow → green instead of the gray you get from linear RGB. Upstream:
<https://github.com/scrtwpns/mixbox>.

The repo flagged in chat ([`0xchaosbi/pigment-mixing`](https://github.com/0xchaosbi/pigment-mixing))
is a third-party **C++ copy** of Mixbox. We use the official build instead.

### Can it run in WebGL? Yes — it's barely a "port"

Mixbox already ships an official **GLSL shader** ([`shaders/mixbox.glsl`](https://github.com/scrtwpns/mixbox/blob/master/shaders/mixbox.glsl)),
and it's about as WebGL-friendly as it gets:

- **One regular 2D RGBA8 texture** — the LUT, **512×512**. Standard 2D
  sampling only: no 3D textures, no float textures, no extensions.
- **GLSL 1.20+ with a `__VERSION__` fallback** → runs unchanged on **WebGL1
  and WebGL2**.
- API drops in: `mixbox_lerp(a, b, t)`, `mixbox_rgb_to_latent`,
  `mixbox_latent_to_rgb`. Optional `MIXBOX_COLORSPACE_LINEAR` flag.

### The one gotcha: the LUT is encoded

The `mixbox_lut.png` distributed for native bindings is **4096×4096 and
*encoded*** — it must be decoded to a 512×512 LUT before the shader can use
it. The official **`mixbox.js` is self-contained**: it embeds and decodes
its own LUT (`mixbox.lutTexture(gl)`) and returns the shader source
(`mixbox.glsl()`). So the WebGL path needs **only `mixbox.js`** — the giant
PNG is not vendored.

### Integration recipe (what the PoC does)

```js
var lut = mixbox.lutTexture(gl);     // upload decoded 512x512 LUT
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, lut);
gl.uniform1i(loc('mixbox_lut'), 0);
// fragment shader = header + `uniform sampler2D mixbox_lut;` + mixbox.glsl() + main()
// then: vec3 c = mixbox_lerp(colorA, colorB, t);
```

Proof of concept lives at **`web/pigment.html`** (vendored library under
`web/vendor/mixbox/`, wired into `make web`). It draws two blend bars —
Mixbox on top, naive linear on the bottom — so blue + yellow visibly goes
green vs. gray. Verified numerically: blue `#002185` + yellow `#feec00` at
t=0.5 → Mixbox `#30953d` (green), linear ≈ `#7f8743` (mud).

### Why latent space matters for the 3D solver

Mixbox's **latent** representation is 7 coefficients (3 Kubelka–Munk pigment
weights + 4 residual). For mixing many colors you **average in latent space**
and convert to RGB once — not pairwise `lerp`. That maps cleanly onto the
grid/particle pipeline:

> Store pigment per particle/cell as the **latent coefficients**, accumulate
> weighted latents during advection / P2G, and convert latent → RGB only at
> render time.

This gives physically-correct N-way pigment mixing essentially for free and
replaces the linear averaging in `src/color.ts`.

### License ⚠️

Mixbox is **CC BY-NC 4.0 — non-commercial only**, with attribution. Fine for
ColorMill as research/education; a commercial release would need a license
from `mixbox@scrtwpns.com`. The third-party C++ copy carries the same terms.

---

## How this maps to the current code

- **Pigment mixing is now Mixbox-backed in the live renderer.** The deployed
  app is the C/emscripten build (`src/main.c`), not the TypeScript seed, so the
  color change landed in `CellColor()`'s "mixed" view via `src/sim/mixbox.c`
  (`Mixbox_PigmentRgb`). It treats each cell as a white silicone base carrying
  red/blue pigment and mixes them in Mixbox **latent space** — the exact
  store-latents-and-convert plan above, just with fixed endpoints precomputed
  offline (no runtime LUT needed yet). The C port is verified bit-exact against
  the official library. Generalizing to per-cell latent storage (more pigments,
  accumulate during advection) is the next step for the 3D solver.
- `MaterialSim` already separates **material mass** from **pigment**
  (`red`/`blue`) — the same split MPM needs (particle mass vs. carried
  pigment). Good foundation.
- Going Eulerian-3D: add `z`/`vz`, extend advection + masks to a third
  axis. Lowest effort; expect smearing at sharp color boundaries.
- Going MPM/APIC: the grid becomes scratch; introduce a particle array and
  the P2G/G2P loop. More work, but folding striations stay crisp — which is
  the entire visual point of a mill.

---

## Reading order (start here)

1. ⭐ Thiffeault, *Mathematics of Taffy Pullers* — <https://arxiv.org/abs/1608.00152>
   (read first; it's the conceptual soul of the mill).
2. `mpm88` taichi demo — get the loop in your hands.
3. SIGGRAPH 2016 MPM course — <https://dl.acm.org/doi/10.1145/2897826.2927348>
4. APIC (2015) → MLS-MPM (2018) for transfers.
5. Mixbox / Kubelka–Munk for the color model.

---

## Solver decision: MPM (implemented)

We chose **MLS-MPM** over Eulerian-3D — folding striations stay crisp because
color rides particles, not a diffusing grid field, which is the whole point of
a mill.

**Status: a working 2D MLS-MPM solver is now the live engine** in
`src/sim/mpm_sim.{c,h}` (the Eulerian `material_sim.c` is retired from the
build but kept for reference). It follows the `mpm88` lineage (Hu et al. 2018)
with APIC transfers (Jiang et al. 2015):

- **Particles carry state** — position, velocity, APIC affine matrix `C`,
  volume ratio `J`, and red/blue pigment. The background grid is scratch,
  rebuilt every substep (P2G → grid solve → G2P → advect).
- **Weakly-compressible fluid** material model (pressure from `J`) — good for a
  silicone-ish pool that shears and folds.
- **Two counter-rotating rollers** are rigid-rotation velocity boundaries
  submerged in the pool; their inner faces sweep down into the nip and knead
  the material. Free-slip box walls; gravity feeds the pool.
- **Pigment is Lagrangian** — it rides particles and mixes purely by transport
  (no diffusion). At render the particle pigment is rasterized to a grid and
  run through **Mixbox** (`Mixbox_PigmentRgb`), so folded red/blue interfaces
  read as real dark-purple blends, not gray.

Verified headlessly (no raylib needed for the physics): stable over thousands
of substeps, no NaNs, particles stay in-domain, mass/particle count conserved,
and rendered frames show the two colors kneading into folded mixing bands.
`main.c` renders the field square with roller overlays and Mixbox color.

### Next steps

1. **3D.** Add a `z` axis to position/velocity/`C`/grid — the loop is otherwise
   identical. Then volumetric rendering (slice or raymarch) instead of the 2D
   texture.
2. **Latents per particle** (see Color model section) — store the 7-float
   Mixbox latent on each particle instead of two pigment scalars, so any number
   of pigments mixes correctly during transport.
3. **Cut/fold operations** and proper viscoelasticity (deformation gradient `F`
   + plasticity) for true dough/taffy behavior.
4. Performance: the solver substeps ~40×/frame; a fixed-timestep accumulator
   and SIMD/threads (or a WebGL/WebGPU compute port) would lift resolution.

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

## How this maps to the current code

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

## Open question: which solver?

**Eulerian-3D** (smallest change, smears sharp color edges) vs.
**MPM/APIC** (more work, but crisp folding striations — the whole point of a
mill). Leaning MPM/APIC at small resolution. Decide before writing solver
code; a short design doc comparing data layouts for `MaterialSim` under each
should come next.

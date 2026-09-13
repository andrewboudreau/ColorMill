# millref — CPU reference of the v2 mill physics: findings

`src/sim/millref.{c,h}` + `tools/millref_main.c` are a headless C99/OpenMP
implementation of `docs/design-v2.md` §1–§5 and §11, literal to the spec
(quadratic B-splines, MLS-MPM APIC, fixed corotated + singular-value clamp,
sticky front roller / separating Coulomb back roller / walls / floor, frame-level
pigment raster + shear-driven dispersion, `addPigment`). Geometry, presets and
the bank seeding (incl. the xorshift jitter) mirror `src/config/mill.ts`
exactly: 60,499 particles at `low`, 205,316 at `medium` (spec: ~60k / ~204k).

```
make ref                                  # -> build/millref
build/millref --selftest                  # 3x3 SVD (2000 random + singular F) and Mixbox port
build/millref --rho 1 --tack 2 --frames 240 --every 20 --out build/ref
build/millref --rho 1 --cells 48 --dt 1.1e-3 --sub 12 --tack 2.4 --frames 240
```

Every spec constant is a flag (`--E --nu --thetaC --thetaS --gravity --omega
--ratio --gap --k --mu --tack --dt --sub`), plus investigation knobs:
`--rho` (density used in the force term; `0` = literal spec), `--tackblend`,
`--damp`, `--floorfric`, `--fixed` (emulate the i32 fixed-point grid),
`--gap2/--gapframe` (gap change mid-run), `--blue/--yellow` (injection frames,
-1 = off), `--bankh/--bankd` (bank size, geometry experiment), `--threads`.
Output: per-frame diagnostics on stdout and `diag.csv`, plus `side_NNNN.png`
(y–z cross-section: grey = all particles, colour = the |x−L/2|<3h slab, red =
back roller, blue = front roller, dark blue = tack band) and `front_NNNN.png`
(x–y, painter's order). PNGs are written directly (no PIL needed).

`build/` is **not** git-ignored (only `node_modules dist .vite coverage`), so
all run output for this investigation went to the session scratchpad:
`/tmp/claude-0/-home-user-ColorMill/aa654445-2361-52fe-9a6c-ffac0127390d/scratchpad/ref/<run>/`
(referred to below as `ref/<run>/`). Add `build/` to `.gitignore` before the
GPU agent starts writing screenshots there.

---

## 1. The big one: the spec's P2G force term is 2.6e5× too weak

§2 says `pMass = 1`, `pVol = h³/8`, "ρ = 8/h³ … the constants are already
expressed so that ρ cancels". It does not cancel. §3.2 multiplies the stress by
`pVol = h³/8` while the inertia is `pMass = 1`, so the acceleration produced by
a stress σ is `∝ σ·h³/8`, i.e. the material has Young's modulus `E/ρ = 60·h³/8
= 2.3e-4` in units where inertia is 1. Sound speed `c = sqrt((λ+2μ)/ρ)` = 0.019
(h/c = 1.6 s!). Gravity 2 with a bank 0.22 high needs a stress of ρgH =
1.2e5 to hold it up; the yield stress is ~2.

Literal run (`ref/spec_literal/`, cells 32, all spec values): the bank's top
surface drops 0.13 in 0.38 s = ½·g·t² — **free fall**. The slab compresses
like dust, is sucked through the nip, and 84% of the material is on the front
roller by t = 0.77 s (`side_0030.png`, `side_0060.png`). No NaN, because the
plastic clamp bounds the stress; it is simply a zero-stiffness dust.

**Fix (one line in P2G):** use the rest volume of a unit-density particle,
i.e. drop `pVol` from the stress term:

```
affine = −dt · (4/h²) · σ + pMass · C        // was −dt · pVol · (4/h²) · σ + pMass·C
```

Equivalently keep the formula and set `pVol = pMass / ρ₀` with `ρ₀ = 1`
(`--rho 1` in millref: `forceVol = pVol · (8/h³) / ρ`). Everything below uses
this scaling; with it E = 60, ν = 0.35 give c = 9.8 and the CFL numbers in §7
of the spec become meaningful (`0.4·h/sqrt(E)` is then 0.5·h/c).

Two nits in the same formula, harmless with the clamp (J ∈ [0.927, 1.023]):
§4 divides by J to get the Cauchy stress and then multiplies by the *rest*
volume; standard MLS-MPM uses `vol₀ · P·Fᵀ` (no /J). ≤ 8% stress error. Either
is fine; the GPU can skip the divide.

## 2. Stability (cells 32, ρ₀ = 1, E = 60): `dt = 1.6e-3` is safe with margin 2

`c = sqrt((λ+2μ)/ρ₀) = 9.81`, `h/c = 3.18e-3`. Each run 1.6 s of sim time, `ref/dt_*`:

| dt | dt/(h/c) | max‖v‖ | min J | verdict |
| --- | --- | --- | --- | --- |
| 1.6e-3 (preset) | 0.50 | 1.16 | 0.958 | clean |
| 2.4e-3 | 0.75 | 1.19 | 0.958 | clean |
| 3.2e-3 | 1.00 | 1.21 | 0.954 | clean (last good) |
| 4.0e-3 | 1.26 | 1.58 | 0.927 (= (1−θc)³, every σ pinned at the clamp) | noisy |
| 4.8e-3 | 1.51 | 3.46 | 0.927 | material ejected, 5% on the tray |
| 6.4e-3 | 2.01 | 1.40 | 0.927 | garbage (8% on the tray), still no NaN |

Largest stable substep ≈ `1.0·h/c` ≈ `0.8·h/sqrt(E)`; the presets use
`0.4·h/sqrt(E)` → **safety factor 2.0** for every preset (they all have the
same dt/(h/c)). Note **NaN never appears**: the yield clamp bounds the stress,
so an unstable run shows up as max‖v‖ ≫ roller speed (0.96), J pinned at
(1−θc)³, and particles on the tray — the §11 NaN check is necessary but not
sufficient; add "max‖v‖ < 2·ωR" and "0 particles below y = 3h".

Fixed-point emulation (`--fixed 1`: mass rounded to 2⁻²⁰, momentum to 2⁻¹⁶)
reproduces the float run to 4 digits (`ref/fixed/`) — the scales in §3.1 are fine.

## 3. Bank, nip feed, sheet, back roller (cells 32, ρ₀ = 1, spec constants otherwise)

`ref/rho1/side_0020/0060/0160/0240.png`:

- **Bank holds** its shape: the slab keeps sharp corners and its seeding
  lattice for the full 3 s; it does not slump at all (yield shear stress
  ≈ 2μ·(θc+θs)/2 ≈ 0.7 vs gravity shear ≈ 0.1). Yielding is confined to the
  layer the rollers drag. E = 30 or halved (θc, θs) behave identically to each
  other (yield ∝ E·θ) and feed the nip ~30% faster; still no visible slump.
  Slight slump would need ~E = 15–20; not recommended (see §5).
- **Nip feed & sheet**: material is dragged into the nip by both rollers, comes
  out as a ~2-cell sheet on the front roller, travels under it, up the front
  and over the top: at t = 2 s the ring is closed (`side_0160`), at 3 s it is
  re-entering the bank from the top. §11 numbers: sheet-on-front
  (d_front < R+4h, y < yc) = 0.23 at 0.5 s, 0.36 at 1.0 s, 0.33 at 1.3 s
  (≥ 0.15 within 3 s ✓); back roller 0.05–0.09 (< ⅓ of front ✓); nip shear
  rate ‖sym C‖ ≈ 4–5 /s.
- **Back roller sheds**: it never carries more than 0.10–0.13 and that is the
  bank overhang, not a sheet.
- No explosions: max‖v‖ ≤ 1.2 for 3 s (roller surface speed 0.96), 0 NaN,
  0 particles out of the domain, particle count constant, `pushOutOfRollers`
  fires on 10–40 particles per frame (0.05%).

## 4. Tack band (front roller), gravity, speed, ratio

`loose` = fraction of particles whose gathered raster mass is < 2 (packed
material gives 8): spray / dust. `high` = fraction above the initial bank top.
All at cells 32, 5 s of sim time unless noted (`ref/tack_*`, `ref/s_*`).

| band | g | result |
| --- | --- | --- |
| 1.0h | 2 | sheet peels off the underside by 1 s (`tack_1.0/side_0080`); by 3 s the domain is a cloud (`side_0240`). **Fails.** |
| 1.5h (spec) | 2 | holds; loose 1.5–4.7%. At g = 4, 5% of the material drops onto the tray by 3 s (`g4/side_0240`). Marginal. |
| 1.5h | 2, cells 48 | 1.5h = 0.031 < gap: fuzz on the underside, loose 3.4%, 1.2% flung (`cells48/side_0240`). |
| 2.0h | 2 | clean ring, bank intact, loose 1.7–3.7%, nothing flung. |
| 2.0h | 4 | holds for 5 s, nothing on the tray, loose ≤ 2.4%. |
| 2.4h = gap, cells 48 | 2 | loose 0.7–1.4%, nothing flung (`s_c48_tack2.4/side_0240`). Cleanest medium run. |
| 3.0h | 2 | clean ring, thicker sheet (more of the bank consumed), loose 3–5%, 3–10× more `pushOut` hits (band overlaps the back-roller test). |
| 3.0h | 4, 6 | holds, nothing on the tray. |
| 1.5h, blend 0.5 | 2 | thick loose sheet, slings the bank onto the back roller (`tackblend_0.5/side_0240`). **Do not blend.** |
| 2.0h, gap 0.10 | 2 | the middle of the nip is held by neither roller: **17% of the material on the tray** by 4 s (`s_tack2_gap.1/side_0200`). |
| 2.0h, gap 0.02 | 2 | fine (57k particles seeded), thin ring. |
| 2.0h + damping 0.002/substep | 2 | loose halves (1.0–2.2%); otherwise identical. Optional. |

Speed and ratio (2.0h): ω = 1.5 / 3 / 6 all clean, max‖v‖ = 0.59 / 1.17 / 2.4
(= 1.2·ωR), nip shear rate 2.5 / 5 / 10 /s; frictionRatio 1.0 / 1.25 / 1.6
indistinguishable except max‖v‖ (the back roller only sheds).

**Recommendation: full stick, `tackBand = max(2h, gap)`** (2h at low for the
default gap; 0.05 at medium/high; 0.1 for the widest gap — the band must
cover the sheet the nip produces, independent of resolution). PENDING:
gap 0.10 with 2.6h / 3.2h (`ref/s_gap.1_tack*`).

## 5. Things the four material constants cannot fix

**(a) The plastic return compacts the material.** Packed material gives a node
mass of 8; the ring around the front roller reaches 22–36 at cells 32 (3–4×
compaction, `maxNodeMass` in every `diag.csv`) — the nip squashes putty
plastically instead of rejecting it. The singular-value clamp is the *snow*
model (θc = 0.025, θs = 0.0075 are literally the Stomakhin 2013 snow numbers):
it allows unbounded plastic volume loss and has no tensile strength. Two
consequences: the bank is swallowed into a thin ring after ~4 s
(`ref/disperse/side_0480.png`) instead of rolling in front of the nip, and the
re-entering sheet flings loose particles (§4 `loose`). Raising θs for cohesion
makes it worse, not better: with θs = 0.03 / 0.06 / 0.12 the slab cannot
tear, gets wrapped onto the front roller as one lump, is compacted 12–16×
(node mass 100–128) and rides around forever with **zero nip shear**
(`s_thetaS.03/side_0200`, `s_thetaS.06/side_0400`). So θs must stay tiny
*with this return rule* — which is the wrong rule for putty.
PENDING: von Mises return on the deviatoric Hencky strain (volume-preserving
plastic flow, `--plastic 1 --yield 0.02`, `ref/p1_*`).

**(b) Not enough material for a bank (spec geometry).** A sheet of thickness t
around the front roller holds `2πR·t = 2.0·t` per unit roller length; the
bank cross-section is `0.5 × 0.22 × 0.72 ≈ 0.08`. With gap 0.05 a gap-thick
ring alone needs 0.10 > 0.08, so even without compaction the steady state is a
ring and no bank. Options (lead's call): `bankHeight` 0.22 → 0.45 (2×
particles) and/or default gap 0.05 → 0.03 (gap 0.03 at low: 58k particles,
clean, `ref/s_gap.03`, but 1 cell wide). PENDING: `ref/tall_bank`.

## 6. Gap change (0.05 → 0.02 at frame 120, `ref/gapchange/`)

1,082 particles are inside a roller the moment the axes move; all are
projected out in the same frame by `pushOutOfRollers`, max‖v‖ stays ≤ 1.5, no
spray from the squeeze itself, the sheet re-forms thinner within 20 frames
(`side_0140`). Clean. `pushOut` hits stay elevated (60–330/frame) while the
old thick sheet is squeezed through the narrower nip.

## 7. Dispersion (k = 0.6, blue at frame 40, yellow at 60, radius 0.09, `ref/disperse/`, `ref/disperse_tack2/`)

The blobs are 640 / 300 particles (1.7% of the material). Blue stays a
coherent blob inside the rigid bank (no shear → no mixing ✓, `side_0160`);
yellow, injected on the front side of the bank top, is dragged into the nip
first. In the sheet every pigmented particle is sheared (k·γ·frameDt ≈
0.03–0.06 per frame) against a 98%-white neighbourhood, so it is **diluted to
white, not mixed to green**: green-dominant sheet particles peak at 6% of the
pigmented ones around 3–4 s, and by 5 s nothing is blue-, yellow- or
green-dominant any more; the pigmented count decays (1130 → 730). The §11
green test cannot pass with two small separated blobs in a white bank — the
kernel behaves as specified, the test setup does not. PENDING: blobs of radius
0.15 at the same x (`ref/disp_big`, `ref/p1_y.02_disp`).

## 8. Performance (4 threads, this container; the numbers are contaminated by the other agent's Chromium load — best quiet frames quoted)

| cells | particles | ms/frame (8 / 12 substeps) | of which P2G / G2P |
| --- | --- | --- | --- |
| 32 | 60,499 | ~113 | 22 / 80 |
| 48 | 205,316 | ~660 | 114 / 415 |

G2P dominates because of the Jacobi SVD (one per particle per substep; the
polar rotation R is cached from it for the next P2G, so there is exactly one
SVD per substep). ~1.2 µs per particle-substep total on CPU; a GPU at 1e9
particle-substeps/s makes `high` (484k × 16) ≈ 8 ms.

## 9. Recommended changes (spec → recommended)

| item | spec | recommended | why |
| --- | --- | --- | --- |
| P2G force term | `−dt·pVol·(4/h²)·σ`, pVol = h³/8 | `−dt·(4/h²)·σ` (ρ₀ = 1) | §1: literal spec is 2.6e5× too soft, bank free-falls |
| E, ν | 60, 0.35 | keep | with ρ₀ = 1 the bank holds and feeds; c = 9.8 |
| θc | 0.025 | keep | compression yield; no puddle |
| θs | 0.0075 | PENDING (0.03–0.06) | cohesion, kills the re-entry spray |
| dt presets | 0.4·h/√E | keep | measured limit 0.8·h/√E → factor 2 |
| tackBand | 1.5h, full stick | `max(2h, 0.6·gap)`, full stick | 1.5h marginal (fails at g = 4, fuzz at cells 48); 1.0h fails; blending fails |
| back roller test | d < R + 0.5h | keep | sheds correctly |
| floor | v.y = 0, tangential × 0.4 | keep | never exercised in a healthy run |
| bankHeight / gap | 0.22 / 0.05 | PENDING (0.45 / 0.03) | §5a: ring volume > bank volume |
| §11 stability check | NaN count | add max‖v‖ < 2ωR and none below y = 3h | §2: unstable runs are NaN-free |
| §11 green test | two blobs r = 0.09 | same (x,z), r ≥ 0.15 | §7: small blobs dilute to white |
| `.gitignore` | — | add `build/` | screenshots/logs |

## 10. Key images

- `ref/spec_literal/side_0030.png`, `side_0060.png` — literal spec: free fall, dust.
- `ref/rho1/side_0000.png`, `side_0060.png`, `side_0160.png`, `side_0240.png` — ρ₀ = 1: bank, sheet, closed ring, re-entry.
- `ref/tack_1.0/side_0080.png` — 1.0h: sheet peels off the underside.
- `ref/g4/side_0240.png` — 1.5h at g = 4: sheet on the tray.
- `ref/gapchange/side_0140.png` — after the 0.05 → 0.02 gap change.
- `ref/disperse/side_0480.png` — 6 s: the bank is consumed into the ring.
- `ref/cells48/side_0240.png` — medium preset, 1.5h: fuzz and spray.

## 11. Known issues / caveats of the reference

- Threaded P2G uses per-thread grids reduced in fixed order: deterministic for
  a fixed thread count, bit-different across thread counts.
- `Stats` samples σ(F) on every 64th particle; J is exact for all.
- Stripes in the front view are the 2-per-cell seeding lattice seen edge-on
  (the bank never deforms along x); the raster will smooth them.
- Kinematic particles (§6) are skipped in P2G/G2P as specified but no
  cut-and-fold script is implemented.

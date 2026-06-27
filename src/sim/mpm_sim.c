#include "mpm_sim.h"

#include <math.h>
#include <string.h>

/* --- simulation constants (mpm88-style normalized [0,1] domain) ------------ */

#define MPM_DX (1.0f / (float)MPM_GRID)
#define MPM_INV_DX ((float)MPM_GRID)
#define MPM_DT_SUB 6.0e-4f          /* stable substep for E below */
#define MPM_MAX_SUBSTEPS 48
#define MPM_E 130.0f                /* bulk stiffness */
#define MPM_P_RHO 1.0f
#define MPM_GRAVITY 6.0f            /* +y is downward on screen */
#define MPM_OMEGA_MAX 6.0f          /* roller angular velocity at full speed */
#define MPM_BOUND 3                 /* wall thickness in cells */
#define MPM_SEED_PER_CELL 4.0f      /* particles per cell when filled */

static const float MPM_P_VOL = (MPM_DX * 0.5f) * (MPM_DX * 0.5f);

static float Clamp(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float Clamp01(float v) {
    return Clamp(v, 0.0f, 1.0f);
}

static int NodeIndex(int i, int j) {
    return i * MPM_NODES + j;
}

static void Rasterize(MpmSim *sim);

int MpmSim_Index(int x, int y) {
    return y * MPM_GRID + x;
}

static bool InsideRoller(const MpmSim *sim, float px, float py,
                         float cx, float r) {
    const float dx = px - cx;
    const float dy = py - sim->rollerCy;
    return dx * dx + dy * dy < r * r;
}

static void UpdateRollerGeometry(MpmSim *sim) {
    sim->rollerRadius = 0.13f;
    const float halfGap = 0.025f + sim->gap * 0.08f;
    sim->rollerCy = 0.70f;
    sim->leftCx = 0.5f - halfGap - sim->rollerRadius;
    sim->rightCx = 0.5f + halfGap + sim->rollerRadius;
}

/* Quadratic B-spline weights for a particle offset fx in [0.5, 1.5). */
static void QuadWeights(float fx, float w[3]) {
    const float a = 1.5f - fx;
    const float b = fx - 1.0f;
    const float c = fx - 0.5f;
    w[0] = 0.5f * a * a;
    w[1] = 0.75f - b * b;
    w[2] = 0.5f * c * c;
}

void MpmSim_Reset(MpmSim *sim) {
    sim->rollerSpeed = 0.85f;
    sim->gap = 0.26f;
    sim->rollerAngle = 0.0f;
    sim->paused = false;
    UpdateRollerGeometry(sim);

    /* Seed a pool of material in the lower domain, two color halves, skipping
       the interiors of the rollers. */
    const int perAxis = (int)sqrtf(MPM_SEED_PER_CELL); /* 2 -> 4 per cell */
    int count = 0;
    for (int cy = (int)(0.42f * MPM_GRID); cy < MPM_GRID - MPM_BOUND; cy++) {
        for (int cx = MPM_BOUND; cx < MPM_GRID - MPM_BOUND; cx++) {
            for (int sy = 0; sy < perAxis; sy++) {
                for (int sx = 0; sx < perAxis; sx++) {
                    if (count >= MPM_MAX_PARTICLES) break;
                    const float px = (cx + (sx + 0.5f) / perAxis) * MPM_DX;
                    const float py = (cy + (sy + 0.5f) / perAxis) * MPM_DX;
                    if (InsideRoller(sim, px, py, sim->leftCx, sim->rollerRadius)) continue;
                    if (InsideRoller(sim, px, py, sim->rightCx, sim->rollerRadius)) continue;

                    MpmParticle *p = &sim->particles[count++];
                    p->x = px;
                    p->y = py;
                    p->vx = 0.0f;
                    p->vy = 0.0f;
                    p->c00 = p->c01 = p->c10 = p->c11 = 0.0f;
                    p->J = 1.0f;
                    p->redFrac = px < 0.5f ? 1.0f : 0.0f;
                    p->blueFrac = px < 0.5f ? 0.0f : 1.0f;
                }
            }
        }
    }
    sim->particleCount = count;
    Rasterize(sim);
}

void MpmSim_Init(MpmSim *sim) {
    memset(sim, 0, sizeof(*sim));
    MpmSim_Reset(sim);
}

void MpmSim_AddPigment(MpmSim *sim, float normalizedX, float normalizedY,
                       float red, float blue, float radius) {
    const float r2 = radius * radius;
    for (int i = 0; i < sim->particleCount; i++) {
        MpmParticle *p = &sim->particles[i];
        const float dx = p->x - normalizedX;
        const float dy = p->y - normalizedY;
        const float d2 = dx * dx + dy * dy;
        if (d2 > r2) continue;
        const float t = 1.0f - sqrtf(d2 / r2);
        p->redFrac = Clamp01(p->redFrac * (1.0f - t) + red * t);
        p->blueFrac = Clamp01(p->blueFrac * (1.0f - t) + blue * t);
    }
}

static void Substep(MpmSim *sim, float dt) {
    memset(sim->gvx, 0, sizeof(sim->gvx));
    memset(sim->gvy, 0, sizeof(sim->gvy));
    memset(sim->gm, 0, sizeof(sim->gm));

    /* --- P2G: scatter particle mass and momentum to the grid --------------- */
    const float pMass = MPM_P_VOL * MPM_P_RHO;
    for (int pi = 0; pi < sim->particleCount; pi++) {
        MpmParticle *p = &sim->particles[pi];
        const float gx = p->x * MPM_INV_DX;
        const float gy = p->y * MPM_INV_DX;
        const int baseX = (int)floorf(gx - 0.5f);
        const int baseY = (int)floorf(gy - 0.5f);
        const float fx = gx - (float)baseX;
        const float fy = gy - (float)baseY;
        float wx[3], wy[3];
        QuadWeights(fx, wx);
        QuadWeights(fy, wy);

        /* fluid stress: isotropic pressure from volume ratio J */
        const float stress = -dt * MPM_P_VOL * (p->J - 1.0f) * 4.0f *
                             MPM_INV_DX * MPM_INV_DX * MPM_E;
        const float a00 = stress + pMass * p->c00;
        const float a01 = pMass * p->c01;
        const float a10 = pMass * p->c10;
        const float a11 = stress + pMass * p->c11;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const float dposX = ((float)i - fx) * MPM_DX;
                const float dposY = ((float)j - fy) * MPM_DX;
                const float weight = wx[i] * wy[j];
                const float pvx = pMass * p->vx + (a00 * dposX + a01 * dposY);
                const float pvy = pMass * p->vy + (a10 * dposX + a11 * dposY);
                const int n = NodeIndex(baseX + i, baseY + j);
                sim->gvx[n] += weight * pvx;
                sim->gvy[n] += weight * pvy;
                sim->gm[n] += weight * pMass;
            }
        }
    }

    /* --- grid update: velocity, gravity, rollers, walls -------------------- */
    const float omega = sim->rollerSpeed * MPM_OMEGA_MAX;
    for (int i = 0; i < MPM_NODES; i++) {
        for (int j = 0; j < MPM_NODES; j++) {
            const int n = NodeIndex(i, j);
            const float m = sim->gm[n];
            if (m <= 0.0f) continue;

            float vx = sim->gvx[n] / m;
            float vy = sim->gvy[n] / m;
            vy += dt * MPM_GRAVITY;

            /* rollers: rigid rotation overrides velocity inside each disk.
               Counter-rotating so both inner faces sweep down into the nip. */
            const float px = (float)i * MPM_DX;
            const float py = (float)j * MPM_DX;
            if (InsideRoller(sim, px, py, sim->leftCx, sim->rollerRadius)) {
                vx = -omega * (py - sim->rollerCy);
                vy = omega * (px - sim->leftCx);
            } else if (InsideRoller(sim, px, py, sim->rightCx, sim->rollerRadius)) {
                vx = omega * (py - sim->rollerCy);
                vy = -omega * (px - sim->rightCx);
            }

            /* domain walls: zero the outward normal component (free slip) */
            if (i < MPM_BOUND && vx < 0.0f) vx = 0.0f;
            if (i > MPM_GRID - MPM_BOUND && vx > 0.0f) vx = 0.0f;
            if (j < MPM_BOUND && vy < 0.0f) vy = 0.0f;
            if (j > MPM_GRID - MPM_BOUND && vy > 0.0f) vy = 0.0f;

            sim->gvx[n] = vx;
            sim->gvy[n] = vy;
        }
    }

    /* --- G2P: gather velocity + affine, advect ----------------------------- */
    const float lo = (float)MPM_BOUND * MPM_DX;
    const float hi = 1.0f - (float)MPM_BOUND * MPM_DX;
    for (int pi = 0; pi < sim->particleCount; pi++) {
        MpmParticle *p = &sim->particles[pi];
        const float gx = p->x * MPM_INV_DX;
        const float gy = p->y * MPM_INV_DX;
        const int baseX = (int)floorf(gx - 0.5f);
        const int baseY = (int)floorf(gy - 0.5f);
        const float fx = gx - (float)baseX;
        const float fy = gy - (float)baseY;
        float wx[3], wy[3];
        QuadWeights(fx, wx);
        QuadWeights(fy, wy);

        float nvx = 0.0f, nvy = 0.0f;
        float nc00 = 0.0f, nc01 = 0.0f, nc10 = 0.0f, nc11 = 0.0f;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const float dposX = ((float)i - fx) * MPM_DX;
                const float dposY = ((float)j - fy) * MPM_DX;
                const float weight = wx[i] * wy[j];
                const int n = NodeIndex(baseX + i, baseY + j);
                const float gvx = sim->gvx[n];
                const float gvy = sim->gvy[n];
                nvx += weight * gvx;
                nvy += weight * gvy;
                const float s = 4.0f * MPM_INV_DX * weight;
                nc00 += s * gvx * dposX;
                nc01 += s * gvx * dposY;
                nc10 += s * gvy * dposX;
                nc11 += s * gvy * dposY;
            }
        }

        p->vx = nvx;
        p->vy = nvy;
        p->c00 = nc00;
        p->c01 = nc01;
        p->c10 = nc10;
        p->c11 = nc11;
        p->J *= 1.0f + dt * (nc00 + nc11);
        p->x += dt * nvx;
        p->y += dt * nvy;
        p->x = Clamp(p->x, lo, hi);
        p->y = Clamp(p->y, lo, hi);
    }
}

static void Rasterize(MpmSim *sim) {
    memset(sim->rMass, 0, sizeof(sim->rMass));
    memset(sim->rRed, 0, sizeof(sim->rRed));
    memset(sim->rBlue, 0, sizeof(sim->rBlue));

    /* bilinear splat of each particle into the cell-centered render grid */
    for (int pi = 0; pi < sim->particleCount; pi++) {
        const MpmParticle *p = &sim->particles[pi];
        const float cx = p->x * MPM_GRID - 0.5f;
        const float cy = p->y * MPM_GRID - 0.5f;
        int x0 = (int)floorf(cx);
        int y0 = (int)floorf(cy);
        const float tx = cx - (float)x0;
        const float ty = cy - (float)y0;
        for (int dy = 0; dy <= 1; dy++) {
            for (int dx = 0; dx <= 1; dx++) {
                const int x = x0 + dx;
                const int y = y0 + dy;
                if (x < 0 || y < 0 || x >= MPM_GRID || y >= MPM_GRID) continue;
                const float w = (dx ? tx : 1.0f - tx) * (dy ? ty : 1.0f - ty);
                const int idx = MpmSim_Index(x, y);
                sim->rMass[idx] += w;
                sim->rRed[idx] += w * p->redFrac;
                sim->rBlue[idx] += w * p->blueFrac;
            }
        }
    }

    /* sample grid node velocity into cells for the debug velocity view */
    for (int y = 0; y < MPM_GRID; y++) {
        for (int x = 0; x < MPM_GRID; x++) {
            const int n = NodeIndex(x, y);
            sim->rVx[MpmSim_Index(x, y)] = sim->gvx[n];
            sim->rVy[MpmSim_Index(x, y)] = sim->gvy[n];
        }
    }
}

void MpmSim_Step(MpmSim *sim, float dt) {
    if (sim->paused) return;

    UpdateRollerGeometry(sim);

    float remaining = dt > 0.05f ? 0.05f : dt;
    int substeps = (int)(remaining / MPM_DT_SUB);
    if (substeps < 1) substeps = 1;
    if (substeps > MPM_MAX_SUBSTEPS) substeps = MPM_MAX_SUBSTEPS;

    for (int s = 0; s < substeps; s++) {
        Substep(sim, MPM_DT_SUB);
    }
    sim->rollerAngle += sim->rollerSpeed * MPM_OMEGA_MAX * (float)substeps * MPM_DT_SUB;

    Rasterize(sim);
}

float MpmSim_MaterialAt(const MpmSim *sim, int x, int y) {
    return Clamp01(sim->rMass[MpmSim_Index(x, y)] / MPM_SEED_PER_CELL);
}

float MpmSim_RedAt(const MpmSim *sim, int x, int y) {
    return sim->rRed[MpmSim_Index(x, y)] / MPM_SEED_PER_CELL;
}

float MpmSim_BlueAt(const MpmSim *sim, int x, int y) {
    return sim->rBlue[MpmSim_Index(x, y)] / MPM_SEED_PER_CELL;
}

float MpmSim_VelocityXAt(const MpmSim *sim, int x, int y) {
    return sim->rVx[MpmSim_Index(x, y)];
}

float MpmSim_VelocityYAt(const MpmSim *sim, int x, int y) {
    return sim->rVy[MpmSim_Index(x, y)];
}

void MpmSim_Rollers(const MpmSim *sim, float *leftCx, float *rightCx,
                    float *centerY, float *radius) {
    *leftCx = sim->leftCx;
    *rightCx = sim->rightCx;
    *centerY = sim->rollerCy;
    *radius = sim->rollerRadius;
}

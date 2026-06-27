#include "mpm3d_sim.h"

#include <math.h>
#include <string.h>

#define MPM3D_DX (1.0f / (float)MPM3D_GRID)
#define MPM3D_INV_DX ((float)MPM3D_GRID)
#define MPM3D_DT_SUB 1.4e-3f
#define MPM3D_MAX_SUBSTEPS 24
#define MPM3D_E 50.0f
#define MPM3D_P_RHO 1.0f
#define MPM3D_GRAVITY 4.0f
#define MPM3D_OMEGA_MAX 6.0f
#define MPM3D_BOUND 3
#define MPM3D_SEED_PER_AXIS 2          /* 2^3 = 8 particles per filled cell */

/* interior container (trough) keeps the pool deep enough to submerge rollers */
#define MPM3D_CXMIN 0.25f
#define MPM3D_CXMAX 0.75f
#define MPM3D_CZMIN 0.30f
#define MPM3D_CZMAX 0.70f

static const float MPM3D_P_VOL =
    (MPM3D_DX * 0.5f) * (MPM3D_DX * 0.5f) * (MPM3D_DX * 0.5f);

static float Clamp(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float Clamp01(float v) {
    return Clamp(v, 0.0f, 1.0f);
}

static int NodeIndex(int i, int j, int k) {
    return (i * MPM3D_NODES + j) * MPM3D_NODES + k;
}

int MpmSim3D_Index(int x, int y, int z) {
    return (z * MPM3D_GRID + y) * MPM3D_GRID + x;
}

static void Rasterize(MpmSim3D *sim);

static bool InsideRoller(const MpmSim3D *sim, float px, float py,
                         float cx, float r) {
    const float dx = px - cx;
    const float dy = py - sim->rollerCy;
    return dx * dx + dy * dy < r * r;  /* cylinder: ignore z */
}

static void UpdateRollerGeometry(MpmSim3D *sim) {
    sim->rollerRadius = 0.13f;
    const float halfGap = 0.025f + sim->gap * 0.08f;
    sim->rollerCy = 0.62f;
    sim->leftCx = 0.5f - halfGap - sim->rollerRadius;
    sim->rightCx = 0.5f + halfGap + sim->rollerRadius;
}

static void QuadWeights(float fx, float w[3]) {
    const float a = 1.5f - fx;
    const float b = fx - 1.0f;
    const float c = fx - 0.5f;
    w[0] = 0.5f * a * a;
    w[1] = 0.75f - b * b;
    w[2] = 0.5f * c * c;
}

void MpmSim3D_Reset(MpmSim3D *sim) {
    sim->rollerSpeed = 0.85f;
    sim->gap = 0.26f;
    sim->rollerAngle = 0.0f;
    sim->paused = false;
    UpdateRollerGeometry(sim);

    const int per = MPM3D_SEED_PER_AXIS;
    int count = 0;
    for (int cz = (int)(0.32f * MPM3D_GRID); cz < (int)(0.68f * MPM3D_GRID); cz++) {
        for (int cy = (int)(0.40f * MPM3D_GRID); cy < (int)(0.84f * MPM3D_GRID); cy++) {
            for (int cx = (int)(0.27f * MPM3D_GRID); cx < (int)(0.73f * MPM3D_GRID); cx++) {
                for (int sz = 0; sz < per; sz++) {
                    for (int sy = 0; sy < per; sy++) {
                        for (int sx = 0; sx < per; sx++) {
                            if (count >= MPM3D_MAX_PARTICLES) break;
                            const float px = (cx + (sx + 0.5f) / per) * MPM3D_DX;
                            const float py = (cy + (sy + 0.5f) / per) * MPM3D_DX;
                            const float pz = (cz + (sz + 0.5f) / per) * MPM3D_DX;
                            if (InsideRoller(sim, px, py, sim->leftCx, sim->rollerRadius)) continue;
                            if (InsideRoller(sim, px, py, sim->rightCx, sim->rollerRadius)) continue;

                            MpmParticle3D *p = &sim->particles[count++];
                            p->x = px; p->y = py; p->z = pz;
                            p->vx = p->vy = p->vz = 0.0f;
                            for (int m = 0; m < 9; m++) p->C[m] = 0.0f;
                            p->J = 1.0f;
                            p->redFrac = px < 0.5f ? 1.0f : 0.0f;
                            p->blueFrac = px < 0.5f ? 0.0f : 1.0f;
                        }
                    }
                }
            }
        }
    }
    sim->particleCount = count;
    Rasterize(sim);
}

void MpmSim3D_Init(MpmSim3D *sim) {
    memset(sim, 0, sizeof(*sim));
    MpmSim3D_Reset(sim);
}

void MpmSim3D_AddPigment(MpmSim3D *sim, float nx, float ny, float nz,
                         float red, float blue, float radius) {
    const float r2 = radius * radius;
    for (int i = 0; i < sim->particleCount; i++) {
        MpmParticle3D *p = &sim->particles[i];
        const float dx = p->x - nx, dy = p->y - ny, dz = p->z - nz;
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > r2) continue;
        const float t = 1.0f - sqrtf(d2 / r2);
        p->redFrac = Clamp01(p->redFrac * (1.0f - t) + red * t);
        p->blueFrac = Clamp01(p->blueFrac * (1.0f - t) + blue * t);
    }
}

static void Substep(MpmSim3D *sim, float dt) {
    memset(sim->gvx, 0, sizeof(sim->gvx));
    memset(sim->gvy, 0, sizeof(sim->gvy));
    memset(sim->gvz, 0, sizeof(sim->gvz));
    memset(sim->gm, 0, sizeof(sim->gm));

    const float pMass = MPM3D_P_VOL * MPM3D_P_RHO;

    /* --- P2G ------------------------------------------------------------- */
    for (int pi = 0; pi < sim->particleCount; pi++) {
        MpmParticle3D *p = &sim->particles[pi];
        const float gx = p->x * MPM3D_INV_DX;
        const float gy = p->y * MPM3D_INV_DX;
        const float gz = p->z * MPM3D_INV_DX;
        const int bx = (int)floorf(gx - 0.5f);
        const int by = (int)floorf(gy - 0.5f);
        const int bz = (int)floorf(gz - 0.5f);
        const float fx = gx - (float)bx, fy = gy - (float)by, fz = gz - (float)bz;
        float wx[3], wy[3], wz[3];
        QuadWeights(fx, wx); QuadWeights(fy, wy); QuadWeights(fz, wz);

        const float stress = -dt * MPM3D_P_VOL * (p->J - 1.0f) * 4.0f *
                             MPM3D_INV_DX * MPM3D_INV_DX * MPM3D_E;
        /* affine = stress*I + pMass*C */
        const float a0 = stress + pMass * p->C[0], a1 = pMass * p->C[1], a2 = pMass * p->C[2];
        const float a3 = pMass * p->C[3], a4 = stress + pMass * p->C[4], a5 = pMass * p->C[5];
        const float a6 = pMass * p->C[6], a7 = pMass * p->C[7], a8 = stress + pMass * p->C[8];

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    const float dpx = ((float)i - fx) * MPM3D_DX;
                    const float dpy = ((float)j - fy) * MPM3D_DX;
                    const float dpz = ((float)k - fz) * MPM3D_DX;
                    const float weight = wx[i] * wy[j] * wz[k];
                    const float pvx = pMass * p->vx + (a0 * dpx + a1 * dpy + a2 * dpz);
                    const float pvy = pMass * p->vy + (a3 * dpx + a4 * dpy + a5 * dpz);
                    const float pvz = pMass * p->vz + (a6 * dpx + a7 * dpy + a8 * dpz);
                    const int n = NodeIndex(bx + i, by + j, bz + k);
                    sim->gvx[n] += weight * pvx;
                    sim->gvy[n] += weight * pvy;
                    sim->gvz[n] += weight * pvz;
                    sim->gm[n] += weight * pMass;
                }
            }
        }
    }

    /* --- grid update ----------------------------------------------------- */
    const float omega = sim->rollerSpeed * MPM3D_OMEGA_MAX;
    for (int i = 0; i < MPM3D_NODES; i++) {
        const float px = (float)i * MPM3D_DX;
        for (int j = 0; j < MPM3D_NODES; j++) {
            const float py = (float)j * MPM3D_DX;
            for (int k = 0; k < MPM3D_NODES; k++) {
                const float pz = (float)k * MPM3D_DX;
                const int n = NodeIndex(i, j, k);
                const float m = sim->gm[n];
                if (m <= 0.0f) continue;

                float vx = sim->gvx[n] / m;
                float vy = sim->gvy[n] / m + dt * MPM3D_GRAVITY;
                float vz = sim->gvz[n] / m;

                if (InsideRoller(sim, px, py, sim->leftCx, sim->rollerRadius)) {
                    vx = -omega * (py - sim->rollerCy);
                    vy = omega * (px - sim->leftCx);
                    vz = 0.0f;
                } else if (InsideRoller(sim, px, py, sim->rightCx, sim->rollerRadius)) {
                    vx = omega * (py - sim->rollerCy);
                    vy = -omega * (px - sim->rightCx);
                    vz = 0.0f;
                }

                /* domain floor + interior container (trough) walls */
                if (j > MPM3D_GRID - MPM3D_BOUND && vy > 0.0f) vy = 0.0f;
                if (px < MPM3D_CXMIN && vx < 0.0f) vx = 0.0f;
                if (px > MPM3D_CXMAX && vx > 0.0f) vx = 0.0f;
                if (pz < MPM3D_CZMIN && vz < 0.0f) vz = 0.0f;
                if (pz > MPM3D_CZMAX && vz > 0.0f) vz = 0.0f;

                sim->gvx[n] = vx;
                sim->gvy[n] = vy;
                sim->gvz[n] = vz;
            }
        }
    }

    /* --- G2P ------------------------------------------------------------- */
    const float lo = (float)MPM3D_BOUND * MPM3D_DX;
    const float hi = 1.0f - (float)MPM3D_BOUND * MPM3D_DX;
    for (int pi = 0; pi < sim->particleCount; pi++) {
        MpmParticle3D *p = &sim->particles[pi];
        const float gx = p->x * MPM3D_INV_DX;
        const float gy = p->y * MPM3D_INV_DX;
        const float gz = p->z * MPM3D_INV_DX;
        const int bx = (int)floorf(gx - 0.5f);
        const int by = (int)floorf(gy - 0.5f);
        const int bz = (int)floorf(gz - 0.5f);
        const float fx = gx - (float)bx, fy = gy - (float)by, fz = gz - (float)bz;
        float wx[3], wy[3], wz[3];
        QuadWeights(fx, wx); QuadWeights(fy, wy); QuadWeights(fz, wz);

        float nvx = 0, nvy = 0, nvz = 0;
        float nc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    const float dpx = ((float)i - fx) * MPM3D_DX;
                    const float dpy = ((float)j - fy) * MPM3D_DX;
                    const float dpz = ((float)k - fz) * MPM3D_DX;
                    const float weight = wx[i] * wy[j] * wz[k];
                    const int n = NodeIndex(bx + i, by + j, bz + k);
                    const float gvx = sim->gvx[n], gvy = sim->gvy[n], gvz = sim->gvz[n];
                    nvx += weight * gvx;
                    nvy += weight * gvy;
                    nvz += weight * gvz;
                    const float s = 4.0f * MPM3D_INV_DX * weight;
                    nc[0] += s * gvx * dpx; nc[1] += s * gvx * dpy; nc[2] += s * gvx * dpz;
                    nc[3] += s * gvy * dpx; nc[4] += s * gvy * dpy; nc[5] += s * gvy * dpz;
                    nc[6] += s * gvz * dpx; nc[7] += s * gvz * dpy; nc[8] += s * gvz * dpz;
                }
            }
        }

        p->vx = nvx; p->vy = nvy; p->vz = nvz;
        for (int m = 0; m < 9; m++) p->C[m] = nc[m];
        p->J *= 1.0f + dt * (nc[0] + nc[4] + nc[8]);
        p->x = Clamp(p->x + dt * nvx, MPM3D_CXMIN, MPM3D_CXMAX);
        p->y = Clamp(p->y + dt * nvy, lo, hi);
        p->z = Clamp(p->z + dt * nvz, MPM3D_CZMIN, MPM3D_CZMAX);
    }
}

static void Rasterize(MpmSim3D *sim) {
    memset(sim->rMass, 0, sizeof(sim->rMass));
    memset(sim->rRed, 0, sizeof(sim->rRed));
    memset(sim->rBlue, 0, sizeof(sim->rBlue));

    for (int pi = 0; pi < sim->particleCount; pi++) {
        const MpmParticle3D *p = &sim->particles[pi];
        const float cx = p->x * MPM3D_GRID - 0.5f;
        const float cy = p->y * MPM3D_GRID - 0.5f;
        const float cz = p->z * MPM3D_GRID - 0.5f;
        const int x0 = (int)floorf(cx), y0 = (int)floorf(cy), z0 = (int)floorf(cz);
        const float tx = cx - (float)x0, ty = cy - (float)y0, tz = cz - (float)z0;
        for (int dz = 0; dz <= 1; dz++) {
            for (int dy = 0; dy <= 1; dy++) {
                for (int dx = 0; dx <= 1; dx++) {
                    const int x = x0 + dx, y = y0 + dy, z = z0 + dz;
                    if (x < 0 || y < 0 || z < 0 ||
                        x >= MPM3D_GRID || y >= MPM3D_GRID || z >= MPM3D_GRID) continue;
                    const float w = (dx ? tx : 1.0f - tx) *
                                    (dy ? ty : 1.0f - ty) *
                                    (dz ? tz : 1.0f - tz);
                    const int idx = MpmSim3D_Index(x, y, z);
                    sim->rMass[idx] += w;
                    sim->rRed[idx] += w * p->redFrac;
                    sim->rBlue[idx] += w * p->blueFrac;
                }
            }
        }
    }
}

void MpmSim3D_Step(MpmSim3D *sim, float dt) {
    if (sim->paused) return;

    UpdateRollerGeometry(sim);

    float remaining = dt > 0.05f ? 0.05f : dt;
    int substeps = (int)(remaining / MPM3D_DT_SUB);
    if (substeps < 1) substeps = 1;
    if (substeps > MPM3D_MAX_SUBSTEPS) substeps = MPM3D_MAX_SUBSTEPS;

    for (int s = 0; s < substeps; s++) {
        Substep(sim, MPM3D_DT_SUB);
    }
    sim->rollerAngle += sim->rollerSpeed * MPM3D_OMEGA_MAX * (float)substeps * MPM3D_DT_SUB;

    Rasterize(sim);
}

float MpmSim3D_MaterialAt(const MpmSim3D *sim, int x, int y, int z) {
    return Clamp01(sim->rMass[MpmSim3D_Index(x, y, z)] /
                   (float)(MPM3D_SEED_PER_AXIS * MPM3D_SEED_PER_AXIS * MPM3D_SEED_PER_AXIS));
}

float MpmSim3D_RedAt(const MpmSim3D *sim, int x, int y, int z) {
    return sim->rRed[MpmSim3D_Index(x, y, z)] /
           (float)(MPM3D_SEED_PER_AXIS * MPM3D_SEED_PER_AXIS * MPM3D_SEED_PER_AXIS);
}

float MpmSim3D_BlueAt(const MpmSim3D *sim, int x, int y, int z) {
    return sim->rBlue[MpmSim3D_Index(x, y, z)] /
           (float)(MPM3D_SEED_PER_AXIS * MPM3D_SEED_PER_AXIS * MPM3D_SEED_PER_AXIS);
}

void MpmSim3D_Rollers(const MpmSim3D *sim, float *leftCx, float *rightCx,
                      float *centerY, float *radius) {
    *leftCx = sim->leftCx;
    *rightCx = sim->rightCx;
    *centerY = sim->rollerCy;
    *radius = sim->rollerRadius;
}

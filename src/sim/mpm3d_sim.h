#ifndef MPM3D_SIM_H
#define MPM3D_SIM_H

#include <stdbool.h>

/*
 * 3D MLS-MPM material simulation for ColorMill.
 *
 * The 3D extension of mpm_sim: particles carry position, velocity, a 3x3 APIC
 * affine matrix, a volume ratio, and red/blue pigment. A background voxel grid
 * is scratch, rebuilt every substep (P2G -> grid solve -> G2P -> advect) with a
 * 3x3x3 quadratic B-spline stencil. The two rollers are counter-rotating
 * cylinders (axis along z) submerged in a pool of material; their inner faces
 * sweep down into the nip and knead the colored material, which folds and mixes
 * by transport and is rendered through Mixbox.
 *
 * References: Hu et al., MLS-MPM (SIGGRAPH 2018); Jiang et al., APIC (2015).
 */

#define MPM3D_GRID 32                          /* cubic sim/render grid (cells) */
#define MPM3D_NODES (MPM3D_GRID + 1)
#define MPM3D_CELLS (MPM3D_GRID * MPM3D_GRID * MPM3D_GRID)
#define MPM3D_NODE_COUNT (MPM3D_NODES * MPM3D_NODES * MPM3D_NODES)
#define MPM3D_MAX_PARTICLES 30000

typedef struct MpmParticle3D {
    float x, y, z;            /* position, normalized [0,1]^3 */
    float vx, vy, vz;         /* velocity */
    float C[9];               /* APIC affine velocity matrix, row-major 3x3 */
    float J;                  /* volume ratio */
    float redFrac, blueFrac;  /* carried pigment concentration, 0..1 */
} MpmParticle3D;

typedef struct MpmSim3D {
    MpmParticle3D particles[MPM3D_MAX_PARTICLES];
    int particleCount;

    float gvx[MPM3D_NODE_COUNT];
    float gvy[MPM3D_NODE_COUNT];
    float gvz[MPM3D_NODE_COUNT];
    float gm[MPM3D_NODE_COUNT];

    float rMass[MPM3D_CELLS];
    float rRed[MPM3D_CELLS];
    float rBlue[MPM3D_CELLS];

    float rollerSpeed;     /* 0..1, scales roller angular velocity */
    float gap;             /* 0..1, roller separation at the nip */
    float rollerAngle;
    bool paused;

    /* roller cylinders: axis along z, centers in the xy plane */
    float rollerRadius;
    float leftCx, rightCx, rollerCy;
} MpmSim3D;

void MpmSim3D_Init(MpmSim3D *sim);
void MpmSim3D_Reset(MpmSim3D *sim);
void MpmSim3D_Step(MpmSim3D *sim, float dt);
void MpmSim3D_AddPigment(MpmSim3D *sim, float nx, float ny, float nz,
                         float red, float blue, float radius);

int MpmSim3D_Index(int x, int y, int z);
float MpmSim3D_MaterialAt(const MpmSim3D *sim, int x, int y, int z);
float MpmSim3D_RedAt(const MpmSim3D *sim, int x, int y, int z);
float MpmSim3D_BlueAt(const MpmSim3D *sim, int x, int y, int z);

void MpmSim3D_Rollers(const MpmSim3D *sim, float *leftCx, float *rightCx,
                      float *centerY, float *radius);

#endif

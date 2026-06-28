#ifndef MPM_SIM_H
#define MPM_SIM_H

#include <stdbool.h>

/*
 * MLS-MPM material simulation for ColorMill (2D, side view of a two-roll mill).
 *
 * Material is carried by particles (the "material points"). A background grid
 * is scratch space rebuilt every substep: scatter particle mass/momentum to the
 * grid (P2G), solve velocity + boundary conditions on the grid, gather motion
 * back to particles (G2P, APIC transfers), then advect. Two counter-rotating
 * rollers act as rotating rigid boundaries that knead the material; each
 * particle carries red/blue pigment that folds and mixes by transport, then is
 * rendered through Mixbox (pigment-correct, not linear RGB).
 *
 * The same loop extends to 3D by adding a z axis to positions, velocities, the
 * affine matrix, and the grid.
 *
 * References: Hu et al., A Moving Least Squares Material Point Method (SIGGRAPH
 * 2018) / mpm88; Jiang et al., APIC (SIGGRAPH 2015).
 */

#define MPM_GRID 64                          /* square sim/render grid (cells) */
#define MPM_NODES (MPM_GRID + 1)
#define MPM_MAX_PARTICLES 12000

typedef struct MpmParticle {
    float x, y;            /* position, normalized [0,1] domain */
    float vx, vy;          /* velocity */
    float c00, c01, c10, c11; /* APIC affine velocity matrix C */
    float J;               /* volume ratio (weakly compressible fluid) */
    float redFrac, blueFrac;  /* carried pigment concentration, 0..1 */
} MpmParticle;

typedef struct MpmSim {
    MpmParticle particles[MPM_MAX_PARTICLES];
    int particleCount;

    /* grid nodes (scratch, rebuilt each substep) */
    float gvx[MPM_NODES * MPM_NODES];
    float gvy[MPM_NODES * MPM_NODES];
    float gm[MPM_NODES * MPM_NODES];

    /* cell-centered fields rasterized from particles for rendering */
    float rMass[MPM_GRID * MPM_GRID];
    float rRed[MPM_GRID * MPM_GRID];
    float rBlue[MPM_GRID * MPM_GRID];
    float rVx[MPM_GRID * MPM_GRID];
    float rVy[MPM_GRID * MPM_GRID];

    float rollerSpeed;     /* 0..1, scales roller angular velocity */
    float gap;             /* 0..1, roller separation at the nip */
    float rollerAngle;     /* accumulated, for drawing roller spokes */
    bool paused;

    /* roller geometry in normalized coordinates */
    float rollerRadius;
    float leftCx, rightCx, rollerCy;
} MpmSim;

void MpmSim_Init(MpmSim *sim);
void MpmSim_Reset(MpmSim *sim);
void MpmSim_Step(MpmSim *sim, float dt);
void MpmSim_AddPigment(MpmSim *sim, float normalizedX, float normalizedY,
                       float red, float blue, float radius);

int MpmSim_Index(int x, int y);
float MpmSim_MaterialAt(const MpmSim *sim, int x, int y);
float MpmSim_RedAt(const MpmSim *sim, int x, int y);
float MpmSim_BlueAt(const MpmSim *sim, int x, int y);
float MpmSim_VelocityXAt(const MpmSim *sim, int x, int y);
float MpmSim_VelocityYAt(const MpmSim *sim, int x, int y);

/* Roller geometry (normalized 0..1) for drawing overlays. */
void MpmSim_Rollers(const MpmSim *sim, float *leftCx, float *rightCx,
                    float *centerY, float *radius);

#endif

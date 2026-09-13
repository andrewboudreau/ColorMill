/*
 * ColorMill v2 - headless CPU reference of the GPU MLS-MPM two-roll mill.
 *
 * Implements docs/design-v2.md sections 1-5 and 11 literally (quadratic
 * B-splines, fixed corotated stress with a plastic singular-value clamp,
 * sticky front roller, separating Coulomb back roller, walls, floor,
 * frame-level pigment raster + shear-driven dispersion) so that the constants
 * and rules can be validated before / alongside the WebGPU implementation.
 *
 * Geometry constants mirror src/config/mill.ts exactly.  Everything the
 * investigation may want to vary is in MillRefOptions; the spec values are
 * the defaults returned by MillRef_DefaultOptions().
 */
#ifndef MILLREF_H
#define MILLREF_H

#include <stddef.h>

#define MILLREF_LATENT 7

/* --- geometry (src/config/mill.ts GEOMETRY) ------------------------------ */
#define MILLREF_DOMAIN_X 1.5f
#define MILLREF_DOMAIN_Y 1.25f
#define MILLREF_DOMAIN_Z 1.5f
#define MILLREF_LENGTH 1.5f
#define MILLREF_RADIUS 0.32f
#define MILLREF_AXIS_Y 0.55f
#define MILLREF_NIP_Z 0.75f
#define MILLREF_BANK_HALF_DEPTH 0.25f
#define MILLREF_BANK_HEIGHT 0.22f
#define MILLREF_BANK_END_MARGIN 0.05f
#define MILLREF_SEED_PER_AXIS 2

typedef struct {
    float E, nu, thetaC, thetaS;       /* design-v2 section 4 */
} MillRefMaterial;

typedef struct {
    float omega;                       /* front roller rad/s */
    float frictionRatio;               /* back / front surface speed */
    float gap;                         /* nip opening */
    float gravity;
    float dispersion;                  /* k of section 5 */
    float backFriction;                /* Coulomb mu on the back roller */
} MillRefParams;

typedef struct {
    int cells;                         /* cells per unit (h = 1/cells) */
    float dt;                          /* substep */
    int sub;                           /* substeps per frame */
    float tackBand;                    /* in units of h (spec 1.5) */
    float tackBlend;                   /* 1 = spec (v = vr in the whole band);
                                          <1 = v = mix(v, vr, tackBlend) in the band beyond R+0.5h */
    float rho;                         /* reference density used in the P2G force term.
                                          <= 0 : literal spec (pVol = h^3/8 with pMass = 1, i.e. rho = 8/h^3).
                                          1    : force term uses pVol*rhoSpec/rho = 1 (density-1 scaling). */
    float floorFriction;               /* tangential scale 1 - f on floor contact (spec 0.6) */
    float damping;                     /* extra grid velocity damping per substep (0 = spec) */
    int fixedPoint;                    /* emulate the i32 fixed-point accumulators (2^20 / 2^16) */
    int threads;                       /* OpenMP threads (0 = default) */
    unsigned seed;                     /* xorshift seed for the bank (1234 as in mill.ts) */
    int plastic;                       /* 0 = spec singular-value clamp (snow); 1 = von Mises return on the
                                          deviatoric Hencky strain (volume-preserving plastic flow, elastic volume);
                                          2 = as 1 plus the volumetric Hencky strain clamped to [ln(1-thetaC), ln(1+thetaS)] */
    float yieldStrain;                 /* plastic 1/2: deviatoric Hencky yield (norm), e.g. 0.02 */
    float jMin, jMax;                  /* plastic 2: volumetric clamp J in [jMin, jMax] (<= 0: use 1-thetaC / 1+thetaS) */
    float bankHeight;                  /* geometry experiment: bank height above the roller top (spec 0.22) */
    float bankHalfDepth;               /* geometry experiment: bank half-depth in z (spec 0.25) */
} MillRefOptions;

typedef struct {
    float x[3], v[3];
    float C[9], F[9];                  /* row-major */
    float lat[MILLREF_LATENT];
    unsigned flags;                    /* bit0 = kinematic */
    float R[9];                        /* cached polar rotation of F (reference-only) */
} MillRefParticle;

typedef struct {
    int count, nan, inDomain, inRoller;
    long pushOuts;                     /* pushOutOfRollers hits during the last frame */
    float maxV, meanV, minJ, maxJ, minSigma, maxSigma;
    float maxNodeMass;
    /* section 11 fractions (0..1) */
    float sheetFront, sheetBack, bank, floor, frontTop, backTop, inNip;
    float nipShear;                    /* mean ||sym(C)||_F in the nip region */
    int nipCount;
    /* colour: among pigmented particles (|lat - white| > 0.05) */
    int pigmented, pigmentedSheet, pigmentedBank;
    float greenSheet, greenBank, blueBank, yellowBank; /* fractions of the pigmented subsets */
    float sheetY0;                     /* lowest y of a front-sheet particle */
    float loose;                       /* fraction with gathered raster mass < 2 (packed = 8): spray / dust */
    float high;                        /* fraction above the initial bank top + h (flung material) */
    float sheetThick;                  /* 2 * mean(d_front - R) over front-sheet particles (~ sheet thickness) */
} MillRefStats;

typedef struct MillRefSim MillRefSim;

MillRefOptions MillRef_DefaultOptions(void);
MillRefMaterial MillRef_DefaultMaterial(void);
MillRefParams MillRef_DefaultParams(void);

MillRefSim *MillRef_Create(const MillRefOptions *opt, const MillRefMaterial *mat, const MillRefParams *par);
void MillRef_Destroy(MillRefSim *sim);

/* live parameters (gap change moves the roller axes) */
void MillRef_SetParams(MillRefSim *sim, const MillRefParams *par);

/* one rendered frame: sub substeps + pigment raster + dispersion */
void MillRef_Frame(MillRefSim *sim);
void MillRef_Substep(MillRefSim *sim);

/* section 5 inject kernel */
void MillRef_AddPigment(MillRefSim *sim, const float center[3], float radius,
                        const float latent[MILLREF_LATENT], float strength);

void MillRef_Stats(const MillRefSim *sim, MillRefStats *st);

/* accessors */
int MillRef_ParticleCount(const MillRefSim *sim);
const MillRefParticle *MillRef_Particles(const MillRefSim *sim);
float MillRef_H(const MillRefSim *sim);
double MillRef_Time(const MillRefSim *sim);
int MillRef_FrameIndex(const MillRefSim *sim);
void MillRef_RollerAxes(const MillRefSim *sim, float *backZ, float *frontZ);
/* timings of the last frame in ms */
void MillRef_Timings(const MillRefSim *sim, double *frameMs, double *p2gMs, double *gridMs, double *g2pMs);

/* Mixbox helpers (port of src/sim/mixbox.c) */
void MillRef_LatentToRgb(const float z[MILLREF_LATENT], float rgb[3]);
extern const float MILLREF_LAT_WHITE[MILLREF_LATENT];
extern const float MILLREF_LAT_RED[MILLREF_LATENT];
extern const float MILLREF_LAT_BLUE[MILLREF_LATENT];
extern const float MILLREF_LAT_YELLOW[MILLREF_LATENT];

/* 3x3 SVD (Jacobi on F^T F); exposed for self-tests */
void MillRef_Svd3(const float F[9], float U[9], float S[3], float V[9]);

#endif

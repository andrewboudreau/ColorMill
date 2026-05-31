#ifndef MATERIAL_SIM_H
#define MATERIAL_SIM_H

#include <stdbool.h>

#define SIM_WIDTH 128
#define SIM_HEIGHT 72
#define SIM_CELL_COUNT (SIM_WIDTH * SIM_HEIGHT)

typedef struct MaterialSim {
    float material[SIM_CELL_COUNT];
    float red[SIM_CELL_COUNT];
    float blue[SIM_CELL_COUNT];
    float vx[SIM_CELL_COUNT];
    float vy[SIM_CELL_COUNT];
    float nextMaterial[SIM_CELL_COUNT];
    float nextRed[SIM_CELL_COUNT];
    float nextBlue[SIM_CELL_COUNT];
    float rollerAngle;
    float rollerSpeed;
    float diffusion;
    float gap;
    bool paused;
} MaterialSim;

void MaterialSim_Init(MaterialSim *sim);
void MaterialSim_Reset(MaterialSim *sim);
void MaterialSim_Step(MaterialSim *sim, float dt);
void MaterialSim_AddPigment(MaterialSim *sim, float normalizedX, float normalizedY, float red, float blue, float radius);
int MaterialSim_Index(int x, int y);
float MaterialSim_MaterialAt(const MaterialSim *sim, int x, int y);
float MaterialSim_RedAt(const MaterialSim *sim, int x, int y);
float MaterialSim_BlueAt(const MaterialSim *sim, int x, int y);
float MaterialSim_VelocityXAt(const MaterialSim *sim, int x, int y);
float MaterialSim_VelocityYAt(const MaterialSim *sim, int x, int y);

#endif
